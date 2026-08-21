"""
Helper: everything derived from a Config -- env construction, save/load,
logging, timing, HPO bookkeeping, the watch_agent viewer. Reachable as
config.<name>. Also defines Timing, StartInCueView and SequenceDataset.
"""

import copy
import gc
import glob
import json
import logging
import os
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv
from minigrid.wrappers import ImgObsWrapper
from torch.utils.data import Dataset
import plotly.graph_objects as go
import minigrid  # noqa: F401
from models.feature_extractor import MLP, GRU
import joblib
from torchinfo import summary
import optuna
import pygame
from models.model import Network

# The two metrics a run reports and the study can rank on: one number per
# seed, aggregated ACROSS seeds, never across the eval episodes within a run.
_HPO_METRICS = ("return_mean", "success_rate")

# Exactly two objectives exist, both scored mean - hpo_lambda*std across seeds.
# Median/IQR were dropped: at five seeds they estimate nothing mean/std doesn't.
_HPO_OBJECTIVES = {
    "return_mean_minus_std": "return_mean",
    "success_rate_mean_minus_std": "success_rate",
}

# Columns of the closing per-seed table (Helper.log_seed_summary): "sampled"
# is the policy sampled from pi, "argmax" the same weights scored greedily.
_SEED_SUMMARY_METRICS = (
    "sampled_return",
    "sampled_success_rate",
    "argmax_return",
    "argmax_success_rate",
)


def parse_hpo_objective(objective):
    """The per-seed metric behind an hpo_objective string.

    Only the two entries of _HPO_OBJECTIVES are accepted; anything else raises
    rather than defaulting silently to a scale the rest of the reports would
    not match.

    Args:
        objective (str): "return_mean_minus-std" or
            "success_rate_mean_minus-std"; case and -/_ are normalized.

    Returns:
        str: the metric key, "return_mean" or "success_rate".

    Raises:
        ValueError: on anything else.
    """
    key = str(objective).strip().lower().replace("-", "_")
    if key not in _HPO_OBJECTIVES:
        raise ValueError(
            f"hpo_objective {objective!r} is not one of "
            f"{sorted(_HPO_OBJECTIVES)}. Both are scored "
            f"mean - hpo_lambda*std across seeds; they differ only in which "
            f"metric is aggregated."
        )
    return _HPO_OBJECTIVES[key]


# when a trial ran, in the study csv. Seconds resolution: these are wall
# clocks on runs that last minutes, and optuna's microseconds only add noise.
_CSV_TIMESTAMP = "%Y-%m-%d %H:%M:%S"


def format_clock(seconds):
    """A duration someone WAITS through: 1h 05m 03s / 5m 03s / 42.1s.

    Args:
        seconds (float): the duration.

    Returns:
        str: the formatted clock.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def format_mean(seconds):
    """A per-call mean duration, in whatever unit keeps it readable.

    Args:
        seconds (float): the mean duration.

    Returns:
        str: the value in us, ms or s.
    """
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f}us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f}ms"
    return f"{seconds:.3f}s"


def log_with(logger, level="info"):
    """The write function to use, so no call site needs a logger/print branch.

    Args:
        logger (logging.Logger | None): the run's logger, or None.
        level (str): which method to take off it -- "info", "warning", "error".

    Returns:
        callable: logger.<level>, or the builtin print when logger is None.
    """
    return print if logger is None else getattr(logger, level)


class Timing:
    """Every clock a training run keeps. The caller only names phases --
    `with timing.phase("sample"):` -- and the arithmetic/formatting lives here.
    Two sets of per-phase [seconds, units, calls] totals: `window` since the
    last report(), `run` for the whole seed; report() folds the first into the
    second so they never double-count. enabled=False makes phase() free."""

    # Phase key -> label, in print order. An unlisted phase is still
    # accumulated, just not shown; a row never timed is skipped.
    ROWS = (
        ("sample", "sample (collect)"),
        ("env_step", "  env.step() + reset()"),
        ("env_loop", "  the worker loop, all of it"),
        ("act_forward", "  act forward (no_grad)"),
        ("act_to_host", "  act -> host copy"),
        ("gae", "gae"),
        ("split_pad_mask", "split_pad_mask"),
        ("update_fwd", "update forward"),
        ("update_bwd", "update backward + step"),
    )

    # printed below the "one whole iteration" line: evaluation is the cost of
    # measuring the policy, not of training it, so it is not part of that total
    ROWS_EVAL = (
        ("evaluate", "evaluate() (report only)"),
        ("eval_forward", "  eval forward (no_grad)"),
        ("eval_step", "  eval env.step() + reset()"),
    )

    def __init__(self, device, enabled=True):
        """Start the clocks for one run.

        Args:
            device (torch.device): what to synchronize before a measurement;
                only a cuda device is ever waited on.
            enabled (bool): False makes phase() a bare yield and prints no
                table -- the master switch, config.calculate_time.
        """
        self.device = device
        self.enabled = bool(enabled)
        self.window = {}  # phase -> [seconds, units], since the last report
        self.run = {}  # phase -> [seconds, units], the whole seed
        self.start()

    def start(self):
        """(Re)start the wall clocks. Called when the training loop begins, so
        building the envs and probing the minibatch size are not counted as
        run time."""
        self.started = time.perf_counter()
        self.reported = self.started

    @property
    def elapsed(self):
        """Seconds since start()."""
        return time.perf_counter() - self.started

    def sync(self, when=True):
        """Wait for queued GPU work, so a timer measures what ran and not what
        was merely launched.

        Args:
            when (bool): False makes it a no-op -- for hot loops where the
                synchronize would cost more than the thing it is timing.
        """
        if when and self.device.type == "cuda":
            torch.cuda.synchronize()

    def add(self, phase, seconds, units=1):
        """Add one measurement to the current window.

        Args:
            phase (str): the phase key, e.g. "sample" or "update_bwd".
            seconds (float): how long this call took.
            units (int): how much work that call handled -- W env steps, one
                minibatch, one forward. 0 for a call that is not work of the
                kind being counted, e.g. an env reset among env steps.
        """
        entry = self.window.setdefault(phase, [0.0, 0, 0])
        entry[0] += seconds
        entry[1] += units
        entry[2] += 1

    @contextmanager
    def phase(self, name, units=1, sync=True):
        """Time the wrapped block and add it to `name`.

        No try/finally: a block that raised never ran to completion, so it is
        not a measurement worth keeping.

        Args:
            name (str): the phase key to add the measurement to.
            units (int): work handled inside the block, as in add().
            sync (bool): False for a region that queues no GPU work, or one
                ending in a copy that forces the sync anyway.

        Yields:
            None: the block runs in the caller's scope.
        """
        if not self.enabled:
            # not even the perf_counter calls: this wraps the innermost loops
            # (per env step, per minibatch), so "off" has to mean free
            yield
            return

        self.sync(sync)
        start = time.perf_counter()
        yield
        self.sync(sync)
        self.add(name, time.perf_counter() - start, units)

    def fold(self):
        """Add the window's totals to the run's and empty the window, so the
        two never double-count.

        Returns:
            dict: what the window held, {phase: [seconds, units, calls]}.
        """
        window = dict(self.window)
        for key, (seconds, units, calls) in window.items():
            total = self.run.setdefault(key, [0.0, 0, 0])
            total[0] += seconds
            total[1] += units
            total[2] += calls
        self.window.clear()
        return window

    def report(self, log):
        """Print the table for everything since the last report, and open a
        new window.

        The numbers are not lost, only moved: fold() keeps them in `run`.

        Args:
            log (callable): where a line goes, logger.info or print.
        """
        if not self.enabled:
            return

        window = self.fold()

        now = time.perf_counter()
        since_report, self.reported = now - self.reported, now

        # an iteration that has not finished yet has nothing to take shares of
        if "iteration" not in window:
            return

        total, _, n = window["iteration"]
        self._table(
            log,
            window,
            f"TIME  {n} iterations in {format_clock(since_report)}"
            f"  --  {format_clock(total / n)} per iteration,"
            f"  {format_clock(now - self.started)} into the run",
        )

    def summary(self, log, seed):
        """Print the table for the whole seed.

        Call it after the last report(), which is what folded the final
        window into `run`.

        Args:
            log (callable): where a line goes, logger.info or print.
            seed (int): the seed being closed out, for the headline.
        """
        if not self.enabled:
            return

        if "iteration" not in self.run:
            return

        total, _, n = self.run["iteration"]
        self._table(
            log,
            self.run,
            f"TIME  SEED {seed}, WHOLE RUN:  {n} iterations in "
            f"{format_clock(self.elapsed)}  --  "
            f"{format_clock(total / n)} per iteration",
        )

    def _table(self, log, timing, headline):
        """Print one phase-by-phase table.

        Args:
            log (callable): where a line goes, logger.info or print.
            timing (dict): {phase: [seconds, units, calls]}. Must hold an
                "iteration" entry -- the share column is a percentage of it.
            headline (str): the line printed above the table.
        """
        iteration_total, _, n_iterations = timing["iteration"]

        def row(label, seconds, units, calls):
            """One line of the table: label (str) plus that phase's
            [seconds, units, calls] (float, int, int)."""
            share = 100.0 * seconds / iteration_total if iteration_total else 0.0

            per_call = format_mean(seconds / calls) if calls else "-"

            # "-" when the two agree, since per-unit would just repeat per
            # call. units=0 marks phases counted in seconds only.
            per_unit = (
                format_mean(seconds / units) if units and units != calls else "-"
            )

            log(
                f"  {label:<30}{seconds:>9.2f}s {share:>7.1f}% "
                f"{calls:>9} {units:>11} {per_call:>11} {per_unit:>11}"
            )

        log("")
        log(headline)
        log(
            f"  {'phase':<30}{'total':>10} {'share':>8} "
            f"{'calls':>9} {'units':>11} {'per call':>11} {'per unit':>11}"
        )

        for key, label in self.ROWS:
            if key in timing:
                row(label, *timing[key])

        log("  " + "-" * 93)
        row("one whole iteration", iteration_total, n_iterations, n_iterations)

        for key, label in self.ROWS_EVAL:
            if key in timing:
                row(label, *timing[key])


# What an out-of-memory RuntimeError says, lowercased. The CPU allocator is
# matched on its class/file name too, since the wording moves between versions.
_OOM_TEXT = (
    "out of memory",
    "can't allocate memory",
    "cannot allocate memory",
    "defaultcpuallocator",
    "alloc_cpu.cpp",
)


class Helper:
    """Builders and small utilities shared by anything that reads the config."""

    @property
    def device(self):
        """cuda when a GPU exists, cpu otherwise. This machine has no GPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_recurrent(self):
        """Does the encoder carry a hidden state between timesteps?"""
        return self.feature_extractor.upper() == "GRU"

    @property
    def path_model(self):
        """The checkpoint this config points at, re-derived every read since dir_pretrained_model moves."""
        return os.path.join(self.dir_pretrained_model, self.name_model)

    def build_env(self, render_mode=None):
        """One MiniGrid game, wrapped as the config asks. Every env in the
        project comes from here.

        Args:
            render_mode (str | None): passed to gym.make; "rgb_array" for the
                pygame viewers, None for training.

        Returns:
            gym.Env: the env, wrapped in StartInCueView when
            force_cue_visible is set.
        """
        # Overriding max_steps with worker_steps is the point: truncation and
        # the success reward both come from it, so no episode outlives a rollout.
        env = gym.make(
            self.name_env,
            render_mode=render_mode,
            max_steps=int(self.worker_steps),
        )

        if self.force_cue_visible:
            env = StartInCueView(env)

        return env

    def build_vector_env(self, n_envs):
        """n_envs MiniGrid games behind one step() call.

        Drops the non-image observation keys so the batch fits in shared
        memory; force_cue_visible still applies, via build_env.

        Args:
            n_envs (int): how many games to run in parallel.

        Returns:
            gym.vector.VectorEnv: AsyncVectorEnv (separate processes), or
            SyncVectorEnv when config.async_envs is False.
        """
        # A closure, not a bound method, so the subprocess pickles this and not
        # the whole Config; gymnasium's cloudpickle carries it across a spawn.
        def make():
            """One env for one worker slot; returns a gym.Env."""
            return ImgObsWrapper(self.build_env())

        if self.async_envs:
            return AsyncVectorEnv(
                [make] * n_envs,
                shared_memory=True,  # the point of dropping mission, above
                autoreset_mode=AutoresetMode.SAME_STEP,
            )

        return SyncVectorEnv([make] * n_envs, autoreset_mode=AutoresetMode.SAME_STEP)

    @property
    def env_max_steps(self):
        """The env's own DEFAULT time limit (int; MemoryEnv: 5 * size^2), read
        off a throwaway env so it can never disagree with name_env.

        Bare gym.make, not build_env: build_env overrides max_steps with
        worker_steps, which would both hand worker_steps back here and make
        this unreadable from the line in Config.__init__ that derives
        worker_steps from it."""
        env = gym.make(self.name_env)
        max_steps = env.unwrapped.max_steps
        env.close()
        return max_steps

    def build_extractor(self):
        """Build the encoder named by config.feature_extractor, at the widths
        the config currently holds.

        Returns:
            nn.Module: an MLP or a GRU; both map
            (batch, seq, 7, 7, 3) -> (batch, seq, hidden_size).

        Raises:
            ValueError: on an unknown feature_extractor.
        """
        name = self.feature_extractor.upper()

        if name == "MLP":
            return MLP(self.input_size, self.hidden_size, self.n_layers_mlp)
        if name == "GRU":
            return GRU(self.input_size, self.hidden_size)

        raise ValueError(f"unknown feature_extractor {self.feature_extractor!r}")

    def config_attributes(self, include_private=False):
        """Every attribute this config carries: the instance attributes plus
        the class-level ones and the @property values defined anywhere in the
        hierarchy (the encoder subclass, Config, Helper).

        Properties are read defensively -- one that raises is reported in
        place rather than taking the caller down.

        Args:
            include_private (bool): also include names starting with "_".

        Returns:
            dict: {name: value}, sorted by name.
        """
        values = {}

        def keep(key):
            """Is this attribute name (str) wanted? Returns bool."""
            return include_private or not key.startswith("_")

        # what __init__ set, and what apply_params() has since overwritten
        for key, value in vars(self).items():
            if keep(key):
                values[key] = value

        # Properties and class attributes aren't in vars(self), hence the MRO
        # walk; most-derived-first, so instance and subclass values win.
        for klass in type(self).__mro__:
            if klass is object:
                continue
            for key, attribute in vars(klass).items():
                if key in values or not keep(key):
                    continue
                if isinstance(attribute, property):
                    try:
                        values[key] = getattr(self, key)
                    except Exception as error:  # a broken property is a value,
                        # not a crash: this runs at the top of every run
                        values[key] = f"<unreadable: {type(error).__name__}: {error}>"
                elif not callable(attribute) and not isinstance(
                    attribute, (staticmethod, classmethod)
                ):
                    values[key] = attribute

        return dict(sorted(values.items()))

    def log_config_attributes(self, logger=None, title="HYPERPARAMETERS"):
        """Write config_attributes() one per line, alphabetically.

        A list of dicts (search_space) goes one entry per line instead of one
        long line.

        Args:
            logger (logging.Logger | None): where the lines go; None prints.
            title (str): the heading above the dump.
        """
        write = log_with(logger)
        attributes = self.config_attributes()
        width = max([len(key) for key in attributes] + [24]) + 2

        write(title)
        for key, value in attributes.items():
            if isinstance(value, (list, tuple)) and any(
                isinstance(item, dict) for item in value
            ):
                write(f"  {key}")
                for item in value:
                    write(f"  {'':<{width}}{item}")
            else:
                write(f"  {key:<{width}}{value}")

    @property
    def run_tag(self):
        """What this run IS, in one filename-safe token (str): the encoder plus
        the backward reach, e.g. MLP, GRU, GRU_tbptt8. Names the log file, so a
        directory of logs can be told apart without opening any of them."""
        return f"{self.feature_extractor.upper()}{self.tbptt_suffix}"

    def build_logger(self, log_dir="logs", name="rl_project"):
        """Build one logger per invocation, writing to a timestamped file and
        to the terminal, with every hyperparameter dumped at the top.

        Args:
            log_dir (str): directory for the log files; created if missing.
            name (str): the logging.Logger name. Loggers are singletons per
                name, so an existing one's handlers are cleared, not added to.

        Returns:
            logging.Logger: pass it to train_agent(logger=...).
        """
        os.makedirs(log_dir, exist_ok=True)

        started = datetime.now()
        # The encoder and backward reach lead the name, so sorting groups logs
        # by run instead of interleaving encoders by start minute.
        stem = f"log_{self.run_tag}_{started:%Y-%m-%d_%H-%M-%S}"
        path = os.path.join(log_dir, f"{stem}.log")

        # _2, _3, ... if the second-precision stamp collides; FileHandler
        # appends, so otherwise one run's log continues into another's.
        collision = 1
        while os.path.exists(path):
            collision += 1
            path = os.path.join(log_dir, f"{stem}_{collision}.log")

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Logger is a singleton per name; without this a second build_logger()
        # call keeps the first call's handlers and doubles every line.
        logger.handlers.clear()
        logger.propagate = False  # don't also hand records to the root logger

        # utf-8 explicitly: torchinfo's summary is drawn with box characters,
        # and a cp1252 default raises UnicodeEncodeError on every such line.
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(file_handler)

        # same reason, for the terminal: a console that can't represent a
        # character should print a placeholder, not drop the whole record.
        stream = sys.stderr
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

        # same format as the file, not a bare message -- so terminal lines are
        # datable on sight and stay aligned with torchinfo's summary output.
        stream_handler = logging.StreamHandler(stream)
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(stream_handler)

        bar = "=" * 78
        logger.info(bar)
        logger.info(f"RUN STARTED  {started:%Y-%m-%d %H:%M:%S}")
        logger.info(f"LOG FILE     {path}")
        # the two things that decide which ablation cell this run belongs to,
        # at the top rather than buried alphabetically in the dump below
        logger.info(
            f"MODEL        {self.feature_extractor.upper()}"
            f"   tbptt {self.tbptt_length}"
            f"   env {self.name_env}"
        )
        logger.info(bar)

        # Attributes and properties together, alphabetically -- the derived
        # ones decide what was built and where it lands, so they belong here.
        self.log_config_attributes(logger)

        logger.info(bar)

        return logger

    # running at the largest mini_batch_size candidate that fits in memory
    def _iter_error_chain(self, error):
        """Walk an exception and everything it was raised from or during, so a
        wrapped OOM (e.g. torchinfo re-raising a generic RuntimeError) is still
        found. Guards against __context__ cycles.

        Args:
            error (BaseException): the exception caught.

        Yields:
            BaseException: error, then each __cause__/__context__ in turn.
        """
        seen = set()
        while error is not None and id(error) not in seen:
            seen.add(id(error))
            yield error
            error = error.__cause__ or error.__context__

    def _is_oom(self, error):
        """Is this an out-of-memory failure, from either allocator?

        Args:
            error (BaseException): the exception caught.

        Returns:
            bool: True if it, or anything it wraps, is a CUDA or CPU-allocator
            out-of-memory error.
        """
        for err in self._iter_error_chain(error):
            if isinstance(err, torch.cuda.OutOfMemoryError):
                return True
            if isinstance(err, RuntimeError) and any(
                text in str(err).lower() for text in _OOM_TEXT
            ):
                return True
        return False

    def _clear_traceback_chain(self, error):
        """Drop the traceback of an exception and every cause it wraps, so a
        retry doesn't keep the failed attempt's tensors referenced and OOM
        again.

        Args:
            error (BaseException): the exception caught.
        """
        for err in self._iter_error_chain(error):
            err.__traceback__ = None

    def _free_memory(self):
        """Give the allocator back whatever the failed attempt left behind."""
        gc.collect()
        if torch.cuda.is_available():
            # Freeing a tensor returns it to torch's caching allocator, not to
            # the driver -- without this it stays unavailable to anything else.
            torch.cuda.empty_cache()

    def run_with_batch_size_fallback(
        self, run_fn, batch_size, logger=None, what="batch size"
    ):
        """Call run_fn at the largest candidate size that doesn't OOM.

        Args:
            run_fn (callable): takes one int, the size to try, and does the
                work; called again at the next size down after an OOM.
            batch_size (int | list[int] | tuple[int]): the candidates; sorted
                largest-first here, so the caller's order doesn't matter.
            logger (logging.Logger | None): where the fallback notes go.
            what (str): what is being sized, for those notes.

        Returns:
            tuple[int, Any]: (size_used, run_fn's result).

        Raises:
            The last OOM if every candidate fails; any non-OOM exception
            propagates untouched.
        """
        sizes = batch_size if isinstance(batch_size, (list, tuple)) else [batch_size]
        candidates = sorted({int(size) for size in sizes}, reverse=True)

        say = log_with(logger)
        warn = log_with(logger, "warning")
        fail = log_with(logger, "error")

        last_error = None

        for i, bs in enumerate(candidates):
            try:
                if len(candidates) > 1:
                    say(f"trying {what} {bs}  (candidate {i + 1}/{len(candidates)})")
                return bs, run_fn(bs)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                if not self._is_oom(error):
                    raise
                self._clear_traceback_chain(error)
                last_error = error
                self._free_memory()

                smaller = candidates[i + 1 :]
                if smaller:
                    warn(f"out of memory at {what} {bs}, falling back to {smaller[0]}")
                else:
                    fail(f"out of memory at {what} {bs}, no smaller one left to try")

        # the loop's last _free_memory already ran; nothing fit
        raise last_error

    def probe_batch_size(self, model, loss_fn, obs_shape, candidates, logger=None):
        """Resolve the largest minibatch size that fits, before training starts.

        One forward/backward/step per candidate on an all-zero worst-case
        batch, not on iteration 0's real data: early rollouts pack fewer
        sequences than later ones, so real data would underestimate.

        Args:
            model (nn.Module): the network about to be trained; deep-copied,
                so the real weights never see the probe.
            loss_fn (callable): (minibatch, model) -> (loss, info), i.e.
                PPOAgent.minibatch_loss.
            obs_shape (tuple[int]): one observation's shape, (7, 7, 3).
            candidates (list[int]): the sizes to try, any order.
            logger (logging.Logger | None): where the fallback notes go.

        Returns:
            int: the size that fit -- logged and written into the checkpoint,
            since it can differ from machine to machine.
        """
        probe = copy.deepcopy(model)
        optimizer = torch.optim.Adam(
            probe.parameters(), lr=self.lr, weight_decay=self.wd
        )
        # The longest sequence split_pad_mask can emit: probing at worker_steps
        # would overestimate the batch and resolve too small a minibatch.
        L = (
            self.worker_steps
            if self.tbptt_length == "max"
            else min(int(self.tbptt_length), self.worker_steps)
        )

        def step(n):
            """One full update at minibatch size n (int), on a worst-case
            all-real batch. Raises if it doesn't fit."""
            # The keys split_pad_mask produces, shaped as SequenceDataset would
            # collate them. All-ones mask: nothing padded is the worst case.
            mb = {
                "obs": torch.zeros(n, L, *obs_shape, dtype=torch.uint8),
                "actions": torch.zeros(n, L, dtype=torch.long),
                "log_probs": torch.zeros(n, L),
                "advantages": torch.zeros(n, L),
                "returns": torch.zeros(n, L),
                "mask": torch.ones(n, L),
            }

            # None for an MLP, which is why the branch does not run for it
            hidden = self.zero_hidden(n)
            if self.is_recurrent:
                mb["hxs"] = hidden.squeeze(0)

            loss, _ = loss_fn(mb, probe)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), self.max_grad_norm)
            optimizer.step()

        resolved, _ = self.run_with_batch_size_fallback(
            step, candidates, logger, what="probe minibatch size"
        )

        # the copy, its gradients and its optimizer state are dead now; hand
        # the memory back before training allocates at the size just settled on
        del probe, optimizer
        self._free_memory()

        return resolved

    def log_model_summary(self, model, logger=None, batch_size=None, seq_len=8):
        """Run torchinfo's summary on the model and log the table -- parameter
        counts, shapes, size.

        Args:
            model (nn.Module): the network to describe.
            logger (logging.Logger | None): where the table goes; None prints.
            batch_size (int | list[int] | None): probe batch size, or
                candidates for the OOM fallback; None uses
                config.mini_batch_size. Shapes the probe pass only -- it does
                not decide what training uses.
            seq_len (int): probe sequence length, same caveat.

        Returns:
            torchinfo.ModelStatistics: the summary object.
        """

        if batch_size is None:
            batch_size = self.mini_batch_size

        # Runs through the same OOM fallback as the real update, since the
        # probe itself can OOM on a wide encoder.
        def probe(bs):
            """One summary pass at batch size bs (int); returns ModelStatistics."""
            return summary(
                model,
                input_size=(bs, seq_len, 7, 7, 3),
                dtypes=[torch.uint8],
                verbose=0,
            )

        batch_size, info = self.run_with_batch_size_fallback(
            probe, batch_size, logger, what="probe batch size"
        )
        shape = (batch_size, seq_len, 7, 7, 3)

        write = log_with(logger)
        write("")
        write(f"MODEL SUMMARY  {self.feature_extractor.upper()}  (probe input {shape})")
        for line in str(info).splitlines():
            write(f"  {line}")
        write("")

        return info

    @property
    def name_model(self):
        """build_model_name(), re-derived every read: it depends on self.seed, which set_seed() fills in later."""
        return self.build_model_name()

    def build_model_name(self):
        """The checkpoint filename for the current seed.

        Returns:
            str: ppo_<seed>_<ENCODER>_<env>.pth, e.g.
            ppo_0_GRU_MiniGrid-DoorKey-8x8-v0.pth. self.seed falls back to
            seed_list[0] if set_seed() hasn't run yet (e.g. watch.py).
        """
        env = self.name_env.replace("/", "-")
        seed = getattr(self, "seed", self.seed_list[0])
        return f"ppo_{seed}_{self.feature_extractor.upper()}_{env}.pth"

    def build_model_path(self):
        """path_model, creating dir_pretrained_model first.

        Call right before torch.save; which directory this is depends on who
        is running -- the encoder's top level, a trial dir under hpo/, or
        no_hpo/.

        Returns:
            str: the full path to write to.
        """
        os.makedirs(self.dir_pretrained_model, exist_ok=True)
        return self.path_model

    def save_model(self, model, optimizer=None, **extra):
        """Write the weights, the architecture they belong to, and the params
        they were trained with, so the file can be reloaded on its own terms.

        Args:
            model (nn.Module): the network to save.
            optimizer (torch.optim.Optimizer | None): also save its state.
            **extra: written verbatim into the checkpoint -- eval_history,
                the headline eval numbers, the resolved mini_batch_size.
                Every value must survive torch.load(weights_only=True), so
                plain tensors, strings, ints, floats and bools only.

        Returns:
            str: the path written.
        """
        path = self.build_model_path()

        checkpoint = {
            "model": model.state_dict(),
            "feature_extractor": self.feature_extractor,
            "hidden_size": self.hidden_size,
            "input_size": self.input_size,
            "name_env": self.name_env,
            "force_cue_visible": self.force_cue_visible,
            # Not architecture, so load_model doesn't validate it -- recorded
            # so two runs differing only in backward reach differ on disk.
            "tbptt_length": self.tbptt_length,
            "params": self.searched_params(),
        }
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        checkpoint.update(extra)

        torch.save(checkpoint, path)
        return path

    def load_model(self, model, path=None):
        """Load weights into an already-built model, refusing a file that was
        trained under a different architecture, env or cue setting.

        Args:
            model (nn.Module): the network to load into. It must already have
                the checkpoint's widths -- apply_params(checkpoint_params())
                first for a tuned run.
            path (str | None): the checkpoint; None uses path_model.

        Returns:
            dict: the whole checkpoint, so the caller can read eval_history
            and the rest.

        Raises:
            FileNotFoundError: no file at path.
            ValueError: the file records a different encoder, width,
                input_size, env or force_cue_visible.
        """
        if path is None:
            path = self.path_model

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no checkpoint at {path}. Train first and call "
                f"config.save_model(agent.model) -- train_agent() does this "
                f"on its own, once, at the end of the run."
            )

        # weights_only=True is the safe default in modern torch and everything
        # save_model writes (tensors, strings, ints, bools) is allowed under it
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        # Everything this project writes goes through save_model. A bare
        # state_dict lacks the "model" key, so wrap it into the same shape.
        if "model" not in checkpoint:
            checkpoint = {"model": checkpoint}

        # force_cue_visible changes no tensor shape but changes the task, so a
        # mismatch must fail loudly. A bare state_dict is loaded unchecked.
        for key in (
            "feature_extractor",
            "hidden_size",
            "input_size",
            "name_env",
            "force_cue_visible",
        ):
            if key in checkpoint and getattr(self, key) != checkpoint[key]:
                raise ValueError(
                    f"{path} was trained with {key}={checkpoint[key]!r}, but "
                    f"the config says {getattr(self, key)!r}. Set "
                    f"config.{key} to match, or load the matching file."
                )

        model.load_state_dict(checkpoint["model"])
        return checkpoint

    # HPO storage and bookkeeping; what a trial does lives in hpo_ppo.py.
    # optuna/joblib import inside these methods so a checkout without them runs.
    def suggest_from_search_space(self, trial):
        """Draw one value per entry of config.search_space.

        Args:
            trial (optuna.Trial): the trial to draw from; each entry's "type"
                picks the suggest_* method, and the rest of the entry is
                passed through as that method's keyword arguments.

        Returns:
            dict: {name: value}, ready for apply_params.
        """
        # anything but "int"/"categorical" -- including a missing type -- is a
        # float, which is what every entry of the shared search_space is
        suggest = {
            "int": trial.suggest_int,
            "categorical": trial.suggest_categorical,
        }

        params = {}

        for spec in self.search_space:
            spec = dict(spec)
            kind = spec.pop("type", "float")
            params[spec["name"]] = suggest.get(kind, trial.suggest_float)(**spec)

        return params

    # both derived from hpo_objective, which parse_hpo_objective validates
    @property
    def hpo_metric(self):
        """The per-seed metric key from hpo_objective: "return_mean" or "success_rate"."""
        return parse_hpo_objective(self.hpo_objective)

    @property
    def hpo_aggregation(self):
        """How the per-seed metrics are reduced. The only aggregation there is."""
        return "mean_minus-std"

    @property
    def score_name(self):
        """ "mean_minus_1std(return_mean)" -- what the study maximizes, for printing next to a score."""
        return f"mean_minus_{self.hpo_lambda:g}std({self.hpo_metric})"

    def aggregate_scores(self, values):
        """Reduce per-seed metrics to the one number a trial is ranked on:
        mean - hpo_lambda * std, both ACROSS SEEDS and never within one run.

        Args:
            values (list[float]): one metric value per seed.

        Returns:
            float: the score.
        """
        values = np.asarray(values, dtype=float)
        # ddof=0, population std -- the seeds ARE the population being compared
        return float(values.mean()) - self.hpo_lambda * float(values.std())

    def apply_params(self, params):
        """Write drawn (or checkpointed) params onto this config's attributes.

        Must be called before PPOAgent(config) is built: the agent reads the
        sizes into the model at construction and never consults the config
        again.

        Args:
            params (dict): {name: value}, from suggest_from_search_space or
                checkpoint_params.

        Returns:
            Config: self, for chaining.

        Raises:
            AttributeError: a name that isn't already an attribute of the
                config -- which would train the untuned default while
                reporting the tuned param.
        """
        for name, value in params.items():
            if not hasattr(self, name):
                raise AttributeError(
                    f"search_space names {name!r}, which is not an attribute of "
                    f"{type(self).__name__}. Every tuned name has to exist on "
                    f"the config already -- otherwise this would set a new "
                    f"attribute that nothing reads, and the trial would train "
                    f"the untuned defaults while reporting the tuned params."
                )
            setattr(self, name, value)

        return self

    def searched_params(self):
        """The current value of every name in search_space -- the inverse of
        apply_params, letting save_model record what a checkpoint was trained
        with.

        Returns:
            dict: {name: value}; empty for configs like ConfigNoHPO whose
            search_space is [].
        """
        return {
            entry["name"]: getattr(self, entry["name"])
            for entry in self.search_space
            if hasattr(self, entry["name"])
        }

    # ----- where a study writes -------------------------------------------
    def build_hpo_dir(self):
        """Make hpo/ and hand it back (str). Optuna won't create the parent
        directory for its own sqlite file, so this runs first."""
        os.makedirs(self.dir_hpo, exist_ok=True)
        return self.dir_hpo

    def dir_hpo_trial(self, number):
        """Where one trial's checkpoints go -- the filenames are otherwise
        identical across trials.

        Args:
            number (int): the trial number.

        Returns:
            str: hpo/trial_<number>/, not created.
        """
        return os.path.join(self.dir_hpo, f"trial_{number}")

    @property
    def dir_hpo_best_trial(self):
        """hpo/best_trial/ -- a copy of the winning trial, same shape as trial_<n>/."""
        return os.path.join(self.dir_hpo, "best_trial")

    def build_hpo_trial_dir(self, number):
        """dir_hpo_trial(number), created.

        Args:
            number (int): the trial number.

        Returns:
            str: the path, which now exists.
        """
        path = self.dir_hpo_trial(number)
        os.makedirs(path, exist_ok=True)
        return path

    def build_hpo_best_trial_dir(self):
        """dir_hpo_best_trial, created (str)."""
        os.makedirs(self.dir_hpo_best_trial, exist_ok=True)
        return self.dir_hpo_best_trial

    def select_run(self, trial=None, seed_index=0):
        """Point this config at exactly one saved checkpoint. The single place
        a seed index is resolved to a path, so the trainer, the reporters and
        the viewer cannot drift apart.

        Args:
            trial (str | int | None): which run under hpo/ -- "best"/"final"
                for the winner, a trial number for one trial. None leaves
                dir_pretrained_model where the config already put it (the
                encoder's top level, or no_hpo/ for ConfigNoHPO).
            seed_index (int): a POSITION in seed_list, not a seed value.

        Returns:
            str: path_model. Nothing is created or read.

        Raises:
            ValueError: trial is neither "best"/"final" nor a number.
            IndexError: seed_index is out of range for seed_list.
        """
        # trial=None leaves dir_pretrained_model wherever the config put it:
        # the encoder's top level, or no_hpo/ for ConfigNoHPO
        if trial is not None:
            name = str(trial).lower()
            if name in ("best", "final"):
                self.dir_pretrained_model = self.dir_hpo_best_trial
            elif name.isdigit():
                self.dir_pretrained_model = self.dir_hpo_trial(int(name))
            else:
                raise ValueError(
                    f"trial={trial!r} is not 'final', 'best' or a trial number"
                )

        if not 0 <= seed_index < len(self.seed_list):
            raise IndexError(
                f"seed index {seed_index} is out of range for seed_list="
                f"{self.seed_list} ({len(self.seed_list)} seed"
                f"{'' if len(self.seed_list) == 1 else 's'}, so the valid "
                f"indices are 0..{len(self.seed_list) - 1}). The index is a "
                f"position in that list, not the seed value."
            )

        # set_seed() is not called here: nothing is being trained, and
        # reseeding the process would only change what the viewer samples.
        self.seed = self.seed_list[seed_index]

        return self.path_model

    def checkpoint_params(self, path=None):
        """The hyperparameters a checkpoint was trained with, read straight off
        the file rather than from a config that may since have changed.

        Args:
            path (str | None): the checkpoint; None uses path_model.

        Returns:
            dict: {name: value}, or {} if the file is missing or records no
            params -- apply_params({}) is then a no-op.
        """
        if path is None:
            path = self.path_model
        if not os.path.exists(path):
            return {}

        # a bare state_dict has no "params" either, and is a dict all the same
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return dict(checkpoint.get("params") or {})

    def copy_best_trial(self, study, logger=None):
        """Copy the winning trial's files into best_trial/, so the winner
        survives deleting trial_*/, and write best_params.json beside them.

        Args:
            study (optuna.Study): the study to read the winner from.
            logger (logging.Logger | None): where the notes go; None prints.

        Returns:
            str | None: the target directory, or None when no trial has
            completed or the winner's directory is gone and best_trial/ can't
            be confirmed to already hold its copy.
        """
        say = log_with(logger)

        try:
            best = study.best_trial
        except ValueError:
            # optuna raises here rather than returning None when no trial has
            # finished
            say("no completed trial yet, nothing to copy into best_trial/")
            return None

        source = self.dir_hpo_trial(best.number)
        target = self.dir_hpo_best_trial
        if not os.path.isdir(source):
            # trial_*/ is gitignored, so a fresh clone lacks it -- but a
            # best_trial/ already matching this winner is still correct.
            best_params_path = os.path.join(target, "best_params.json")
            if os.path.isfile(best_params_path):
                with open(best_params_path) as f:
                    recorded = json.load(f)
                if recorded.get("best_trial") == best.number:
                    say(
                        f"trial {best.number} has no directory at {source}, "
                        f"but {target} already holds its copy -- keeping it"
                    )
                    return target
            say(f"trial {best.number} has no directory at {source}, nothing to copy")
            return None
        if os.path.isdir(target):
            shutil.rmtree(target)
        os.makedirs(target, exist_ok=True)

        for name in sorted(os.listdir(source)):
            src = os.path.join(source, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(target, name))

        self.save_json(
            os.path.join(target, "best_params.json"),
            {
                "feature_extractor": self.feature_extractor,
                "name_env": self.name_env,
                "objective": self.hpo_objective,
                "metric": self.hpo_metric,
                "aggregation": self.hpo_aggregation,
                "score_name": self.score_name,
                "direction": self.hpo_direction,
                "best_trial": best.number,
                "best_value": best.value,
                "best_params": best.params,
                "user_attrs": best.user_attrs,
                "copied_from": source,
            },
        )

        say(f"best trial {best.number} (value {best.value}) copied to {target}")
        return target

    def save_sampler(self, sampler, path=None):
        """Pickle the sampler, so a resumed study exploits what the finished
        trials showed instead of re-exploring.

        Args:
            sampler (optuna.samplers.BaseSampler): the study's sampler.
            path (str | None): where to write; None uses path_hpo_sampler.

        Returns:
            str: the path written.
        """
        if path is None:
            path = self.path_hpo_sampler
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(sampler, path)
        return path

    def load_sampler(self, path=None):
        """Restore a pickled sampler.

        Args:
            path (str | None): where to read; None uses path_hpo_sampler.

        Returns:
            optuna.samplers.BaseSampler | None: the sampler, or None when
            there isn't one yet -- the caller then builds a fresh one.
        """
        if path is None:
            path = self.path_hpo_sampler
        if not os.path.exists(path):
            return None
        return joblib.load(path)

    def build_pruner(self):
        """The study's pruner, wired from this config's hpo_pruner_* knobs.

        A step here is one SEED, not one training iteration: HPOPPO reports
        once per finished run, between two seeds, so the whole trial only ever
        has len(seed_list) steps. All four knobs are therefore small integers,
        and n_min_trials is passed rather than left at optuna's default of 1 --
        a median over one other trial is not a median, and with so few steps
        there is no later check to correct an early bad call.

        Plain optuna MedianPruner: tbptt_length is fixed for a whole study now,
        so every trial in one study is directly comparable and there is nothing
        left to group by.

        Returns:
            optuna.pruners.MedianPruner: wired from hpo_pruner_startup_trials,
            hpo_pruner_warmup, hpo_pruner_interval and hpo_pruner_min_trials.
        """
        return optuna.pruners.MedianPruner(
            n_startup_trials=self.hpo_pruner_startup_trials,
            n_warmup_steps=self.hpo_pruner_warmup,
            interval_steps=self.hpo_pruner_interval,
            n_min_trials=self.hpo_pruner_min_trials,
        )

    def save_json(self, path, data):
        """Write data as indented json, creating the directory if needed.

        Args:
            path (str): the file to write.
            data (dict | list): anything json can take; default=str catches
                the numpy scalars that would otherwise raise.

        Returns:
            str: the path written.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as handle:
            json.dump(data, handle, indent=2, default=str)
        return path

    def csv_study_export(self, study, path_csv=None):
        """Export the study as a csv: one row per trial, holding the number,
        the objective value, every drawn param, the four across-seed numbers
        HPOPPO records, the state, and when the trial ran.

        Written at the START of each trial, so an interrupted study still
        leaves a csv of how far it got.

        Args:
            study (optuna.Study): the study to export.
            path_csv (str | None): where to write; None uses path_hpo_csv.

        Returns:
            str | None: the path written, or None when pandas is missing.
        """
        if path_csv is None:
            path_csv = self.path_hpo_csv

        os.makedirs(os.path.dirname(path_csv) or ".", exist_ok=True)
        try:
            # Named attrs, not the default -- the default's system/user attrs
            # bury the params. Clock columns are named last, so params lead.
            frame = study.trials_dataframe(
                attrs=(
                    "number",
                    "value",
                    "params",
                    "user_attrs",
                    "state",
                    "datetime_start",
                    "datetime_complete",
                    "duration",
                )
            )
        except (ImportError, AttributeError):
            return None

        frame.columns = [self._strip_prefix(name) for name in frame.columns]
        frame = frame.rename(columns={"value": "objective_value"})

        # Left raw, pandas writes microseconds and '0 days 00:12:34.567890'.
        # A trial with no end yet leaves the cell empty rather than 'NaT'.
        for column in ("datetime_start", "datetime_complete"):
            if column in frame:
                frame[column] = [
                    "" if self._is_missing(value) else value.strftime(_CSV_TIMESTAMP)
                    for value in frame[column]
                ]

        if "duration" in frame:
            # pandas' rendering with the sub-second tail cut off. str().split,
            # not .dt.floor('s'), which raises on an all-None object column.
            frame["duration_seconds"] = [
                "" if self._is_missing(value) else int(value.total_seconds())
                for value in frame["duration"]
            ]
            frame["duration"] = [
                "" if self._is_missing(value) else str(value).split(".")[0]
                for value in frame["duration"]
            ]

        frame.to_csv(path_csv, index=False)
        return path_csv

    @staticmethod
    def _is_missing(value):
        """Is this cell empty? Written as a self-inequality so this file still
        never imports pandas itself.

        Args:
            value: any cell out of the trials dataframe.

        Returns:
            bool: True for None and for pandas' NaT/NaN.
        """
        return value is None or value != value

    @staticmethod
    def _strip_prefix(name):
        """Shorten one optuna dataframe column name.

        Args:
            name (str): e.g. "params_lr" or "user_attrs_return_std".

        Returns:
            str: e.g. "lr" or "return_std"; anything else unchanged.
        """
        for prefix in ("params_", "user_attrs_"):
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name

    # Every checkpoint carries its eval_history, so a directory of them is a
    # set of curves: one mean+-std figure per _HPO_METRICS entry, html and svg.
    def load_eval_histories(self, directory):
        """Read every checkpoint's learning curve out of one directory.

        Args:
            directory (str): a trial dir, best_trial/ or no_hpo/.

        Returns:
            dict: {seed: eval_history}, the seed parsed off each filename
            (the filename itself if unparseable), ints first and in order.
            Files that aren't checkpoints, or carry no eval_history, are
            skipped rather than raising; a missing directory gives {}.
        """
        histories = {}
        if not os.path.isdir(directory):
            return histories

        for name in sorted(os.listdir(directory)):
            if not name.endswith(".pth"):
                continue

            try:
                checkpoint = torch.load(
                    os.path.join(directory, name),
                    map_location="cpu",
                    weights_only=True,
                )
            except Exception:
                continue

            if not isinstance(checkpoint, dict):
                continue
            history = checkpoint.get("eval_history")
            if not history:
                continue

            parts = name.split("_")
            try:
                seed = int(parts[1])
            except (IndexError, ValueError):
                seed = name

            histories[seed] = list(history)

        # ints first and in order, anything unparseable after them
        return dict(
            sorted(
                histories.items(),
                key=lambda kv: (
                    (1, str(kv[0])) if isinstance(kv[0], str) else (0, kv[0])
                ),
            )
        )

    @staticmethod
    def curve_table(histories, metric):
        """Line up several seeds' curves on a shared iteration axis.

        Args:
            histories (dict): {seed: eval_history}, from load_eval_histories.
            metric (str): which key to pull out of each entry, e.g.
                "return_mean" or "success_rate".

        Returns:
            tuple: (iterations, seeds, values) -- a sorted list[int], the
            seeds in order, and a float array (n_iterations, n_seeds) with
            nan where a seed has no entry at that iteration.
        """
        by_iteration = {}
        for seed, history in histories.items():
            for entry in history:
                if metric in entry and entry[metric] is not None:
                    step = int(entry["iteration"])
                    by_iteration.setdefault(step, {})[seed] = float(entry[metric])

        iterations = sorted(by_iteration)
        seeds = list(histories)
        values = np.array(
            [[by_iteration[i].get(s, np.nan) for s in seeds] for i in iterations],
            dtype=float,
        )
        return iterations, seeds, values

    def plot_eval_curves(
        self,
        directory,
        metrics=None,
        name=None,
        title=None,
        logger=None,
        include_plotlyjs="cdn",
    ):
        """Draw one mean+-std learning curve per metric for a directory of
        checkpoints, as .html and .svg. The raw per-seed lines are drawn too,
        hidden behind a legend click.

        Args:
            directory (str): the checkpoints to read, and where the figures
                are written.
            metrics (str | tuple[str] | None): which metrics to draw; None
                does both of _HPO_METRICS.
            name (str | None): label for the title; None uses the directory's
                own name.
            title (str | None): override the whole title line.
            logger (logging.Logger | None): where the notes go; None prints.
            include_plotlyjs (str | bool): passed to plotly's write_html --
                "cdn" keeps the files small, True inlines the library so they
                work offline.

        Returns:
            list[str]: the paths written; [] when there is no curve to draw.
            A missing kaleido costs the .svg only, not the .html.
        """
        say = log_with(logger)

        if metrics is None:
            metrics = _HPO_METRICS
        elif isinstance(metrics, str):
            metrics = (metrics,)
        if name is None:
            name = os.path.basename(os.path.normpath(directory))

        histories = self.load_eval_histories(directory)
        if not histories:
            say(f"no eval_history in {directory}, nothing to plot")
            return []

        paths = []
        for metric in metrics:
            iterations, seeds, values = self.curve_table(histories, metric)
            if not iterations:
                say(f"no {metric!r} in the curves under {directory}, skipped")
                continue

            # nan-aware everywhere: a seed missing at one iteration must not
            # turn the whole row into nan and blank out the plot
            mean = np.nanmean(values, axis=1)
            std = np.nanstd(values, axis=1)
            low, high = mean - std, mean + std
            centre_label, band_label, colour = (
                "mean over seeds",
                "+- 1 std",
                "#2f6fdb",
            )

            header = (
                f"{self.feature_extractor.upper()} on {self.name_env}"
                f"  --  {name}  --  {len(seeds)} seed(s) {list(seeds)}"
            )

            fig = go.Figure()

            # band drawn first so the centre line sits on top; filled "toself"
            # polygon, up the upper edge and back down the lower one reversed
            fig.add_trace(
                go.Scatter(
                    x=list(iterations) + list(iterations)[::-1],
                    y=list(high) + list(low)[::-1],
                    fill="toself",
                    fillcolor=self._rgba(colour, 0.18),
                    # spelled out: plotly defaults to "lines+markers" under 20
                    # points, which would make the band's edges dotted
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    name=band_label,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=mean,
                    mode="lines+markers",
                    line=dict(color=colour, width=2.5),
                    marker=dict(size=6),
                    name=centre_label,
                    # the band's own trace is hoverinfo="skip", so the spread
                    # has to ride along on the centre line to be readable
                    customdata=np.stack([std, low, high], axis=-1),
                    hovertemplate="iteration %{x}<br>"
                    + metric
                    + " %{y:.3f} +- %{customdata[0]:.3f}"
                    + "<br>band [%{customdata[1]:.3f}, %{customdata[2]:.3f}]"
                    + "<extra></extra>",
                )
            )

            # raw seeds, drawn but hidden (legendonly) -- one click brings
            # them back if the band looks wrong
            for index, seed in enumerate(seeds):
                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=values[:, index],
                        mode="lines",
                        line=dict(color="#8b8b8b", width=1, dash="dot"),
                        name=f"seed {seed}",
                        visible="legendonly",
                        hovertemplate=(
                            f"seed {seed}<br>iteration %{{x}}<br>"
                            + metric
                            + " %{y:.3f}<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(
                title=dict(
                    text=(title or f"{metric}  --  {centre_label} and {band_label}")
                    + f"<br><sub>{header}</sub>"
                ),
                xaxis_title="iteration",
                yaxis_title=metric,
                template="plotly_white",
                hovermode="x unified",
                width=1000,
                height=560,
                legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
                margin=dict(t=110, r=30, b=60, l=70),
            )

            os.makedirs(directory, exist_ok=True)
            stem = os.path.join(directory, f"curve_{metric}_mean_std")

            # a median/IQR figure left by an earlier run would sit next to this
            # one looking current; there is no code left that can refresh it
            for stale in glob.glob(
                os.path.join(directory, f"curve_{metric}_median_iqr.*")
            ):
                os.remove(stale)

            fig.write_html(f"{stem}.html", include_plotlyjs=include_plotlyjs)
            paths.append(f"{stem}.html")

            # needs kaleido; a missing static exporter shouldn't lose the html
            # already written or kill an otherwise-finished study
            try:
                fig.write_image(f"{stem}.svg")
                paths.append(f"{stem}.svg")
            except Exception as error:
                say(f"could not write {stem}.svg ({type(error).__name__}: {error})")

        say(
            f"plotted {', '.join(metrics)} for {name}: "
            f"{len(paths)} file(s) in {directory}"
        )
        return paths

    @staticmethod
    def _rgba(hex_colour, alpha):
        """A translucent colour in the string form plotly wants.

        Args:
            hex_colour (str): "#2f6fdb", with or without the hash.
            alpha (float): opacity, 0 to 1.

        Returns:
            str: e.g. "rgba(47,111,219,0.18)".
        """
        hex_colour = hex_colour.lstrip("#")
        r, g, b = (int(hex_colour[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"

    def plot_hpo(self, metrics=None, logger=None, include_trials=True):
        """Draw curves for every trial directory under hpo/, plus best_trial/.

        Args:
            metrics (str | tuple[str] | None): passed to plot_eval_curves;
                None does both of _HPO_METRICS.
            logger (logging.Logger | None): where the notes go; None prints.
            include_trials (bool): False skips the individual trials and does
                best_trial/ alone.

        Returns:
            list[str]: every path written.
        """
        say = log_with(logger)

        directories = []
        if include_trials and os.path.isdir(self.dir_hpo):
            trials = [
                entry
                for entry in os.listdir(self.dir_hpo)
                if entry.startswith("trial_")
                and os.path.isdir(os.path.join(self.dir_hpo, entry))
            ]
            # trial_2 before trial_10: sorted() on the string would not
            directories += [
                os.path.join(self.dir_hpo, entry)
                for entry in sorted(trials, key=lambda e: int(e.split("_")[1]))
            ]

        directories.append(self.dir_hpo_best_trial)

        paths = []
        for directory in directories:
            paths += self.plot_eval_curves(directory, metrics=metrics, logger=logger)

        say(f"plot_hpo: {len(paths)} file(s) written under {self.dir_hpo}")
        return paths

    @staticmethod
    def seed_result_row(seed, sampled, argmax):
        """One seed's row for log_seed_summary, from its two evaluations of the
        same weights.

        Args:
            seed (int): the seed, which labels the row.
            sampled (dict): the evaluation with actions drawn from pi; needs
                return_mean and success_rate.
            argmax (dict): the same weights evaluated greedily, same keys.

        Returns:
            dict: seed plus the four _SEED_SUMMARY_METRICS as floats.
        """
        return {
            "seed": seed,
            "sampled_return": float(sampled["return_mean"]),
            "sampled_success_rate": float(sampled["success_rate"]),
            "argmax_return": float(argmax["return_mean"]),
            "argmax_success_rate": float(argmax["success_rate"]),
        }

    def log_seed_summary(self, logger, rows, header=None):
        """The closing table: one row per seed, then each of the four metrics
        aggregated across seeds.

        main_no_hpo.py and HPOPPO.final both end with this, so a hand-picked
        run and a tuned one are read the same way.

        Args:
            logger (logging.Logger | None): where the table goes; None prints.
            rows (list[dict]): one seed_result_row per seed.
            header (str | None): the line above the table; None builds one
                from the encoder and the row count.

        Returns:
            dict: {metric: {"mean": float, "std": float}}, both ACROSS SEEDS.
        """
        write = log_with(logger)
        width = max(len(metric) for metric in _SEED_SUMMARY_METRICS) + 2

        write("")
        write(header or f"{self.feature_extractor.upper()}  over {len(rows)} seed(s)")
        write(
            f"{'seed':>6}" + "".join(f"{m:>{width}}" for m in _SEED_SUMMARY_METRICS)
        )
        for row in rows:
            write(
                f"{row['seed']:>6}"
                + "".join(f"{row[m]:>{width}.3f}" for m in _SEED_SUMMARY_METRICS)
            )

        write("")
        write("  (spread is across seeds, not across the episodes of one run;")
        write("   sampled and argmax measure the same weights -- read down a")
        write("   column, not across)")

        stats = {}
        for metric in _SEED_SUMMARY_METRICS:
            values = [row[metric] for row in rows]
            stats[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
            summary = stats[metric]
            write(
                f"{metric:>{width}}  mean {summary['mean']:.3f}  "
                f"std {summary['std']:.3f}"
            )

        return stats

    def print_separate_lines(self, logger, n=10):
        """Print n (int) rules to the logger, to set a phase of the run apart
        in a long log."""
        for _ in range(n):
            logger.info("=" * 78)

    def callback_optuna_report_function(self, kind_training, logger, study, trial):
        """Log one line per finished trial, plus the best so far.

        Passed as an optuna study.optimize callback, so it runs after every
        trial including pruned and failed ones -- hence the last two
        arguments' order, which optuna fixes.

        Args:
            kind_training (str): the encoder being tuned, for the line.
            logger (logging.Logger): where the lines go.
            study (optuna.Study): the running study.
            trial (optuna.trial.FrozenTrial): the trial that just finished.
        """
        logger.info(
            f"HPO {kind_training}: trial {trial.number} finished "
            f"({trial.state.name}) value {trial.value} params {trial.params}"
        )
        try:
            logger.info(
                f"  best so far: trial {study.best_trial.number} "
                f"value {study.best_value} params {study.best_params}"
            )
        except ValueError:
            # best_trial raises when nothing has completed yet
            logger.info("  best so far: none, no trial has completed yet")
        logger.info("-" * 78)

    def hpo_optimize(self, study, n_trials, objective, logger, kind_training):
        """study.optimize, resume-aware: run only the trials still needed to
        reach n_trials, and re-queue the last one's params if it crashed or
        was killed mid-run.

        Args:
            study (optuna.Study): the study to run.
            n_trials (int): a TOTAL, counted as complete + pruned -- not "run
                N more".
            objective (callable): takes an optuna.Trial, returns a float.
            logger (logging.Logger): where the progress lines go.
            kind_training (str): the encoder being tuned, for those lines.

        Returns:
            int: how many trials this call ran, 0 when the study was already
            finished.
        """
        states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
        done = study.get_trials(deepcopy=False, states=list(states))
        remaining = n_trials - len(done)

        if remaining <= 0:
            logger.info(
                f"{len(done)} of {n_trials} trials already done, nothing to run."
            )
            return 0

        self.print_separate_lines(logger)
        logger.info(f"Start HPO {kind_training} with {remaining} remaining trials")

        # .params is empty if the trial died before suggesting anything, in
        # which case there is nothing to repeat and the sampler just draws again
        last = study.trials[-1] if study.trials else None
        if (
            last is not None
            and last.params
            and last.state
            in (optuna.trial.TrialState.FAIL, optuna.trial.TrialState.RUNNING)
        ):
            logger.info(
                f"trial {last.number} is {last.state.name} -- "
                f"re-queueing its params: {last.params}"
            )
            study.enqueue_trial(last.params)

        study.optimize(
            objective,
            n_trials=remaining,
            callbacks=[
                lambda study, trial: self.callback_optuna_report_function(
                    kind_training, logger, study, trial
                )
            ],
        )
        return remaining

    def summary_hpo(self, logger, study, path_csv=None):
        """Close the search out: export the csv, count the trials by state,
        and print the winner and its params.

        Args:
            logger (logging.Logger): where the summary goes.
            study (optuna.Study): the finished (or interrupted) study.
            path_csv (str | None): where the export goes; None uses
                path_hpo_csv.

        Returns:
            optuna.trial.FrozenTrial | None: the best trial, or None if
            nothing completed.
        """
        self.csv_study_export(study, path_csv)

        trials = study.trials
        counted = {state: 0 for state in optuna.trial.TrialState}
        for trial in trials:
            counted[trial.state] += 1

        bar = "=" * 78
        logger.info(bar)
        logger.info(f"SUMMARY HPO  study {study.study_name}")
        logger.info(
            f"  total {len(trials)}   "
            + "   ".join(
                f"{state.name.lower()} {n}" for state, n in counted.items() if n
            )
        )
        logger.info(f"  csv   {path_csv or self.path_hpo_csv}")
        logger.info(bar)

        try:
            best = study.best_trial
        except ValueError:
            logger.info("no trial completed -- there is no best trial to report")
            self.print_separate_lines(logger)
            return None

        logger.info(f"BEST  trial {best.number}   {self.score_name} {best.value}")
        for key, value in best.params.items():
            logger.info(f"    {key:<24}{value}")
        # Max of n_trials noisy measurements, so biased upward by the selection
        # itself (winner's curse). final() reports it rather than re-estimating.
        logger.info(
            "  (the max over trials, so biased upward by the selection itself; "
            "final() reports these same runs and does not re-estimate it)"
        )
        self.print_separate_lines(logger)

        return best

    def zero_hidden(self, batch_size=None):
        """A fresh hidden state -- no memory of anything.

        Args:
            batch_size (int | None): how many parallel streams; None uses
                n_workers.

        Returns:
            torch.Tensor | None: zeros (1, batch_size, hidden_size) on the
            config's device, or None for a memoryless encoder.
        """
        if not self.is_recurrent:
            return None

        if batch_size is None:
            batch_size = self.n_workers

        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

    def reset_hidden_of(self, hidden, w):
        """Zero one worker's hidden state in place, when its episode ends,
        leaving the other workers' memory untouched.

        Args:
            hidden (torch.Tensor | None): the batch's hidden state,
                (1, n_workers, hidden_size).
            w (int): which worker slot to clear.

        Returns:
            torch.Tensor | None: the same tensor, for chaining.
        """
        if not self.is_recurrent:
            return hidden

        hidden[:, w] = 0.0
        return hidden

    def watch_agent(
        self, path_model=None, deterministic=None, steps_per_sec=2.5, fullscreen=False
    ):
        """Open a pygame window that plays a saved policy over the fixed
        eval-set mazes, showing the full maze, the agent's 7x7 observation,
        its action distribution and its value estimate.

        Trains nothing and writes nothing. Blocks until the window closes.

        Args:
            path_model (str | None): the checkpoint to load; None uses
                path_model. Its tuned architecture is applied first, so a
                trial with a different width loads cleanly.
            deterministic (bool | None): True plays the argmax action, False
                samples; None uses config.eval_deterministic.
            steps_per_sec (float): how fast the agent acts. The window still
                redraws at _VIEW_FPS.
            fullscreen (bool): open filling the screen; F11 toggles either
                way, and the layout is drawn at a fixed size and scaled to
                fit (see _ViewWindow).

        Raises:
            FileNotFoundError: no checkpoint at that path.
        """
        if deterministic is None:
            deterministic = self.eval_deterministic

        # Apply the checkpoint's architecture before the builders read config
        # defaults, or a tuned checkpoint fails load_state_dict.
        if path_model is None:
            path_model = self.path_model
        params = self.checkpoint_params(path_model)
        if params:
            self.apply_params(params)

        env = self.build_env(render_mode="rgb_array")  # same builder as training
        n_actions = env.action_space.n

        model = Network(self.build_extractor(), self.hidden_size, n_actions)
        checkpoint = self.load_model(model, path_model)
        model.to(self.device)
        model.eval()

        pygame.init()

        maze_px = _VIEW_MAZE_PX
        win_w = maze_px + _VIEW_SIDEBAR_W
        win_h = max(maze_px, _VIEW_MIN_H)

        # everything below draws into window.canvas at these fixed
        # coordinates; the window itself can be any size
        window = _ViewWindow(
            (win_w, win_h),
            f"{self.feature_extractor.upper()} on {self.name_env}",
            fullscreen=fullscreen,
        )
        screen = window.canvas
        clock = pygame.time.Clock()

        fonts = {
            "head": pygame.font.SysFont("monospace", 13, bold=True),
            "body": pygame.font.SysFont("monospace", 11),
            "tiny": pygame.font.SysFont("monospace", 10),
            "big": pygame.font.SysFont("monospace", 15, bold=True),
        }

        # Everything the buttons reset lives in this dict, written only here.
        # max_steps is read off the live env, not config.env_max_steps.
        ep = {"max_steps": env.unwrapped.max_steps}

        def think():
            """One forward pass on the current observation, advancing the
            hidden state exactly once. Writes probs, value and action into
            `ep`; returns nothing."""
            # (7, 7, 3) -> (1, 1, 7, 7, 3): batch 1, seq_len 1, as in evaluate()
            obs_t = torch.from_numpy(ep["obs"]).to(self.device)[None, None]

            with torch.no_grad():
                dist, value, ep["hidden"] = model(obs_t, ep["hidden"])

            ep["probs"] = dist.probs[0, 0].cpu().numpy()
            ep["value"] = float(value[0, 0])
            ep["action"] = (
                int(ep["probs"].argmax()) if deterministic else int(dist.sample()[0, 0])
            )

        def start(move=0, keep_trail=False):
            """Begin an episode, resetting everything `ep` holds.

            Args:
                move (int): +1/-1/0 for the next/previous/current eval maze,
                    wrapping both ways.
                keep_trail (bool): preserve the recorded action trail across
                    the reset -- what lets go_to() replay it.
            """
            index = ep.get("index", -1)
            if move:
                index = (index + move) % self.n_eval_episodes

            trail = list(ep["trail"]) if keep_trail else []

            obs_state, _ = env.reset(seed=self.eval_seed + index)  # same seed evaluate() uses

            ep.update(
                index=index,
                obs=obs_state["image"],
                hidden=self.zero_hidden(batch_size=1),
                step=0,
                total_reward=0.0,
                done=False,
                outcome="",
                history=[],
                # every action taken this episode, in order; lets STEP -1 work
                # by resetting to the same seed and re-walking this list
                trail=trail,
                # read once from the first observation and kept on screen: the
                # cue is out of view after step 0, only in the hidden state
                cue=_find_cue(obs_state["image"]),
            )
            think()

        def act(forced=None):
            """Take one action and update the episode state.

            Args:
                forced (int | None): a recorded action to replay; None takes
                    the one think() just chose.
            """
            action = ep["action"] if forced is None else forced
            obs_state, reward, terminated, truncated, _ = env.step(action)

            ep["obs"] = obs_state["image"]
            ep["step"] += 1
            ep["total_reward"] += reward
            ep["history"] = (ep["history"] + [(ep["step"], action, reward)])[-8:]

            # only a genuinely new action extends the record; re-walking one
            # already in the trail must not append it again
            if ep["step"] > len(ep["trail"]):
                ep["trail"].append(action)

            if terminated or truncated:
                ep["done"] = True
                # MemoryEnv pays >0 for the correct object, 0 for the wrong one
                ep["outcome"] = (
                    "SOLVED"
                    if reward > 0
                    else "WRONG OBJECT" if terminated else "OUT OF TIME"
                )
            else:
                think()

        def advance():
            """One step forward: replays the recorded action if we have
            rewound into the trail, otherwise acts fresh."""
            if ep["step"] < len(ep["trail"]):
                act(forced=ep["trail"][ep["step"]])
            else:
                act()

        def go_to(target):
            """Put the episode back at an earlier step by resetting to the
            same seed and replaying its recorded actions -- neither the env
            nor the hidden state can be stepped backward.

            Args:
                target (int): the step to land on; clamped at 0.
            """
            target = max(0, target)

            start(move=0, keep_trail=True)  # same maze, trail preserved
            for i in range(target):
                act(forced=ep["trail"][i])

            # replay the recorded action, not a fresh sample, so STEP -1 then
            # STEP +1 lands exactly where it started
            if target < len(ep["trail"]):
                ep["action"] = ep["trail"][target]

        start(move=+1)  # index starts at -1, so this opens on maze 1

        paused = False
        auto_new = False  # when an episode ends, roll straight into the next
        step_once = False  # STEP +1 was pressed: take exactly one action
        since_step = 0.0  # seconds of unspent time towards the next action
        since_done = 0.0  # seconds spent sitting on a finished episode
        buttons = {}  # filled by the sidebar each frame: name -> pygame.Rect
        running = True

        def press(name):
            """Act on one control, whether it came from a click or a key.

            Args:
                name (str): the control -- "new", "last", "replay", "pause",
                    "step", "back" or "auto".
            """
            nonlocal paused, step_once, auto_new
            if name == "new":
                start(move=+1)
            elif name == "last":
                start(move=-1)
            elif name == "replay":
                start(move=0)
            elif name == "pause":
                paused = not paused
            elif name == "step":
                # stepping implies pausing: it would be useless otherwise,
                # the clock would just run on top of the single step
                paused, step_once = True, True
            elif name == "back" and ep["step"] > 0:
                paused = True
                go_to(ep["step"] - 1)
            elif name == "auto":
                auto_new = not auto_new

        keys = {
            pygame.K_n: "new",
            pygame.K_p: "last",  # p for previous; n and p are the usual pair
            pygame.K_r: "replay",
            pygame.K_SPACE: "pause",
            pygame.K_RIGHT: "step",
            pygame.K_LEFT: "back",
            pygame.K_a: "auto",
        }

        while running:
            dt = clock.tick(_VIEW_FPS) / 1000.0

            for event in pygame.event.get():
                if window.handle(event):  # resize / F11: not a control
                    continue

                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key in keys:
                        press(keys[event.key])

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Hit-tested against what the sidebar drew last frame, so
                    # the layout lives in one place. Window px vs canvas px.
                    click = window.to_canvas(event.pos)
                    for name, rect in buttons.items():
                        if rect.collidepoint(click):
                            press(name)
                            break

            if ep["done"]:
                # delay before auto-advancing, so the outcome badge is readable
                if auto_new and not paused:
                    since_done += dt
                    if since_done >= _VIEW_AUTO_DELAY:
                        since_done = 0.0
                        start(move=+1)
            else:
                since_done = 0.0

                if step_once:
                    step_once = False
                    advance()
                elif not paused:
                    # act on a clock, not once per frame: redraw stays 60fps
                    # while the agent moves at a human-readable speed
                    since_step += dt
                    if since_step >= 1.0 / steps_per_sec:
                        since_step = 0.0
                        advance()

            screen.fill(_VIEW_BG)

            # left: the whole maze, centred vertically (sidebar is taller than
            # the frame is square, so pinned-to-top leaves a black shelf)
            frame = env.render()
            surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
            screen.blit(
                pygame.transform.scale(surface, (maze_px, maze_px)),
                (0, (win_h - maze_px) // 2),
            )

            buttons = _draw_viewer_sidebar(
                screen,
                fonts,
                maze_px,
                win_h,
                self,
                ep,
                checkpoint,
                deterministic,
                {
                    "paused": paused,
                    "auto_new": auto_new,
                    # None unless a countdown is actually running
                    "auto_in": (
                        max(0.0, _VIEW_AUTO_DELAY - since_done)
                        if ep["done"] and auto_new and not paused
                        else None
                    ),
                },
                window.mouse,
            )

            window.flip()

        env.close()
        pygame.quit()

    def play_env(self, detail=False, fullscreen=False):
        """Open a pygame window that lets YOU play one env from the keyboard:
        the maze on the left, the agent's 7x7 observation and what is in it on
        the right.

        Trains nothing and loads nothing -- this is the env explorer, and it
        plays exactly the env the config names. Blocks until the window closes.

        Args:
            detail (bool): add a third column with every number of
                obs["image"], raw and decoded.
            fullscreen (bool): open filling the screen; F11 toggles either
                way, and the layout is drawn at a fixed size and scaled to fit.
        """
        env = self.build_env(render_mode="rgb_array")  # same builder as training

        pygame.init()

        maze_px = _VIEW_MAZE_PX
        win_w = maze_px + _PLAY_SIDEBAR_W + (_PLAY_CHANNEL_W if detail else 0)
        win_h = max(maze_px, _PLAY_MIN_H)

        # fixed-size canvas, scaled into whatever the window is -- as in
        # watch_agent, the drawing below never sees the real window size
        window = _ViewWindow(
            (win_w, win_h),
            f"MiniGrid interactive -- {self.name_env}",
            fullscreen=fullscreen,
        )
        screen = window.canvas
        clock = pygame.time.Clock()

        fonts = {
            "head": pygame.font.SysFont("monospace", 13, bold=True),
            "body": pygame.font.SysFont("monospace", 11),
            "tiny": pygame.font.SysFont("monospace", 10),
            "micro": pygame.font.SysFont("monospace", 9),
            "big": pygame.font.SysFont("monospace", 15, bold=True),
        }

        # Everything an episode owns, so R restarts via start() alone.
        # max_steps comes off the live env -- build_env set it to worker_steps.
        ep = {"max_steps": env.unwrapped.max_steps}

        def start():
            """Begin an episode, resetting everything `ep` holds. Takes no
            arguments -- R restarts by calling this alone."""
            obs_state, _ = env.reset()
            ep.update(
                obs=obs_state["image"],  # (7, 7, 3): object, colour, state ids
                step=0,
                total_reward=0.0,
                done=False,
                history=[],
                message="",
            )

        start()

        print("\nCONTROLS")
        print("--------")
        for key, description in _PLAY_KEYS:
            print(f"  {key:<14} {description}")
        print()

        running = True
        while running:
            action = None

            for event in pygame.event.get():
                if window.handle(event):  # resize / F11: not a control
                    continue

                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        start()
                    elif event.key in _PLAY_ACTION_KEYS and not ep["done"]:
                        action = _PLAY_ACTION_KEYS[event.key]

            if action is not None:
                obs_state, reward, terminated, truncated, _ = env.step(action)

                ep["obs"] = obs_state["image"]
                ep["step"] += 1
                ep["total_reward"] += reward
                ep["done"] = terminated or truncated
                ep["history"].append((ep["step"], action, reward, ep["done"]))

                status = (
                    "TERMINATED"
                    if terminated
                    else "TRUNCATED" if truncated else "ongoing"
                )
                print(
                    f"step {ep['step']:03d} | {_ACTION_NAME[action]:<11} | "
                    f"reward={reward:+.3f} | total={ep['total_reward']:+.3f} | {status}"
                )

                if ep["done"]:
                    # MemoryEnv pays >0 for the correct object, 0 for the wrong
                    # one; running out of time also pays 0
                    outcome = (
                        "SOLVED"
                        if reward > 0
                        else "WRONG OBJECT" if terminated else "OUT OF TIME"
                    )
                    ep["message"] = (
                        f"{outcome} -- total reward {ep['total_reward']:+.3f}"
                        f"   press R to play again"
                    )

            screen.fill(_VIEW_BG)

            # left: the whole maze, centred vertically against the taller sidebar
            frame = env.render()
            screen.blit(
                _scale_frame(pygame, frame, maze_px, maze_px),
                (0, (win_h - maze_px) // 2),
            )

            _draw_play_sidebar(screen, pygame, fonts, maze_px, win_h, self, ep)

            if detail:
                _draw_channel_panel(
                    screen,
                    pygame,
                    maze_px + _PLAY_SIDEBAR_W,
                    win_h,
                    ep["obs"],
                    fonts,
                )

            if ep["message"]:
                banner = fonts["big"].render(
                    ep["message"],
                    True,
                    _VIEW_GOOD if ep["message"].startswith("SOLVED") else _VIEW_BAD,
                )
                screen.blit(banner, (10, (win_h + maze_px) // 2 - 30))

            window.flip()
            clock.tick(_PLAY_FPS)

        env.close()
        pygame.quit()


class StartInCueView(gym.Wrapper):
    """Spawns the agent in MiniGrid MemoryEnv where the cue is actually
    visible, working around its random start position hiding the cue in most
    episodes. Rebuilds the observation after moving the agent."""

    def __init__(self, env):
        """Wrap one env.

        Args:
            env (gym.Env): a MiniGrid MemoryEnv; the wrapper assumes its
                layout, and checks that assumption on the first reset.
        """
        super().__init__(env)
        self._checked = False

    def reset(self, **kwargs):
        """Reset, then move the agent to the middle of the start room facing
        east, where the cue is in view.

        Args:
            **kwargs: passed straight to the wrapped env's reset, seed
                included.

        Returns:
            tuple[dict, dict]: (observation, info), the observation rebuilt
            after the move so it shows what the agent can now see.
        """
        self.env.reset(**kwargs)

        env = self.env.unwrapped

        # middle row of the start room; the cue sits one tile above at
        # (1, height//2 - 1), always empty since the room is >= 3 tall
        env.agent_pos = np.array((1, env.height // 2))
        env.agent_dir = 0  # east, same heading _gen_grid uses

        # checked once, not every episode -- a cheap guard the wrapper's
        # assumption about MemoryEnv's layout still holds
        if not self._checked:
            assert env.in_view(1, env.height // 2 - 1), (
                f"{env.spec.id}: the cue is not visible from the forced spawn "
                f"-- this wrapper assumes MemoryEnv's layout"
            )
            self._checked = True

        return env.gen_obs(), {}


class SequenceDataset(Dataset):
    """A torch Dataset over split_pad_mask's output. One item is one whole
    padded sequence (not a timestep), so a batch is (batch_size, L, ...).
    """

    def __init__(self, batch):
        """Wrap one rollout.

        Args:
            batch (dict): split_pad_mask's output. The hidden states come in
                as (1, n_seq, H) and are unwrapped to (n_seq, H) here, so
                default_collate can stack them like everything else.
        """
        self.data = {k: (v[0] if k in ("hxs", "cxs") else v) for k, v in batch.items()}
        self.n_seq = self.data["mask"].shape[0]

    def __len__(self):
        """How many sequences the rollout split into (int)."""
        return self.n_seq

    def __getitem__(self, i):
        """One whole padded sequence.

        Args:
            i (int): index into the sequences.

        Returns:
            dict: every key of the batch, sliced at i.
        """
        # default_collate stacks these dicts into (mb, L, ...) tensors
        return {k: v[i] for k, v in self.data.items()}


# pygame viewer drawing helpers. The sizes below are the layout: fixed canvas
# pixels, which _ViewWindow scales to fit the actual window for both viewers.

_VIEW_SIDEBAR_W = 440  # px, the right-hand panel
_VIEW_MAZE_PX = 560  # px, the square the env frame is scaled into
_VIEW_MIN_H = 880  # px, tall enough for the whole sidebar
_VIEW_CELL = 34  # px per cell of the 7x7 observation
_VIEW_FPS = 60  # redraw rate; agent's step rate is steps_per_sec
_VIEW_AUTO_DELAY = 1.5  # s to sit on a finished episode before auto-advancing

_VIEW_BG = (18, 18, 18)
_VIEW_PANEL = (26, 26, 26)
_VIEW_LINE = (65, 65, 65)
_VIEW_TEXT = (210, 210, 210)
_VIEW_DIM = (130, 130, 130)
_VIEW_HEAD = (255, 210, 80)
_VIEW_GOOD = (110, 220, 130)
_VIEW_BAD = (245, 100, 100)


class _ViewWindow:
    """A fixed-size drawing canvas, shown scaled to fit the real window.

    Both viewers lay themselves out in absolute pixels (the sidebar offsets,
    the 7x7 cell size, the font sizes are all constants above), so they cannot
    simply be handed a bigger window. Instead they keep drawing into `canvas`
    at exactly the coordinates they always used, and `flip()` blits that canvas
    -- scaled, aspect preserved, letterboxed -- into whatever size the window
    currently is. Fullscreen and resizing therefore cost the drawing code
    nothing; the one thing that has to know about the scale is the mouse, and
    `to_canvas`/`mouse` map it back.
    """

    def __init__(self, size, caption, fullscreen=False):
        """Open the window and make the canvas the viewers draw into.

        Args:
            size (tuple[int, int]): the canvas's fixed (width, height) in
                pixels -- the coordinates all the drawing code uses.
            caption (str): the window title.
            fullscreen (bool): open filling the screen instead of at `size`.
        """
        self.size = size
        self._fullscreen = bool(fullscreen)
        self._open()  # opens the real window, so convert() below has a display
        pygame.display.set_caption(caption)
        self.canvas = pygame.Surface(size).convert()

    def _open(self):
        """(Re)creates the OS window for the current fullscreen state."""
        if self._fullscreen:
            # (0, 0) means "the desktop resolution" -- the point of the flag
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(self.size, pygame.RESIZABLE)
        self._fit()

    def _fit(self):
        """Recomputes scale and letterbox offset from the window's real size."""
        win_w, win_h = self.screen.get_size()
        width, height = self.size
        # the smaller ratio is the one that fits: the other axis gets the bars
        self.scale = max(min(win_w / width, win_h / height), 0.05)
        self._dest = (max(1, int(width * self.scale)), max(1, int(height * self.scale)))
        self._origin = ((win_w - self._dest[0]) // 2, (win_h - self._dest[1]) // 2)

    def handle(self, event):
        """Consume the events that are about the window itself.

        Args:
            event (pygame.event.Event): one event off the queue.

        Returns:
            bool: True if it was window plumbing (a resize, or F11) and the
            caller should skip it; False if it is the caller's to handle.
        """
        if event.type == pygame.VIDEORESIZE:
            if not self._fullscreen:
                self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            self._fit()
            return True

        if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
            self._fullscreen = not self._fullscreen
            self._open()
            return True

        return False

    def to_canvas(self, pos):
        """Window pixel -> canvas pixel, for hit-testing a click against the
        rects the drawing code laid out.

        Args:
            pos (tuple[int, int]): (x, y) in the real window.

        Returns:
            tuple[float, float]: (x, y) on the canvas. Floats are fine --
            Rect.collidepoint takes them.
        """
        return (
            (pos[0] - self._origin[0]) / self.scale,
            (pos[1] - self._origin[1]) / self.scale,
        )

    @property
    def mouse(self):
        """The cursor in canvas pixels, tuple[float, float] -- what the hover
        highlights hit-test against."""
        return self.to_canvas(pygame.mouse.get_pos())

    def flip(self):
        """Show the frame: scale the canvas into the window, aspect preserved
        and letterboxed, then present it."""
        self.screen.fill((0, 0, 0))  # the letterbox bars
        if self._dest == self.size:
            self.screen.blit(self.canvas, self._origin)  # 1:1, no resample
        else:
            self.screen.blit(
                pygame.transform.smoothscale(self.canvas, self._dest), self._origin
            )
        pygame.display.flip()

# MiniGrid's encoding, channel 0. Copied from minigrid.core.constants rather
# than imported, so this file still needs nothing but torch/gym.
_OBJ_NAME = {
    0: "unseen",
    1: "empty",
    2: "wall",
    3: "floor",
    4: "door",
    5: "key",
    6: "ball",
    7: "box",
    8: "goal",
    9: "lava",
    10: "agent",
}
_COLOR_NAME = {0: "red", 1: "green", 2: "blue", 3: "purple", 4: "yellow", 5: "grey"}

# channel 2, minigrid.core.constants.STATE_TO_IDX. Only a door ever has one:
# everywhere else the number is 0 and means nothing.
_STATE_NAME = {0: "open", 1: "closed", 2: "locked"}

# channel 1 -> rgb, for the objects that are drawn in their own colour
_MG_RGB = [
    (220, 50, 50),
    (50, 200, 50),
    (60, 120, 220),
    (160, 50, 220),
    (240, 220, 0),
    (140, 140, 140),
]
_COLOR_DRIVEN = {4, 5, 6, 7}  # door, key, ball, box take the colour channel

# everything else has a fixed display colour
_OBJ_RGB = {
    0: (30, 30, 30),
    1: (210, 210, 210),
    2: (75, 85, 105),
    3: (185, 175, 145),
    8: (0, 200, 80),
    9: (255, 90, 0),
    10: (255, 50, 50),
}
_CELL_LABEL = {2: "W", 4: "D", 5: "K", 6: "O", 7: "[]", 8: "G", 9: "!"}

# MiniGrid's Discrete(7). Only the first three matter on MemoryEnv, which is
# itself worth seeing: a good policy puts almost no mass on the other four.
_ACTION_NAME = {
    0: "turn left",
    1: "turn right",
    2: "forward",
    3: "pick up",
    4: "drop",
    5: "toggle",
    6: "done",
}
_ACTION_USED = (0, 1, 2)  # the ones that do anything in this task


def _lum(rgb):
    """Perceived brightness of a colour, so text on top can be black or white.

    Args:
        rgb (tuple[int, int, int]): the colour, 0-255 per channel.

    Returns:
        float: luminance, 0-255.
    """
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def _cell_rgb(obj_idx, color_idx):
    """What colour to draw one observation cell.

    Args:
        obj_idx (int): channel 0, the object id.
        color_idx (int): channel 1, the colour id -- used only for the
            objects in _COLOR_DRIVEN, which are the ones MiniGrid tints.

    Returns:
        tuple[int, int, int]: the rgb to fill the cell with.
    """
    if obj_idx in _COLOR_DRIVEN:
        return _MG_RGB[color_idx] if color_idx < len(_MG_RGB) else (180, 180, 180)
    return _OBJ_RGB.get(obj_idx, (180, 180, 180))


def _find_cue(image):
    """The cue object in view, named for the sidebar.

    Only meaningful at step 0: later there are two such objects, one per
    branch of the T.

    Args:
        image (np.ndarray): the observation, (7, 7, 3) uint8.

    Returns:
        str | None: e.g. "green ball", or None when no key/ball is in view.
    """
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            obj, color, _ = image[x, y]
            if obj in (5, 6):
                return f"{_COLOR_NAME.get(color, '')} {_OBJ_NAME[obj]}".strip()
    return None


def _visible_objects(image):
    """Everything in view worth naming, with its bearing from the agent.

    Args:
        image (np.ndarray): the observation, (7, 7, 3) uint8.

    Returns:
        list[str]: one line per non-structural cell, e.g.
        "green ball -- 3 ahead, 2 left"; a single "nothing but walls in
        view" when there is none.
    """
    n = image.shape[0]
    agent_col, agent_row = n // 2, n - 1  # the agent sits bottom-centre
    lines = []

    for row in range(n):
        for col in range(n):
            # MiniGrid indexes the observation [x, y]; row 0 is farthest ahead
            obj, color, state = image[col, row]
            if obj in (0, 1, 2, 3, 10):
                continue

            ahead = agent_row - row
            side = col - agent_col
            where = f"{ahead} ahead" if ahead else "beside you"
            if side:
                where += f", {abs(side)} {'left' if side < 0 else 'right'}"

            name = f"{_COLOR_NAME.get(color, '')} {_OBJ_NAME.get(obj, '?')}".strip()
            if obj == 4:
                name += f" ({_STATE_NAME.get(state, '?')})"
            lines.append(f"{name} -- {where}")

    return lines or ["nothing but walls in view"]


def _draw_obs_grid(screen, pygame, image, ox, oy, font):
    """Draw the agent's 7x7 egocentric window as coloured cells, its own cell
    outlined and an arrow marking the way it faces.

    Args:
        screen (pygame.Surface): the canvas to draw on.
        pygame (module): the pygame module, passed in rather than imported.
        image (np.ndarray): the observation, (7, 7, 3) uint8.
        ox (int): left edge of the grid, in canvas pixels.
        oy (int): top edge of the grid, in canvas pixels.
        font (pygame.font.Font): for the one-letter cell labels.
    """
    n = image.shape[0]
    agent_col, agent_row = n // 2, n - 1

    for row in range(n):
        for col in range(n):
            obj, color, _ = image[col, row]
            rgb = _cell_rgb(obj, color)
            rect = pygame.Rect(
                ox + col * _VIEW_CELL,
                oy + row * _VIEW_CELL,
                _VIEW_CELL - 1,
                _VIEW_CELL - 1,
            )
            pygame.draw.rect(screen, rgb, rect)

            # Faint outline on every cell: "unseen" nearly matches the panel
            # behind it, so the grid shape would vanish when most is unseen.
            pygame.draw.rect(screen, (58, 58, 58), rect, 1)

            if row == agent_row and col == agent_col:
                pygame.draw.rect(screen, _VIEW_HEAD, rect, 2)

            label = _CELL_LABEL.get(obj, "")
            if label:
                ink = (20, 20, 20) if _lum(rgb) > 128 else (240, 240, 240)
                surf = font.render(label, True, ink)
                screen.blit(
                    surf,
                    (
                        rect.x + (rect.w - surf.get_width()) // 2,
                        rect.y + (rect.h - surf.get_height()) // 2,
                    ),
                )

    # the agent always faces "up" in its own view, so the arrow is fixed
    ax = ox + agent_col * _VIEW_CELL + _VIEW_CELL // 2
    ay = oy + agent_row * _VIEW_CELL - 2
    pygame.draw.polygon(screen, _VIEW_HEAD, [(ax, ay - 7), (ax - 5, ay), (ax + 5, ay)])
    pygame.draw.rect(
        screen, (110, 110, 110), (ox, oy, n * _VIEW_CELL, n * _VIEW_CELL), 1
    )


def _draw_policy(screen, pygame, probs, chosen, x, y, width, font):
    """Draw one bar per action for pi(a|s), highlighting the one about to be
    taken and dimming the four actions this task never needs.

    Args:
        screen (pygame.Surface): the canvas to draw on.
        pygame (module): the pygame module.
        probs (np.ndarray): the action distribution, (7,).
        chosen (int): the action about to be taken.
        x (int): left edge, in canvas pixels.
        y (int): top edge, in canvas pixels.
        width (int): how wide the block may be.
        font (pygame.font.Font): for the labels and percentages.

    Returns:
        int: the y the drawing left off at.
    """
    for a, p in enumerate(probs):
        used = a in _ACTION_USED
        is_next = a == chosen

        label = font.render(
            f"{_ACTION_NAME[a]:<10}", True, _VIEW_TEXT if used else (95, 95, 95)
        )
        screen.blit(label, (x, y))

        bx = x + 78
        bw = width - 78 - 46
        pygame.draw.rect(screen, (45, 45, 45), (bx, y + 2, bw, 9))
        if p > 0.001:
            fill = (
                _VIEW_HEAD if is_next else ((120, 170, 220) if used else (80, 80, 80))
            )
            pygame.draw.rect(screen, fill, (bx, y + 2, max(1, int(bw * p)), 9))

        pct = font.render(f"{p:5.1%}", True, _VIEW_TEXT if is_next else _VIEW_DIM)
        screen.blit(pct, (bx + bw + 6, y))
        y += 15

    return y


def _draw_button(screen, pygame, rect, label, font, mouse, accent, enabled=True):
    """Draw one button, lit up while the cursor is over it.

    Args:
        screen (pygame.Surface): the canvas to draw on.
        pygame (module): the pygame module.
        rect (pygame.Rect): where the button goes, in canvas pixels.
        label (str): the text on it.
        font (pygame.font.Font): for that text.
        mouse (tuple[float, float]): the cursor in CANVAS pixels; drives the
            hover highlight only.
        accent (tuple[int, int, int]): the button's colour.
        enabled (bool): False draws it flat and grey, and it stops reacting.

    Returns:
        pygame.Rect: the same rect, so the caller can hit-test clicks on it.
    """
    if not enabled:
        pygame.draw.rect(screen, (34, 34, 34), rect, border_radius=6)
        pygame.draw.rect(screen, (58, 58, 58), rect, 2, border_radius=6)
        surf = font.render(label, True, (78, 78, 78))
        screen.blit(
            surf,
            (
                rect.x + (rect.w - surf.get_width()) // 2,
                rect.y + (rect.h - surf.get_height()) // 2,
            ),
        )
        return rect

    hot = rect.collidepoint(mouse)
    body = accent if hot else tuple(int(c * 0.42) for c in accent)
    pygame.draw.rect(screen, body, rect, border_radius=6)
    pygame.draw.rect(screen, accent, rect, 2, border_radius=6)

    ink = (15, 15, 15) if hot else (235, 235, 235)
    surf = font.render(label, True, ink)
    screen.blit(
        surf,
        (
            rect.x + (rect.w - surf.get_width()) // 2,
            rect.y + (rect.h - surf.get_height()) // 2,
        ),
    )
    return rect


def _draw_viewer_sidebar(
    screen,
    fonts,
    maze_px,
    win_h,
    config,
    ep,
    checkpoint,
    deterministic,
    ui,
    mouse,
):
    """Draw the whole right-hand info/control panel for watch_agent.

    Args:
        screen (pygame.Surface): the canvas to draw on.
        fonts (dict): {name: pygame.font.Font} -- "head", "body", "tiny", "big".
        maze_px (int): width of the maze on the left, so the panel's x origin.
        win_h (int): canvas height, which the controls are pinned to.
        config (Config): read for the encoder name, the env and the eval set.
        ep (dict): the live episode state watch_agent maintains.
        checkpoint (dict): the loaded file, for the "what is loaded" line.
        deterministic (bool): whether actions are argmax or sampled.
        ui (dict): paused (bool), auto_new (bool) and auto_in (float | None),
            the countdown to the next maze.
        mouse (tuple[float, float]): the cursor in CANVAS pixels (see
            _ViewWindow), not window pixels; drives the hover highlight only.

    Returns:
        dict: {name: pygame.Rect} for every button, for click hit-testing.
    """
    import pygame

    paused = ui["paused"]

    sx = maze_px
    pygame.draw.rect(screen, _VIEW_PANEL, (sx, 0, _VIEW_SIDEBAR_W, win_h))

    pad = sx + 12
    inner = _VIEW_SIDEBAR_W - 24
    y = 12

    def put(text, font="body", color=_VIEW_TEXT, dy=3):
        """One line down the panel: text (str) in fonts[font] (str) and colour
        (rgb tuple), leaving dy (int) pixels of gap. Advances y."""
        nonlocal y
        surf = fonts[font].render(text, True, color)
        screen.blit(surf, (pad, y))
        y += surf.get_height() + dy

    def rule():
        """A horizontal divider across the panel. Advances y."""
        nonlocal y
        y += 5
        pygame.draw.line(screen, _VIEW_LINE, (sx + 6, y), (sx + _VIEW_SIDEBAR_W - 6, y))
        y += 7

    # ---- what is loaded -------------------------------------------------
    put(
        f"{config.feature_extractor.upper()}  on  {config.name_env}", "head", _VIEW_HEAD
    )
    trained = checkpoint.get("iteration")
    scored = checkpoint.get("eval_success_rate")
    if trained is not None or scored is not None:
        bits = []
        if trained is not None:
            bits.append(f"iter {trained}")
        if scored is not None:
            bits.append(f"eval success {scored:.2f}")
        put("checkpoint: " + "   ".join(bits), "tiny", _VIEW_DIM)
    put(
        f"actions: {'argmax (deterministic)' if deterministic else 'sampled from pi'}",
        "tiny",
        _VIEW_DIM,
    )
    rule()

    # ---- where we are ---------------------------------------------------
    put(
        f"eval maze {ep['index'] + 1} / {config.n_eval_episodes}"
        f"    seed {config.eval_seed + ep['index']}",
        "body",
        _VIEW_TEXT,
    )
    put(
        f"step {ep['step']} / {ep['max_steps']}"
        f"    reward {ep['total_reward']:+.3f}",
        "body",
        _VIEW_TEXT,
    )

    if ep["cue"]:
        put(f"CUE AT STEP 0:  {ep['cue']}", "head", (150, 200, 255))
    if ep["done"]:
        put(
            ep["outcome"], "big", _VIEW_GOOD if ep["outcome"] == "SOLVED" else _VIEW_BAD
        )
        if ui["auto_in"] is not None:
            put(f"next maze in {ui['auto_in']:.1f}s", "tiny", _VIEW_DIM)
    elif paused:
        put("PAUSED", "big", _VIEW_HEAD)
    rule()

    # ---- the agent's actual input ---------------------------------------
    put("WHAT THE AGENT SEES  (7x7, its ENTIRE input)", "head", _VIEW_HEAD)
    y += 2
    grid_px = 7 * _VIEW_CELL
    _draw_obs_grid(
        screen,
        pygame,
        ep["obs"],
        sx + (_VIEW_SIDEBAR_W - grid_px) // 2,
        y,
        fonts["tiny"],
    )
    y += grid_px + 6

    for line in _visible_objects(ep["obs"])[:3]:
        put(line, "tiny", _VIEW_DIM, dy=1)
    rule()

    # ---- the model's internals ------------------------------------------
    put("POLICY  pi(a | s)", "head", _VIEW_HEAD)
    y += 2
    y = _draw_policy(
        screen, pygame, ep["probs"], ep["action"], pad, y, inner, fonts["body"]
    )
    y += 4
    put(f"critic  V(s) = {ep['value']:+.3f}", "body", _VIEW_DIM)
    rule()

    # ---- what it just did -----------------------------------------------
    put("LAST ACTIONS", "head", _VIEW_HEAD)
    y += 2
    for step_i, action, reward in reversed(ep["history"][-6:]):
        color = _VIEW_GOOD if reward > 0 else _VIEW_TEXT
        put(
            f"s{step_i:03d}  {_ACTION_NAME[action]:<10} r={reward:+.3f}",
            "tiny",
            color,
            dy=1,
        )

    # Three rows: transport, episode, and the persistent toggle at the bottom
    # (shorter, since it changes future behavior rather than acting now).
    bh, th, gap = 42, 32, 12
    row_toggle = win_h - th - 34
    row_bottom = row_toggle - bh - gap
    row_top = row_bottom - bh - gap

    # Both action rows share these columns so they line up vertically:
    #   STEP -1 / PAUSE / STEP +1  over  LAST GAME / REPLAY / NEW GAME
    side = 104
    middle = inner - 2 * side - 2 * gap
    col_left = pad
    col_mid = pad + side + gap
    col_right = col_mid + middle + gap

    buttons = {
        # disabled at step 0, nothing to go back to
        "back": _draw_button(
            screen,
            pygame,
            pygame.Rect(col_left, row_top, side, bh),
            "STEP -1",
            fonts["head"],
            mouse,
            (170, 170, 190),
            enabled=ep["step"] > 0,
        ),
        "pause": _draw_button(
            screen,
            pygame,
            pygame.Rect(col_mid, row_top, middle, bh),
            "PLAY" if paused else "PAUSE",
            fonts["head"],
            mouse,
            (120, 230, 140) if paused else (200, 200, 200),
        ),
        "step": _draw_button(
            screen,
            pygame,
            pygame.Rect(col_right, row_top, side, bh),
            "STEP +1",
            fonts["head"],
            mouse,
            (170, 170, 190),
            enabled=not ep["done"],
        ),
        # wraps round the eval set; never disabled, unlike STEP -1
        "last": _draw_button(
            screen,
            pygame,
            pygame.Rect(col_left, row_bottom, side, bh),
            "LAST GAME",
            fonts["head"],
            mouse,
            (90, 190, 255),
        ),
        "replay": _draw_button(
            screen,
            pygame,
            pygame.Rect(col_mid, row_bottom, middle, bh),
            "REPLAY",
            fonts["head"],
            mouse,
            (255, 190, 90),
        ),
        "new": _draw_button(
            screen,
            pygame,
            pygame.Rect(col_right, row_bottom, side, bh),
            "NEW GAME",
            fonts["head"],
            mouse,
            (90, 190, 255),
        ),
        # a setting, not an action: auto-starts the next eval maze on episode end
        "auto": _draw_button(
            screen,
            pygame,
            pygame.Rect(pad, row_toggle, inner, th),
            f"AUTO NEW GAME:  {'ON' if ui['auto_new'] else 'OFF'}",
            fonts["body"],
            mouse,
            (120, 230, 140) if ui["auto_new"] else (120, 120, 130),
        ),
    }

    # two lines, not one: the full list no longer fits the sidebar width
    hint_y = row_toggle + th + 6
    for line in (
        "SPACE pause   <- -> step   P/N maze   R replay   A auto",
        "F11 fullscreen   Q quit",
    ):
        surf = fonts["tiny"].render(line, True, (105, 105, 105))
        screen.blit(surf, (pad, hint_y))
        hint_y += surf.get_height() + 1

    return buttons


# ---------------------------------------------------------------------------
# The human-playable viewer, Helper.play_env: watch_agent's env and drawing
# primitives with a keyboard where the policy was, to show the 7x7 observation.
# ---------------------------------------------------------------------------

_PLAY_SIDEBAR_W = 430  # px, the info panel beside the maze
_PLAY_CHANNEL_W = 380  # px, the extra column --detail adds
_PLAY_MIN_H = 820  # px, tall enough for the three channel tables
_PLAY_FPS = 30  # nothing moves between keystrokes, so 30 is plenty
_PLAY_CELL_W = 44  # px per cell of a channel table
_PLAY_CELL_H = 26
_PLAY_ROW_LABEL_W = 28  # px reserved for the "y=3" row labels

# key label -> what it does. One list, printed to the terminal on start and
# drawn at the bottom of the sidebar, so the two can never disagree.
_PLAY_KEYS = (
    ("Arrow Left", "turn left"),
    ("Arrow Right", "turn right"),
    ("Arrow Up", "move forward"),
    ("P", "pick up"),
    ("D", "drop"),
    ("Space", "toggle / open door"),
    ("Enter", "done"),
    ("R", "reset episode"),
    ("F11", "fullscreen on / off"),
    ("Q / Esc", "quit"),
)

# the same seven, as pygame keycode -> index into MiniGrid's Discrete(7)
# (_ACTION_NAME lists them in order). Only 0-2 do anything on MemoryEnv.
_PLAY_ACTION_KEYS = {
    pygame.K_LEFT: 0,
    pygame.K_RIGHT: 1,
    pygame.K_UP: 2,
    pygame.K_p: 3,
    pygame.K_d: 4,
    pygame.K_SPACE: 5,
    pygame.K_RETURN: 6,
}

# swatch label, colour, meaning -- the cells MemoryEnv actually produces.
# Labels match _CELL_LABEL, which is what _draw_obs_grid stamps on them.
_PLAY_LEGEND = (
    ("^", _VIEW_HEAD, "you (agent)"),
    ("W", _OBJ_RGB[2], "wall"),
    ("D", (140, 80, 20), "door"),
    ("K", (0, 200, 80), "key"),
    ("O", (0, 200, 80), "ball"),
    ("", _OBJ_RGB[0], "unseen"),
)

# axis 2 of obs["image"]: channel index, name, and the table decoding it
_PLAY_CHANNELS = (
    (0, "object id", _OBJ_NAME),
    (1, "color id", _COLOR_NAME),
    (2, "state id", _STATE_NAME),
)


def _scale_frame(pygame, frame, width, height):
    """A rendered env frame as a pygame surface of the size wanted.

    Args:
        pygame (module): the pygame module.
        frame (np.ndarray): what env.render() returned, (h, w, 3) uint8.
        width (int): target width in canvas pixels.
        height (int): target height in canvas pixels.

    Returns:
        pygame.Surface: the scaled frame, ready to blit.
    """
    surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
    return pygame.transform.scale(surface, (width, height))


def _front_cell_hint(image):
    """The one line a human at the keyboard needs: what is directly ahead and
    which key acts on it.

    Args:
        image (np.ndarray): the observation, (7, 7, 3) uint8.

    Returns:
        str | None: the hint, or None when there is nothing worth saying.
    """
    n = image.shape[0]
    if n < 2:
        return None

    # one cell forward of the agent, which sits bottom-centre facing "up"
    obj, color, state = image[n // 2, n - 2]
    name = f"{_COLOR_NAME.get(color, '')} {_OBJ_NAME.get(obj, '?')}".strip()

    if obj == 2:
        return ">> wall directly ahead -- cannot move forward"
    if obj == 4:
        opened = _STATE_NAME.get(state, "?")
        return (
            f">> {name} ahead ({opened}) -- Space to "
            f"{'close' if opened == 'open' else 'open'}"
        )
    if obj in (5, 6, 7):
        return f">> {name} directly ahead -- P to pick it up"
    if obj == 8:
        return ">> goal tile directly ahead -- forward to reach it"
    if obj == 9:
        return ">> lava directly ahead -- forward ends the episode"
    return None


def _draw_play_legend(screen, pygame, ox, y, font):
    """The swatch row under the observation grid, saying what each colour is.

    Args:
        screen (pygame.Surface): the canvas to draw on.
        pygame (module): the pygame module.
        ox (int): the panel's left edge, in canvas pixels.
        y (int): where to start, in canvas pixels.
        font (pygame.font.Font): for the swatch labels.

    Returns:
        int: the y it left off at, having wrapped as needed.
    """
    x = ox + 6
    for label, color, name in _PLAY_LEGEND:
        if x + 100 > ox + _PLAY_SIDEBAR_W - 6:
            x = ox + 6
            y += 18

        pygame.draw.rect(screen, color, (x, y, 13, 13))
        if label:
            ink = (20, 20, 20) if _lum(color) > 128 else (240, 240, 240)
            surf = font.render(label, True, ink)
            screen.blit(
                surf,
                (x + (13 - surf.get_width()) // 2, y + (13 - surf.get_height()) // 2),
            )

        text = font.render(f" {name}", True, (160, 160, 160))
        screen.blit(text, (x + 14, y))
        x += 14 + text.get_width() + 10

    return y + 18


def _draw_channel_block(screen, pygame, ox, y, image, channel, name, table, fonts):
    """One channel of obs["image"] as a 7x7 table: the raw id above the name it
    decodes to, in every cell. Cells whose number is filler rather than
    information are dimmed.

    Indexed image[x, y, channel], drawn with rows = y (0 = farthest ahead) and
    cols = x (3 = straight ahead).

    Args:
        screen (pygame.Surface): the canvas to draw on.
        pygame (module): the pygame module.
        ox (int): the panel's left edge, in canvas pixels.
        y (int): where to start, in canvas pixels.
        image (np.ndarray): the observation, (7, 7, 3) uint8.
        channel (int): which of the three to draw, 0-2.
        name (str): that channel's name, for the header.
        table (dict): {id: name}, the decoding for this channel.
        fonts (dict): {name: pygame.font.Font}; uses "tiny" and "micro".

    Returns:
        int: the y it left off at.
    """
    n = image.shape[0]
    agent_col, agent_row = n // 2, n - 1

    present = ",".join(
        str(value) for value in sorted({int(v) for v in image[:, :, channel].ravel()})
    )
    header = fonts["tiny"].render(
        f"c={channel}  {name}   present: {present}", True, _VIEW_HEAD
    )
    screen.blit(header, (ox, y))
    y += header.get_height() + 3

    gx = ox + _PLAY_ROW_LABEL_W
    for col in range(n):
        label = fonts["micro"].render(f"x={col}", True, (120, 120, 120))
        x = gx + col * _PLAY_CELL_W + (_PLAY_CELL_W - label.get_width()) // 2
        screen.blit(label, (x, y))
    y += 12

    for row in range(n):
        label = fonts["micro"].render(f"y={row}", True, (120, 120, 120))
        screen.blit(label, (ox, y + (_PLAY_CELL_H - label.get_height()) // 2))

        for col in range(n):
            value = int(image[col, row, channel])
            obj = int(image[col, row, 0])

            # Dim the cells whose number is filler: colour is 0 on unseen/empty
            # (decodes "red", means nothing), state is 0 except on a door.
            if channel == 1:
                filler = obj in (0, 1)
            elif channel == 2:
                filler = obj != 4
            else:
                filler = False

            rect = pygame.Rect(
                gx + col * _PLAY_CELL_W, y, _PLAY_CELL_W - 2, _PLAY_CELL_H - 2
            )
            pygame.draw.rect(screen, (30, 30, 34) if filler else (46, 48, 56), rect)
            if row == agent_row and col == agent_col:
                pygame.draw.rect(screen, _VIEW_HEAD, rect, 1)

            number = fonts["micro"].render(
                str(value), True, (85, 85, 85) if filler else (240, 240, 240)
            )
            decoded = fonts["micro"].render(
                table.get(value, "?"), True, (75, 75, 75) if filler else (150, 205, 150)
            )
            screen.blit(
                number, (rect.x + (rect.w - number.get_width()) // 2, rect.y + 1)
            )
            screen.blit(
                decoded, (rect.x + (rect.w - decoded.get_width()) // 2, rect.y + 12)
            )

        y += _PLAY_CELL_H

    return y


def _draw_channel_panel(screen, pygame, ox, win_h, image, fonts):
    """--detail's third column: all three channel tables, one under the other,
    with a note on how to read the axes and the dimming.

    Args:
        screen (pygame.Surface): the canvas to draw on.
        pygame (module): the pygame module.
        ox (int): the column's left edge, in canvas pixels.
        win_h (int): canvas height, so the panel fills it.
        image (np.ndarray): the observation, (7, 7, 3) uint8.
        fonts (dict): {name: pygame.font.Font}; uses "head", "tiny", "micro".
    """
    pygame.draw.rect(screen, (20, 20, 20), (ox, 0, _PLAY_CHANNEL_W, win_h))
    pad = ox + 10
    y = 10

    title = fonts["head"].render(
        "CHANNEL VALUES  obs['image'][x, y, c]", True, _VIEW_HEAD
    )
    screen.blit(title, (pad, y))
    y += title.get_height() + 4

    for line in (
        f"shape {image.shape} = {image.size} uint8 numbers, axis 2 is NOT RGB",
        "rows = y: 0 = farthest ahead, 6 = the agent's own row",
        "cols = x: 0 = far left, 3 = straight ahead, 6 = far right",
    ):
        surf = fonts["micro"].render(line, True, (140, 140, 140))
        screen.blit(surf, (pad, y))
        y += surf.get_height() + 1
    y += 6

    for channel, name, table in _PLAY_CHANNELS:
        y = _draw_channel_block(
            screen, pygame, pad, y, image, channel, name, table, fonts
        )
        y += 10

    pygame.draw.line(screen, _VIEW_LINE, (ox + 5, y), (ox + _PLAY_CHANNEL_W - 5, y))
    y += 6
    for line in (
        "dimmed = filler, not information. colour is 0 (-> 'red') on",
        "unseen/empty cells; state is only real on a door, so in this",
        "env c=2 is all filler. Yellow border = the agent's own cell.",
    ):
        surf = fonts["micro"].render(line, True, (120, 120, 120))
        screen.blit(surf, (pad, y))
        y += surf.get_height() + 1


def _draw_play_sidebar(screen, pygame, fonts, ox, win_h, config, ep):
    """play_env's info panel: where we are, the 7x7 observation, what is in it,
    and the last few actions. The key list is pinned to the bottom.

    Args:
        screen (pygame.Surface): the canvas to draw on.
        pygame (module): the pygame module.
        fonts (dict): {name: pygame.font.Font} -- "head", "body", "tiny".
        ox (int): width of the maze on the left, so the panel's x origin.
        win_h (int): canvas height, which the key list is pinned to.
        config (Config): read for the env's name.
        ep (dict): the live episode state play_env maintains.
    """
    pygame.draw.rect(screen, _VIEW_PANEL, (ox, 0, _PLAY_SIDEBAR_W, win_h))

    pad = ox + 10
    y = 10

    def put(text, font="body", color=_VIEW_TEXT):
        """One line down the panel: text (str) in fonts[font] (str) and colour
        (rgb tuple). Advances y."""
        nonlocal y
        surf = fonts[font].render(text, True, color)
        screen.blit(surf, (pad, y))
        y += surf.get_height() + 3

    def rule():
        """A horizontal divider across the panel. Advances y."""
        nonlocal y
        y += 5
        pygame.draw.line(screen, _VIEW_LINE, (ox + 5, y), (ox + _PLAY_SIDEBAR_W - 5, y))
        y += 7

    # ---- what is loaded -------------------------------------------------
    put(f"ENV: {config.name_env}", "head", _VIEW_HEAD)
    put(
        f"step {ep['step']} / {ep['max_steps']}"
        f"    total reward {ep['total_reward']:+.3f}",
        "body",
        _VIEW_GOOD,
    )
    rule()

    # ---- the agent's actual input ---------------------------------------
    put("PARTIAL OBSERVATION  (7x7 egocentric view)", "head", _VIEW_HEAD)
    y += 3
    grid_px = 7 * _VIEW_CELL
    _draw_obs_grid(
        screen,
        pygame,
        ep["obs"],
        ox + (_PLAY_SIDEBAR_W - grid_px) // 2,
        y,
        fonts["tiny"],
    )
    y += grid_px + 8

    y = _draw_play_legend(screen, pygame, ox, y, fonts["tiny"])
    y += 2
    rule()

    # ---- what that means ------------------------------------------------
    put("WHAT YOU SEE", "head", _VIEW_HEAD)
    y += 2

    hint = _front_cell_hint(ep["obs"])
    for line in ([hint] if hint else []) + _visible_objects(ep["obs"]):
        # soft word-wrap: these lines name a colour, an object and a bearing,
        # and run past the panel on the wider observations
        row = ""
        for word in line.split():
            candidate = f"{row} {word}".strip()
            if fonts["body"].size(candidate)[0] > _PLAY_SIDEBAR_W - 20:
                put(row, "body", (200, 200, 200))
                row = word
            else:
                row = candidate
        if row:
            put(row, "body", (200, 200, 200))
    y += 4
    rule()

    # ---- what you just did ----------------------------------------------
    put("LAST ACTIONS", "head", _VIEW_HEAD)
    y += 2
    for step_i, action, reward, done in reversed(ep["history"][-6:]):
        put(
            f"s{step_i:03d}  {_ACTION_NAME[action]:<11} r={reward:+.3f}"
            f"{'  END' if done else ''}",
            "body",
            _VIEW_BAD if done else (190, 190, 190),
        )

    # ---- controls, pinned to the bottom ---------------------------------
    y = win_h - len(_PLAY_KEYS) * 14 - 20
    pygame.draw.line(screen, _VIEW_LINE, (ox + 5, y), (ox + _PLAY_SIDEBAR_W - 5, y))
    y += 6
    for key, description in _PLAY_KEYS:
        surf = fonts["tiny"].render(f"{key:<14} {description}", True, (120, 120, 120))
        screen.blit(surf, (pad, y))
        y += surf.get_height() + 2
