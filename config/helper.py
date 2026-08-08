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
from models.feature_extractor import MLP, LSTM, GRU, Transformer
import joblib
from torchinfo import summary
import optuna
import pygame
from models.model import Network

# The two metrics a run reports and the study can rank on. Both are one
# number per seed; mean and std then aggregate ACROSS seeds, never across the
# eval episodes within one run.
_HPO_METRICS = ("return_mean", "success_rate")

# Exactly two objectives exist, both scored mean - hpo_lambda*std across seeds.
# Median/IQR were dropped: with a handful of seeds they estimate nothing the
# mean/std pair doesn't, and two objectives keep every report comparable.
_HPO_OBJECTIVES = {
    "return_mean_minus_std": "return_mean",
    "success_rate_mean_minus_std": "success_rate",
}

# the columns of the closing per-seed table (Helper.log_seed_summary), shared
# by the hand-picked run and the search's final report. "sampled" is the policy
# sampled from pi, "argmax" the same weights evaluated greedily.
_SEED_SUMMARY_METRICS = (
    "sampled_return",
    "sampled_success_rate",
    "argmax_return",
    "argmax_success_rate",
)


def parse_hpo_objective(objective):
    """The per-seed metric behind hpo_objective. Only the two entries of
    _HPO_OBJECTIVES are accepted; anything else raises rather than defaulting
    silently to a scale the rest of the reports would not match."""
    key = str(objective).strip().lower().replace("-", "_")
    if key not in _HPO_OBJECTIVES:
        raise ValueError(
            f"hpo_objective {objective!r} is not one of "
            f"{sorted(_HPO_OBJECTIVES)}. Both are scored "
            f"mean - hpo_lambda*std across seeds; they differ only in which "
            f"metric is aggregated."
        )
    return _HPO_OBJECTIVES[key]


def format_clock(seconds):
    """A duration someone WAITS through: 1h 05m 03s / 5m 03s / 42.1s."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def format_mean(seconds):
    """A per-call mean duration, in whatever unit (us/ms/s) keeps it readable."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f}us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f}ms"
    return f"{seconds:.3f}s"


def log_with(logger, level="info"):
    """logger.<level>, or print when there is no logger -- so everything here
    is usable from a scratch script and from a real run without a branch at
    every call site."""
    return print if logger is None else getattr(logger, level)


class Timing:
    """Every clock a training run keeps. The caller only names phases --
    `with timing.phase("sample"):` -- and the arithmetic/formatting lives here.
    Two sets of per-phase [seconds, units, calls] totals: `window` since the
    last report(), `run` for the whole seed; report() folds the first into the
    second so they never double-count. enabled=False makes phase() free."""

    # phase key -> label, in the order the table prints them. A phase that has
    # no row here is still accumulated, just not shown; a row whose phase was
    # never timed is skipped.
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
        was merely launched. when=False makes it a no-op -- for hot loops where
        the synchronize would cost more than the thing it is timing."""
        if when and self.device.type == "cuda":
            torch.cuda.synchronize()

    def add(self, phase, seconds, units=1):
        """Add one measurement to `phase`: one call, processing `units` of
        whatever that call handles (W env steps, one minibatch, one forward)."""
        entry = self.window.setdefault(phase, [0.0, 0, 0])
        entry[0] += seconds
        entry[1] += units
        entry[2] += 1

    @contextmanager
    def phase(self, name, units=1, sync=True):
        """Time the block and add it to `name`. sync=False for a region that
        queues no GPU work, or one ending in a copy that forces the sync anyway.
        No try/finally: a block that raised never ran to completion, so it is
        not a measurement worth keeping."""
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
        """Add the window's totals to the run's and empty it; returns what the window held."""
        window = dict(self.window)
        for key, (seconds, units, calls) in window.items():
            total = self.run.setdefault(key, [0.0, 0, 0])
            total[0] += seconds
            total[1] += units
            total[2] += calls
        self.window.clear()
        return window

    def report(self, log):
        """Print the table for everything since the last report and open a new
        window. The numbers are not lost, only moved: fold() keeps them in `run`."""
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
        """Print the table for the whole seed. Call it after the last report(),
        which is what folded the final window into `run`."""
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
        """One phase-by-phase table. `timing` is {phase: [seconds, units, calls]}
        and must hold an "iteration" entry: the share column is a percentage of it."""
        iteration_total, _, n_iterations = timing["iteration"]

        def row(label, seconds, units, calls):
            share = 100.0 * seconds / iteration_total if iteration_total else 0.0

            per_call = format_mean(seconds / calls) if calls else "-"

            # "-" when the two agree (one forward per call, one iteration per
            # call): a per-unit column would just repeat per call. units=0 is
            # deliberate for phases counted in seconds only -- an env reset is
            # not an env step -- and has no per-unit either.
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


# what an out-of-memory RuntimeError says, lowercased. The CPU allocator is
# matched on its class/file name too, since the wording ("can't"/"cannot") has
# changed between torch versions.
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
        return self.feature_extractor.upper() in ("LSTM", "GRU")

    @property
    def is_lstm(self):
        """An LSTM needs (h, c). A GRU needs only h. An MLP needs neither."""
        return self.feature_extractor.upper() == "LSTM"

    @property
    def path_model(self):
        """The checkpoint this config points at, re-derived every read since dir_pretrained_model moves."""
        return os.path.join(self.dir_pretrained_model, self.name_model)

    def build_env(self, render_mode=None):
        """One MiniGrid game, wrapped as the config asks. Every env in the project comes from here."""
        # Overriding max_steps with worker_steps is the point, not an option:
        # MiniGrid computes truncation AND the success reward
        # (1 - 0.9*step_count/max_steps) from it, so an episode can never
        # outlive one rollout and the reward scale stays pinned to T.
        env = gym.make(
            self.name_env,
            render_mode=render_mode,
            max_steps=int(self.worker_steps),
        )

        if self.force_cue_visible:
            env = StartInCueView(env)

        return env

    def build_vector_env(self, n_envs):
        """n_envs MiniGrid games behind one step() call, via AsyncVectorEnv
        (or SyncVectorEnv if async_envs=False). Applies force_cue_visible and
        drops non-image obs so the batch fits in shared memory."""
        # a closure, not a bound method, so the subprocess pickles this and
        # not the whole Config. gymnasium wraps env_fns in cloudpickle, so a
        # lambda survives the spawn start method macOS defaults to.
        def make():
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
        """The env's own DEFAULT time limit (MemoryEnv: 5 * size^2), read off a
        throwaway env so it can never disagree with name_env. Bare gym.make, not
        build_env: build_env overrides max_steps with worker_steps, which would
        both hand worker_steps back here and make this unreadable from the line
        in Config.__init__ that derives worker_steps from it."""
        env = gym.make(self.name_env)
        max_steps = env.unwrapped.max_steps
        env.close()
        return max_steps

    def build_extractor(self):
        """The encoder named by self.feature_extractor. All four map (batch, seq, 7, 7, 3) -> (batch, seq, hidden_size)."""
        name = self.feature_extractor.upper()

        if name == "MLP":
            return MLP(self.input_size, self.hidden_size, self.n_layers_mlp)
        if name == "LSTM":
            return LSTM(self.input_size, self.hidden_size)
        if name == "GRU":
            return GRU(self.input_size, self.hidden_size)
        if name == "TRANSFORMER":
            # is_recurrent is False here: no hidden state, so zero_hidden /
            # reset_hidden_of are no-ops for it. Also: sample() calls it with
            # seq_len=1, so while acting it attends over nothing and behaves
            # like an MLP -- only the update sees real history. See
            # feature_extractor module docstring.
            return Transformer(
                self.input_size,
                self.hidden_size,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers_transformer,
                d_ff=self.d_ff,
                p_drop=self.p_drop,
                max_seq_length=self.max_seq_length,
            )

        raise ValueError(f"unknown feature_extractor {self.feature_extractor!r}")

    def config_attributes(self, include_private=False):
        """Every attribute this config carries, as a dict sorted by name: the
        instance attributes plus the class-level ones and the @property values
        defined anywhere in the hierarchy (the encoder subclass, Config,
        Helper). Properties are read defensively -- one that raises is reported
        in place rather than taking the caller down."""
        values = {}

        def keep(key):
            return include_private or not key.startswith("_")

        # what __init__ set, and what apply_params() has since overwritten
        for key, value in vars(self).items():
            if keep(key):
                values[key] = value

        # @property values and plain class attributes live on the class, not in
        # vars(self), so they need the MRO walk. It runs most-derived-first and
        # the `key in values` guard keeps the first hit, so an instance
        # attribute wins over a class default and a subclass property (d_ff)
        # wins over a base-class one of the same name.
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
        """Writes config_attributes() one per line, alphabetically. A list of
        dicts (search_space) goes one entry per line instead of one long
        line."""
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

    def build_logger(self, log_dir="logs", name="rl_project"):
        """Builds one logger per invocation: writes to
        logs/log_<date>_<time>.log and the terminal, with every hyperparameter
        dumped at the top. Returns a logging.Logger; pass to train_agent(logger=...)."""
        os.makedirs(log_dir, exist_ok=True)

        started = datetime.now()
        path = os.path.join(log_dir, f"log_{started:%Y-%m-%d_%H-%M-%S}.log")

        # _2, _3, ... if the second-precision stamp collides (two commands
        # started back to back); FileHandler opens in append mode, so without
        # this the second run's log would silently continue the first's.
        collision = 1
        while os.path.exists(path):
            collision += 1
            path = os.path.join(
                log_dir, f"log_{started:%Y-%m-%d_%H-%M-%S}_{collision}.log"
            )

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Logger is a singleton per name; without this a second build_logger()
        # call keeps the first call's handlers and doubles every line.
        logger.handlers.clear()
        logger.propagate = False  # don't also hand records to the root logger

        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(file_handler)

        # same format as the file, not a bare message -- so terminal lines are
        # datable on sight and stay aligned with torchinfo's summary output.
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(stream_handler)

        bar = "=" * 78
        logger.info(bar)
        logger.info(f"RUN STARTED  {started:%Y-%m-%d %H:%M:%S}")
        logger.info(f"LOG FILE     {path}")
        logger.info(bar)

        # attributes and @property values together, alphabetically -- the
        # derived ones (device, path_model, ...) decide what was built and
        # where it lands, so they belong in the dump next to what they came from
        self.log_config_attributes(logger)

        logger.info(bar)

        return logger

    # running at the largest mini_batch_size candidate that fits in memory
    def _iter_error_chain(self, error):
        """Yields error and everything it was raised from or during (via
        __cause__/__context__), so a wrapped OOM (e.g. torchinfo re-raising a
        generic RuntimeError) is still found. Guards against __context__ cycles.
        """
        seen = set()
        while error is not None and id(error) not in seen:
            seen.add(id(error))
            yield error
            error = error.__cause__ or error.__context__

    def _is_oom(self, error):
        """True if error, or anything it wraps, is a CUDA or CPU-allocator out-of-memory error."""
        for err in self._iter_error_chain(error):
            if isinstance(err, torch.cuda.OutOfMemoryError):
                return True
            if isinstance(err, RuntimeError) and any(
                text in str(err).lower() for text in _OOM_TEXT
            ):
                return True
        return False

    def _clear_traceback_chain(self, error):
        """Drops the traceback of error and every cause it wraps, so a retry
        doesn't keep the failed attempt's tensors referenced and OOM again."""
        for err in self._iter_error_chain(error):
            err.__traceback__ = None

    def _free_memory(self):
        """Give the allocator back whatever the failed attempt left behind."""
        gc.collect()
        if torch.cuda.is_available():
            # python freeing a tensor returns it to torch's caching allocator,
            # not to the driver. Without this the memory is free as far as
            # torch is concerned and still unavailable to anything else.
            torch.cuda.empty_cache()

    def run_with_batch_size_fallback(
        self, run_fn, batch_size, logger=None, what="batch size"
    ):
        """Calls run_fn(size) at the largest batch_size candidate that doesn't
        OOM, falling back on failure. Returns (size_used, result); re-raises
        the last OOM if every candidate fails. Non-OOM exceptions propagate."""
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
        """Resolve the largest of `candidates` that fits, before training starts.
        One forward/backward/step per candidate on an all-zero worst-case batch,
        not on iteration 0's real data: early rollouts pack fewer sequences than
        later ones, so real data would underestimate. loss_fn gets a throwaway
        deep copy of `model`, so the real weights never see the probe."""
        probe = copy.deepcopy(model)
        optimizer = torch.optim.Adam(
            probe.parameters(), lr=self.lr, weight_decay=self.wd
        )
        # the longest sequence split_pad_mask can emit: a chunk is capped at
        # tbptt_length, so probing at worker_steps would overestimate the batch
        # by T/tbptt_length and resolve a needlessly small minibatch
        L = (
            self.worker_steps
            if self.tbptt_length == "max"
            else min(int(self.tbptt_length), self.worker_steps)
        )

        def step(n):
            # the keys split_pad_mask produces, in the shapes SequenceDataset
            # would have collated them into. All-ones mask: nothing padded is
            # the worst case, every slot is real work.
            mb = {
                "obs": torch.zeros(n, L, *obs_shape, dtype=torch.uint8),
                "actions": torch.zeros(n, L, dtype=torch.long),
                "log_probs": torch.zeros(n, L),
                "advantages": torch.zeros(n, L),
                "returns": torch.zeros(n, L),
                "mask": torch.ones(n, L),
            }

            # None for an MLP, which is why neither branch runs for it
            hidden = self.zero_hidden(n)
            if self.is_lstm:
                mb["hxs"], mb["cxs"] = hidden[0].squeeze(0), hidden[1].squeeze(0)
            elif self.is_recurrent:
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
        """Runs torchinfo's summary on the model and logs the table (param
        counts, shapes, size). batch_size/seq_len only shape the probe pass.
        Returns the ModelStatistics object."""

        if batch_size is None:
            batch_size = self.mini_batch_size

        # runs through the same OOM fallback as the real update: this probe
        # forward pass can itself OOM on a wide transformer. Doesn't decide
        # what training uses -- train() resolves its own size separately.
        def probe(bs):
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
        """ppo_<seed>_<ENCODER>_<env>.pth -- the checkpoint filename, e.g.
        ppo_0_GRU_MiniGrid-DoorKey-8x8-v0.pth. self.seed falls back to
        seed_list[0] if set_seed() hasn't run yet (e.g. watch.py)."""
        env = self.name_env.replace("/", "-")
        seed = getattr(self, "seed", self.seed_list[0])
        return f"ppo_{seed}_{self.feature_extractor.upper()}_{env}.pth"

    def build_model_path(self):
        """path_model, creating dir_pretrained_model first. Call right before
        torch.save; which directory this is depends on who is running (the
        encoder's top level, a trial dir under hpo/, or no_hpo/)."""
        os.makedirs(self.dir_pretrained_model, exist_ok=True)
        return self.path_model

    def save_model(self, model, optimizer=None, **extra):
        """Writes model's (and optimizer's) state, architecture attributes,
        and searched params to build_model_path(). Returns the path. **extra
        is written verbatim and must survive torch.load(weights_only=True)."""
        path = self.build_model_path()

        checkpoint = {
            "model": model.state_dict(),
            "feature_extractor": self.feature_extractor,
            "hidden_size": self.hidden_size,
            "input_size": self.input_size,
            "name_env": self.name_env,
            "force_cue_visible": self.force_cue_visible,
            "params": self.searched_params(),
        }
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        checkpoint.update(extra)

        torch.save(checkpoint, path)
        return path

    def load_model(self, model, path=None):
        """Loads weights into an already-built model, from path (default
        path_model). Returns the checkpoint dict. Raises FileNotFoundError if
        missing, ValueError if trained under a different architecture/env."""
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

        # everything this project writes goes through save_model, so this is
        # its dict. A bare state_dict from torch.save(model.state_dict(), ...)
        # is a dict too, just without the "model" key -- wrapped here so the
        # rest of the method has one shape to deal with.
        if "model" not in checkpoint:
            checkpoint = {"model": checkpoint}

        # force_cue_visible changes no tensor shape but changes what the agent
        # could see during training; checked here so a mismatch fails loudly
        # instead of silently scoring the wrong task. A bare state_dict carries
        # none of these keys and is loaded unchecked.
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

    # HPO: what to search over and how it's written/resumed lives here; what a
    # trial does (train seeds, score them) lives in agents/hpo_ppo.py. optuna
    # and joblib are imported inside these methods so a checkout without them
    # still trains, watches and scores.
    def suggest_from_search_space(self, trial):
        """Draws one value per entry of config.search_space via trial.suggest_*
        (picked by each entry's "type"). Returns {name: value}."""
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
        name = f"mean_minus_{self.hpo_lambda:g}std({self.hpo_metric})"
        if self.searches_tbptt:
            name += f"_minus_{self.tbptt_lambda:g}(tbptt/T)"
        return name

    @property
    def searches_tbptt(self):
        """True when tbptt_length is in this config's search_space -- i.e. the
        tbptt study. The max study never pays the penalty."""
        return any(entry["name"] == "tbptt_length" for entry in self.search_space)

    def aggregate_scores(self, values):
        """Reduces per-seed metric values to the score the study maximizes:
        mean(values) - hpo_lambda * std(values), both across seeds and not
        within one run. Returns a float."""
        values = np.asarray(values, dtype=float)
        # ddof=0, population std -- the seeds ARE the population being compared
        return float(values.mean()) - self.hpo_lambda * float(values.std())

    def trial_score(self, values, params=None):
        """What a trial is ranked on: aggregate_scores, minus the tbptt penalty
        when this trial drew a tbptt_length. In a "max" run params carries no
        such key, so nothing is subtracted and this is aggregate_scores."""
        score = self.aggregate_scores(values)

        length = (params or {}).get("tbptt_length")
        if length is None or length == "max":
            return score

        # normalized by T so the penalty lands in [0, tbptt_lambda] -- the same
        # scale as the return, which makes tbptt_lambda readable as "the most
        # return I will trade away to go from full BPTT to none"
        return score - self.tbptt_lambda * (int(length) / self.worker_steps)

    def apply_params(self, params):
        """Writes a trial's drawn params onto this config's attributes. Returns
        self. Must be called before PPOAgent(config) is built. Raises
        AttributeError on a name that isn't already an attribute of the config.
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
        """Current value of every name in search_space -- the inverse of
        apply_params, letting save_model record what a checkpoint was trained
        with. Empty for configs like ConfigNoHPO whose search_space is []."""
        return {
            entry["name"]: getattr(self, entry["name"])
            for entry in self.search_space
            if hasattr(self, entry["name"])
        }

    # ----- where a study writes -------------------------------------------
    def build_hpo_dir(self):
        """Make hpo/ and hand it back."""
        os.makedirs(self.dir_hpo, exist_ok=True)
        return self.dir_hpo

    def dir_hpo_trial(self, number):
        """hpo/trial_<number>/ -- one directory per trial, since the filenames are otherwise identical."""
        return os.path.join(self.dir_hpo, f"trial_{number}")

    @property
    def dir_hpo_best_trial(self):
        """hpo/best_trial/ -- a copy of the winning trial, same shape as trial_<n>/."""
        return os.path.join(self.dir_hpo, "best_trial")

    def build_hpo_trial_dir(self, number):
        """dir_hpo_trial(n), created."""
        path = self.dir_hpo_trial(number)
        os.makedirs(path, exist_ok=True)
        return path

    def build_hpo_best_trial_dir(self):
        """dir_hpo_best_trial, created."""
        os.makedirs(self.dir_hpo_best_trial, exist_ok=True)
        return self.dir_hpo_best_trial

    def select_run(self, trial=None, seed_index=0):
        """Points this config at one saved checkpoint: sets
        dir_pretrained_model (trial=None/"best"/"final"/int) and self.seed
        (via seed_index). Returns path_model; nothing is created or read."""
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
        """The hyperparameters a checkpoint (default path_model) was trained
        with, read straight off the file. Returns {} if the file is missing or
        has no recorded params -- apply_params({}) is then a no-op.
        """
        if path is None:
            path = self.path_model
        if not os.path.exists(path):
            return {}

        # a bare state_dict has no "params" either, and is a dict all the same
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return dict(checkpoint.get("params") or {})

    def copy_best_trial(self, study, logger=None):
        """Copies the winning trial's files into best_trial/ (clearing it
        first) and writes best_params.json beside them. Returns the target
        path, or None if no trial has completed yet."""
        say = log_with(logger)

        try:
            best = study.best_trial
        except ValueError:
            # optuna raises here rather than returning None when no trial has
            # finished
            say("no completed trial yet, nothing to copy into best_trial/")
            return None

        source = self.dir_hpo_trial(best.number)
        if not os.path.isdir(source):
            say(f"trial {best.number} has no directory at {source}, nothing to copy")
            return None

        target = self.dir_hpo_best_trial
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
        """Pickle the TPE sampler. Called after every trial -- see path_hpo_sampler."""
        if path is None:
            path = self.path_hpo_sampler
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(sampler, path)
        return path

    def load_sampler(self, path=None):
        """The pickled sampler, or None if there is not one yet."""
        if path is None:
            path = self.path_hpo_sampler
        if not os.path.exists(path):
            return None
        return joblib.load(path)

    def save_json(self, path, data):
        """Write data as indented json. default=str so numpy scalars survive."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as handle:
            json.dump(data, handle, indent=2, default=str)
        return path

    def csv_study_export(self, study, path_csv=None):
        """One row per trial: number, objective value, every drawn param, the
        four across-seed numbers HPOPPO records, and state. Written at the
        start of each trial. Needs pandas; returns None if missing.
        """
        if path_csv is None:
            path_csv = self.path_hpo_csv

        os.makedirs(os.path.dirname(path_csv) or ".", exist_ok=True)
        try:
            # named attrs, not the default: the default also dumps datetimes
            # and every user attr, which buries the params in the wide table
            frame = study.trials_dataframe(
                attrs=("number", "value", "params", "user_attrs", "state")
            )
        except (ImportError, AttributeError):
            return None

        frame.columns = [self._strip_prefix(name) for name in frame.columns]
        frame = frame.rename(columns={"value": "objective_value"})
        frame.to_csv(path_csv, index=False)
        return path_csv

    @staticmethod
    def _strip_prefix(name):
        """'params_lr' -> 'lr', 'user_attrs_return_std' -> 'return_std'."""
        for prefix in ("params_", "user_attrs_"):
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name

    # Every checkpoint carries its own eval_history, so a directory of them is
    # a set of curves. One mean+-std figure per metric in _HPO_METRICS, as html
    # and svg. plotly is imported inside these methods, so a checkout without
    # it still trains.
    def load_eval_histories(self, directory):
        """Reads every checkpoint's eval_history in `directory`. Returns
        {seed: eval_history}, seed parsed from the filename. Skips files that
        aren't checkpoints or have no eval_history rather than raising.
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
        """{seed: history} -> (iterations, seeds, values), values shaped
        (n_iterations, n_seeds) with nan where a seed has no entry at that
        iteration."""
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
        """Writes one mean+-std learning curve per metric for a directory of
        checkpoints, as .html and .svg. metrics defaults to both of
        _HPO_METRICS. Returns the paths written, or [] if there's no curve."""
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
        """'#2f6fdb', 0.18 -> 'rgba(47,111,219,0.18)'. Plotly wants the string."""
        hex_colour = hex_colour.lstrip("#")
        r, g, b = (int(hex_colour[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"

    def plot_hpo(self, metrics=None, logger=None, include_trials=True):
        """Plots every trial directory and best_trial/ under hpo/ that holds
        checkpoints. Returns the paths written. include_trials=False skips the
        individual trials and does only best_trial/.
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
        same weights: `sampled` drawn from pi, `argmax` greedy."""
        return {
            "seed": seed,
            "sampled_return": float(sampled["return_mean"]),
            "sampled_success_rate": float(sampled["success_rate"]),
            "argmax_return": float(argmax["return_mean"]),
            "argmax_success_rate": float(argmax["success_rate"]),
        }

    def log_seed_summary(self, logger, rows, header=None):
        """The closing table: one row per seed, then each of the four metrics
        aggregated across seeds. main_no_hpo.py and HPOPPO.final both end with
        this, so a hand-picked run and a tuned one are read the same way.
        Returns {metric: {mean, std}}, both across seeds."""
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
        for _ in range(n):
            logger.info("=" * 78)

    def callback_optuna_report_function(self, kind_training, logger, study, trial):
        """Logs one line per finished trial (state, value, params), plus the
        best trial so far. Passed as an optuna study.optimize callback, so it
        runs after every trial including pruned and failed ones.
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
        """study.optimize, resume-aware: runs only the trials still needed to
        reach n_trials (complete + pruned), re-queueing the last trial's
        params if it crashed or was killed mid-run. Returns trials run."""
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
        """The closing table: counts by state, then the winner. Returns it, or None."""
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
        # max of n_trials noisy measurements -- biased upward by the selection
        # itself (winner's curse). final() reports this value rather than
        # re-estimating it. See HPOPPO.final.
        logger.info(
            "  (the max over trials, so biased upward by the selection itself; "
            "final() reports these same runs and does not re-estimate it)"
        )
        self.print_separate_lines(logger)

        return best

    def zero_hidden(self, batch_size=None):
        """Zeroed h_0 (and c_0 for LSTM), (1, batch_size, hidden_size); None for MLP."""
        if not self.is_recurrent:
            return None

        if batch_size is None:
            batch_size = self.n_workers

        h = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        return (h, h.clone()) if self.is_lstm else h

    def reset_hidden_of(self, hidden, w):
        """Zeros the hidden state of worker w only, in place, leaving other
        workers' state untouched. Returns hidden for chaining."""
        if not self.is_recurrent:
            return hidden

        if self.is_lstm:
            hidden[0][:, w] = 0.0  # h
            hidden[1][:, w] = 0.0  # c
        else:
            hidden[:, w] = 0.0

        return hidden

    def watch_agent(self, path_model=None, deterministic=None, steps_per_sec=2.5):
        """Opens a pygame window that plays a saved policy over the fixed
        eval-set mazes, showing the full maze, the agent's 7x7 observation,
        action distribution and value estimate. Blocks until closed."""
        if deterministic is None:
            deterministic = self.eval_deterministic

        # apply the checkpoint's tuned architecture params before build_env /
        # build_extractor read config defaults -- otherwise a tuned checkpoint
        # (different hidden_size, n_layers, ...) fails load_state_dict. No-op
        # for a no_hpo checkpoint. See checkpoint_params / searched_params.
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

        screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption(
            f"{self.feature_extractor.upper()} on {self.name_env}"
        )
        clock = pygame.time.Clock()

        fonts = {
            "head": pygame.font.SysFont("monospace", 13, bold=True),
            "body": pygame.font.SysFont("monospace", 11),
            "tiny": pygame.font.SysFont("monospace", 10),
            "big": pygame.font.SysFont("monospace", 15, bold=True),
        }

        # everything the buttons reset lives in this dict; start() is the only
        # place it's written. max_steps read once off the live env, not via
        # config.env_max_steps (which builds a throwaway env every call).
        ep = {"max_steps": env.unwrapped.max_steps}

        def think():
            """One forward pass on the current observation; advances the hidden state exactly once."""
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
            """Begins an episode. move = +1/-1/0 selects the next/previous/
            current eval maze, wrapping both ways. keep_trail is for go_to():
            it preserves the recorded action trail across the reset.
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
            """Takes one action -- think()'s choice, or forced, a recorded one -- and updates the episode state."""
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
            """One step forward, replaying the trail first if we have rewound."""
            if ep["step"] < len(ep["trail"]):
                act(forced=ep["trail"][ep["step"]])
            else:
                act()

        def go_to(target):
            """Puts the episode back at step `target` by resetting to the same
            seed and replaying its recorded actions, since neither the env nor
            the hidden state can be stepped backward.
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
            """One place for a control, whether it came from a click or a key."""
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
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key in keys:
                        press(keys[event.key])

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # hit-tested against whatever the sidebar drew last frame,
                    # so the layout lives in exactly one place
                    for name, rect in buttons.items():
                        if rect.collidepoint(event.pos):
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
            )

            pygame.display.flip()

        env.close()
        pygame.quit()


class StartInCueView(gym.Wrapper):
    """Spawns the agent in MiniGrid MemoryEnv where the cue is actually
    visible, working around its random start position hiding the cue in most
    episodes. Rebuilds the observation after moving the agent."""

    def __init__(self, env):
        super().__init__(env)
        self._checked = False

    def reset(self, **kwargs):
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
        self.data = {k: (v[0] if k in ("hxs", "cxs") else v) for k, v in batch.items()}
        self.n_seq = self.data["mask"].shape[0]

    def __len__(self):
        return self.n_seq

    def __getitem__(self, i):
        # default_collate stacks these dicts into (mb, L, ...) tensors
        return {k: v[i] for k, v in self.data.items()}


# pygame viewer drawing helpers, used only by Helper.watch_agent. pygame is
# not imported at module level, so a machine without it can still train.

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

# MiniGrid's encoding, channel 0. Kept here rather than imported so this file
# still needs nothing but torch/gym; the numbers are in
# minigrid.core.constants.OBJECT_TO_IDX and have not moved in years.
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
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def _cell_rgb(obj_idx, color_idx):
    if obj_idx in _COLOR_DRIVEN:
        return _MG_RGB[color_idx] if color_idx < len(_MG_RGB) else (180, 180, 180)
    return _OBJ_RGB.get(obj_idx, (180, 180, 180))


def _find_cue(image):
    """The one key/ball visible in the observation, as 'green ball'. None if
    absent. Only meaningful at step 0 -- later there are two (one per branch).
    """
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            obj, color, _ = image[x, y]
            if obj in (5, 6):
                return f"{_COLOR_NAME.get(color, '')} {_OBJ_NAME[obj]}".strip()
    return None


def _visible_objects(image):
    """['green ball -- 3 ahead, 2 left', ...] for each non-structural cell in view."""
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
                name += (
                    f" ({['open', 'closed', 'locked'][state] if state < 3 else '?'})"
                )
            lines.append(f"{name} -- {where}")

    return lines or ["nothing but walls in view"]


def _draw_obs_grid(screen, pygame, image, ox, oy, font):
    """The agent's 7x7 egocentric window, as coloured cells."""
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

            # faint outline on every cell: "unseen" is nearly the same colour
            # as the panel behind it, so without this the grid shape vanishes
            # exactly when most of it is unseen
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
    """Draws one bar per action for pi(a|s), highlighting the action about to be taken."""
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
    """A filled rounded button that lights up on hover; enabled=False draws it flat and grey."""
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
):
    """Draws the whole right-hand info/control panel for watch_agent. Returns
    {name: pygame.Rect} for every button, used for click hit-testing.
    """
    import pygame

    paused = ui["paused"]

    sx = maze_px
    pygame.draw.rect(screen, _VIEW_PANEL, (sx, 0, _VIEW_SIDEBAR_W, win_h))

    pad = sx + 12
    inner = _VIEW_SIDEBAR_W - 24
    y = 12
    mouse = pygame.mouse.get_pos()

    def put(text, font="body", color=_VIEW_TEXT, dy=3):
        nonlocal y
        surf = fonts[font].render(text, True, color)
        screen.blit(surf, (pad, y))
        y += surf.get_height() + dy

    def rule():
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

    # three rows: transport, episode, and the persistent toggle at the bottom
    # (shorter than the action buttons, since it changes future behavior
    # rather than acting immediately)
    bh, th, gap = 42, 32, 12
    row_toggle = win_h - th - 34
    row_bottom = row_toggle - bh - gap
    row_top = row_bottom - bh - gap

    # both action rows share these columns so they line up vertically:
    #     STEP -1     PAUSE/PLAY   STEP +1      (within one episode)
    #     LAST GAME   REPLAY       NEW GAME     (between episodes)
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

    hint = fonts["tiny"].render(
        "SPACE pause   <- -> step   P/N maze   R replay   A auto   Q quit",
        True,
        (105, 105, 105),
    )
    screen.blit(hint, (pad, row_toggle + th + 8))

    return buttons
