"""
Helper: the things that are DECIDED BY the config but are not PPO itself.

Config inherits from this class, so every one of these is reachable as
config.something and can read config's own attributes directly:

    config.device              cuda if there is one, else cpu
    config.is_recurrent        True for LSTM and GRU, False for MLP
    config.is_lstm             True only for LSTM (it has a cell state too)
    config.build_env()         one MiniGrid game, wrapped as the config asks
    config.build_vector_env()  n of them behind ONE step(), each in a process
    config.env_max_steps       that env's own time limit, 5 * size^2
    config.build_extractor()   the encoder named by config.feature_extractor
    config.build_logger()      logs/log_<date>_<time>.log, hyperparameters first
    config.log_model_summary() torchinfo's table: layers, params, size in MB
    config.build_model_path()  the encoder's directory, created, + the filename
    config.save_model()        weights + the architecture they belong to
    config.load_model()        the same, back into a built model, checked
    config.zero_hidden()       h_0 (and c_0) full of zeros
    config.reset_hidden_of()   zero the hidden state of ONE worker
    config.watch_agent()       a pygame window that plays a saved policy

    config.run_with_batch_size_fallback(fn, sizes, logger)
                               fn(size) at the largest size that fits in memory

And, for agents/hpo_ppo.py only -- everything a study needs that is DECIDED BY
the config rather than by the search:

    config.suggest_from_search_space(trial)   config.search_space -> params
    config.apply_params(params)               params -> config attributes
    config.build_hpo_dir()                    hpo/, and the trial dirs under it
    config.copy_best_trial(study)             winner's files -> best_trial/
    config.hpo_optimize(...)                  resume-aware study.optimize
    config.summary_hpo(...)                   the final table
    config.save_sampler / load_sampler        the TPE state, pickled

Plus two standalone classes, imported directly rather than through config:

    StartInCueView             spawn the agent where the cue is actually visible
    SequenceDataset            split_pad_mask's output as a torch Dataset

None of this knows what an advantage, a ratio or a clip is. Swapping GRU for
LSTM is a change here and in the config, never in the agent.
"""

import gc
import json
import logging
import os
import shutil
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv
from minigrid.wrappers import ImgObsWrapper
from torch.utils.data import Dataset

# looks unused, but importing it is what registers MiniGrid-* with gymnasium.
# Delete it and gym.make raises NameNotFound. Do not let a linter remove it.
import minigrid  # noqa: F401

from models.feature_extractor import MLP, LSTM, GRU, Transformer

# ----- what a study maximizes, as ONE string ---------------------------------
#
# config.hpo_objective is  <metric>_<center>_<spread>  -- three fields:
#
#     return_mean_minus-std           mean over seeds, minus their std
#     success-rate_median_minus-iqr   median over seeds, minus their IQR
#     success-rate_median_None        the plain median, nothing subtracted
#
# THE METRIC IS ONE NUMBER PER TRAINING RUN, i.e. per seed. The CENTER and the
# SPREAD say how the len(seed_list) of those become the single number optuna
# compares, and BOTH ARE TAKEN ACROSS SEEDS -- never across the eval episodes
# inside one run. That distinction is the reason this is spelled out rather
# than left to a "std" somewhere: the within-run spread of a bimodal return is
# pinned at about 0.9*sqrt(p(1-p)), a function of the success rate itself, so
# subtracting it would score a policy that works 20% of the time BELOW one that
# never works at all. The across-seed spread is 0 for anything that behaves the
# same way every time, whatever its score, which is the property that makes it
# worth penalising.
#
# ONE SETTING RATHER THAN TWO. This used to be hpo_objective plus a separate
# hpo_aggregation, which could disagree with each other in a log ("return_mean"
# next to "mean_minus_std", and nothing in either saying they belonged to one
# score). One string cannot be half-updated.
# BOTH ARE KEYS OF AN eval_history ENTRY, which is what lets the same name be
# the thing the study ranks on AND the y axis of the learning-curve plots.
#
# THERE WAS A THIRD, "aulc" -- the mean of the success_rate curve over the whole
# run, meant to reward learning fast rather than merely ending well. It is gone
# because a mean over the curve is PERMUTATION-INVARIANT: a run that scored
# 1, 1, 0 at its three report iterations gets exactly the same aulc as one that
# scored 0, 1, 1, and those are not the same run. The second learned and held;
# the first learned and then collapsed. Worse, the checkpoint kept is the LAST
# iteration's, so the winning run of that pair would have been the one whose
# saved weights no longer solve the task -- the score and the file on disk would
# have described different policies. Ordering is exactly what "learned fast"
# means, and an average discards it.
_HPO_METRICS = ("return_mean", "success_rate")

# what the metric field may be written as. "return" alone is accepted because
# return_mean_minus-std parses its "mean" as the CENTER -- and both readings of
# that string ("metric return, centred by the mean" and "metric return_mean")
# mean the same thing, so neither has to win.
_HPO_METRIC_ALIASES = {
    "return": "return_mean",
    "success": "success_rate",
    "success_rate": "success_rate",
    "return_mean": "return_mean",
}

_HPO_CENTERS = ("mean", "median")

# spread -> the penalty subtracted, weighted by config.hpo_lambda. None means
# no penalty at all, which is what "None" in the string spells.
_HPO_SPREADS = ("std", "iqr", None)


def parse_hpo_objective(objective):
    """ "success-rate_median_minus-iqr" -> ("success_rate", "median", "iqr").

    Returns (metric, center, spread). spread is None for "no penalty".

    PARSED FROM THE RIGHT, because the metric is the only field that can
    contain an underscore -- return_mean is one name, not two. Dashes and
    underscores are interchangeable, so success-rate and success_rate are the
    same field, and the dashes exist only so that a reader can see where one
    field stops:

        return_mean_minus-std          -> return_mean,  mean,   std
        success-rate_median_minus-iqr  -> success_rate, median, iqr
        success-rate_median_None       -> success_rate, median, None

    BOTH TRAILING FIELDS ARE OPTIONAL and default to mean / None, so the old
    one-field spellings still parse: "return_mean" is the plain mean over
    seeds. Note that a bare "return_mean" therefore no longer subtracts a std
    -- write return_mean_minus-std for that, which is what config.py now says.

    Raises ValueError on an unknown field rather than falling back to a
    default: a typo here would otherwise run a whole study -- hours of
    training -- under an objective nobody chose, and the log would faithfully
    report the objective that was asked for.
    """
    text = str(objective).strip().lower().replace("-", "_")
    fields = [field for field in text.split("_") if field]
    if not fields:
        raise ValueError(
            "hpo_objective is empty. Write it as <metric>_<center>_<spread>, "
            "e.g. return_mean_minus-std, success-rate_median_minus-iqr or "
            "success-rate_median_None."
        )

    original = list(fields)

    # ---- the spread, last ------------------------------------------------
    spread = None
    if fields[-2:] == ["minus", "std"]:
        spread, fields = "std", fields[:-2]
    elif fields[-2:] == ["minus", "iqr"]:
        spread, fields = "iqr", fields[:-2]
    elif fields[-1] in ("none", "null"):
        fields = fields[:-1]
    elif fields[-1] == "minus" or (len(fields) > 1 and fields[-2] == "minus"):
        # "..._minus" alone, or "..._minus_something-else"
        raise ValueError(
            f"hpo_objective {objective!r}: {'_'.join(original[-2:])!r} is not a "
            f"spread. Use minus-std, minus-iqr or None."
        )

    # ---- the center, next-to-last ---------------------------------------
    center = "mean"
    if fields and fields[-1] in _HPO_CENTERS:
        center, fields = fields[-1], fields[:-1]

    # ---- everything left is the metric ----------------------------------
    metric = "_".join(fields)
    metric = _HPO_METRIC_ALIASES.get(metric, metric)
    if metric not in _HPO_METRICS:
        raise ValueError(
            f"hpo_objective {objective!r} names the metric {metric or '(nothing)'!r}, "
            f"which is not one of {list(_HPO_METRICS)}. The format is "
            f"<metric>_<center>_<spread>, where <center> is one of "
            f"{list(_HPO_CENTERS)} and <spread> is one of minus-std, minus-iqr "
            f"or None -- e.g. return_mean_minus-std."
        )

    return metric, center, spread


class Helper:
    """Builders and small utilities shared by anything that reads the config."""

    # ------------------------------------------------------------------
    # what the config implies
    # ------------------------------------------------------------------
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
        """The checkpoint this config currently points at.

        DERIVED ON EVERY READ, not stored in __init__, because
        dir_pretrained_model moves: hpo_ppo.py points it at hpo/trial_7/ for
        the duration of a trial, and ConfigNoHPO points it at no_hpo/. A path
        frozen at construction would send every trial's checkpoint back to
        whichever directory was set at that moment, and thirty trials would
        overwrite one file -- they all share an encoder, an env and a seed, so
        the filename cannot tell them apart. The DIRECTORY is what separates
        them.

        name_model is still a plain attribute: the encoder and the env are
        fixed once the config is built.
        """
        return os.path.join(self.dir_pretrained_model, self.name_model)

    # ------------------------------------------------------------------
    # builders
    # ------------------------------------------------------------------
    def build_env(self, render_mode=None):
        """One MiniGrid game, wrapped the way the config asks. Never gym.make.

        Every env in the project comes from here -- the W rollout workers,
        evaluate()'s private one and watch_agent()'s -- so training, scoring
        and watching cannot silently end up playing three different games.

        force_cue_visible adds StartInCueView (below). Turn it off to get the
        raw registered env back, which is what the first runs used and what
        the "no encoder beats any other" logs came from.

        render_mode stays None everywhere except watch_agent(), which asks for
        "rgb_array" and blits the frame into a pygame window. It is an argument
        rather than a config attribute because it is a property of ONE env
        instance, not of the experiment: rendering during training would cost
        real time for a picture nobody looks at.
        """
        env = gym.make(self.name_env, render_mode=render_mode)

        if self.force_cue_visible:
            env = StartInCueView(env)

        return env

    def build_vector_env(self, n_envs):
        """n_envs games behind ONE step() call. Images only, SAME_STEP reset.

        WHY, when build_env() already works. Stepping W games meant a python
        loop over W of them, one at a time, on one core -- 10,240 sequential
        env.step() calls per iteration at W=16, T=640. MiniGrid's step is
        ~85us of pure python (89% of it inside gen_obs_grid, walking the
        7x7 view cell by cell), so that loop was over half the run.

        AsyncVectorEnv gives each game its own process and steps them all at
        once. THE WIN IS NOT n_envs-FOLD, and it is worth knowing why before
        reading the timing table: gymnasium still sends the action and
        receives (reward, terminated, truncated, info) down one pipe per
        worker, in a python loop, so a batched step keeps an O(n_envs) serial
        part. Measured on 8 cores: 1.5x at n_envs=4, 1.8x at 8, 1.7x at 16,
        and 0.8x -- SLOWER -- at 32, where the workers outnumber the cores and
        the processes fight each other. More cores move that ceiling; nothing
        in this file does.

        async_envs=False falls back to SyncVectorEnv, which is the old python
        loop wearing the same interface. Keep it for debugging (one process,
        real tracebacks, no pickling) and for machines where the processes
        cost more than they save.

        IMAGE OBSERVATIONS ONLY. MiniGrid's observation is a Dict of image,
        direction and mission, and mission is a MissionSpace, which cannot go
        into shared memory -- with the full Dict, AsyncVectorEnv raises.
        shared_memory=False would "work" by pickling a mission string per env
        per step, forever, for a constant nothing here reads. ImgObsWrapper
        drops it and obs arrives as one (n_envs, 7, 7, 3) uint8 block written
        straight into shared memory. Everything in this project already reads
        obs["image"] and nothing else, so this loses nothing; the wrapper goes
        on the OUTSIDE, so StartInCueView still applies underneath.

        SAME_STEP autoreset, which is gymnasium's old behaviour and NOT its
        current default. A worker that finishes at step t returns the final
        reward together with the first observation of its next episode --
        exactly what the hand-rolled loop did with env.step() then env.reset().
        The default, NEXT_STEP, instead burns a whole extra step per episode
        returning the reset observation with a zero reward and an ignored
        action: a transition that never happened, which would have to be
        masked out of the rollout buffer, the advantages and the mask by hand.
        """
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
        """The env's own time limit: MemoryEnv sets max_steps = 5 * size^2.

            MemoryS7    245
            MemoryS11   605
            MemoryS13   845

        Read off the env rather than hardcoded, so changing name_env changes
        this too and the two can never disagree.

        Builds a throwaway env to ask. That is why Config reads it ONCE, into
        self.worker_steps, instead of using it as a property everywhere: this
        is cheap but not free, and nothing here changes during a run.

        env.unwrapped, not env: gym.make wraps in OrderEnforcing and this
        wrapper adds StartInCueView on top. max_steps belongs to MiniGridEnv
        itself, at the bottom of that stack. (spec.max_episode_steps is None
        for these envs -- MiniGrid enforces the limit in its own step(), so
        there is no gymnasium TimeLimit wrapper to read it from.)
        """
        env = self.build_env()
        max_steps = env.unwrapped.max_steps
        env.close()
        return max_steps

    def build_extractor(self):
        """MLP / LSTM / GRU / TRANSFORMER, picked by self.feature_extractor.

        All four take (batch, seq_len, 7, 7, 3) and give back
        (batch, seq_len, hidden_size), so the agent never has to care
        which one it got.
        """
        name = self.feature_extractor.upper()

        if name == "MLP":
            return MLP(self.input_size, self.hidden_size, self.n_layers_mlp)
        if name == "LSTM":
            return LSTM(self.input_size, self.hidden_size)
        if name == "GRU":
            return GRU(self.input_size, self.hidden_size)
        if name == "TRANSFORMER":
            # is_recurrent stays False for this one, and that is correct: it
            # takes no hidden state, so zero_hidden and reset_hidden_of have
            # nothing to do and Network calls it without one.
            #
            # READ THIS BEFORE TRUSTING A TRANSFORMER RUN. sample() picks
            # actions one step at a time, seq_len = 1. A transformer remembers
            # by attending to earlier POSITIONS of the sequence it is given,
            # and a length-1 sequence has none -- so while ACTING it is exactly
            # an MLP, however much history the update later shows it. It will
            # train and score, but the number is not a memory result. Fixing
            # that means caching the last K observations per worker in the
            # rollout buffer; see the feature_extractor module docstring.
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

    def build_logger(self, log_dir="logs", name="rl_project"):
        """One log file per COMMAND: logs/log_<date>_<time>.log, hyperparameters first.

        Called ONCE per invocation -- main.py builds one for a whole study,
        main_no_hpo.py one for the whole seed list -- and handed down to
        everything that runs underneath. Not once per trial or once per seed:
        a run's cross-seed summary has to land in the same file as the seeds
        it summarises, or the number the command exists to produce belongs to
        no file at all.

        The file name carries the timestamp, so two runs never collide and the
        directory sorts chronologically:

            logs/log_2026-07-31_14-03-27.log

        WHY THE HYPERPARAMETERS GO IN FIRST. A log of returns is worthless six
        runs later if you cannot tell which run it was. Every attribute of the
        config is dumped at the top, so the file answers "what was I running?"
        on its own -- no need to remember what config.py looked like that day.
        It is read straight off vars(self), so a hyperparameter added to
        Config.__init__ appears here with no change to this function. The
        three @property values (device, is_recurrent, is_lstm) are not in
        vars() and are asked for by name.

        THE DUMP SHOWS seed_list, NOT seed, because this is built before any
        PPOAgent -- it is set_seed(), called from the agent's __init__, that
        puts a single resolved seed into vars(self). That is the right way
        round for a command that trains every seed in the list: one `seed`
        line would describe whichever agent happened to be built first. Each
        seed announces itself in the body of the log instead. (Build it after
        an agent, as a one-seed script might, and `seed` appears too.)

        Two handlers, on purpose:
            FileHandler     timestamped, the permanent record
            StreamHandler   bare message, so the terminal still looks like the
                            plain print()s it replaced

        Returns a logging.Logger. Pass it to agent.train_agent(logger=...) and
        the per-iteration report lands in the file too.
        """
        os.makedirs(log_dir, exist_ok=True)

        started = datetime.now()
        path = os.path.join(log_dir, f"log_{started:%Y-%m-%d_%H-%M-%S}.log")

        # _2, _3, ... if that name is taken. The stamp is only accurate to the
        # second, and this is now built at the TOP of a command rather than
        # after an agent has been constructed, so two commands started back to
        # back really can land on the same name -- and a FileHandler opens in
        # append mode, so the second would silently continue the first, its
        # hyperparameter dump landing in the middle of someone else's run.
        # Sub-second precision in the name would fix it too, at the cost of
        # making every filename harder to read for a case this rare.
        collision = 1
        while os.path.exists(path):
            collision += 1
            path = os.path.join(
                log_dir, f"log_{started:%Y-%m-%d_%H-%M-%S}_{collision}.log"
            )

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # a Logger is a SINGLETON per name: logging.getLogger("rl_project")
        # twice gives the same object, still carrying the first call's
        # handlers. Without this, a second build_logger() in one process
        # writes every line to both files and twice to the terminal.
        logger.handlers.clear()

        # do not also hand the record to the root logger, which would print
        # it a second time if anything ever calls logging.basicConfig()
        logger.propagate = False

        file_handler = logging.FileHandler(path)
        # DATE AND TIME, not just time. A run that starts at 23:50 and finishes
        # at 00:40 reads as going backwards otherwise, and a 1000-iteration
        # MemoryS11 run is long enough for that to happen. The date is also in
        # the filename, but a line pasted into notes or a report arrives
        # without its filename.
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        # THE SAME FORMAT AS THE FILE, not a bare message. Two reasons: a line
        # copied out of the terminal carries its own date, and a report row is
        # datable on sight -- which is how you see that iterations 300..400
        # took three times as long as 0..100 without waiting for the run to
        # end. Every record gets the same fixed-width prefix, so the report
        # table and torchinfo's summary stay aligned; they just start 21
        # columns further right. Set this back to "%(message)s" for a bare
        # terminal. (train_agent also prints the elapsed time between report
        # iterations outright, so the subtraction does not have to be done by
        # eye -- this is the absolute clock, that is the delta.)
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(stream_handler)

        bar = "=" * 78
        logger.info(bar)
        logger.info(f"RUN STARTED  {started:%Y-%m-%d %H:%M:%S}")
        logger.info(f"LOG FILE     {path}")
        logger.info(bar)
        logger.info("HYPERPARAMETERS")

        for key, value in vars(self).items():
            logger.info(f"  {key:<24}{value}")

        # @property, so not in vars(self) -- but they decide what was built and
        # where it lands
        logger.info("  " + "-" * 40)
        for key in ("device", "is_recurrent", "is_lstm", "name_model", "path_model"):
            logger.info(f"  {key:<24}{getattr(self, key)}")

        logger.info(bar)

        return logger

    # ------------------------------------------------------------------
    # running at the largest batch size that fits
    #
    # The point of this section: mini_batch_size stops being a number you have
    # to know the machine to choose. The config names CANDIDATES, largest
    # first, and whatever is running here tries them in order until one does
    # not run out of memory. A laptop and a GPU box then run the same config
    # file and neither needs it edited.
    # ------------------------------------------------------------------
    def _iter_error_chain(self, error):
        """The error, and everything it was raised from or during.

        Needed because an OOM does not always arrive as itself. torchinfo, for
        one, catches the OOM from its probe forward pass and re-raises a
        generic RuntimeError("Failed to run torchinfo ...") whose own message
        no longer says anything about memory -- the real cause survives only on
        __cause__ / __context__. Checking the top-level message alone would
        miss it and turn a recoverable OOM into a crashed run.

        The seen set is not decoration: __context__ chains can form a cycle
        (raise A, handle it, raise B, handle THAT and re-raise A), and without
        it this loops forever.
        """
        seen = set()
        while error is not None and id(error) not in seen:
            seen.add(id(error))
            yield error
            error = error.__cause__ or error.__context__

    def _is_oom(self, error):
        """Is this -- or anything it wraps -- an out-of-memory error?

        THREE SHAPES, because torch raises a different one per situation:

            torch.cuda.OutOfMemoryError        the modern CUDA one. Note it is
                                               a SUBCLASS of RuntimeError, so
                                               the except clause below catching
                                               RuntimeError already covers it;
                                               it is named separately for
                                               clarity, not for reach.
            RuntimeError("... out of memory")  older / edge CUDA builds
            RuntimeError("... DefaultCPUAllocator: can't allocate memory")
                                               the CPU one, which is what this
                                               machine would ever actually hit

        Matching on message text is not elegant, and it is what torch gives us.
        """
        for err in self._iter_error_chain(error):
            if isinstance(err, torch.cuda.OutOfMemoryError):
                return True
            if isinstance(err, RuntimeError):
                text = str(err).lower()
                if "out of memory" in text:
                    return True
                # the CPU allocator's wording. Matched on the allocator's NAME
                # as well as the phrase, because the phrase is the part torch
                # has reworded before ("can't" / "cannot") and the class name
                # in the message is what has stayed put.
                if "can't allocate memory" in text or "cannot allocate memory" in text:
                    return True
                if "defaultcpuallocator" in text or "alloc_cpu.cpp" in text:
                    return True
        return False

    def _clear_traceback_chain(self, error):
        """Drop the traceback of the error and of every cause it wraps.

        NOT a tidiness step -- the retry depends on it. A traceback holds every
        frame it passed through, and those frames hold the failed attempt's
        tensors. Keep the traceback and that memory is still referenced when
        the next, smaller batch size is tried, so the smaller one runs out of
        memory in exactly the same place and the fallback walks all the way
        down its candidate list failing for a reason it created itself.
        """
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
        """Call run_fn(size) at the largest size that does not run out of memory.

            size_used, result = config.run_with_batch_size_fallback(
                lambda bs: agent.learn(batch, bs), config.mini_batch_size, logger
            )

        what NAMES WHAT IS BEING SIZED, and it exists because this runs twice
        per run over the same candidate list for two unrelated reasons: once
        for torchinfo's probe forward, which only decides how wide a table to
        print, and once for the real update, which decides what the run trains
        with. Identical wording for both put two lines in the log that look
        like the same event and are not -- and the second is the one that
        matters.

        batch_size may be a single int -- in which case there is nothing to
        fall back to and this is just a call -- or a list/tuple of candidates,
        which are sorted DESCENDING and deduplicated, so the order they are
        written in the config does not matter.

        Returns (size_used, whatever run_fn returned). Raises the last OOM if
        every candidate runs out.

        ONLY OOM IS CAUGHT. Everything else -- a shape bug, a KeyboardInterrupt,
        optuna's TrialPruned -- propagates untouched. That matters most for
        pruning: catching it here would turn "stop this trial" into "retry this
        trial smaller", and the study would never prune anything.

        THE HONEST COST OF A MID-RUN RETRY. run_fn is called again from the
        top, and if the first attempt already stepped the optimizer on some
        minibatches those steps are NOT undone -- the retry re-walks the same
        rollout, so a few sequences get updated twice. That is a real, small
        distortion of one iteration. It is accepted because the alternative is
        losing the run, and it can happen at most once per run: the resolved
        size is written back to self.mini_batch_size, so every later iteration
        starts from the size that is already known to fit.
        """
        if isinstance(batch_size, (list, tuple)):
            candidates = sorted({int(b) for b in batch_size}, reverse=True)
        else:
            candidates = [int(batch_size)]

        # print() when there is no logger, so this is usable from a scratch
        # script and from a real run without a branch at every call site
        say = logger.info if logger is not None else print
        warn = logger.warning if logger is not None else print
        fail = logger.error if logger is not None else print

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

                # reclaim what the FAILED attempt left referenced, before the
                # next candidate asks for more. In the except branch and not at
                # the top of the loop: there is nothing to reclaim before the
                # first try, and this runs once per ITERATION for the whole run
                # -- the resolved size is written back, so from iteration 2
                # onwards there is a single candidate that has never failed.
                # gc.collect() plus empty_cache() measured 41ms/iteration here
                # and 92ms on the reporter's box, which was every millisecond
                # the phase table could not account for. empty_cache() is worse
                # than its own clock says: it hands the cached blocks back to
                # the driver, so the next iteration re-cudaMallocs them.
                self._free_memory()

                if i + 1 < len(candidates):
                    warn(
                        f"out of memory at {what} {bs}, "
                        f"falling back to {candidates[i + 1]}"
                    )
                else:
                    fail(
                        f"out of memory at {what} {bs}, no smaller one left to try"
                    )

        self._free_memory()
        raise last_error

    def log_model_summary(self, model, logger=None, batch_size=None, seq_len=8):
        """torchinfo's table for the built model, into the log file.

        Called from main.py right after build_logger(), so the run's permanent
        record says not just which hyperparameters were used but how big the
        thing they built actually was:

            Total params        201,352
            Trainable params    201,352
            Params size (MB)       0.77

        WHY IT NEEDS A PROBE INPUT. torchinfo works by running a forward pass
        and watching the shapes go by, so it has to be handed something to
        pass. That is the only reason batch_size and seq_len exist here.
        Parameter counts do NOT depend on either -- an nn.Linear has the same
        weights whatever you push through it -- so the numbers above are exact
        for any probe. What the probe does change is the Output Shape column
        and the mult-adds estimate, which are per-batch quantities.

        seq_len matters most for TRANSFORMER, where attention is quadratic in
        it: the mult-adds at seq_len=8 are not a hundredth of the mult-adds at
        seq_len=640. Read that row as an illustration, not as the cost of a
        real update. Params are still exact.

        dtypes=[torch.uint8] is not optional. The observation stays uint8 all
        the way to flatten_obs, which one-hots it -- torchinfo's default float
        probe would be handed to F.one_hot and raise.

        hidden is left at its default of None, which is the same thing every
        first step of a rollout does: the model builds its own zeros. Nothing
        is trained, no gradient is kept, and the probe never touches the envs.

        torchinfo is imported HERE rather than at the top of the file, so a
        machine without it can still train -- the summary is a convenience,
        not a dependency of the experiment. Returns the ModelStatistics object
        (so .total_params and .trainable_params can be read), or None if
        torchinfo is missing.
        """
        try:
            from torchinfo import summary
        except ImportError:
            message = (
                "torchinfo not installed, skipping the model summary "
                "(pip install torchinfo)"
            )
            (logger.warning if logger is not None else print)(message)
            return None

        if batch_size is None:
            # the update's batch counts SEQUENCES, which is what a forward
            # pass during optimization actually receives. Now a LIST of
            # candidates, so the probe goes through the same fallback the real
            # update does -- see below.
            batch_size = self.mini_batch_size

        # THE PROBE RUNS THROUGH THE FALLBACK TOO, for two reasons. It is a
        # real forward pass and can genuinely run out of memory on a wide
        # transformer, and a crash HERE would kill a run before a single
        # iteration -- over a table that is only ever informational. And since
        # mini_batch_size is now a list, `input_size=(a_list, ...)` would not
        # even be a valid shape.
        #
        # It does NOT decide what training uses: train() resolves its own size
        # against the real update, which allocates far more than this does.
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

        write = logger.info if logger is not None else print
        write("")
        write(f"MODEL SUMMARY  {self.feature_extractor.upper()}  (probe input {shape})")
        for line in str(info).splitlines():
            write(f"  {line}")
        write("")

        return info

    @property
    def name_model(self):
        """build_model_name(), re-read every time. See the property path_model.

        A PROPERTY and not an attribute frozen in Config.__init__, because the
        name now carries the seed and the seed is not known until set_seed()
        runs -- which happens inside PPOAgent.__init__, after the config is
        built. Frozen at construction, all three seeds of a run would share one
        filename and the third would be the only one left on disk.
        """
        return self.build_model_name()

    def build_model_name(self):
        """ppo_<seed>_<ENCODER>_<env>.pth -- the ONE place the filename is spelled.

            ppo_0_GRU_MiniGrid-DoorKey-8x8-v0.pth
            ppo_26_MLP_MiniGrid-MemoryS7-v0.pth

        ALL THREE PARTS change what the weights mean.

        The encoder decides the architecture; the env decides what the agent
        was trained to do, and a DoorKey policy loaded against MemoryS11 is not
        a worse agent, it is a meaningless one.

        The SEED is in there because a result is seed_list as a whole -- three
        runs, deliberately -- and they are otherwise the same encoder on the
        same env, so without it the three would be one filename written three
        times and only the last would survive. That in turn would make the
        spread over seeds, which is the actual uncertainty about a config,
        impossible to go back and inspect.

        self.seed is set by set_seed(), which PPOAgent.__init__ calls. Before
        that has happened -- watch.py, which builds a config and loads a
        checkpoint without ever training -- it falls back to seed_list[0], so
        `python watch.py GRU` finds the first seed's file with no argument.

        Keying on the encoder alone -- what this used to do -- meant a GRU run
        on MemoryS11 silently overwrote a GRU run on DoorKey. train_agent()
        saves on every improvement and starts each run from best_success =
        -1.0, so the first evaluation of the new run, however bad, lands on top
        of a finished result from the old one. Nothing warns, because the
        filename is the only thing that ever distinguished them.

        It is a METHOD, not an attribute set in Config.__init__, so that both
        halves are read WHEN IT IS CALLED. That is what lets watch.py override
        feature_extractor from the command line and get the matching file --
        an f-string evaluated once in __init__ would still be spelling the
        encoder that was set at import time.

        The replace() is for gymnasium's namespaced ids ("ALE/Pong-v5"), which
        MiniGrid does not use but which would otherwise put a directory
        separator in the middle of a filename and fail confusingly.
        """
        env = self.name_env.replace("/", "-")
        seed = getattr(self, "seed", self.seed_list[0])
        return f"ppo_{seed}_{self.feature_extractor.upper()}_{env}.pth"

    def build_model_path(self):
        """dir_pretrained_model/ppo_<encoder>_<env>.pth, with the directory made.

        The path itself is the path_model property (dir_pretrained_model +
        name_model); this only creates the directory and hands the path back,
        the same split build_logger uses for logs/.

        Call it right before torch.save. Making the directory at import time
        instead would litter agents/pretrained_model_*/ into every checkout
        that merely imports a config without ever training anything -- which is
        why hpo/, no_hpo/ and the trial directories appear only once something
        is actually saved into them.

        WHICH directory this is depends on who is running: the encoder's top
        level normally, hpo/trial_<n>/ inside a trial, no_hpo/ under
        ConfigNoHPO. It reads dir_pretrained_model every call, so redirecting
        that one attribute is all any of them has to do.
        """
        os.makedirs(self.dir_pretrained_model, exist_ok=True)
        return self.path_model

    def save_model(self, model, optimizer=None, **extra):
        """Write model (+ optimizer) to build_model_path(). Returns the path.

        The file is a dict, not a bare state_dict, because a bare one cannot be
        loaded without already knowing what shape to load it into. Alongside
        the weights it carries the FOUR attributes that decide the
        architecture:

            feature_extractor  GRU / LSTM / MLP   -> different modules entirely
            hidden_size                         -> every layer width
            input_size                          -> the encoder's first layer
            name_env                            -> which game it can play

        load_model checks those against the live config and refuses a
        mismatch, so "trained a GRU, config now says LSTM" is a clear error
        instead of a size-mismatch traceback or, worse, silent nonsense.

        optimizer is optional: pass it to be able to RESUME training, leave it
        out for a checkpoint that is only ever going to be watched or scored.
        Adam's state is two moment tensors per parameter, so including it
        roughly triples the file.

        **extra goes in verbatim -- iteration=, eval_success_rate=,
        eval_history= and so on. Everything here has to survive
        torch.load(weights_only=True), which allows numbers, strings, bools,
        None, and lists / tuples / dicts of those. That is enough for
        train_agent's eval_history (a list of dicts of floats), which is how a
        checkpoint carries its own learning curve. It is NOT enough for
        arbitrary objects -- numpy scalars included, so cast with float().

        "params" makes the file SELF-DESCRIBING, which is what lets anything
        reload it without being told which trial it came from. See
        searched_params() for why that is not optional.
        """
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
        """Load weights INTO an already-built model. Returns the checkpoint dict.

        The model has to exist first -- this fills it in, it does not build it.
        That is deliberate: building needs n_actions, which only an env can
        say, and this class does not know whose model it is being handed.

            model = Network(config.build_extractor(), config.hidden_size, 7)
            checkpoint = config.load_model(model)

        path defaults to self.path_model, i.e. the encoder the config is
        currently set to. Pass one to look at a different file.

        Raises rather than guesses:
            FileNotFoundError  no checkpoint -- train and save one first
            ValueError         the checkpoint was trained under a different
                               architecture or a different env

        A bare state_dict (torch.save(model.state_dict(), ...)) still loads,
        it just cannot be checked -- there is nothing in it to check against.
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

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state = checkpoint["model"]

            # force_cue_visible IS checked, even though it changes no tensor
            # shape and so would never raise on its own. It changes what the
            # agent could SEE: a policy trained with the cue forced into view
            # every episode, watched on an env where it is visible in one
            # episode in eight, is being scored on a task it never played.
            # That is the failure this guard exists to make loud, and it is
            # the one case of it that a shape check cannot notice.
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
        else:
            # a bare state_dict from torch.save(model.state_dict(), ...)
            state = checkpoint
            checkpoint = {"model": state}

        model.load_state_dict(state)
        return checkpoint

    # ------------------------------------------------------------------
    # HPO -- everything a study needs that the CONFIG decides
    #
    # The split with agents/hpo_ppo.py is the same one as everywhere else in
    # this file: what to search over, where it is written and how it resumes
    # are config questions and live here; what a trial actually DOES -- train
    # three seeds and score them -- is the agent's, and lives there.
    #
    # optuna and joblib are imported INSIDE these methods, never at module
    # level, so that a checkout without them still trains, watches and scores.
    # A search is one way to use this project, not a dependency of it.
    # ------------------------------------------------------------------
    def suggest_from_search_space(self, trial):
        """config.search_space -> {name: value} drawn for this trial.

        Each entry is a dict passed almost verbatim to trial.suggest_*; "type"
        picks the method and everything else is forwarded, so step=, log= and
        choices= work without this function knowing they exist. Adding a knob
        is a line in a config file, never a change here.

        The dict is COPIED before "type" is popped. Without that, the first
        trial would strip "type" out of the config's own list and every
        following trial would silently fall through to the float default --
        a bug that only appears from trial 1 onwards, and only for int and
        categorical knobs.
        """
        params = {}

        for spec in self.search_space:
            spec = dict(spec)
            kind = spec.pop("type", "float")

            if kind == "int":
                params[spec["name"]] = trial.suggest_int(**spec)
            elif kind == "categorical":
                params[spec["name"]] = trial.suggest_categorical(**spec)
            else:
                params[spec["name"]] = trial.suggest_float(**spec)

        return params

    # ----- the score: hpo_objective, taken apart -------------------------
    #
    # Properties, all four derived from the one string, so there is no second
    # place for any of this to be set and nothing to keep in sync. See
    # parse_hpo_objective above for the format.
    @property
    def hpo_metric(self):
        """The PER-SEED key: "return_mean" or "success_rate".

        This is the one that gets looked up in a run's result dict, so it has
        to be a real key of what hpo_ppo.run_split returns -- which is what
        parse_hpo_objective checks, at config-build time rather than three
        hours into a study.

        IT IS ALSO A KEY OF AN eval_history ENTRY, which is what lets the
        learning-curve plots default their y axis to it: the study ranks on the
        last point of the very line they draw. That used not to hold -- "aulc"
        was a summary of the whole curve rather than a point on it, and needed
        a separate hpo_curve_metric to say what to plot instead. See
        _HPO_METRICS for why it is gone.
        """
        return parse_hpo_objective(self.hpo_objective)[0]

    @property
    def hpo_center(self):
        """ "mean" or "median" -- how the per-seed metrics are centred."""
        return parse_hpo_objective(self.hpo_objective)[1]

    @property
    def hpo_spread(self):
        """ "std", "iqr", or None -- what is subtracted from the center."""
        return parse_hpo_objective(self.hpo_objective)[2]

    @property
    def hpo_aggregation(self):
        """ "median_minus-iqr" -- the two aggregation fields, back as one string.

        Read-only and DERIVED, where it used to be a setting of its own. It is
        kept because the json reports and the best_params.json record it as a
        field, and because "median_minus-iqr" is more readable in a table than
        re-deriving it from the objective every time it is printed.
        """
        _, center, spread = parse_hpo_objective(self.hpo_objective)
        return center if spread is None else f"{center}_minus-{spread}"

    @property
    def score_name(self):
        """ "mean_minus_1std(return_mean)" -- what the study actually maximizes.

        Spelled out wherever a value is printed, because "0.42" alone does not
        say whether the across-seed spread has already been subtracted from it,
        nor whether the number is a mean or a median.
        """
        metric, center, spread = parse_hpo_objective(self.hpo_objective)
        if spread is None:
            return f"{center}({metric})"
        weight = getattr(self, "hpo_lambda", 1.0)
        return f"{center}_minus_{weight:g}{spread}({metric})"

    def aggregate_scores(self, values):
        """The per-seed metrics -> the one number the study maximizes.

        Both halves come from config.hpo_objective:

            <center>   mean or median of the per-seed values
            <spread>   std, iqr or nothing, subtracted with weight hpo_lambda

            score = center(values) - hpo_lambda * spread(values)

        THE SPREAD HERE IS OVER SEEDS. It is the run-to-run variation -- "does
        this config work every time, or only sometimes?" -- and NOT the spread
        over the eval episodes inside one run, which is a different quantity
        that this function never sees. evaluate() reports that one as
        return_std, and it must not be substituted here: for a bimodal return
        it is a function of the mean itself, so subtracting it would score a
        policy that succeeds 20% of the time BELOW one that never succeeds.

        MEDIAN + IQR IS THE ROBUST PAIR, and on these tasks that is not a
        stylistic choice. The outcome is bimodal -- a seed either finds the
        reward or never does -- so with seed_list = [0, 26, 98] one dead seed
        moves the mean by a third of the score while the median does not move
        at all. Which behaviour you want is a real decision: mean_minus-std
        asks for a config that works on EVERY seed, median_minus-iqr asks for
        one that works on MOST of them.

        WITH THREE SEEDS THE IQR IS NOT A QUANTILE ESTIMATE. numpy interpolates
        it between two of the three values, so for [0, 0, x] it is x/2 and for
        [0, x, x] it is x/2 as well -- read it as "a spread", the same way the
        median+IQR plot is read. It also inherits the sign problem the
        hpo_lambda comment in config.py describes: at lambda = 1 a config that
        works on one seed out of three scores BELOW one that never works.

        ddof=0 for the std, so a single value gives 0 rather than nan; the IQR
        of a single value is 0 for the same reason. That matters for pruning,
        where this is called on a running list that starts at length one: the
        first report is then simply the raw metric, and every trial is equally
        optimistic at that step, so the comparison stays fair.

        Returns a float.
        """
        values = np.asarray(values, dtype=float)
        _, center, spread = parse_hpo_objective(self.hpo_objective)

        score = float(np.median(values)) if center == "median" else float(values.mean())

        if spread is None:
            return score

        if spread == "iqr":
            penalty = float(np.percentile(values, 75) - np.percentile(values, 25))
        else:
            penalty = float(values.std())  # ddof=0 -- see above

        return score - getattr(self, "hpo_lambda", 1.0) * penalty

    def apply_params(self, params):
        """Write a trial's draw onto this config. Returns self, for chaining.

        MUST BE CALLED BEFORE PPOAgent(config). The agent copies every value it
        needs out of the config in __init__ -- lr and wd into the optimizer,
        hidden_size and d_model into the encoder it builds -- and never reads
        the config again. Applied after the agent exists, a trial would train
        the DEFAULT hyperparameters and report them under the drawn ones, which
        is worse than crashing: the study would run to completion and its
        results would be meaningless.

        The hasattr check turns a typo in search_space into an error at the top
        of the first trial, rather than a run that quietly tunes nothing
        because setattr happily creates any attribute it is given.
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
        """The CURRENT value of every name in search_space. The inverse of apply_params.

        apply_params writes a trial's draw onto the config; this reads it back
        off, so save_model can put it in the checkpoint and anything reloading
        that checkpoint can put it back. Round-trips:

            config.apply_params(trial.params)
            ... train, save ...
            config.apply_params(checkpoint["params"])   # the same architecture

        WHY THE CHECKPOINT NEEDS THIS AT ALL. search_space tunes the
        ARCHITECTURE, not just the optimiser -- hidden_size for all four
        encoders, n_layers_mlp for the MLP, d_model / n_heads /
        n_layers_transformer / d_ff_mult for the transformer. A config built
        fresh from make_config() carries the DEFAULTS, so building a model from
        it and loading trial 7's weights into that model is a shape mismatch.
        load_model catches the hidden_size case (it is in the guard tuple) but
        n_layers_mlp is not in any checkpoint field at all, and a 2-layer file
        loaded into a 3-layer model is a raw load_state_dict traceback.

        DERIVED FROM search_space rather than stored when apply_params runs, so
        there is nothing to keep in sync: whatever the study is searching over
        is exactly what gets written. Empty under ConfigNoHPO, whose
        search_space is deliberately [] -- and correctly so, because that
        config's values are hand-written constants, so rebuilding it already
        reproduces the architecture its checkpoints were trained with.

        Everything optuna can draw (int, float, categorical) survives
        torch.load(weights_only=True), which is what save_model needs.
        """
        return {
            entry["name"]: getattr(self, entry["name"])
            for entry in getattr(self, "search_space", [])
            if hasattr(self, entry["name"])
        }

    # ----- where a study writes -------------------------------------------
    def build_hpo_dir(self):
        """Make hpo/ and hand it back."""
        os.makedirs(self.dir_hpo, exist_ok=True)
        return self.dir_hpo

    def dir_hpo_trial(self, number):
        """hpo/trial_7/ -- where trial 7's checkpoint goes.

        ONE DIRECTORY PER TRIAL, because every trial writes the identical
        filename: name_model is built from the encoder and the env, both of
        which are fixed for the whole study. Thirty trials would be one file,
        thirty times overwritten, and copy_best_trial would have nothing to
        copy. The trial number cannot go in the FILENAME instead -- watch.py
        and load_model rebuild that name from the config alone and have no
        trial number to put in it.
        """
        return os.path.join(self.dir_hpo, f"trial_{number}")

    @property
    def dir_hpo_best_trial(self):
        """hpo/best_trial/ -- a copy of whichever trial won, and THE RESULT.

        One set of runs and nothing else, the same shape as trial_<n>/, which
        is what lets one loader and one plotter read either without being told
        which it has -- see load_eval_histories.

        THERE IS NO SEPARATE final/ ANY MORE. It used to hold a fresh retrain
        at the winning params; that retrain trained the same encoder, on the
        same env, from the same seeds, at the same hyperparameters as the
        trial already in here, so it cost three full training runs to produce
        another sample of a run that had already been made. final() now reads
        THESE checkpoints instead and writes its report json beside them. See
        hpo_ppo.final for what is lost by that (the retrain was also the one
        unbiased estimate of the winner's score) and why it is worth it.
        """
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

    # ----- pointing a fresh config at ONE saved run ------------------------
    def select_run(self, trial=None, seed_index=0):
        """Aim this config at one checkpoint on disk. Returns its path.

        The reader's counterpart to what the trainer does implicitly. A run is
        identified by exactly two things once the encoder and env are fixed:

            WHICH RUN     the directory   -> dir_pretrained_model
            WHICH SEED    the filename    -> self.seed, via build_model_name

        so this sets those two attributes and hands back path_model. Nothing
        is created and nothing is read -- the file may not exist yet, and
        load_model is what says so.

            trial=None      leave dir_pretrained_model where the config put it.
                            That is no_hpo/ under ConfigNoHPO and the encoder's
                            top level otherwise -- i.e. "the run this config
                            already describes", which is what watch.py wants
                            when no study is involved.
            trial="best"    hpo/best_trial/  the winning trial. THE RESULT.
            trial="final"   the same directory. "final" was the retrain's name
                            back when there was one; there is no separate
                            retrain any more, and the accepted alias is what
                            keeps an old command from failing on a directory
                            that no longer exists. See dir_hpo_best_trial.
            trial=7         hpo/trial_7/     one particular draw

        seed_index INDEXES seed_list, it is not the seed itself. seed_list is
        [0, 26, 98] for a study and whatever ConfigNoHPO says for a hand-picked
        run, so 1 is a valid choice under one and out of range under the other
        -- hence the explicit message rather than a bare IndexError from deep
        inside a property.
        """
        if trial is not None:
            if isinstance(trial, str) and trial.lower() in ("best", "final"):
                self.dir_pretrained_model = self.dir_hpo_best_trial
            else:
                try:
                    number = int(trial)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"trial={trial!r} is not 'final', 'best' or a trial number"
                    ) from None
                self.dir_pretrained_model = self.dir_hpo_trial(number)

        if not 0 <= seed_index < len(self.seed_list):
            raise IndexError(
                f"seed index {seed_index} is out of range for seed_list="
                f"{self.seed_list} ({len(self.seed_list)} seed"
                f"{'' if len(self.seed_list) == 1 else 's'}, so the valid "
                f"indices are 0..{len(self.seed_list) - 1}). The index is a "
                f"position in that list, not the seed value."
            )

        # build_model_name reads self.seed and falls back to seed_list[0] when
        # it is unset. Setting it here is what makes the filename name the seed
        # asked for -- set_seed() is NOT called, because nothing is being
        # trained and reseeding the process would only change what the viewer
        # happens to sample.
        self.seed = self.seed_list[seed_index]

        return self.path_model

    def checkpoint_params(self, path=None):
        """The hyperparameters a checkpoint was TRAINED with. {} if it has none.

        Opens the file for its "params" entry alone and applies nothing --
        callers hand the result to apply_params, in that order, BEFORE building
        the model. See searched_params for why this exists at all.

        Missing key rather than error for anything written before save_model
        started recording params, and for ConfigNoHPO, whose search_space is
        empty by design. In both cases {} is right: apply_params({}) is a
        no-op, which leaves the config's own values in place, which is exactly
        what those files were trained with.
        """
        if path is None:
            path = self.path_model
        if not os.path.exists(path):
            return {}

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            return {}
        return dict(checkpoint.get("params") or {})

    def copy_best_trial(self, study, logger=None):
        """Copy the winning trial's files into best_trial/, and its params beside them.

        A COPY, not a pointer or a symlink, so best_trial/ is still readable
        after trial_12/ is deleted to reclaim space -- which is the normal
        thing to do with thirty of them.

        THE TARGET IS EMPTIED FIRST. A resumed study can pick a NEW winner, and
        writing the new one's files over the old one's leaves any file the new
        trial did not happen to write sitting there from the previous winner --
        a best_trial/ that is half one run and half another, with nothing
        saying so.

        Returns the path, or None if no trial has completed yet.
        """
        say = logger.info if logger is not None else print

        try:
            best = study.best_trial
        except ValueError:
            # optuna RAISES here rather than returning None when nothing has
            # finished -- an empty study, or one whose every trial failed
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

        # what the search FOUND, written beside what it produced. Without this
        # the directory holds a .pth whose hyperparameters are recoverable only
        # by cross-referencing the trial number against the csv.
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

    # ----- resuming --------------------------------------------------------
    def save_sampler(self, sampler, path=None):
        """Pickle the TPE sampler. Called after every trial -- see path_hpo_sampler."""
        import joblib

        if path is None:
            path = self.path_hpo_sampler
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(sampler, path)
        return path

    def load_sampler(self, path=None):
        """The pickled sampler, or None if there is not one yet."""
        import joblib

        if path is None:
            path = self.path_hpo_sampler
        if not os.path.exists(path):
            return None
        return joblib.load(path)

    # ----- reporting -------------------------------------------------------
    def save_json(self, path, data):
        """Write data as indented json. default=str so numpy scalars survive."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as handle:
            json.dump(data, handle, indent=2, default=str)
        return path

    def csv_study_export(self, study, path_csv=None):
        """Every trial as one csv row: params, value, state, duration, user attrs.

        Called at the START of each trial rather than at the end of the study,
        so an interrupted search still leaves a readable table of how far it
        got -- which is exactly when one is wanted.

        Needs pandas (optuna's trials_dataframe does). Missing it is not worth
        failing a study over, so it is caught and reported.
        """
        if path_csv is None:
            path_csv = self.path_hpo_csv

        os.makedirs(os.path.dirname(path_csv) or ".", exist_ok=True)
        try:
            study.trials_dataframe().to_csv(path_csv, index=False)
        except ImportError:
            return None
        return path_csv

    # ----- plots -----------------------------------------------------------
    #
    # WHAT IS PLOTTED. train_agent() evaluates on every report iteration and
    # keeps the result in eval_history, which save_model puts INTO the
    # checkpoint. So every .pth in this project carries its own learning
    # curve, and a directory of them -- one per seed -- is a set of curves
    # over the same x axis, ready to aggregate. trial_<n>/ and best_trial/ are
    # both exactly that, which is why one loader and one plotter cover both.
    #
    # TWO FIGURES, NOT ONE, and deliberately not two bands on one axis. They
    # answer different questions and disagreeing is the interesting case:
    #
    #   mean +- std      what the AVERAGE seed did, and how far the seeds
    #                    spread around it. Sensitive to one seed that never
    #                    learns -- which on a sparse-reward MiniGrid task is
    #                    a common outcome, not an outlier to be discarded.
    #   median + IQR     what the TYPICAL seed did. Unmoved by that one dead
    #                    seed, so a median well above the mean is the plot
    #                    saying "most seeds solved it, one did not".
    #
    # With three seeds the IQR is a wide interpolation between two of them --
    # readable as a spread, not as a quantile estimate. Both figures are drawn
    # anyway, because the GAP between them is the diagnostic.
    #
    # plotly is imported inside these methods, the same as optuna and joblib
    # above: a checkout without it still trains, searches and scores.
    # ------------------------------------------------------------------
    def load_eval_histories(self, directory):
        """Every checkpoint in `directory` -> {seed: eval_history}.

        Reads the curve straight out of the .pth files, so it works on any of
        hpo/trial_<n>/ and hpo/best_trial/ without being told which it is
        looking at, and it works on a study that finished weeks ago with no
        log file left.

        The seed comes from the FILENAME -- ppo_<seed>_<ENC>_<env>.pth, see
        build_model_name -- which is the whole reason the seed is in there.

        Skips silently rather than raising: a file that is not a checkpoint, a
        checkpoint from before eval_history existed, or a half-written one
        from a crashed run. An empty dict back means "nothing to plot", which
        the callers treat as a normal outcome for a pruned trial.
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
        """{seed: history} -> (iterations, seeds, values).

        values is (n_iterations, n_seeds) with nan where a seed has no entry
        at that iteration. Nan rather than a dropped row, because a pruned
        trial can hold two seeds that both ran the full 500 iterations and one
        that did not run at all -- and the aggregate should be over the seeds
        that HAVE a number there, not over a truncated x axis.
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
        metric=None,
        name=None,
        title=None,
        logger=None,
        include_plotlyjs="cdn",
    ):
        """Two figures from one directory of checkpoints. Returns the paths written.

            hpo/best_trial/curve_return_mean_mean_std.html   .svg
            hpo/best_trial/curve_return_mean_median_iqr.html .svg

        directory  either of hpo/trial_<n>/ and hpo/best_trial/ -- see
                   load_eval_histories
        metric     the y axis, defaulting to config.hpo_metric, so the plot
                   shows the quantity the study was actually ranked on. Any
                   key of an eval_history entry works: success_rate,
                   return_mean, timeout_rate, length_mean.
        name       what to call this run in the title. Defaults to the
                   directory's basename, which is already "trial_7" or
                   "best_trial".

        BOTH FORMATS, because they are for different readers. The .html keeps
        the hover and the legend, so the per-seed traces can be switched on
        one at a time -- they are drawn but start hidden, since three raw
        curves under a band is unreadable at a glance and invaluable once you
        are asking why the band is wide. The .svg is vector and static, which
        is what goes in the report.

        include_plotlyjs is "cdn" on purpose. The alternative inlines ~3 MB of
        javascript into EVERY file, and a thirty-trial study writes sixty of
        them -- 180 MB of duplicated library. The cost is that the .html needs
        a network connection to render; the .svg never does, and that is the
        one that gets published.

        Returns [] and logs a line if the directory holds no curve -- a pruned
        trial that died on its first seed is the normal case, not an error.
        """
        import plotly.graph_objects as go

        say = logger.info if logger is not None else print

        if metric is None:
            metric = self.hpo_metric
        if name is None:
            name = os.path.basename(os.path.normpath(directory))

        histories = self.load_eval_histories(directory)
        if not histories:
            say(f"no eval_history in {directory}, nothing to plot")
            return []

        iterations, seeds, values = self.curve_table(histories, metric)
        if not iterations:
            say(f"no {metric!r} in the curves under {directory}, nothing to plot")
            return []

        # nan-aware everywhere: a seed missing at one iteration must not turn
        # the whole row into nan and blank out the plot
        mean = np.nanmean(values, axis=1)
        std = np.nanstd(values, axis=1)
        median = np.nanmedian(values, axis=1)
        q25 = np.nanpercentile(values, 25, axis=1)
        q75 = np.nanpercentile(values, 75, axis=1)

        header = (
            f"{self.feature_extractor.upper()} on {self.name_env}"
            f"  --  {name}  --  {len(seeds)} seed(s) {list(seeds)}"
        )

        paths = []
        for suffix, centre, low, high, centre_label, band_label, colour in (
            (
                "mean_std",
                mean,
                mean - std,
                mean + std,
                "mean over seeds",
                "+- 1 std",
                "#2f6fdb",
            ),
            (
                "median_iqr",
                median,
                q25,
                q75,
                "median over seeds",
                "IQR (q25 - q75)",
                "#c2410c",
            ),
        ):
            fig = go.Figure()

            # THE BAND FIRST, so the centre line draws on top of it rather
            # than under. A filled "toself" polygon: up the upper edge, back
            # down the lower one reversed.
            fig.add_trace(
                go.Scatter(
                    x=list(iterations) + list(iterations)[::-1],
                    y=list(high) + list(low)[::-1],
                    fill="toself",
                    fillcolor=self._rgba(colour, 0.18),
                    # mode SPELLED OUT. Left to plotly it defaults to
                    # "lines+markers" for a trace under 20 points, and the
                    # polygon is 2 * len(iterations) = 12 -- so the band would
                    # come out dotted along both edges and its legend swatch
                    # would carry a marker it does not mean.
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    name=band_label,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=centre,
                    mode="lines+markers",
                    line=dict(color=colour, width=2.5),
                    marker=dict(size=6),
                    name=centre_label,
                    hovertemplate="iteration %{x}<br>"
                    + metric
                    + " %{y:.3f}<extra></extra>",
                )
            )

            # the raw seeds, DRAWN BUT HIDDEN. legendonly keeps them out of
            # the svg and out of the first look, and one click puts them back
            # -- which is what you want the moment the band looks wrong.
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
            stem = os.path.join(directory, f"curve_{metric}_{suffix}")

            fig.write_html(f"{stem}.html", include_plotlyjs=include_plotlyjs)
            paths.append(f"{stem}.html")

            # needs kaleido. It is in requirements.txt, but a missing static
            # exporter should not lose the html that was just written or kill
            # a study that is otherwise finished.
            try:
                fig.write_image(f"{stem}.svg")
                paths.append(f"{stem}.svg")
            except Exception as error:
                say(f"could not write {stem}.svg ({type(error).__name__}: {error})")

        say(f"plotted {metric} for {name}: {len(paths)} file(s) in {directory}")
        return paths

    @staticmethod
    def _rgba(hex_colour, alpha):
        """'#2f6fdb', 0.18 -> 'rgba(47,111,219,0.18)'. Plotly wants the string."""
        hex_colour = hex_colour.lstrip("#")
        r, g, b = (int(hex_colour[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"

    def plot_hpo(self, metric=None, logger=None, include_trials=True):
        """Plot every trial and the best trial. Returns the paths.

        Walks hpo/ and plots whichever of trial_<n>/ and best_trial/ actually
        hold checkpoints, so it is safe to call on a study that is half
        finished or on one whose trials were pruned. Re-running overwrites --
        the plots are derived from the .pth files and nothing about them is
        cumulative.

        include_trials=False skips the thirty trial directories and does only
        best_trial/, which is the fast version when the individual trials are
        not what is being looked at.
        """
        say = logger.info if logger is not None else print

        if metric is None:
            metric = self.hpo_metric

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
            paths += self.plot_eval_curves(directory, metric=metric, logger=logger)

        say(f"plot_hpo: {len(paths)} file(s) written under {self.dir_hpo}")
        return paths

    def print_separate_lines(self, logger, n=10):
        for _ in range(n):
            logger.info("=" * 78)

    def callback_optuna_report_function(self, kind_training, logger, study, trial):
        """One line per finished trial, plus the best so far.

        Passed to study.optimize(callbacks=[...]), so it runs after EVERY
        trial including pruned and failed ones -- which is the point: a study
        that silently drops trials is one you cannot debug afterwards.
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
            # best_trial RAISES when nothing has completed. Common and fine
            # early on, or if the first trials were pruned.
            logger.info("  best so far: none, no trial has completed yet")
        logger.info("-" * 78)

    # ----- the loop --------------------------------------------------------
    def hpo_optimize(self, study, n_trials, objective, logger, kind_training):
        """study.optimize, but resume-aware. Safe to run repeatedly.

        TWO THINGS IT ADDS over calling study.optimize directly.

        1. BUDGET IS WHAT IS LEFT, not n_trials again. Re-running a study that
           already did 12 of 30 runs 18 more, not 30 more. The count is over
           COMPLETE + PRUNED only, so a trial that CRASHED does not spend
           budget -- a machine that ran out of memory or was killed gets that
           trial back rather than paying for the interruption.

        2. THE INTERRUPTED TRIAL IS RE-QUEUED. If the last trial is FAIL or
           still marked RUNNING (what a hard kill leaves behind), its exact
           params go back on the queue with enqueue_trial, so the run that was
           lost is the run that is retried -- rather than the sampler drawing
           somewhere else and that point never being measured. This is the
           mechanism that makes a crash cost time and not coverage.

        Returns the number of trials this call actually ran.
        """
        import optuna

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

        last = study.trials[-1] if study.trials else None
        if last is not None and last.state in (
            optuna.trial.TrialState.FAIL,
            optuna.trial.TrialState.RUNNING,
        ):
            # .params is empty if it died before suggesting anything, in which
            # case there is nothing to repeat and the sampler just draws again
            if last.params:
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
        import optuna

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
        # THE MAXIMUM OF n_trials NOISY MEASUREMENTS, so it is biased upward by
        # the selection itself -- the winner's curse. final() no longer
        # retrains, so it reports THIS number rather than an independent
        # estimate of it; the caveat travels with the value instead of being
        # cancelled by a second sample. See HPOPPO.final.
        logger.info(
            "  (the max over trials, so biased upward by the selection itself; "
            "final() reports these same runs and does not re-estimate it)"
        )
        self.print_separate_lines(logger)

        return best

    # ------------------------------------------------------------------
    # hidden state
    # ------------------------------------------------------------------
    def zero_hidden(self, batch_size=None):
        """h_0 (and c_0) of shape (1, batch_size, hidden_size), None for MLP.

        The leading 1 is num_layers * num_directions -- NOT the batch. That
        stays 1 because both encoders are single-layer and one-directional.
        batch_size defaults to n_workers, which is what the rollout needs;
        evaluate() passes 1, because it plays one episode at a time.

        Zeros are what the paper starts every episode from (Section 6.4 calls
        this "naive" but uses it anyway; learnable initial states are hard
        because the gradient is truncated).
        """
        if not self.is_recurrent:
            return None

        if batch_size is None:
            batch_size = self.n_workers

        h = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        return (h, h.clone()) if self.is_lstm else h

    def reset_hidden_of(self, hidden, w):
        """Zero the hidden state of worker w only, in place.

        Called when worker w's game ends. The other workers are still in the
        middle of their own games and must keep remembering. Column w is the
        worker axis of (1, n_workers, hidden_size).

        Returns hidden so the caller can write  h = config.reset_hidden_of(h, w).
        """
        if not self.is_recurrent:
            return hidden

        if self.is_lstm:
            hidden[0][:, w] = 0.0  # h
            hidden[1][:, w] = 0.0  # c
        else:
            hidden[:, w] = 0.0

        return hidden

    # ------------------------------------------------------------------
    # watching a trained policy
    # ------------------------------------------------------------------
    def watch_agent(self, path_model=None, deterministic=None, steps_per_sec=2.5):
        """A pygame window that plays a saved policy. NOBODY DRIVES THE AGENT.

        The human-controlled viewer in test_enviroment/ answers "what is this
        task like?". This one answers "what did my agent learn?", so nobody
        chooses the actions and the controls are four buttons instead:

            STEP -1     go back one action. The env has no reverse gear, so
                        this resets to the same seed and re-walks the recorded
                        actions -- see go_to() below
            PAUSE/PLAY  stop the clock. The panel keeps drawing, so this is
                        how you read pi(a|s) for the step you are on
            STEP +1     take exactly one action, then pause again. Pausing on
                        its own is not enough to follow a policy -- by the
                        time you hit it the step you wanted has gone by, so
                        the controls that actually work are the single steps
            LAST GAME   go back to the PREVIOUS maze of the eval set
            REPLAY      play the SAME maze again from step 0
            NEW GAME    advance to the NEXT maze of the eval set and play it
            AUTO NEW    a toggle, not an action: when an episode ends, wait
             GAME       _VIEW_AUTO_DELAY seconds and start the next maze on
                        its own. Left on, the window walks the whole eval set
                        unattended, which is how you find WHICH mazes a 0.94
                        policy is losing without pressing anything 50 times

        The two button rows are deliberately parallel, and reading them that
        way is the whole layout:

            STEP -1     PAUSE/PLAY   STEP +1      move within ONE episode
            LAST GAME   REPLAY       NEW GAME     move between EPISODES

        Left goes back, right goes forward, the middle one holds still. So
        LAST GAME is to mazes what STEP -1 is to steps, and the index wraps
        both ways -- LAST GAME on maze 1 lands on maze 50.

        WHY LAST GAME IS NEEDED AT ALL. Without it the eval set is a one-way
        street: overshoot the maze you wanted and the only way back is 49 more
        presses of NEW GAME. That matters most with AUTO NEW GAME on, which is
        exactly when a maze goes by before you have read the outcome -- the
        loss you are hunting for scrolls past and cannot be recovered.

        STEP -1 and STEP +1 are exact inverses: the actions taken are on
        record, so going back and forward again re-walks the SAME trajectory
        rather than re-sampling a new one. LAST GAME and NEW GAME are NOT
        inverses in that sense -- each one restarts an episode from step 0,
        so the maze comes back but the trajectory through it is drawn again
        (identically if deterministic, freshly sampled if not).

        WHY THE EVAL SET AND NOT RANDOM MAZES. The mazes are exactly
        evaluate()'s: seed = eval_seed + i for i in 0..n_eval_episodes-1. So
        the window is a walkthrough of the number in the log -- a run reporting
        success 0.94 has three losing mazes in those 50, and NEW GAME will
        eventually land on them. Random mazes would show a different
        distribution from the one being reported.

        WHAT REPLAY IS FOR. Same seed means the same maze, the same cue and the
        same start. With deterministic=True the trajectory repeats exactly, so
        it is a rewind. With deterministic=False the policy is re-SAMPLED, so
        pressing it a few times shows how much of the behaviour is the policy
        and how much is luck -- which on a bimodal task like this is worth more
        than one more number.

        The sidebar shows what the ENV knows on the left and what the AGENT
        knows on the right: the full maze is rendered for the human, but the
        7x7 panel is the agent's entire input. The cue is only in that panel at
        step 0. After that, anything the agent still does right about it is
        coming out of the hidden state -- which is the whole experiment, made
        visible.

        It also draws the actor's full action distribution and the critic's
        V(s) every step. Those are the two things a return curve cannot show:
        a policy that is right but unsure looks identical to one that is right
        and certain, until you watch the bars.

        Arguments:
            path_model      defaults to config.path_model, i.e. the encoder the
                            config is set to. load_model refuses a file whose
                            architecture disagrees with the config.
            deterministic   defaults to config.eval_deterministic. True =
                            argmax, the policy's actual decision. False =
                            sample, the same way training and the logged eval
                            curve do.
            steps_per_sec   how fast the agent acts. NOT the frame rate -- the
                            window redraws smoothly at 60 either way. 2.5 is
                            400 ms a step, slow enough to read the action
                            distribution as it changes without pausing.

        Keys: SPACE pause/resume, LEFT/RIGHT ARROW step one action back or
        forward, P previous maze, N next maze, R replay, A toggle auto new
        game, Q or Esc quit.

        Blocks until the window is closed. Never called from main.py or from
        training -- it is a separate thing you run against a saved file.
        """
        # imported HERE, not at the top of the module: training must not need
        # pygame installed, and importing it opens an SDL connection
        import pygame

        from models.model import Network

        if deterministic is None:
            deterministic = self.eval_deterministic

        # ---- FIRST, the architecture the file was trained with -----------
        # BEFORE build_env and before build_extractor, both of which read the
        # config and would otherwise read the DEFAULTS. A tuned checkpoint was
        # trained at drawn values -- hidden_size for every encoder, plus
        # n_layers_mlp / d_model / n_heads / n_layers_transformer -- so a
        # config built fresh from make_config() describes a different network
        # than the one on disk. Skipping this gives a load_state_dict shape
        # error at best; the layer-count mismatches are not covered by
        # load_model's guard at all. A no_hpo checkpoint returns {} and this
        # is a no-op. See checkpoint_params / searched_params.
        if path_model is None:
            path_model = self.path_model
        params = self.checkpoint_params(path_model)
        if params:
            self.apply_params(params)

        # ---- the agent -------------------------------------------------
        # render_mode is what makes env.render() give back a picture. Same
        # builder as training's, so this is the same game, wrappers and all.
        env = self.build_env(render_mode="rgb_array")
        n_actions = env.action_space.n

        model = Network(self.build_extractor(), self.hidden_size, n_actions)
        checkpoint = self.load_model(model, path_model)
        model.to(self.device)
        model.eval()  # no dropout or batchnorm here, but it is the contract

        # ---- the window ------------------------------------------------
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

        # ---- episode state ---------------------------------------------
        # everything the two buttons reset lives in this dict, so start() is
        # the ONLY place it is written and there is no chance of a button
        # clearing four of the five things it should.
        #
        # max_steps is read ONCE, off the live env, and start() never touches
        # it. Not config.env_max_steps: that property builds a throwaway env
        # to answer, and the sidebar asks every frame -- 60 envs a second.
        ep = {"max_steps": env.unwrapped.max_steps}

        def think():
            """One forward pass on the CURRENT observation. Advances hidden ONCE.

            This is the one place the recurrence moves, and that is not an
            accident: a second forward pass "just to draw the bars" would feed
            the same observation through the GRU twice, and the hidden state
            the next action is chosen from would have consumed a step that
            never happened. Draw from what this stores, never by re-running.
            """
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
            """Begin an episode. move = +1 next maze, -1 previous, 0 this one.

            An integer rather than the old next_maze boolean, because there
            are now three destinations and a second boolean beside the first
            would allow the meaningless "next and previous at once".

            The modulo wraps BOTH ways -- Python's -1 % 50 is 49, not -1 -- so
            LAST GAME on maze 1 lands on maze 50 and the eval set behaves as a
            ring rather than a street with two dead ends.

            keep_trail is for go_to() only: it re-runs this to get back to a
            clean reset and then re-walks the recorded actions, so the trail
            must survive the wipe.
            """
            index = ep.get("index", -1)
            if move:
                index = (index + move) % self.n_eval_episodes

            trail = list(ep["trail"]) if keep_trail else []

            # the same seed evaluate() uses for episode `index`
            obs_state, _ = env.reset(seed=self.eval_seed + index)

            ep.update(
                index=index,
                obs=obs_state["image"],
                hidden=self.zero_hidden(batch_size=1),  # a new game remembers nothing
                step=0,
                total_reward=0.0,
                done=False,
                outcome="",
                history=[],
                # every action this episode has ever taken, in order, never
                # truncated. It is what makes STEP -1 possible: MiniGrid has no
                # reverse gear, so going back means resetting to the same seed
                # and re-walking this list.
                trail=trail,
                # read ONCE, from the first observation, and then kept on
                # screen for the rest of the episode. After step 0 the cue is
                # out of view and the only copy left is inside the hidden
                # state, so this label is what the agent's choice at the
                # junction has to be judged against.
                cue=_find_cue(obs_state["image"]),
            )
            think()

        def act(forced=None):
            """Take one action, then think about what came back.

            forced replays a RECORDED action instead of the one think() just
            chose. Only go_to() and advance() pass it, and it is what keeps a
            rewind faithful: with deterministic=False, re-sampling on the way
            back would walk a different trajectory and the step you were trying
            to look at again would not be there.
            """
            action = ep["action"] if forced is None else forced
            obs_state, reward, terminated, truncated, _ = env.step(action)

            ep["obs"] = obs_state["image"]
            ep["step"] += 1
            ep["total_reward"] += reward
            ep["history"] = (ep["history"] + [(ep["step"], action, reward)])[-8:]

            # only a genuinely NEW action extends the record. Re-walking one
            # that is already in the trail must not append it a second time.
            if ep["step"] > len(ep["trail"]):
                ep["trail"].append(action)

            if terminated or truncated:
                ep["done"] = True
                # MemoryEnv pays 1 - 0.9*(steps/max_steps) for the correct
                # object and exactly 0 for the wrong one, so reward > 0 IS
                # success. A truncation means it never reached either.
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
            """Put the episode back at step `target`, however far back that is.

            THERE IS NO REVERSE GEAR. env.step() cannot be undone, and neither
            can a GRU's hidden state -- h_t is not invertible. So "back one
            step" is really "reset to the same seed and replay target actions",
            which is exact precisely because the maze is seeded and the actions
            are on record.

            The cost is O(target) env steps and forward passes per press, so
            stepping back from step 240 replays 239 of them. That is a few tens
            of milliseconds on a click, and it buys a scrubber that cannot
            drift out of sync with what actually happened.
            """
            target = max(0, target)

            start(move=0, keep_trail=True)  # same maze, trail preserved
            for i in range(target):
                act(forced=ep["trail"][i])

            # show the action history took from here, not a fresh sample, so
            # STEP -1 followed by STEP +1 lands exactly where it started
            if target < len(ep["trail"]):
                ep["action"] = ep["trail"][target]

        start(move=+1)  # index starts at -1, so this opens on maze 1

        # ---- the loop --------------------------------------------------
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
                # AUTO NEW GAME. The wait is not politeness: the outcome badge
                # and the final position are the whole reason to watch, and
                # cutting to the next maze the instant an episode ends means
                # never seeing either. Paused still means paused.
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
                    # act on a CLOCK, not once per frame: the window still
                    # redraws at 60 fps while the agent moves at a speed a
                    # human can read
                    since_step += dt
                    if since_step >= 1.0 / steps_per_sec:
                        since_step = 0.0
                        advance()

            screen.fill(_VIEW_BG)

            # left: the whole maze, which is FAR more than the agent can see.
            # Centred vertically because the sidebar is taller than the frame
            # is square -- pinned to the top it leaves an odd black shelf.
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
                    # None unless a countdown is actually running, so the
                    # sidebar does not have to re-derive the same condition
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
    """Spawn the agent in the start room, where the cue is actually visible.

    THE BUG THIS FIXES IS IN MINIGRID, NOT IN THIS PROJECT. MemoryEnv._gen_grid
    says "Fix the player's start position and orientation" and then does:

        self.agent_pos = np.array((self._rand_int(1, hallway_end + 1),
                                   height // 2))
        self.agent_dir = 0                       # east

    hallway_end is width - 3, so on MemoryS11 the agent starts at a UNIFORMLY
    RANDOM x in 1..8, facing east, anywhere along the hallway. The cue -- the
    object it is supposed to memorize -- is fixed at (1, height // 2 - 1),
    inside the walled start room behind it. Measured over 2000 resets of
    MemoryS11:

        start_x  episodes  cue visible in the first observation
              1       233                                   233
              2       229                                     0
              3       272                                     0
           4..8      1266                                     0

    Only x = 1 sees it: one episode in eight. In the other seven the cue is
    never observed at all, so there is nothing to remember, and a GRU, an LSTM
    and an MLP are mathematically the same agent. That is why the ablation came
    out flat -- all three converged on "sprint east and always turn the same
    way", the best a memoryless policy can do.

    THE FIX. After reset, put the agent at (1, height // 2) facing east and
    re-derive the observation. Verified on S7, S11 and S13: 200/200 resets have
    the cue in view and 200/200 have the spawn tile free.

    Why the observation must be REBUILT. reset() returns an obs computed from
    the position _gen_grid chose. Move the agent afterwards and that obs is a
    photograph of somewhere the agent no longer is -- the first step would be
    taken on a stale view. gen_obs() renders the 7x7 window from the CURRENT
    agent_pos and agent_dir, which is what makes the move real.

    Nothing else is touched: the maze, the cue, the two split objects and
    success_pos / failure_pos are all still drawn by _gen_grid from the env's
    own seeded RNG, so evaluate()'s fixed per-episode seeds keep giving the
    same mazes. Only where the agent wakes up changes.

    What it costs: every episode now starts at the far end, so the walk is
    longer -- ~10 steps minimum on S11 instead of as few as 3. Against
    max_steps = 5 * size^2 = 605 that is nothing, and the reward
    1 - 0.9 * (steps / max_steps) barely moves.

    What it buys: the memoryless ceiling stays where it was (guess one side,
    0.62 on the 50 eval mazes) while a policy that remembers can now reach
    ~1.0. THAT GAP IS THE ABLATION. Without it there is no experiment.
    """

    def __init__(self, env):
        super().__init__(env)
        self._checked = False

    def reset(self, **kwargs):
        # let MiniGrid build the maze and place everything as usual, then
        # override only where the agent stands
        self.env.reset(**kwargs)

        env = self.env.unwrapped

        # (1, height // 2) is the middle row of the start room. The cue sits
        # one tile above it at (1, height // 2 - 1) and the room is always at
        # least 3 tall, so this tile is empty at every registered size.
        env.agent_pos = np.array((1, env.height // 2))
        env.agent_dir = 0  # east, the same heading _gen_grid uses

        # once, not every episode: a cheap guard that the whole point of this
        # wrapper still holds, without paying for gen_obs_grid twice per reset
        # forever. in_view is the env's own line-of-sight test.
        if not self._checked:
            assert env.in_view(1, env.height // 2 - 1), (
                f"{env.spec.id}: the cue is not visible from the forced spawn "
                f"-- this wrapper assumes MemoryEnv's layout"
            )
            self._checked = True

        # rebuilt from the position set two lines up, not the one reset()
        # returned. Info is regenerated too, so nothing describes the old spot.
        return env.gen_obs(), {}


class SequenceDataset(Dataset):
    """split_pad_mask's output as a torch Dataset. ONE ITEM = ONE SEQUENCE.

    This is the one thing to be careful about coming from supervised deep
    learning: the sample is not a timestep, it is a whole padded sequence of
    up to L of them. A DataLoader with batch_size=8 therefore hands back 8
    SEQUENCES, i.e. (8, L, ...) -- somewhere between 8 and 8*L real steps,
    depending on how long those episodes happened to be. That varying step
    count is exactly why every loss is reduced with masked_mean, never
    .mean().

    A sequence cannot be split. Its steps are computed from one h_0 in order,
    so half a sequence would start from a hidden state nobody ever computed.
    Shuffling ACROSS sequences is fine, and is what the DataLoader does.

    Nothing here is PPO-specific: it takes a dict of (n_seq, L, ...) tensors
    and serves rows of it. It lives beside the other shape-and-plumbing code
    for that reason.

    hxs and cxs arrive as (1, n_seq, H) -- the leading 1 is nn.GRU's
    num_layers axis, not a batch. It is dropped here so every tensor is
    indexed on axis 0 like any other dataset, and put back by the caller
    right before the model call.
    """

    def __init__(self, batch):
        self.data = {k: (v[0] if k in ("hxs", "cxs") else v) for k, v in batch.items()}
        self.n_seq = self.data["mask"].shape[0]

    def __len__(self):
        return self.n_seq

    def __getitem__(self, i):
        # default_collate stacks these dicts back into (mb, L, ...) tensors,
        # so no collate_fn is needed
        return {k: v[i] for k, v in self.data.items()}


# ======================================================================
# pygame viewer -- drawing only, used by nothing except Helper.watch_agent
#
# None of this imports pygame at module level: watch_agent imports it and
# passes the surface in, so a machine without pygame can still train. These
# are module functions rather than methods because they know about pixels
# and nothing about the config.
# ======================================================================

_VIEW_SIDEBAR_W = 440  # px, the right-hand panel
_VIEW_MAZE_PX = 560  # px, the square the env frame is scaled into
_VIEW_MIN_H = 880  # px, tall enough for the whole sidebar
_VIEW_CELL = 34  # px per cell of the 7x7 observation
_VIEW_FPS = 60  # REDRAW rate. The agent's step rate is steps_per_sec.
_VIEW_AUTO_DELAY = 1.5  # s to sit on a finished episode before AUTO NEW GAME
#                         moves on, so the outcome is actually readable

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
    """The one key/ball in the FIRST observation, as 'green ball'. None if absent.

    Called only at step 0, where MemoryEnv's layout guarantees exactly one such
    object in view: the cue in the start room. At the junction there are two
    (one per branch), which is why this is not a general-purpose scan -- run it
    later and it would report whichever one it hit first.
    """
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            obj, color, _ = image[x, y]
            if obj in (5, 6):
                return f"{_COLOR_NAME.get(color, '')} {_OBJ_NAME[obj]}".strip()
    return None


def _visible_objects(image):
    """['green ball 3 ahead, 2 left', ...] for every non-structural cell in view.

    Skips unseen/empty/wall/floor/agent: the maze walls are already obvious in
    the rendered frame on the left, and listing them would bury the one or two
    cells that actually matter.
    """
    n = image.shape[0]
    agent_col, agent_row = n // 2, n - 1  # the agent sits bottom-centre
    lines = []

    for row in range(n):
        for col in range(n):
            # MiniGrid indexes the observation [x, y], and row 0 is the FARTHEST
            # cell ahead, so the agent's own cell is the bottom-centre one
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

            # a faint outline on every cell. "unseen" is nearly the same colour
            # as the panel behind it, so without this the 7x7 shape disappears
            # exactly when most of the view is unseen -- which is most of the
            # time, and precisely when the human wants to see how little the
            # agent has to work with.
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
    """One bar per action, pi(a|s). The action about to be taken is highlighted.

    This is the readout a return curve cannot give. A policy that turns the
    right way at 0.35 and one that turns it at 0.99 score identically for that
    episode, and only one of them has actually learned the task.
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
    """A filled rounded rect that lights up under the cursor.

    enabled=False draws it flat and grey and, crucially, does NOT light up on
    hover -- a button that highlights but does nothing is worse than one that
    is visibly dead. The press handler ignores it independently; this is only
    the picture.
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
):
    """The whole right-hand panel. Returns {name: pygame.Rect} for every button.

    Returning the rects is what wires the buttons up: watch_agent's event loop
    hit-tests against whatever this drew, so the layout lives in one place and
    the click handling cannot drift out of sync with it.

    ui carries the transient view state that is not part of the episode --
    paused, auto_new, and auto_in (seconds left before the next maze starts).
    Bundled rather than passed one by one so adding a control does not mean
    re-threading another argument through the call.
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
        # the single most useful line in the window: what the agent was shown
        # at step 0, still on screen at step 40 when only its memory has it
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

    # ---- the buttons, pinned to the bottom -------------------------------
    # Three rows: transport, episode, and the one persistent setting at the
    # bottom. The toggle is shorter than the action buttons on purpose -- it
    # changes what happens LATER, it does not do anything when pressed, and
    # looking different is what keeps that distinction readable.
    bh, th, gap = 42, 32, 12
    row_toggle = win_h - th - 34
    row_bottom = row_toggle - bh - gap
    row_top = row_bottom - bh - gap

    # BOTH action rows use these same three columns, so the buttons line up
    # vertically and the two rows read as the same control at two scales:
    #
    #     STEP -1     PAUSE/PLAY   STEP +1      <- moves within one episode
    #     LAST GAME   REPLAY       NEW GAME     <- moves between episodes
    #
    # left goes back, right goes forward, the middle one stays put. Giving
    # the episode row its own geometry would break that reading for no gain.
    side = 104
    middle = inner - 2 * side - 2 * gap
    col_left = pad
    col_mid = pad + side + gap
    col_right = col_mid + middle + gap

    buttons = {
        # greyed out at step 0, where there is nothing behind us to go back to
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
        # one button, two labels. Green while paused reads as "press to go",
        # which is the state you are in when you have stopped to read the bars.
        "pause": _draw_button(
            screen,
            pygame,
            pygame.Rect(col_mid, row_top, middle, bh),
            "PLAY" if paused else "PAUSE",
            fonts["head"],
            mouse,
            (120, 230, 140) if paused else (200, 200, 200),
        ),
        # advance exactly one action. Pausing alone is not enough to read a
        # policy: by the time you hit it the step you wanted is already gone,
        # so the useful control is one that moves in single steps.
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
        # the previous maze of the eval set, wrapping round to 50 from 1. Never
        # disabled: unlike STEP -1, which runs out at step 0, there is always
        # another maze behind this one.
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
        # a SETTING, not an action: when an episode ends, start the next eval
        # maze on its own after a short pause. Turn it on to watch all 50 go
        # by without touching anything -- which is the fastest way to see
        # WHICH mazes a 0.94 policy is losing.
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
