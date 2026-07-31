"""
Helper: the things that are DECIDED BY the config but are not PPO itself.

Config inherits from this class, so every one of these is reachable as
config.something and can read config's own attributes directly:

    config.device              cuda if there is one, else cpu
    config.is_recurrent        True for LSTM and GRU, False for MLP
    config.is_lstm             True only for LSTM (it has a cell state too)
    config.build_env()         one MiniGrid game, wrapped as the config asks
    config.env_max_steps       that env's own time limit, 5 * size^2
    config.build_extractor()   the encoder named by config.recurrent_model
    config.build_logger()      logs/log_<date>_<time>.log, hyperparameters first
    config.build_model_path()  agents/pretrained_model/, created, + the filename
    config.zero_hidden()       h_0 (and c_0) full of zeros
    config.reset_hidden_of()   zero the hidden state of ONE worker

Plus two standalone classes, imported directly rather than through config:

    StartInCueView             spawn the agent where the cue is actually visible
    SequenceDataset            split_pad_mask's output as a torch Dataset

None of this knows what an advantage, a ratio or a clip is. Swapping GRU for
LSTM is a change here and in the config, never in the agent.
"""

import logging
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import Dataset

# looks unused, but importing it is what registers MiniGrid-* with gymnasium.
# Delete it and gym.make raises NameNotFound. Do not let a linter remove it.
import minigrid  # noqa: F401

from models.feature_extractor import MLP, LSTM, GRU


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
        return self.recurrent_model.upper() in ("LSTM", "GRU")

    @property
    def is_lstm(self):
        """An LSTM needs (h, c). A GRU needs only h. An MLP needs neither."""
        return self.recurrent_model.upper() == "LSTM"

    # ------------------------------------------------------------------
    # builders
    # ------------------------------------------------------------------
    def build_env(self):
        """One MiniGrid game, wrapped the way the config asks. Never gym.make.

        Every env in the project comes from here -- the W rollout workers and
        evaluate()'s private one -- so training and evaluation cannot silently
        end up playing two different games.

        force_cue_visible adds StartInCueView (below). Turn it off to get the
        raw registered env back, which is what the first runs used and what
        the "no encoder beats any other" logs came from.
        """
        env = gym.make(self.name_env)

        if self.force_cue_visible:
            env = StartInCueView(env)

        return env

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
        """MLP / LSTM / GRU, picked by self.recurrent_model.

        All three take (batch, seq_len, 7, 7, 3) and give back
        (batch, seq_len, hidden_size), so the agent never has to care
        which one it got.
        """
        name = self.recurrent_model.upper()

        if name == "MLP":
            return MLP(self.input_size, self.hidden_size, self.n_layers_mlp)
        if name == "LSTM":
            return LSTM(self.input_size, self.hidden_size)
        if name == "GRU":
            return GRU(self.input_size, self.hidden_size)

        raise ValueError(f"unknown recurrent_model {self.recurrent_model!r}")

    def build_logger(self, log_dir="logs", name="rl_project"):
        """One log file per run: logs/log_<date>_<time>.log, hyperparameters first.

        Called once, from main.py, right after the agent is built. The file
        name carries the timestamp, so two runs never collide and the
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

        Build it AFTER PPOAgent, not before: the agent's __init__ calls
        set_seed(), which is what puts the actual seed into vars(self). Built
        first, the dump would show seed_default and no seed.

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
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

        bar = "=" * 78
        logger.info(bar)
        logger.info(f"RUN STARTED  {started:%Y-%m-%d %H:%M:%S}")
        logger.info(f"LOG FILE     {path}")
        logger.info(bar)
        logger.info("HYPERPARAMETERS")

        for key, value in vars(self).items():
            logger.info(f"  {key:<24}{value}")

        # @property, so not in vars(self) -- but they decide what was built
        logger.info("  " + "-" * 40)
        for key in ("device", "is_recurrent", "is_lstm"):
            logger.info(f"  {key:<24}{getattr(self, key)}")

        logger.info(bar)

        return logger

    def build_model_path(self):
        """agents/pretrained_model/ppo_<encoder>.pth, with the directory made.

        The path itself is decided in Config (dir_pretrained_model +
        name_model); this only creates the directory and hands the path back,
        the same split build_logger uses for logs/.

        Call it right before torch.save. Making the directory at import time
        instead would litter agents/pretrained_model/ into every checkout that
        merely imports the config without ever training anything.
        """
        os.makedirs(self.dir_pretrained_model, exist_ok=True)
        return self.path_model

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
        self.data = {
            k: (v[0] if k in ("hxs", "cxs") else v) for k, v in batch.items()
        }
        self.n_seq = self.data["mask"].shape[0]

    def __len__(self):
        return self.n_seq

    def __getitem__(self, i):
        # default_collate stacks these dicts back into (mb, L, ...) tensors,
        # so no collate_fn is needed
        return {k: v[i] for k, v in self.data.items()}
