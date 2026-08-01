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
    config.log_model_summary() torchinfo's table: layers, params, size in MB
    config.build_model_path()  agents/pretrained_model_feature_extractor/,
                               created, + the filename
    config.save_model()        weights + the architecture they belong to
    config.load_model()        the same, back into a built model, checked
    config.zero_hidden()       h_0 (and c_0) full of zeros
    config.reset_hidden_of()   zero the hidden state of ONE worker
    config.watch_agent()       a pygame window that plays a saved policy

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

from models.feature_extractor import MLP, LSTM, GRU, Transformer


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
        """MLP / LSTM / GRU / TRANSFORMER, picked by self.recurrent_model.

        All four take (batch, seq_len, 7, 7, 3) and give back
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

        # @property, so not in vars(self) -- but they decide what was built.
        # hasattr because d_ff exists on the transformer config only.
        logger.info("  " + "-" * 40)
        for key in (
            "device",
            "is_recurrent",
            "is_lstm",
            "d_ff",
            "name_model",
            "path_model",
        ):
            if hasattr(self, key):
                logger.info(f"  {key:<24}{getattr(self, key)}")

        logger.info(bar)

        return logger

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
            # pass during optimization actually receives
            batch_size = self.mini_batch_size

        shape = (batch_size, seq_len, 7, 7, 3)
        info = summary(model, input_size=shape, dtypes=[torch.uint8], verbose=0)

        write = logger.info if logger is not None else print
        write("")
        write(f"MODEL SUMMARY  {self.recurrent_model.upper()}  (probe input {shape})")
        for line in str(info).splitlines():
            write(f"  {line}")
        write("")

        return info

    def build_model_name(self):
        """ppo_<seed>_<ENCODER>_<env>.pth -- the ONE place the filename is spelled.

            ppo_42_GRU_MiniGrid-DoorKey-8x8-v0.pth
            ppo_0_MLP_MiniGrid-MemoryS7-v0.pth

        ALL THREE halves are in the name because all three change what the
        weights mean. The encoder decides the architecture; the env decides
        what the agent was trained to do, and a DoorKey policy loaded against
        MemoryS11 is not a worse agent, it is a meaningless one; the seed
        decides WHICH RUN this is, and two seeds of the same encoder on the
        same env are two independent samples that must not share a file.

        Each of the three was added after the corresponding collision. Keying
        on the encoder alone meant a GRU run on MemoryS11 silently overwrote a
        GRU run on DoorKey. Keying on encoder + env still meant every seed of a
        multi-seed run overwrote the one before it -- and a seeded study is
        exactly what makes a result reportable, so that one destroys the whole
        point of running three. train_agent() saves on every improvement and
        starts each run from best_success = -1.0, so the first evaluation of
        the new run, however bad, lands on top of a finished result from the
        old one. Nothing warns, because the filename is the only thing that
        ever distinguished them.

        It is a METHOD, not an attribute set in Config.__init__, so that all
        three parts are read WHEN IT IS CALLED. That is what lets watch.py
        override recurrent_model from the command line and get the matching
        file -- and, for the seed, it is not optional: self.seed does not hold
        the run's real seed until set_seed() has run, which PPOAgent.__init__
        does AFTER the config exists. An f-string evaluated once in __init__
        would spell seed_default for every run ever. See Config.name_model,
        which is a property for the same reason.

        The replace() is for gymnasium's namespaced ids ("ALE/Pong-v5"), which
        MiniGrid does not use but which would otherwise put a directory
        separator in the middle of a filename and fail confusingly.
        """
        env = self.name_env.replace("/", "-")
        return f"ppo_{self.seed}_{self.recurrent_model.upper()}_{env}.pth"

    def build_model_path(self):
        """pretrained_model_feature_extractor/ppo_<seed>_<encoder>_<env>.pth,
        with the directory made.

        The path itself is decided in Config (dir_pretrained_model +
        name_model, both live -- name_model is a property); this only creates
        the directory and hands the path back, the same split build_logger
        uses for logs/.

        Call it right before torch.save. Making the directory at import time
        instead would litter agents/pretrained_model_feature_extractor/ into
        every checkout that merely imports the config without ever training
        anything.
        """
        os.makedirs(self.dir_pretrained_model, exist_ok=True)
        return self.path_model

    def save_model(self, model, optimizer=None, **extra):
        """Write model (+ optimizer) to build_model_path(). Returns the path.

        The file is a dict, not a bare state_dict, because a bare one cannot be
        loaded without already knowing what shape to load it into. Alongside
        the weights it carries the FOUR attributes that decide the
        architecture:

            recurrent_model  GRU / LSTM / MLP   -> different modules entirely
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

        **extra goes in verbatim -- iteration=, eval_success_rate= and so on.
        Keep it to plain numbers and strings: everything here has to survive
        torch.load(weights_only=True).
        """
        path = self.build_model_path()

        checkpoint = {
            "model": model.state_dict(),
            "recurrent_model": self.recurrent_model,
            "hidden_size": self.hidden_size,
            "input_size": self.input_size,
            "name_env": self.name_env,
            "force_cue_visible": self.force_cue_visible,
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
                f"on its own whenever eval_success_rate improves."
            )

        # weights_only=True is the safe default in modern torch and everything
        # save_model writes (tensors, strings, ints, bools) is allowed under it
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state = checkpoint["model"]

            for key in ("recurrent_model", "hidden_size", "input_size", "name_env"):
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
            f"{self.recurrent_model.upper()} on {self.name_env}"
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
                int(ep["probs"].argmax())
                if deterministic
                else int(dist.sample()[0, 0])
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
                    "SOLVED" if reward > 0 else "WRONG OBJECT"
                    if terminated else "OUT OF TIME"
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
                screen, fonts, maze_px, win_h, self, ep, checkpoint, deterministic,
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
    0: "unseen", 1: "empty", 2: "wall", 3: "floor", 4: "door",
    5: "key", 6: "ball", 7: "box", 8: "goal", 9: "lava", 10: "agent",
}
_COLOR_NAME = {0: "red", 1: "green", 2: "blue", 3: "purple", 4: "yellow", 5: "grey"}

# channel 1 -> rgb, for the objects that are drawn in their own colour
_MG_RGB = [
    (220, 50, 50), (50, 200, 50), (60, 120, 220),
    (160, 50, 220), (240, 220, 0), (140, 140, 140),
]
_COLOR_DRIVEN = {4, 5, 6, 7}  # door, key, ball, box take the colour channel

# everything else has a fixed display colour
_OBJ_RGB = {
    0: (30, 30, 30), 1: (210, 210, 210), 2: (75, 85, 105),
    3: (185, 175, 145), 8: (0, 200, 80), 9: (255, 90, 0), 10: (255, 50, 50),
}
_CELL_LABEL = {2: "W", 4: "D", 5: "K", 6: "O", 7: "[]", 8: "G", 9: "!"}

# MiniGrid's Discrete(7). Only the first three matter on MemoryEnv, which is
# itself worth seeing: a good policy puts almost no mass on the other four.
_ACTION_NAME = {
    0: "turn left", 1: "turn right", 2: "forward",
    3: "pick up", 4: "drop", 5: "toggle", 6: "done",
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
                name += f" ({['open', 'closed', 'locked'][state] if state < 3 else '?'})"
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
                ox + col * _VIEW_CELL, oy + row * _VIEW_CELL,
                _VIEW_CELL - 1, _VIEW_CELL - 1,
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
                screen.blit(surf, (
                    rect.x + (rect.w - surf.get_width()) // 2,
                    rect.y + (rect.h - surf.get_height()) // 2,
                ))

    # the agent always faces "up" in its own view, so the arrow is fixed
    ax = ox + agent_col * _VIEW_CELL + _VIEW_CELL // 2
    ay = oy + agent_row * _VIEW_CELL - 2
    pygame.draw.polygon(screen, _VIEW_HEAD, [(ax, ay - 7), (ax - 5, ay), (ax + 5, ay)])
    pygame.draw.rect(screen, (110, 110, 110), (ox, oy, n * _VIEW_CELL, n * _VIEW_CELL), 1)


def _draw_policy(screen, pygame, probs, chosen, x, y, width, font):
    """One bar per action, pi(a|s). The action about to be taken is highlighted.

    This is the readout a return curve cannot give. A policy that turns the
    right way at 0.35 and one that turns it at 0.99 score identically for that
    episode, and only one of them has actually learned the task.
    """
    for a, p in enumerate(probs):
        used = a in _ACTION_USED
        is_next = a == chosen

        label = font.render(f"{_ACTION_NAME[a]:<10}", True,
                            _VIEW_TEXT if used else (95, 95, 95))
        screen.blit(label, (x, y))

        bx = x + 78
        bw = width - 78 - 46
        pygame.draw.rect(screen, (45, 45, 45), (bx, y + 2, bw, 9))
        if p > 0.001:
            fill = _VIEW_HEAD if is_next else ((120, 170, 220) if used else (80, 80, 80))
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
        screen.blit(surf, (
            rect.x + (rect.w - surf.get_width()) // 2,
            rect.y + (rect.h - surf.get_height()) // 2,
        ))
        return rect

    hot = rect.collidepoint(mouse)
    body = accent if hot else tuple(int(c * 0.42) for c in accent)
    pygame.draw.rect(screen, body, rect, border_radius=6)
    pygame.draw.rect(screen, accent, rect, 2, border_radius=6)

    ink = (15, 15, 15) if hot else (235, 235, 235)
    surf = font.render(label, True, ink)
    screen.blit(surf, (
        rect.x + (rect.w - surf.get_width()) // 2,
        rect.y + (rect.h - surf.get_height()) // 2,
    ))
    return rect


def _draw_viewer_sidebar(
    screen, fonts, maze_px, win_h, config, ep, checkpoint, deterministic, ui,
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
    put(f"{config.recurrent_model.upper()}  on  {config.name_env}", "head", _VIEW_HEAD)
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
        "tiny", _VIEW_DIM,
    )
    rule()

    # ---- where we are ---------------------------------------------------
    put(
        f"eval maze {ep['index'] + 1} / {config.n_eval_episodes}"
        f"    seed {config.eval_seed + ep['index']}",
        "body", _VIEW_TEXT,
    )
    put(
        f"step {ep['step']} / {ep['max_steps']}"
        f"    reward {ep['total_reward']:+.3f}",
        "body", _VIEW_TEXT,
    )

    if ep["cue"]:
        # the single most useful line in the window: what the agent was shown
        # at step 0, still on screen at step 40 when only its memory has it
        put(f"CUE AT STEP 0:  {ep['cue']}", "head", (150, 200, 255))
    if ep["done"]:
        put(ep["outcome"], "big", _VIEW_GOOD if ep["outcome"] == "SOLVED" else _VIEW_BAD)
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
        screen, pygame, ep["obs"], sx + (_VIEW_SIDEBAR_W - grid_px) // 2, y, fonts["tiny"]
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
        put(f"s{step_i:03d}  {_ACTION_NAME[action]:<10} r={reward:+.3f}", "tiny",
            color, dy=1)

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
            screen, pygame, pygame.Rect(col_left, row_top, side, bh),
            "STEP -1", fonts["head"], mouse, (170, 170, 190),
            enabled=ep["step"] > 0,
        ),
        # one button, two labels. Green while paused reads as "press to go",
        # which is the state you are in when you have stopped to read the bars.
        "pause": _draw_button(
            screen, pygame, pygame.Rect(col_mid, row_top, middle, bh),
            "PLAY" if paused else "PAUSE", fonts["head"], mouse,
            (120, 230, 140) if paused else (200, 200, 200),
        ),
        # advance exactly one action. Pausing alone is not enough to read a
        # policy: by the time you hit it the step you wanted is already gone,
        # so the useful control is one that moves in single steps.
        "step": _draw_button(
            screen, pygame, pygame.Rect(col_right, row_top, side, bh),
            "STEP +1", fonts["head"], mouse, (170, 170, 190),
            enabled=not ep["done"],
        ),
        # the previous maze of the eval set, wrapping round to 50 from 1. Never
        # disabled: unlike STEP -1, which runs out at step 0, there is always
        # another maze behind this one.
        "last": _draw_button(
            screen, pygame, pygame.Rect(col_left, row_bottom, side, bh),
            "LAST GAME", fonts["head"], mouse, (90, 190, 255),
        ),
        "replay": _draw_button(
            screen, pygame, pygame.Rect(col_mid, row_bottom, middle, bh),
            "REPLAY", fonts["head"], mouse, (255, 190, 90),
        ),
        "new": _draw_button(
            screen, pygame, pygame.Rect(col_right, row_bottom, side, bh),
            "NEW GAME", fonts["head"], mouse, (90, 190, 255),
        ),
        # a SETTING, not an action: when an episode ends, start the next eval
        # maze on its own after a short pause. Turn it on to watch all 50 go
        # by without touching anything -- which is the fastest way to see
        # WHICH mazes a 0.94 policy is losing.
        "auto": _draw_button(
            screen, pygame, pygame.Rect(pad, row_toggle, inner, th),
            f"AUTO NEW GAME:  {'ON' if ui['auto_new'] else 'OFF'}",
            fonts["body"], mouse,
            (120, 230, 140) if ui["auto_new"] else (120, 120, 130),
        ),
    }

    hint = fonts["tiny"].render(
        "SPACE pause   <- -> step   P/N maze   R replay   A auto   Q quit",
        True, (105, 105, 105),
    )
    screen.blit(hint, (pad, row_toggle + th + 8))

    return buttons
