"""Config: every hyperparameter the project shares, plus the paths and names
derived from them. Abstract -- an encoder subclass fills in _configure_model().
The toolbox it inherits (env builders, checkpoints, HPO bookkeeping, viewers)
lives in helper.py."""

import random
import os
import re
import numpy as np
import torch

from config.helper import Helper


class Config(Helper):
    """Hyperparameters shared by every encoder. Abstract -- subclass
    (ConfigMLP/ConfigGRU) and fill in _configure_model(), or build via
    make_config("MLP")."""

    feature_extractor = None  # set by each subclass in _configure_model()

    def __init__(self, tbptt_length=None):
        """Set every shared hyperparameter, call the subclass's
        _configure_model(), then derive the directories this run writes to.

        Args:
            tbptt_length (int | None): how many steps the gradient may flow
                back through. None (stored as "max") = cut on episode
                boundaries only, i.e. full BPTT. GRU only.

        Raises:
            ValueError: a length on a non-recurrent encoder, or one below 1.
        """
        self.seed_list = [0, 15, 12, 97, 98]

        self.input_size = 7 * 7 * 20  # 980 after flatten_obs one-hots each cell

        # How far the gradient may flow back; "max" = episode boundaries only.
        # Never searched: each length is its own study, in its own directory.
        self.tbptt_length = "max" if tbptt_length is None else int(tbptt_length)

        self.name_env = "MiniGrid-MemoryS17Random-v0"  # "MiniGrid-MemoryS11-v0" # "MiniGrid-DoorKey-8x8-v0"
        self.env_size = int(re.search(r"(\d+)", self.name_env).group(1))
        self.force_cue_visible = False

        self.n_workers = 32  # W: games played in parallel

        # AsyncVectorEnv (separate processes) vs SyncVectorEnv. False is
        # required in notebooks and scripts without an __main__ guard.
        self.async_envs = True

        # T: build_env feeds this back as the env's max_steps, so truncation
        # and the success reward (1 - 0.9*step_count/max_steps) move with it.
        self.worker_steps = self.env_max_steps // 4
        self.n_total_steps = self.n_workers * self.worker_steps  # W * T

        # Declared, not chosen: drawn per trial and written by apply_params.
        # None, not a default, so a path that forgets to apply params fails.
        self.lr = None
        self.wd = None
        self.gamma = None
        self.gae_lambda = None
        self.clip_eps = None
        self.value_coef = None
        self.entropy_coef = None
        self.max_grad_norm = None

        # not searched; PPOAgent reads `is not None` as "early-stop on KL".
        self.target_kl = None

        # decay lr linearly to 0 across n_iterations; falsy = constant lr.
        # Not searched either. With it on, `lr` is the INITIAL rate.
        self.lr_anneal = None

        # candidates, largest first: run_with_batch_size_fallback uses the largest that fits
        self.mini_batch_size = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4]

        # The budget every encoder is compared at: change it for all or none.
        # Never searched -- a trial must not be able to win by buying compute.
        self.n_epochs = 3
        self.n_iterations = 1000
        self.n_iterations_report = 10

        # Master switch for every clock (Timing in helper.py). Off: phases
        # become bare yields, no cuda.synchronize(), no TIME table printed.
        self.calculate_time = False

        self.n_eval_episodes = 50
        self.eval_seed = 10_000
        # False = sample actions, True = argmax. Sampling is safe: entropy is
        # ~0 by the time the task is solved, so both take the same actions.
        self.eval_deterministic = False

        self.n_trials = 50
        self.seed_hpo = 42
        self.hpo_direction = "maximize"

        # MedianPruner knobs, in HPOPPO's units: one step is one SEED. 0 means
        # the first check runs after seed 0 -- aggressive; 1 judges on two.
        self.hpo_pruner_warmup = 0

        # How often to check, offset by the warmup. 1 = at every seed; with so
        # few steps per trial there is nothing to gain by skipping any.
        self.hpo_pruner_interval = 1

        # No pruning until this many trials COMPLETED, and none at a step
        # unless that many reported there -- a median over two is not a median.
        self.hpo_pruner_startup_trials = 5
        self.hpo_pruner_min_trials = 5

        # Exactly two values: "return_mean_minus-std" or
        # "success_rate_mean_minus-std". Both score mean - lambda*std.
        self.hpo_objective = "return_mean_minus-std"

        # score = mean - lambda * std, both across seeds.
        self.hpo_lambda = 1.0

        # The PPO half, shared by both encoders; subclasses append their
        # architecture knobs. Every "name" must be an attribute above.
        self.search_space = [
            {"type": "float", "name": "lr", "low": 1e-5, "high": 1e-2, "log": True},
            {
                # step must divide (high - low): with step=0.001 Optuna
                # silently clipped high to 0.999, so 0.9999 was never drawn.
                "type": "float",
                "name": "gamma",
                "low": 0.99,
                "high": 0.9999,
                "step": 0.0001,
            },
            {
                "type": "float",
                "name": "gae_lambda",
                "low": 0.9,
                "high": 0.99,
                "step": 0.01,
            },
            {
                "type": "float",
                "name": "entropy_coef",
                "low": 1e-4,
                "high": 1e-1,
                "log": True,
            },
            {
                "type": "float",
                "name": "value_coef",
                "low": 0.01,
                "high": 1.0,
                "log": True,
            },
            {
                "type": "float",
                "name": "clip_eps",
                "low": 0.1,
                "high": 0.3,
                "step": 0.01,
            },
            {"type": "float", "name": "wd", "low": 1e-8, "high": 1e-2, "log": True},
            {
                "type": "float",
                "name": "max_grad_norm",
                "low": 0.1,
                "high": 2.0,
                "log": True,
            },
        ]

        self._configure_model()

        # An MLP has no time dependence to truncate, so a length would silently
        # mean nothing but would still split its results across directories.
        if self.tbptt_length != "max" and not self.is_recurrent:
            raise ValueError(
                f"tbptt_length={self.tbptt_length} makes no sense for "
                f"{self.feature_extractor}: there is no recurrence to truncate. "
                f"Drop --tbptt, or run it on GRU."
            )
        if self.tbptt_length != "max" and self.tbptt_length < 1:
            raise ValueError(f"tbptt_length={self.tbptt_length} must be at least 1")

        # Full BPTT keeps the plain pretrained_model_<ENCODER>/; each truncated
        # length gets a sibling, so studies never write over each other.
        self.tbptt_suffix = (
            "" if self.tbptt_length == "max" else f"_tbptt{self.tbptt_length}"
        )
        self.dir_model = os.path.join(
            "agents",
            f"pretrained_model_{self.feature_extractor.upper()}{self.tbptt_suffix}",
        )
        self.dir_pretrained_model = self.dir_model
        self.dir_hpo = os.path.join(self.dir_model, "hpo")

    def _configure_model(self):
        """Overridden by each subclass to set feature_extractor and its
        architecture knobs. Config itself is abstract."""
        raise NotImplementedError(
            "Config is abstract -- build a ConfigMLP / ConfigGRU, or call "
            "make_config('MLP')."
        )

    @property
    def name_hpo(self):
        """What this study IS, in one filename-safe token: encoder, env and
        backward reach, e.g. GRU_MiniGrid-MemoryS17Random-v0_tbptt8 (str)."""
        env = self.name_env.replace("/", "-")
        return f"{self.feature_extractor.upper()}_{env}{self.tbptt_suffix}"

    @property
    def name_study(self):
        """The optuna study name, "ppo_" + name_hpo (str). Two runs agreeing on
        this resume one study; differing on it start two."""
        return f"ppo_{self.name_hpo}"

    @property
    def path_hpo_csv(self):
        """Where csv_study_export writes the one-row-per-trial table (str)."""
        return os.path.join(self.dir_hpo, f"hpo_csv_{self.name_hpo}.csv")

    @property
    def path_hpo_db(self):
        """The study's sqlite URL (str) -- optuna's storage, and the thing that
        makes a search resumable."""
        path = os.path.join(self.dir_hpo, f"hpo_db_{self.name_hpo}.db")
        return "sqlite:///" + path

    @property
    def path_hpo_sampler(self):
        """Where the pickled TPE sampler lives (str). Restoring it is what
        makes a resumed study exploit rather than re-explore."""
        return os.path.join(self.dir_hpo, f"hpo_sampler_{self.name_hpo}.pkl")

    def set_seed(self, env=None, seed=None):
        """Seed python, numpy, torch (and cudnn) and optionally an env, then
        record the seed on the config as self.seed.

        Args:
            env (gym.Env | None): also seed this env's reset and its two
                spaces. None seeds the libraries only.
            seed (int | None): the seed; None uses seed_list[0].

        Returns:
            int: the seed applied.
        """
        if seed is None:
            seed = self.seed_list[0]
        self.seed = seed

        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if env is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)

        return seed
