import random
import os
import re
import numpy as np
import torch

from config.helper import Helper


class Config(Helper):
    """Hyperparameters shared by every encoder. Abstract -- subclass
    (ConfigMLP/ConfigLSTM/ConfigGRU/ConfigTransformer) and fill in
    _configure_model(), or build via make_config("MLP")."""

    feature_extractor = None  # set by each subclass in _configure_model()

    def __init__(self):
        self.seed_list = [0]# , 26, 98

        self.input_size = 7 * 7 * 20  # 980 after flatten_obs one-hots each cell

        self.tbptt_length = "max"

        self.name_env = "MiniGrid-MemoryS17Random-v0" #"MiniGrid-MemoryS11-v0" # "MiniGrid-DoorKey-8x8-v0"
        self.env_size = int(re.search(r"(\d+)", self.name_env).group(1))
        self.force_cue_visible = False

        self.n_workers = 32  # W: games played in parallel

        # AsyncVectorEnv (separate processes) vs SyncVectorEnv. False is
        # required in notebooks and scripts without an __main__ guard.
        self.async_envs = True

        # T: build_env feeds this back as the env's max_steps, so MiniGrid's
        # truncation and success reward (1 - 0.9*step_count/max_steps) move
        # with it. Derived from env_max_steps so it tracks name_env.
        self.worker_steps = self.env_max_steps // 4
        self.n_total_steps = self.n_workers * self.worker_steps  # W * T

        # Declared, not chosen: every name below is drawn per trial and written
        # by apply_params, which refuses names the config lacks -- hence the
        # placeholders. None, not a default, so a path that forgets to apply
        # params fails loudly instead of training something it didn't report.
        # Hand-picked runs get real numbers from ConfigNoHPO, only there.
        self.lr = None
        self.wd = None
        self.gamma = None
        self.gae_lambda = None
        self.clip_eps = None
        self.value_coef = None
        self.entropy_coef = None
        self.max_grad_norm = None
        self.n_epochs = None

        # not searched; PPOAgent reads `is not None` as "early-stop on KL".
        self.target_kl = None

        # decay lr linearly to 0 across n_iterations; falsy = constant lr.
        # Not searched either. With it on, `lr` is the INITIAL rate.
        self.lr_anneal = None

        # candidates, largest first: run_with_batch_size_fallback uses the largest that fits
        self.mini_batch_size = [1024,512,256,128, 64, 32, 16, 8, 4]

        # the budget every encoder is compared at: change it for all or none.
        # Raising it is safe -- the lr schedule spans n_iterations, so it just
        # decays more slowly.
        self.n_iterations = 1000
        self.n_iterations_report = 10

        # master switch for every clock (Timing in helper.py). Off: each
        # `timing.phase(...)` becomes a bare yield, no cuda.synchronize() is
        # issued to measure, no TIME table printed. "took ..." stays either way.
        self.calculate_time = True

        self.n_eval_episodes = 50
        self.eval_seed = 10_000
        # False = sample actions, True = argmax. Sampling is safe: entropy_coef
        # plus the lr decay drive entropy to ~0 by the time the task is solved,
        # so a sampled and an argmax episode take the same actions.
        self.eval_deterministic = False

        self.n_trials = 30
        self.seed_hpo = 42
        self.hpo_direction = "maximize"

        # <metric>_<center>_<spread>, e.g. "return_mean_minus-std". metric is
        # "return" or "success-rate"; center is "mean"/"median" across seeds;
        # spread is "minus-std"/"minus-iqr"/"None", weighted by hpo_lambda.
        self.hpo_objective = "return_mean_minus-std"

        # score = center - lambda * spread, both across seeds.
        self.hpo_lambda = 1.0

        # The PPO half, shared by all four encoders so tuned values stay
        # comparable. Each subclass appends its architecture knobs in
        # _configure_model(); ConfigNoHPO replaces it with []. Every "name"
        # must already be an attribute above -- apply_params raises otherwise.
        self.search_space = [
            {"type": "float", "name": "lr", "low": 1e-5, "high": 1e-2, "log": True},
            {
                "type": "float",
                "name": "gamma",
                "low": 0.99,
                "high": 0.9999,
                "step": 0.001,
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
            {"type": "int", "name": "n_epochs", "low": 1, "high": 8},
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
        self.dir_model = os.path.join(
            "agents", f"pretrained_model_{self.feature_extractor.upper()}"
        )
        self.dir_pretrained_model = self.dir_model
        self.dir_hpo = os.path.join(self.dir_model, "hpo")

    def _configure_model(self):
        """Overridden by each subclass to set feature_extractor and its
        architecture knobs. Config itself is abstract."""
        raise NotImplementedError(
            "Config is abstract -- build a ConfigMLP / ConfigLSTM / ConfigGRU / "
            "ConfigTransformer, or call make_config('MLP')."
        )

    @property
    def name_hpo(self):
        env = self.name_env.replace("/", "-")
        return f"{self.feature_extractor.upper()}_{env}"

    @property
    def name_study(self):
        return f"ppo_{self.name_hpo}"

    @property
    def path_hpo_csv(self):
        return os.path.join(self.dir_hpo, f"hpo_csv_{self.name_hpo}.csv")

    @property
    def path_hpo_db(self):
        path = os.path.join(self.dir_hpo, f"hpo_db_{self.name_hpo}.db")
        return "sqlite:///" + path

    @property
    def path_hpo_sampler(self):
        return os.path.join(self.dir_hpo, f"hpo_sampler_{self.name_hpo}.pkl")

    def set_seed(self, env=None, seed=None):
        """Seed python/numpy/torch and env. Defaults to seed_list[0]."""
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
