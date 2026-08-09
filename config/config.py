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

    def __init__(self, hpo_tag=None):
        self.seed_list = [0, 15, 12, 97, 98]

        self.input_size = 7 * 7 * 20  # 980 after flatten_obs one-hots each cell

        # Which of the two GRU/LSTM studies this is. "max" keeps tbptt_length
        # fixed below; "tbptt" puts it in the search space (config_gru.py).
        # Narrowed to "" for MLP/Transformer once _configure_model names the
        # encoder -- they share one untagged directory, as before.
        self.hpo_tag = "max" if hpo_tag is None else str(hpo_tag).lower()

        # how many steps the gradient may flow back through; "max" = cut on
        # episode boundaries only. Read by PPOAgent.split_pad_mask.
        self.tbptt_length = "max"

        # only bites in a "tbptt" run: score loses tbptt_lambda * (L / T), so
        # among lengths that score the same the shortest wins. See trial_score.
        self.tbptt_lambda = 0.1

        self.name_env = "MiniGrid-MemoryS17Random-v0"  # "MiniGrid-MemoryS11-v0" # "MiniGrid-DoorKey-8x8-v0"
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

        # not searched; PPOAgent reads `is not None` as "early-stop on KL".
        self.target_kl = None

        # decay lr linearly to 0 across n_iterations; falsy = constant lr.
        # Not searched either. With it on, `lr` is the INITIAL rate.
        self.lr_anneal = None

        # candidates, largest first: run_with_batch_size_fallback uses the largest that fits
        self.mini_batch_size = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4]

        # the budget every encoder is compared at: change it for all or none.
        # Raising n_iterations is safe -- the lr schedule spans it, so it just
        # decays more slowly. Deliberately NOT searched: how much compute a
        # trial gets to spend must not be a tuned hyperparameter, or a study
        # could "win" by buying more passes over the same rollout instead of by
        # having the better encoder.
        self.n_epochs = 3
        self.n_iterations = 1000
        self.n_iterations_report = 10

        # master switch for every clock (Timing in helper.py). Off: each
        # `timing.phase(...)` becomes a bare yield, no cuda.synchronize() is
        # issued to measure, no TIME table printed. "took ..." stays either way.
        self.calculate_time = False

        self.n_eval_episodes = 50
        self.eval_seed = 10_000
        # False = sample actions, True = argmax. Sampling is safe: entropy_coef
        # plus the lr decay drive entropy to ~0 by the time the task is solved,
        # so a sampled and an argmax episode take the same actions.
        self.eval_deterministic = False

        self.n_trials = 100
        self.seed_hpo = 42
        self.hpo_direction = "maximize"

        # MedianPruner, in the units HPOPPO's step uses -- one step is one
        # training ITERATION, offset per seed (step = seed_index * n_iterations
        # + iteration), so all four numbers below read in iterations/trials.
        #
        # warmup is the one that must not be 0. Optuna judges a trial on its
        # best intermediate value so far against the median of completed trials
        # at the same step; at iteration 0 that is one untrained network vs the
        # median of other untrained networks, i.e. a coin flip, so a warmup of 0
        # would kill roughly half of every trial after its first evaluation.
        # 200 = a fifth of the budget, enough for the curves to separate while
        # still cutting a hopeless draw off before it burns four more seeds.
        self.hpo_pruner_warmup = 200

        # how often to actually check, offset by the warmup. Tied to the eval
        # cadence because a check is only possible where a value was reported:
        # optuna postpones a check that lands on a step with no value, so any
        # interval from 1 to n_iterations_report means the same thing -- "check
        # at every evaluation". Writing it as n_iterations_report says that on
        # purpose instead of by accident, and follows if the cadence changes.
        # Cost is not a reason to raise it: should_prune() measures ~4 ms
        # against a 50-trial sqlite study, i.e. ~2 s over a whole trial.
        self.hpo_pruner_interval = self.n_iterations_report

        # no pruning at all until this many trials have COMPLETED, and no
        # pruning at a given step unless this many completed trials reported a
        # value there -- a median over one or two trials is not a median.
        #
        # Both counts are per GROUP, and in a tbptt study the group is the drawn
        # tbptt_length (GroupedMedianPruner in hpo_ppo.py). So with 6 lengths
        # and n_trials=100 -- ~17 trials per length -- pruning only starts
        # biting a length once 5 trials of that length have finished, roughly a
        # third of the way in. That lateness is the price of not letting the
        # pruner answer the question the tbptt study is asking.
        self.hpo_pruner_startup_trials = 5
        self.hpo_pruner_min_trials = 5

        # Exactly two values are accepted -- "return_mean_minus-std" or
        # "success_rate_mean_minus-std". Both score mean - hpo_lambda*std
        # across seeds; they differ only in the metric aggregated.
        self.hpo_objective = "return_mean_minus-std"

        # score = mean - lambda * std, both across seeds.
        self.hpo_lambda = 1.0

        # The PPO half, shared by all four encoders so tuned values stay
        # comparable. Each subclass appends its architecture knobs in
        # _configure_model(); ConfigNoHPO replaces it with []. Every "name"
        # must already be an attribute above -- apply_params raises otherwise.
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

        # only the recurrent encoders split into a max/tbptt pair: an MLP has
        # no time dependence to truncate, and the Transformer is out of scope
        # for this comparison. Both keep pretrained_model_<ENCODER>/.
        if self.feature_extractor not in ("GRU", "LSTM"):
            self.hpo_tag = ""

        suffix = f"_{self.hpo_tag}" if self.hpo_tag else ""
        self.dir_model = os.path.join(
            "agents", f"pretrained_model_{self.feature_extractor.upper()}{suffix}"
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
        tag = f"_{self.hpo_tag}" if self.hpo_tag else ""
        return f"{self.feature_extractor.upper()}_{env}{tag}"

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
