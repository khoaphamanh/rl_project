import random
import os
import numpy as np
import torch

from config.helper import Helper


class Config(Helper):
    """Hyperparameters shared by every encoder. NOT built directly.

    The four architecture configs subclass this and fill in _configure_model()
    with the one encoder they are for:

        config_mlp.py          ConfigMLP          hidden_size, n_layers_mlp
        config_lstm.py         ConfigLSTM         hidden_size
        config_gru.py          ConfigGRU          hidden_size
        config_transformer.py  ConfigTransformer  hidden_size, d_model, n_heads,
                                                  n_layers, d_ff, p_drop, ...

    Everything that stays the SAME whichever encoder runs -- the env, the
    rollout size, the PPO knobs, evaluation -- lives here, so that an ablation
    changes exactly one thing between runs: the encoder. That is the whole
    point of the split; a hyperparameter that belongs here but got copied into
    four files would be four chances for the runs to stop being comparable.

    Build one with make_config("MLP"), never Config() -- see config/__init__.py.
    Config on its own is abstract: _configure_model() raises until a subclass
    overrides it.

    Inheriting Helper means config.build_extractor(), config.zero_hidden() and
    config.is_recurrent read these attributes directly.
    """

    # Set by each subclass in _configure_model(). Named here so is_recurrent
    # and build_model_name have an attribute to read, and so a stray Config()
    # fails in the hook below rather than later on a None.upper().
    recurrent_model = None

    def __init__(self):

        # one training run per seed. The unit of a result is this whole list,
        # not any single run: main.py trains all of them and reports the mean.
        # A one-element list, [42], is the quick debug run.
        self.base_seed_list = [42, 0, 92]

        # the seed actually in use -- set_seed() overwrites it per run, and
        # build_model_name spells it into the checkpoint. Until a run starts it
        # is the first of the list, so a config that never reaches an agent
        # (watch.py, a bare load_model) still names a file.
        self.seed = self.base_seed_list[0]

        # fix hyperparameters
        self.input_size = 7 * 7 * 20  # 980, NOT 147: flatten_obs one-hots the
        #   observation's three CATEGORY channels (11 objects + 6 colours +
        #   3 door states = 20 numbers per cell). Handing the raw indices to
        #   the encoder instead makes key(5) vs ball(6) a difference of 1.0 in
        #   one of 147 inputs, along the same axis as empty(1) vs wall(2) --
        #   and then no encoder can read the cue, so all three score alike.
        #   See feature_extractor.flatten_obs for the measurements.

        # model hyperparameters shared by every encoder. recurrent_model,
        # hidden_size and the per-architecture knobs (n_layers_mlp, d_model,
        # n_heads, ...) are NOT here -- each subclass sets its own in
        # _configure_model(). hidden_size in particular is per-encoder on
        # purpose: it is the one width every encoder has, so owning it is what
        # lets an LSTM be widened without dragging the MLP along.
        self.lr = 1e-3
        self.wd = 0.0  # Adam weight_decay, read in PPOAgent.__init__

        self.tbptt_length = "max"

        # env
        self.name_env = "MiniGrid-DoorKey-8x8-v0"  # "MiniGrid-DoorKey-5x5-v0"  # "MiniGrid-MemoryS11-v0""MiniGrid-DoorKey-8x8-v0"
        self.force_cue_visible = False  # wrap the env in StartInCueView, which
        #   spawns the agent at (1, height//2) instead of a random x along the
        #   hallway. MiniGrid's own MemoryEnv shows the cue from x = 1 ONLY, so
        #   without this it is unobservable in 7 of 8 episodes -- and an
        #   unobservable cue makes GRU, LSTM and MLP the same agent. This is
        #   what makes the ablation an ablation. See helper.StartInCueView.

        # sampling: W games played in parallel, T steps each per iteration
        self.n_workers = 16  # W

        # T = the env's OWN time limit, 5 * size^2 (245 on S7, 605 on S11),
        # read off the env instead of typed in. NOT a free choice.
        #
        # WHY T = 256 SOLVED S7 AND NOT S11, on identical everything else. The
        # limit is what differs: 245 on S7, 605 on S11. An unsolved episode runs
        # to that limit, so one 8 x 256 = 2048-step rollout contains
        #
        #     S7    2048 / 245  ~  8 complete episodes  -> 8 reward signals
        #     S11   2048 / 605  ~  3 complete episodes  -> 3 reward signals
        #
        # per update. On a task that pays nothing until the right object is
        # touched, three outcomes is not enough to estimate an advantage from,
        # and S7 got nearly three times as many for the same compute. T = 245
        # happened to sit just under 256; that coincidence is the whole reason
        # one size worked and the other did not.
        #
        # Setting T = max_steps makes that structural rather than lucky: the
        # longest episode the env can produce still fits inside one rollout at
        # every size, so each worker contributes at least one complete episode
        # per update and split_pad_mask only ever cuts at a real done -- where
        # truncating BPTT is correct, because the hidden state was zeroed there
        # anyway. The cost is rollout size, which now grows with the maze:
        # 8 * 605 = 4840 steps per iteration on S11 against 1960 on S7.
        self.worker_steps = self.env_max_steps  # T

        # total env steps collected per iteration. NOT called batch_size on
        # purpose: mini_batch_size below is handed to a torch DataLoader, where
        # batch_size counts SEQUENCES, and two sibling attributes counting
        # different things under the same word is a trap. This one counts steps.
        self.n_total_steps = self.n_workers * self.worker_steps  # W * T

        # advantage estimation
        self.gamma = 0.99  # discount: how far ahead a reward still counts
        self.gae_lambda = 0.95  # 0 = TD(0), low variance / high bias
        #                         1 = Monte Carlo, high variance / no bias

        # the update
        self.clip_eps = 0.2  # how far pi_new may drift from pi_old before
        #                      the gradient is cut off. The paper's value.
        self.value_coef = 0.1  # weight of the critic's regression term. Damped,
        #                        or a squared error on returns would dominate
        #                        the shared encoder and the policy learns nothing.
        self.entropy_coef = 0.03  # weight of the exploration bonus. Small on
        #                           purpose: a brake against premature collapse,
        #                           not a goal.
        self.max_grad_norm = 0.5  # rescale the gradient if it is longer than this

        # how hard each rollout is reused
        self.n_epochs = 3  # passes over the SAME batch. The whole reason the
        #                    clip exists: after pass 1 the data is off-policy.
        self.mini_batch_size = 4  # DataLoader batch size, counted in SEQUENCES
        #                           (not timesteps): one sample is one whole
        #                           padded episode fragment of up to L steps.
        self.target_kl = None  # e.g. 0.015 to abandon the remaining epochs once
        #                        pi_new has drifted too far. None = off.

        # training length
        self.n_iterations = 500  # sample -> update, this many times
        self.n_iterations_report = 100  # every this many iterations, run a full
        #                                evaluate() and print a line. NOT every
        #                                iteration: 50 unsolved episodes cost up
        #                                to 50 * max_steps env steps, which is
        #                                more than the n_total_steps one
        #                                iteration collects, and the policy
        #                                barely moves in a single update anyway.

        # evaluation: no gradients, no training, COMPLETE episodes only
        self.n_eval_episodes = 50  # played to their own end, never truncated
        self.eval_seed = 10_000  # fixed and far from every training seed, so the same
        #                          50 mazes are replayed at every evaluation.
        #                          That removes the maze draw as a source of
        #                          noise -- but NOT the action sampling, which
        #                          still varies unless deterministic=True.
        self.eval_deterministic = False  # False = sample from pi, the same way
        #                                  training does, so the curve is
        #                                  comparable to train()'s.
        #                                  True = argmax: the cleaner final
        #                                  number, but it deadlocks on a
        #                                  half-trained policy (one action
        #                                  repeated until the time limit).
        #                                  evaluate(deterministic=True)
        #                                  overrides this for one call.

        # ----- HPO ---------------------------------------------------------
        # one trial = one hyperparameter draw, trained once per base_seed_list
        # entry, so a trial costs len(base_seed_list) full runs.
        self.n_trials = 10  # testing only; a real search wants 25-40 per encoder

        # the search space. Every "name" MUST be an attribute set above, and it
        # must be set on the config BEFORE PPOAgent is built -- the agent copies
        # each one into its own attribute in __init__ and never looks back.
        #
        # The keys are optuna's own: suggest_float(name, low, high, step=, log=),
        # suggest_int(name, low, high, step=, log=), suggest_categorical(name,
        # choices). "type" is the only key that is not optuna's -- it picks
        # which suggest_* to call, and defaults to "float".
        #
        # step ONLY on log=False entries: suggest_float raises if given both.
        # Each range is a whole number of steps wide, or optuna lowers "high"
        # to the nearest one and warns.
        #
        # The four encoder configs APPEND their own architecture knobs to this
        # list in _configure_model(), which runs below.
        self.search_space = [
            {"name": "lr", "low": 1e-5, "high": 1e-2, "log": True},
            {"name": "entropy_coef", "low": 1e-4, "high": 1e-1, "log": True},
            {"name": "wd", "low": 0.00001, "high": 0.5, "log": True},
            {"name": "value_coef", "low": 0.1, "high": 1.0, "log": True},
            {"name": "gamma", "low": 0.95, "high": 0.9995, "step": 0.0001, "log": False},
            {
                "name": "gae_lambda",
                "low": 0.90,
                "high": 0.99,
                "step": 0.005,
                "log": False,
            },
            {"name": "clip_eps", "low": 0.1, "high": 0.3, "step": 0.01, "log": False},
            {
                "name": "n_epochs",
                "type": "int",
                "low": 2,
                "high": 10,
                "step": 1,
                "log": False,
            },
        ]

        # where a trained model is written
        self.dir_pretrained_model = os.path.join(
            "agents", "pretrained_model_feature_extractor"
        )

        # ----- the encoder, filled by the subclass -----------------------
        # ConfigMLP / ConfigLSTM / ConfigGRU / ConfigTransformer override this
        # to set recurrent_model and whatever architecture knobs their encoder
        # needs. Called HERE, near the end, for two reasons that decide the
        # ordering:
        #   - AFTER worker_steps exists, because the transformer's
        #     max_seq_length is derived from it.
        self._configure_model()

    # PROPERTIES, not attributes: self.seed is not the run's real seed until
    # set_seed() runs, which happens after __init__. Read-only -- to move the
    # file, set dir_pretrained_model (or seed) and these follow.
    @property
    def name_model(self):
        return self.build_model_name()

    @property
    def path_model(self):
        return os.path.join(self.dir_pretrained_model, self.name_model)

    def _configure_model(self):
        """Set recurrent_model and this encoder's architecture knobs.

        Overridden by ConfigMLP / ConfigLSTM / ConfigGRU / ConfigTransformer.
        Config itself is abstract, so building one directly stops here with a
        clear message instead of failing later inside build_extractor or on a
        None.upper() in build_model_name.
        """
        raise NotImplementedError(
            "Config is abstract -- build a ConfigMLP / ConfigLSTM / ConfigGRU / "
            "ConfigTransformer, or call make_config('MLP'). Each one fills in "
            "recurrent_model and its own encoder hyperparameters in "
            "_configure_model()."
        )

    def set_seed(self, env=None, seed=None):
        """Seed every source of randomness and return the seed that was used.

        seed=None falls back to base_seed_list[0], so set_seed() with no
        argument is always reproducible. Normally the seed IS passed --
        PPOAgent(config, seed=s) hands over one seed of base_seed_list.

        The CUDA calls are no-ops on a machine without a GPU. They stay in so
        the exact same config also reproduces on a machine that has one.

        env is optional. In Gymnasium the env RNG is seeded through
        reset(seed=...), so this consumes one reset and throws its
        observation away. Call set_seed first, then reset() again without a
        seed for the real first observation -- that keeps the stream going
        from the seeded state instead of restarting it.
        """
        if seed is None:
            seed = self.base_seed_list[0]
        self.seed = seed

        # affects subprocesses only (the parallel workers), not this process
        os.environ["PYTHONHASHSEED"] = str(seed)

        random.seed(seed)  # python's own random
        np.random.seed(seed)  # numpy, used by gymnasium internally

        torch.manual_seed(seed)  # cpu tensors, weight init, dropout
        torch.cuda.manual_seed(seed)  # one gpu
        torch.cuda.manual_seed_all(seed)  # all gpus

        if torch.cuda.is_available():
            # without these two, cudnn picks algorithms by benchmarking and
            # the same seed can still give different numbers
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if env is not None:
            env.reset(seed=seed)  # seeds the env RNG (layout, cue position)
            env.action_space.seed(seed)  # seeds env.action_space.sample()
            env.observation_space.seed(seed)

        return seed
