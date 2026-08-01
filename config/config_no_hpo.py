"""
The hand-picked run. NO SEARCH -- you choose every number in this file.

This is config.py and the four encoder configs collapsed into one place, set
back to the values that were chosen BY HAND before any of it was handed to
optuna. It exists because a search is not always what you want:

    - a smoke test, where 30 trials x 3 seeds is absurd
    - reproducing the pre-HPO numbers that are already in the logs
    - trying one idea, on purpose, without a sampler having opinions
    - a baseline for the search to be measured against -- "the tuned GRU beat
      the hand-picked GRU by X" is a sentence the search cannot produce alone

WHAT YOU EDIT. Two things:

    1. FEATURE_EXTRACTOR below -- ONE of MLP / LSTM / GRU / TRANSFORMER
    2. whichever hyperparameters you want to change, in _configure_model()

Every encoder's knobs are listed, not only the chosen one's, and that is
deliberate: setting d_model while FEATURE_EXTRACTOR is "MLP" is harmless
(build_extractor never reads it), and having all four visible side by side is
what makes it obvious which quantities are being held equal across the
ablation. The unused ones cost nothing.

WHAT IS DIFFERENT FROM config.py. Only three things:

    - search_space is emptied. Nothing here is tuned, so leaving it populated
      would be a lie about what this config is for -- and anything that
      iterates it would act on a search that is not happening.
    - dir_pretrained_model points at pretrained_model_<ENC>/no_hpo/, PARALLEL
      to hpo/ rather than inside it, so a hand-picked run and the study's
      final model never land on the same filename. They would otherwise:
      same encoder, same env, same seeds, same name.
    - the values are the pre-HPO ones, spelled out rather than inherited.

Run it with main_no_hpo.py. Everything else -- the agent, the env, the
checkpoint format, watch.py -- is unchanged and does not know the difference.
"""

import os

from config.config import Config


# THE ONE LINE YOU EDIT to switch encoder: MLP / LSTM / GRU / TRANSFORMER.
# main_no_hpo.py can override it from the command line; this stays the default
# so the file alone is enough to say what a bare run does.
FEATURE_EXTRACTOR = "MLP"


class ConfigNoHPO(Config):
    """Every hyperparameter, hand-picked, for whichever encoder is chosen.

    ConfigNoHPO("GRU") overrides FEATURE_EXTRACTOR for one build. Built with no
    argument it uses the module constant above.

    Subclasses Config DIRECTLY rather than ConfigGRU / ConfigMLP / ... because
    it has to work for all four, and each of those would drag in its own
    _configure_model with its own search_space.
    """

    def __init__(self, feature_extractor=None):
        # stashed BEFORE super().__init__(), which is what calls
        # _configure_model() partway through its own body
        self._chosen_model = (feature_extractor or FEATURE_EXTRACTOR).upper()
        super().__init__()

        # AFTER super().__init__(), because that is where dir_model is derived
        # from feature_extractor and dir_pretrained_model is first set.
        # no_hpo/ sits PARALLEL to hpo/, not inside it: this is not a trial.
        self.dir_pretrained_model = os.path.join(self.dir_model, "no_hpo")

    @property
    def d_ff(self):
        """The transformer's feed-forward width, derived so d_model is enough.

        Repeated from ConfigTransformer because this class subclasses Config
        directly -- it has to work for all four encoders, and only one of them
        has a d_model at all. Harmless for the other three: nothing reads it.
        """
        return self.d_ff_mult * self.d_model

    def _configure_model(self):
        # ==============================================================
        # 1. WHICH ENCODER
        # ==============================================================
        self.feature_extractor = self._chosen_model

        # ==============================================================
        # 2. THE RUN
        # ==============================================================
        # one training run per seed. Trim to [0] for a quick debug run --
        # main_no_hpo.py trains every entry and reports the spread over them.
        self.seed_list = [0, 26, 98]

        self.n_iterations = 500  # sample -> update, this many times
        self.n_iterations_report = 100  # evaluate() and print every this many

        # ==============================================================
        # 3. THE ENV
        # ==============================================================
        self.name_env = "MiniGrid-DoorKey-8x8-v0"
        self.force_cue_visible = False  # True on a Memory env, or the cue is
        #   unobservable in 7 of 8 episodes and all four encoders score alike

        # ==============================================================
        # 4. SAMPLING
        # ==============================================================
        self.n_workers = 16  # W, games played in parallel
        self.worker_steps = self.env_max_steps  # T, the env's own time limit
        # recomputed HERE and not inherited: Config.__init__ works it out from
        # n_workers and worker_steps BEFORE this hook runs, so changing either
        # above without this line would leave n_total_steps describing the old
        # pair
        self.n_total_steps = self.n_workers * self.worker_steps  # W * T

        # ==============================================================
        # 5. PPO -- the pre-HPO values
        # ==============================================================
        self.lr = 1e-3
        self.wd = 0.0  # Adam weight_decay. 0.0 = plain Adam

        self.gamma = 0.99  # discount
        self.gae_lambda = 0.95  # GAE bias/variance knob

        self.clip_eps = 0.2  # the paper's value
        self.value_coef = 0.1  # damped, or the critic dominates the encoder
        self.entropy_coef = 0.03  # a brake against collapse, not a goal
        self.max_grad_norm = 0.5

        self.n_epochs = 3  # passes over the SAME rollout
        self.mini_batch_size = [128, 64, 32, 16, 8, 4]  # candidates, largest
        #   first -- the update runs at the biggest one that fits in memory
        self.target_kl = None  # e.g. 0.015 to stop early. None = off.

        # ==============================================================
        # 6. EVALUATION
        # ==============================================================
        self.n_eval_episodes = 50
        self.eval_seed = 10_000
        self.eval_deterministic = False

        # ==============================================================
        # 7. THE ENCODERS -- all four listed, one of them used
        #
        # hidden_size is shared here because these are the values the ablation
        # was run at and they were equal across all four: equal WIDTH, NOT
        # equal parameter count (an MLP layer costs h^2 weights, a GRU 3h^2,
        # an LSTM 4h^2). Give them different widths to match parameters
        # instead -- just say which of the two you controlled in the writeup.
        # ==============================================================
        self.hidden_size = 64  # every encoder: the width fc_actor / fc_critic
        #                        are built for

        # ---- MLP ----------------------------------------------------
        self.n_layers_mlp = 3  # hidden Linear+ReLU blocks

        # ---- LSTM / GRU ---------------------------------------------
        # nothing else: build_extractor builds LSTM(input_size, hidden_size)
        # and GRU(input_size, hidden_size), one layer each.

        # ---- TRANSFORMER --------------------------------------------
        self.d_model = 64  # the width the attention stack runs at
        self.n_heads = 4  # MUST DIVIDE d_model. 64 / 4 = 16 per head.
        self.n_layers_transformer = 1  # 2 is the smallest stack that can
        #   really compose "find the cue" with "relate it to here"; 1 is the
        #   cheapest first test
        self.d_ff_mult = 4  # d_ff = d_ff_mult * d_model, see the property
        self.p_drop = 0.0  # NOT free: dropout makes the same observation give
        #   two answers, so PPO's ratio pi_new/pi_old is noise. Leave at 0.
        self.max_seq_length = self.worker_steps  # also not free: the
        #   positional codes and causal mask are built once, this long

        # ==============================================================
        # 8. NOT TUNED HERE
        # ==============================================================
        # emptied on purpose. Nothing in this file is searched over, and a
        # populated search_space would suggest otherwise to anyone reading it
        # -- or, worse, to anything that iterates it.
        self.search_space = []

        # n_trials, seed_hpo, hpo_direction and hpo_objective are inherited
        # from Config and simply unused. They are not deleted: build_logger
        # dumps whatever the config happens to carry, and an attribute that
        # exists and is ignored is safer than one that raises.
