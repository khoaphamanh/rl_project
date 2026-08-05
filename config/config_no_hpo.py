"""The hand-picked run: no optuna, every hyperparameter for every encoder
spelled out by hand in _configure_model(). Used for smoke tests, reproducing
pre-HPO numbers, and as the baseline main_no_hpo.py compares a tuned run
against. Writes to no_hpo/, parallel to the search's hpo/."""

import os

from config.config import Config

FEATURE_EXTRACTOR = "MLP"  # default encoder; main_no_hpo.py can override via CLI


class ConfigNoHPO(Config):
    """Every hyperparameter, hand-picked, for whichever encoder is chosen.
    ConfigNoHPO("GRU") overrides FEATURE_EXTRACTOR for one build; with no
    argument it uses the module constant above."""

    def __init__(self, feature_extractor=None):
        # stashed before super().__init__(), which calls _configure_model()
        self._chosen_model = (feature_extractor or FEATURE_EXTRACTOR).upper()
        super().__init__()

        # no_hpo/ sits parallel to hpo/, not inside it: this is not a trial
        self.dir_pretrained_model = os.path.join(self.dir_model, "no_hpo")

    @property
    def d_ff(self):
        return self.d_ff_mult * self.d_model

    def _configure_model(self):
        # 1. which encoder
        self.feature_extractor = self._chosen_model

        # 2. the run
        # self.seed_list = [0]
        # self.n_iterations = 500
        # self.n_iterations_report = 100

        # 5. PPO -- the pre-HPO values
        self.lr = 1e-3
        self.wd = 0.0

        self.gamma = 0.99
        self.gae_lambda = 0.95

        self.clip_eps = 0.2
        self.value_coef = 0.1
        self.entropy_coef = 0.03
        self.max_grad_norm = 0.5

        self.n_epochs = 3
        self.target_kl = None

        # 6. evaluation
        self.n_eval_episodes = 50
        self.eval_seed = 10_000
        self.eval_deterministic = False

        # 7. the encoders -- all four listed, one of them used. hidden_size
        # is shared here because the ablation ran with equal WIDTH across
        # encoders, not equal parameter count.
        self.hidden_size = 64

        self.n_layers_mlp = 3  # MLP only

        # LSTM/GRU: nothing else, build_extractor builds a single layer

        self.d_model = 64  # transformer only
        self.n_heads = 4  # must divide d_model
        self.n_layers_transformer = 1
        self.d_ff_mult = 4
        self.p_drop = 0.0  # must stay 0: breaks PPO's log-prob ratio otherwise
        self.max_seq_length = self.worker_steps

        # 8. not tuned here
        self.search_space = []
