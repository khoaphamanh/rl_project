"""The hand-picked run: no optuna, every hyperparameter spelled out in
_configure_model(). Used for smoke tests and as main_no_hpo.py's baseline.
Writes to no_hpo/, parallel to the search's hpo/."""

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
        # untagged even for GRU/LSTM: max/tbptt splits the SEARCH into two
        # studies, and a hand-picked run is neither. Keeps no_hpo/ where it was.
        super().__init__(hpo_tag="")

        # no_hpo/ sits parallel to hpo/, not inside it: this is not a trial
        self.dir_pretrained_model = os.path.join(self.dir_model, "no_hpo")

    @property
    def d_ff(self):
        return self.d_ff_mult * self.d_model

    def _configure_model(self):
        # 1. which encoder
        self.feature_extractor = self._chosen_model

        # 2. PPO -- hand-picked hyperparameters (tunable params, fixed here)
        self.lr = 1e-3
        self.wd = 0.0
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_eps = 0.2
        self.value_coef = 0.1
        self.entropy_coef = 0.005
        self.max_grad_norm = 0.5
        self.n_epochs = 3
        self.target_kl = 0.02
        # Without annealing this run solves the task then walks back off it
        # (1.00 -> 0.38 -> 0.98): once every episode succeeds the advantage
        # spread collapses ~100x, so normalized advantages are mostly noise
        # and a constant step size keeps acting on it.
        self.lr_anneal = True

        # 3. the encoders -- all four listed, one used. hidden_size is shared:
        # the ablation equalizes WIDTH across encoders, not parameter count.
        self.hidden_size = 64
        self.n_layers_mlp = 3  # MLP only

        # LSTM/GRU: nothing else, build_extractor builds a single layer

        self.d_model = 64  # transformer only
        self.n_heads = 4  # must divide d_model
        self.n_layers_transformer = 1
        self.d_ff_mult = 4
        self.p_drop = 0.0  # must stay 0: breaks PPO's log-prob ratio otherwise
        self.max_seq_length = self.worker_steps

        # 4. no tuning for hand-picked runs
        self.search_space = []
