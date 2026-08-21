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

    def __init__(self, feature_extractor=None, tbptt_length=None):
        """Build the hand-picked config for one encoder.

        Args:
            feature_extractor (str | None): "MLP" or "GRU", case-insensitive.
                None falls back to the FEATURE_EXTRACTOR constant above.
            tbptt_length (int | None): the GRU's backward reach in steps;
                None = full BPTT. GRU only, and it names the directory.
        """
        # stashed before super().__init__(), which calls _configure_model()
        self._chosen_model = (feature_extractor or FEATURE_EXTRACTOR).upper()
        super().__init__(tbptt_length=tbptt_length)

        # no_hpo/ sits parallel to hpo/, not inside it: this is not a trial.
        # The backward reach is already in dir_model, so lengths can't collide.
        self.dir_pretrained_model = os.path.join(self.dir_model, "no_hpo")

    def _configure_model(self):
        """Spells out every hyperparameter a study would otherwise search, plus
        the encoder and a single-seed, longer budget. Called by Config.__init__;
        sets attributes, returns nothing."""
        # 1. which encoder
        self.feature_extractor = self._chosen_model
        self.seed_list = [0]  # five seeds for each encode
        self.n_iterations = 2000
        self.mini_batch_size = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4]
        self.calculate_time = True

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
        # Without annealing this run solves the task then walks back off it:
        # once the advantage spread collapses, a constant step acts on noise.
        self.lr_anneal = True

        # tbptt_length is NOT set here -- Config.__init__ already took it from
        # --tbptt, and writing it again would silently override the flag.

        # 3. the encoders -- both listed, one used. hidden_size is shared: the
        # ablation equalizes WIDTH across encoders, not parameter count.
        self.hidden_size = 64
        self.n_layers_mlp = 3  # MLP only
        # GRU: nothing else, build_extractor builds a single layer

        # 4. no tuning for hand-picked runs
        self.search_space = []
