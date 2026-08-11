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
        # stashed before super().__init__(), which calls _configure_model()
        self._chosen_model = (feature_extractor or FEATURE_EXTRACTOR).upper()
        super().__init__(tbptt_length=tbptt_length)

        # no_hpo/ sits parallel to hpo/, not inside it: this is not a trial.
        # The backward reach is already in dir_model (pretrained_model_GRU_tbptt8/),
        # so a truncated run cannot overwrite the full-BPTT baseline it exists to
        # be compared against even though both write the same filename.
        self.dir_pretrained_model = os.path.join(self.dir_model, "no_hpo")

    def _configure_model(self):
        # 1. which encoder
        self.feature_extractor = self._chosen_model
        self.seed_list = [0]  # five seeds for each encode
        self.n_iterations = 2000
        self.mini_batch_size = [256]
        self.calculate_time = True
        self.force_gpu = True

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

        # tbptt_length is NOT set here -- Config.__init__ already took it from
        # the --tbptt flag, and writing it again would silently override the
        # flag with whatever was hard-coded. "max" = full BPTT, cut on episode
        # boundaries only; an int also cuts every L steps, so the encoder still
        # sees the whole history forward (each chunk is seeded from the hidden
        # state sample() recorded) and only the backward pass shrinks.

        # 3. the encoders -- both listed, one used. hidden_size is shared:
        # the ablation equalizes WIDTH across encoders, not parameter count.
        self.hidden_size = 64
        self.n_layers_mlp = 3  # MLP only
        # GRU: nothing else, build_extractor builds a single layer

        # 4. no tuning for hand-picked runs
        self.search_space = []
