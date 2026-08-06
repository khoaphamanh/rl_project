from config.config import Config


class ConfigTransformer(Config):
    """Transformer encoder: sets the attention-stack knobs, everything else is
    shared in Config. Note it acts one step at a time, so with no rollout cache
    of past observations it behaves like an MLP while acting."""

    @property
    def d_ff(self):
        """Feed-forward width per block, tracking whatever d_model is drawn."""
        return self.d_ff_mult * self.d_model

    def _configure_model(self):
        self.feature_extractor = "TRANSFORMER"

        self.hidden_size = 64  # what fc_out projects down to
        self.d_model = 64  # width the attention stack runs at
        self.n_heads = 4  # must divide d_model
        self.n_layers_transformer = 1  # 2+ to actually compose layers
        self.d_ff_mult = 4

        # must stay 0.0: PPO compares rollout log_probs against ones recomputed
        # during the update, and dropout makes those differ on the same input.
        self.p_drop = 0.0

        # step=8 on d_model with n_heads in {2,4,8} keeps every combination
        # divisible -- MultiHeadAttention asserts d_model % n_heads == 0, and a
        # violation crashes the trial instead of scoring it badly.
        self.search_space += [
            {"type": "int", "name": "d_model", "low": 32, "high": 256, "step": 8},
            {"type": "categorical", "name": "n_heads", "choices": [2, 4, 8]},
            {"type": "int", "name": "n_layers_transformer", "low": 1, "high": 3},
            {"type": "categorical", "name": "d_ff_mult", "choices": [2, 4]},
            {"type": "int", "name": "hidden_size", "low": 32, "high": 512, "step": 8},
        ]

        self.max_seq_length = self.worker_steps
