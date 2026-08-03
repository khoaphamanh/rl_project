from config.config import Config


class ConfigTransformer(Config):
    """Transformer encoder. Sets the attention-stack knobs (d_model, n_heads,
    n_layers_transformer, d_ff_mult, p_drop); everything else is shared in
    Config. Note: acting samples one step at a time, so without a rollout
    cache of past observations it behaves like an MLP while acting."""

    @property
    def d_ff(self):
        """Feed-forward width inside each block, derived from d_model so it
        tracks whatever d_model a trial draws."""
        return self.d_ff_mult * self.d_model

    def _configure_model(self):
        self.feature_extractor = "TRANSFORMER"

        self.hidden_size = 64  # what fc_out projects down to
        self.d_model = 64  # width the attention stack runs at
        self.n_heads = 4  # must divide d_model
        self.n_layers_transformer = 1  # cheap first test; 2+ to actually compose layers
        self.d_ff_mult = 4  # d_ff = d_ff_mult * d_model, see the property above

        # p_drop must stay 0.0: PPO compares log_probs from the rollout
        # against log_probs recomputed during the update, and dropout (with
        # no model.eval() anywhere in this project) would make those differ
        # on the same input for reasons unrelated to learning.
        self.p_drop = 0.0

        # step=8 on d_model and {2,4,8} on n_heads keeps every combination
        # divisible, since MultiHeadAttention asserts d_model % n_heads == 0
        # and a violation crashes the trial instead of scoring it badly.
        # d_ff is tuned via d_ff_mult, not directly, so it always tracks
        # d_model through the property above.
        self.search_space += [
            {"type": "int", "name": "d_model", "low": 32, "high": 256, "step": 8},
            {"type": "categorical", "name": "n_heads", "choices": [2, 4, 8]},
            {"type": "int", "name": "n_layers_transformer", "low": 1, "high": 3},
            {"type": "categorical", "name": "d_ff_mult", "choices": [2, 4]},
            {"type": "int", "name": "hidden_size", "low": 32, "high": 512, "step": 8},
        ]

        self.max_seq_length = self.worker_steps
