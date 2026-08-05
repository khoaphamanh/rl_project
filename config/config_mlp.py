from config.config import Config


class ConfigMLP(Config):
    """MLP encoder, the no-recurrence arm: sees one observation at a time,
    with no hidden state. Sets hidden_size and n_layers_mlp."""

    def _configure_model(self):
        self.feature_extractor = "MLP"

        # Encoder architecture. The PPO hyperparameters and their half of the
        # search space are shared, in Config -- edit them there, once, so all
        # four encoders keep searching the same ranges.
        self.hidden_size = 64  # width of every hidden layer
        self.n_layers_mlp = 3  # hidden Linear+ReLU blocks

        # Encoder-specific search space (same low/high/step as other encoders
        # so the widths the search settles on are comparable across the ablation)
        self.search_space += [
            {"type": "int", "name": "hidden_size", "low": 32, "high": 512, "step": 8},
            {"type": "int", "name": "n_layers_mlp", "low": 1, "high": 4},
        ]
