"""The MLP arm of the ablation: the memoryless control the GRU is measured
against. Everything but the architecture comes from Config."""

from config.config import Config


class ConfigMLP(Config):
    """MLP encoder, the no-recurrence arm: sees one observation at a time,
    with no hidden state. Sets hidden_size and n_layers_mlp."""

    def _configure_model(self):
        """Names the encoder, sets its two architecture knobs (hidden_size,
        n_layers_mlp) and appends both to the shared search_space. Called by
        Config.__init__; sets attributes, returns nothing."""
        self.feature_extractor = "MLP"

        # Architecture only -- the PPO half lives in Config, edited there once.
        self.hidden_size = 64  # width of every hidden layer
        self.n_layers_mlp = 3  # hidden Linear+ReLU blocks

        # same low/high/step as the other encoders, so widths stay comparable
        self.search_space += [
            {"type": "int", "name": "hidden_size", "low": 32, "high": 512, "step": 8},
            {"type": "int", "name": "n_layers_mlp", "low": 1, "high": 4},
        ]
