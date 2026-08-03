from config.config import Config


class ConfigGRU(Config):
    """GRU encoder, the recurrent arm carrying only h. Only architecture
    knob is hidden_size, set in _configure_model()."""

    def _configure_model(self):
        self.feature_extractor = "GRU"

        self.hidden_size = 64  # width of h

        # same low/high/step as the other three encoders, so the widths the
        # search settles on are comparable across the ablation
        self.search_space += [
            {"type": "int", "name": "hidden_size", "low": 32, "high": 512, "step": 8},
        ]
