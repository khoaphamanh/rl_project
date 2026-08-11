from config.config import Config


class ConfigGRU(Config):
    """GRU encoder, the recurrent arm carrying only h. Only architecture
    knob is hidden_size, set in _configure_model()."""

    def _configure_model(self):
        self.feature_extractor = "GRU"

        self.hidden_size = 64  # width of h

        # same low/high/step as the MLP, so widths stay comparable
        self.search_space += [
            {"type": "int", "name": "hidden_size", "low": 32, "high": 512, "step": 8},
        ]

        # tbptt_length is deliberately NOT here. It is fixed per run (--tbptt L)
        # and each length is its own study writing to its own directory, so no
        # sampler or pruner ever ranks one length against another -- comparing
        # them is the experiment, and it happens between studies, not inside one.
