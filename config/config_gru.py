"""The GRU arm of the ablation: the recurrent encoder that can beat the state
aliasing the MLP cannot. Everything but the architecture comes from Config; the
backward reach comes from the --tbptt flag, not from here."""

from config.config import Config


class ConfigGRU(Config):
    """GRU encoder, the recurrent arm carrying only h. Only architecture
    knob is hidden_size, set in _configure_model()."""

    def _configure_model(self):
        """Names the encoder, sets its one architecture knob (hidden_size) and
        appends it to the shared search_space. Called by Config.__init__; sets
        attributes, returns nothing."""
        self.feature_extractor = "GRU"

        self.hidden_size = 64  # width of h

        # same low/high/step as the MLP, so widths stay comparable
        self.search_space += [
            {"type": "int", "name": "hidden_size", "low": 32, "high": 512, "step": 8},
        ]

        # tbptt_length is deliberately NOT here: it is fixed per run (--tbptt L)
        # and each length is its own study, compared between studies not inside.
