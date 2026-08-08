from config.config import Config


class ConfigLSTM(Config):
    """LSTM encoder, the recurrent arm carrying (h, c). Only architecture
    knob is hidden_size, set in _configure_model()."""

    def _configure_model(self):
        self.feature_extractor = "LSTM"

        self.hidden_size = 64  # width of h and c

        # same low/high/step as the other encoders, so widths stay comparable
        self.search_space += [
            {"type": "int", "name": "hidden_size", "low": 32, "high": 512, "step": 8},
        ]

        # same five choices as ConfigGRU, so the two recurrent encoders are
        # searched over the same lengths. Only the tbptt study draws it.
        if self.hpo_tag == "tbptt":
            self.search_space += [
                {
                    "type": "categorical",
                    "name": "tbptt_length",
                    "choices": [8, 16, 32, 64, 128, self.worker_steps],
                },
            ]
