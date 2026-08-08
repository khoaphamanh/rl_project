from config.config import Config


class ConfigGRU(Config):
    """GRU encoder, the recurrent arm carrying only h. Only architecture
    knob is hidden_size, set in _configure_model()."""

    def _configure_model(self):
        self.feature_extractor = "GRU"

        self.hidden_size = 64  # width of h

        # same low/high/step as the other encoders, so widths stay comparable
        self.search_space += [
            {"type": "int", "name": "hidden_size", "low": 32, "high": 512, "step": 8},
        ]

        # only the tbptt study searches it; the max study leaves tbptt_length
        # at "max", so the two differ in exactly one thing. Categorical, not an
        # int range: TPE can resolve five buckets from n_trials, not T of them.
        # The last choice IS "max" (no chunk can exceed T), so the search can
        # rediscover full BPTT and the two studies overlap at one point.
        if self.hpo_tag == "tbptt":
            self.search_space += [
                {
                    "type": "categorical",
                    "name": "tbptt_length",
                    "choices": [8, 16, 32, 64, 128, self.worker_steps],
                },
            ]
