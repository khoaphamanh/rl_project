from config.config import Config


class ConfigGRU(Config):
    """GRU encoder -- the "With Recurrence" arm that carries only h.

    Like the LSTM, build_extractor builds GRU(input_size, hidden_size), so
    hidden_size is the only architecture knob -- set HERE rather than in Config
    so the GRU can be resized without touching the other three.

    Depth is one layer. For a stacked GRU add self.n_layers_gru here and read it
    in helper.build_extractor's GRU branch, the same way the transformer's
    knobs are wired in config_transformer.py.
    """

    def _configure_model(self):
        self.recurrent_model = "GRU"

        self.hidden_size = 64  # width of h, and the size handed to fc_actor /
        #   fc_critic. A GRU costs 3h^2 weights against the LSTM's 4h^2 and the
        #   MLP's h^2, so equal values across the four encoders match them on
        #   WIDTH, not on parameter count. Change this one to match parameters
        #   instead -- but say which you did, the comparison depends on it.
