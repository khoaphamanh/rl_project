from config.config import Config


class ConfigLSTM(Config):
    """LSTM encoder -- the "With Recurrence" arm that carries (h, c).

    build_extractor builds LSTM(input_size, hidden_size), so hidden_size is the
    only architecture knob this encoder has -- and it is set HERE rather than in
    Config so the LSTM can be resized without touching the other three.

    Depth is one layer. If a stacked LSTM is ever wanted, add self.n_layers_lstm
    here AND read it in helper.build_extractor's LSTM branch -- keeping the knob
    and its reader in step is the reason each encoder owns its own file.
    """

    def _configure_model(self):
        self.recurrent_model = "LSTM"

        self.hidden_size = 64  # width of h AND of c, and the size handed to
        #   fc_actor / fc_critic. An LSTM carries two states of this width and
        #   costs 4h^2 weights against the GRU's 3h^2 -- so if the ablation is
        #   meant to be parameter-matched rather than width-matched, this is the
        #   number that moves. Equal across encoders = matched on width.

        # APPENDED to Config's shared space, never replacing it
        self.search_space += [
            {
                "name": "hidden_size",
                "type": "int",
                "low": 32,
                "high": 256,
                "step": 8,
                "log": False,
            },
        ]
