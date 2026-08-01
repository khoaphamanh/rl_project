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
        self.feature_extractor = "LSTM"

        self.hidden_size = 64  # width of h AND of c, and the size handed to
        #   fc_actor / fc_critic. An LSTM carries two states of this width and
        #   costs 4h^2 weights against the GRU's 3h^2 -- so if the ablation is
        #   meant to be parameter-matched rather than width-matched, this is the
        #   number that moves. Equal across encoders = matched on width.

        # the LSTM's only architecture knob, appended to the shared PPO ones.
        #
        # A STEPPED INT RANGE, 32..512 in steps of 8: 61 candidate widths, so
        # the search can land between the powers of two. The step keeps it from
        # wasting trials on differences no training run could resolve.
        #
        # IDENTICAL low/high/step TO THE OTHER THREE ENCODERS, so a difference
        # in the width the search settles on is a difference between the
        # encoders and not between four differently-shaped grids. Note the
        # LSTM carries TWO states of this width and costs 4h^2 weights, the
        # most of the four -- at the top of this range that is the biggest
        # model in the ablation.
        self.search_space += [
            {"type": "int", "name": "hidden_size", "low": 32, "high": 512, "step": 8},
        ]
