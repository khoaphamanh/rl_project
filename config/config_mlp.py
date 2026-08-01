from config.config import Config


class ConfigMLP(Config):
    """MLP encoder. Everything shared lives in Config; only the architecture
    knobs the MLP branch of build_extractor reads are set here.

    The MLP is the "No Recurrence" arm of the ablation: no hidden state, so
    while it acts it sees one observation and nothing before it. On a task
    where the cue is still on screen at decision time (DoorKey, or a Memory env
    without force_cue_visible) that is enough and it scores like the recurrent
    ones -- which is exactly why force_cue_visible exists, to make the cue gone
    by the time it matters and force the comparison to mean something.
    """

    def _configure_model(self):
        self.feature_extractor = "MLP"

        self.hidden_size = 64  # width of every hidden layer, and the size the
        #   encoder hands to fc_actor / fc_critic. Owned per encoder, not
        #   shared: an MLP layer of width h costs h^2 weights where a GRU costs
        #   3h^2, so matching PARAMETER COUNT across the ablation means giving
        #   them different widths. Keep them equal to match width instead --
        #   just be clear in the writeup which of the two was controlled.

        self.n_layers_mlp = 3  # hidden Linear+ReLU blocks between the flattened
        #   observation and the hidden_size output. build_extractor passes this
        #   to MLP(input_size, hidden_size, n_layers_mlp); the LSTM and GRU take
        #   no such depth argument, which is why it lives here and not in Config.

        # the MLP's own knobs, appended to the shared PPO ones in Config.
        #
        # hidden_size is a STEPPED INT RANGE, 32..512 in steps of 8 -- 61
        # candidate widths rather than a handful of powers of two. The step is
        # what keeps that from being silly: it stops the sampler spending
        # trials on the difference between 97 and 98, which no training run
        # could resolve at this noise level, while still letting it land
        # somewhere the powers of two would have skipped entirely.
        #
        # THE SAME low/high/step IS USED FOR ALL FOUR ENCODERS, on purpose. If
        # the search settles the MLP at 96 and the GRU at 320, that is a fact
        # about the encoders; if they had been given different grids it would
        # only be a fact about the grids.
        self.search_space += [
            {"type": "int", "name": "hidden_size", "low": 32, "high": 512, "step": 8},
            {"type": "int", "name": "n_layers_mlp", "low": 1, "high": 4},
        ]
