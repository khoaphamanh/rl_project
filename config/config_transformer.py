from config.config import Config


class ConfigTransformer(Config):
    """Transformer encoder. Only the attention-stack knobs live here; the rest
    is shared in Config.

    READ THIS BEFORE TRUSTING A TRANSFORMER RUN. sample() acts one step at a
    time (seq_len = 1), and a transformer remembers by attending to earlier
    POSITIONS of the sequence it is handed -- a length-1 sequence has none, so
    while ACTING it degrades to an MLP however much history the update later
    shows it. It trains and scores, but the number is not a memory result until
    the rollout caches the last K observations per worker. See the
    feature_extractor module docstring and helper.build_extractor.
    """

    # derived, so setting d_model (or d_ff_mult) is enough and d_ff follows
    @property
    def d_ff(self):
        return self.d_ff_mult * self.d_model

    def _configure_model(self):
        self.recurrent_model = "TRANSFORMER"

        self.hidden_size = 64  # what fc_out projects DOWN to, and the size
        #   fc_actor / fc_critic are built for. Set here with the other three
        #   encoders' widths so all four are independent -- this one is not
        #   optional even though the transformer also has d_model, because
        #   build_extractor passes BOTH (input_size -> d_model -> hidden_size).

        # ------------------------------------------------------------------
        # SMALL ON PURPOSE: a first test of the encoder, not a tuned run.
        # Measured on DoorKey-8x8, one fwd+bwd over mini_batch_size=4
        # sequences of L=640, it is CHEAPER than the GRU beside it:
        #
        #     GRU                                200,832 params     97 ms
        #     Transformer d_model=64, 2 layers   166,912 params     74 ms
        #     Transformer d_model=128, 4 layers  926,912 params    279 ms
        #
        # A GRU walks 640 steps strictly one after another; attention does all
        # 640 in one matmul. What does bite is that attention is quadratic in
        # L, so the third line above is where it stops being free.
        # ------------------------------------------------------------------
        self.d_model = 64  # the width the attention stack runs at. Genuinely
        #   independent of hidden_size above -- fc_out projects d_model ->
        #   hidden_size -- but set equal to it here so the test adds no width
        #   anywhere and the comparison against the GRU stays honest. This is
        #   the knob to raise for a bigger transformer; hidden_size only decides
        #   what the heads see, and raising it would move all four encoders'
        #   comparison point rather than just this one's capacity.
        self.n_heads = 4  # must DIVIDE d_model. 64 / 4 = 16 numbers per head.
        self.n_layers_transformer = 1  # one layer is a single lookup and cannot
        #   compose two of them ("find the cue" then "relate it to here"), so 2
        #   is the smallest stack that is still really a transformer. Set to 1
        #   here anyway for the cheapest possible first test; raise it to 2 once
        #   the rollout-cache fix makes the memory number trustworthy.
        self.d_ff_mult = 4  # the ratio the original paper uses. A MULTIPLIER and
        #   not d_ff itself, because d_ff is a property below: tuning d_model
        #   would otherwise leave a stale d_ff sized for the old width.

        self.p_drop = 0.0  # NOT a free choice. PPO compares log_probs recorded
        #   during the rollout against log_probs recomputed during the update.
        #   Dropout makes the same observation give two different answers, so
        #   the ratio pi_new / pi_old is noise before any learning happens --
        #   and nothing in this project calls model.eval(), so a nonzero value
        #   would be live during both passes. Leave at 0.0.

        # APPENDED to Config's shared space, never replacing it. p_drop and
        # max_seq_length are absent on purpose -- see the comments on each.
        #
        # d_model AND n_heads ARE COUPLED: MultiHeadAttention asserts
        # d_model % n_heads == 0, optuna samples the two independently, and it
        # has no way to express a constraint between them. So every width one
        # can propose must divide by every head count the other can, which
        # means d_model's step must be a multiple of lcm(n_heads choices).
        #
        #     n_heads 2, 4, 6, 8   lcm 24   d_model step 24   ->  7 x 4 =  28
        #     n_heads 2, 4         lcm  4   d_model step  4   -> 37 x 2 =  74
        #
        # The second is taken here: a fine grid on the width, which decides
        # capacity, bought with head counts, which mostly decide how that same
        # width is sliced up. Put 6 and 8 heads back and d_model's step MUST
        # return to 24 -- a step of 4 would hand the assert 52 / 6.
        self.search_space += [
            {
                "name": "hidden_size",
                "type": "int",
                "low": 32,
                "high": 256,
                "step": 8,
                "log": False,
            },
            {
                "name": "d_model",
                "type": "int",
                "low": 48,
                "high": 192,
                "step": 4,
                "log": False,
            },
            {
                "name": "n_heads",
                "type": "int",
                "low": 2,
                "high": 4,
                "step": 2,
                "log": False,
            },
            {
                "name": "n_layers_transformer",
                "type": "int",
                "low": 1,
                "high": 4,
                "step": 1,
                "log": False,
            },
        ]

        self.max_seq_length = self.worker_steps  # also NOT free. The positional
        #   codes and the causal mask are built once, this long, and forward
        #   raises on anything longer. split_pad_mask pads to L = the longest
        #   unbroken stretch of steps, which is at most T -- so tying this to
        #   worker_steps (shared, set in Config before this hook runs) makes it
        #   follow the env instead of being one more number to change per size.
