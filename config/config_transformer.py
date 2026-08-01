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
        self.d_ff = 4 * self.d_model  # 256, the ratio the original paper uses

        self.p_drop = 0.0  # NOT a free choice. PPO compares log_probs recorded
        #   during the rollout against log_probs recomputed during the update.
        #   Dropout makes the same observation give two different answers, so
        #   the ratio pi_new / pi_old is noise before any learning happens --
        #   and nothing in this project calls model.eval(), so a nonzero value
        #   would be live during both passes. Leave at 0.0.

        self.max_seq_length = self.worker_steps  # also NOT free. The positional
        #   codes and the causal mask are built once, this long, and forward
        #   raises on anything longer. split_pad_mask pads to L = the longest
        #   unbroken stretch of steps, which is at most T -- so tying this to
        #   worker_steps (shared, set in Config before this hook runs) makes it
        #   follow the env instead of being one more number to change per size.
