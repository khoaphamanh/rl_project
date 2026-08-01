"""
Feature extractors for comparing memory against no memory under a POMDP.

MiniGrid-MemoryS11-v0 gives the agent a 7x7x3 egocentric view, so the cue
shown at the start of the episode is out of sight by the time the agent has
to choose a branch. Only an encoder that carries information across timesteps
can solve it. This is the ablation of the RPPO paper (Figure 6, "With
Recurrence" vs "No Recurrence"), where Section 5 concludes that recurrence is
essential for the memory-dependent task.

Four independent, interchangeable encoders:

    MLP           no memory  -- each timestep is encoded on its own
    LSTM          memory     -- carries hidden state h and cell state c
    GRU           memory     -- carries hidden state h
    Transformer   memory     -- carries nothing; RE-READS the whole sequence
                               through causally masked self-attention

They all share one contract, so PPO can swap between them without any other
change:

    input : (batch, seq_len, *obs_shape)   e.g. (B, T, 7, 7, 3) uint8
    output: (batch, seq_len, hidden_size)

All three call flatten_obs first, which one-hots the observation's three
category channels into 7 * 7 * 20 = 980 numbers. Read its docstring before
changing anything about it -- feeding the raw indices instead is what made
this ablation come out flat, for every encoder at once.

batch is the number of sequences, seq_len the truncation length used for
truncated BPTT (paper Section 3.1). While collecting rollouts, seq_len = 1.

THAT LAST SENTENCE IS WHY THE TRANSFORMER IS NOT YET WIRED INTO build_extractor.
LSTM and GRU survive seq_len = 1 because everything they remember is in the
hidden state the rollout loop hands back in. A transformer remembers by looking
at earlier POSITIONS of the sequence it was given -- hand it one timestep and
there are no earlier positions, so it silently degrades to an MLP. Running it
in PPO needs the rollout to carry a window of past observations the way it
currently carries h; see Transformer.forward for what that would take.

Run with:
    python models/feature_extractor.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# MiniGrid encodes every cell as three CATEGORY INDICES, not as measurements:
#
#     channel 0   object   11 kinds   unseen 0, empty 1, wall 2, ... key 5, ball 6
#     channel 1   colour    6 kinds
#     channel 2   state     3 kinds   a door being open / closed / locked
#
# These are the sizes of minigrid.core.constants.OBJECT_TO_IDX, COLOR_TO_IDX
# and the three door states. Hardcoded rather than imported so this file needs
# nothing but torch and still runs on its own. An index that outgrew its range
# would make F.one_hot raise, so the failure would be loud, not silent.
N_OBJECT, N_COLOR, N_STATE = 11, 6, 3
OBS_CHANNEL_SIZES = (N_OBJECT, N_COLOR, N_STATE)
CELL_SIZE = sum(OBS_CHANNEL_SIZES)  # 20 numbers describe one cell


def flatten_obs(x):
    """(batch, seq_len, 7, 7, 3) -> (batch, seq_len, 7*7*20), one-hot float.

    WHY THIS IS NOT JUST A RESHAPE. It used to be, and that single line is what
    made the whole GRU-vs-LSTM-vs-MLP ablation come out flat.

    The indices above are CATEGORIES wearing the costume of numbers. Fed in
    raw, the cue the agent has to memorize -- key(5) versus ball(6) -- is a
    difference of 1.0 in ONE of 147 inputs, and it points along the very same
    axis as empty(1) versus wall(2). The network cannot separate "which object"
    from "how much object", and the one bit that matters is buried under the
    large constant background of walls and unseen tiles that fills most of the
    7x7 window.

    Measured, on a SUPERVISED version of the task (walk east 9 steps, predict
    which branch is correct -- no exploration, no credit assignment, the label
    handed over 2500 times), a GRU of exactly this size scores:

        raw indices, as before             0.51  0.49  0.49   <- chance
        x / 10 - 0.5                       0.51  0.49  0.49   <- chance
        per-channel rescale, centred       0.51  0.49  0.49   <- chance
        one-hot, below                     1.00  1.00  1.00

    So it was never a matter of scale. Rescaling moves the numbers, it does not
    make them categories. One-hot does: each kind of object gets its OWN
    orthogonal input, "is it a ball" becomes a coordinate rather than a
    magnitude, and a single weight can read it. This is what the MiniGrid
    baselines all do, whether by one-hot or by an embedding table.

    With it the ablation finally says something, on the same probe:

        MLP    0.49  0.49  0.49   chance, and it MUST be -- no memory
        GRU    1.00  1.00  1.00
        LSTM   0.49  1.00  1.00   fails 1 seed in 3

    which is the paper's Figure 6, down to Section 4's remark that the GRU came
    out slightly ahead of the LSTM.

    The cost is width: 147 inputs become 7 * 7 * 20 = 980. Only the encoder's
    first layer grows, and the rollout buffer is untouched -- it still stores
    uint8 (7, 7, 3) and the expansion happens here, per forward pass.
    """
    # long() because F.one_hot indexes with it; uint8 is rejected
    x = x.long()

    # x[..., c] is (batch, seq_len, 7, 7); one_hot appends the class axis, so
    # each of these is (batch, seq_len, 7, 7, n) and they concatenate into
    # (batch, seq_len, 7, 7, 20) -- one cell, described by 20 numbers
    channels = [F.one_hot(x[..., c], n) for c, n in enumerate(OBS_CHANNEL_SIZES)]

    onehot = torch.cat(channels, dim=-1)
    return onehot.reshape(x.shape[0], x.shape[1], -1).float()


def random_obs(*shape):
    """A random but VALID observation, (*shape, 7, 7, 3) uint8.

    torch.randint(0, 11, ...) across all three channels would be invalid: the
    colour channel only has 6 kinds and the state channel 3, so it would hand
    F.one_hot an out-of-range index. Each channel is drawn from its own range.
    """
    channels = [torch.randint(0, n, (*shape, 7, 7)) for n in OBS_CHANNEL_SIZES]
    return torch.stack(channels, dim=-1).to(torch.uint8)


class MLP(nn.Module):
    """Memoryless baseline: n_layers x (Linear -> ReLU).

    The first layer maps input_size -> hidden_size, every following layer
    maps hidden_size -> hidden_size.

    nn.Linear only touches the last dimension, so each timestep is encoded
    independently and nothing flows from t-1 to t. That is exactly the
    property being tested: under a POMDP this encoder cannot recall the cue.
    """

    def __init__(self, input_size, hidden_size, n_layers):
        super().__init__()

        layers = []
        for i in range(n_layers):
            in_features = input_size if i == 0 else hidden_size
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(flatten_obs(x))


class LSTM(nn.Module):
    """Single-layer LSTM returning only its output.

    hidden is the (h_0, c_0) pair of the sequence, each of shape
    (1, batch, hidden_size). During optimization it comes from the buffer,
    so consecutive sequences continue where the previous one stopped.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x, hidden=None, return_hidden=False):
        """return_hidden=True also gives back (h_n, c_n) of the last step.

        The rollout loop needs it: c_n cannot be recovered from output, so it
        would be lost forever. Default stays output-only, so Network and
        memory_test are unaffected.
        """
        output, hidden = self.lstm(flatten_obs(x), hidden)
        return (output, hidden) if return_hidden else output


class GRU(nn.Module):
    """Single-layer GRU returning only its output.

    hidden is h_0 of the sequence, of shape (1, batch, hidden_size). The GRU
    is the recurrent layer the paper settled on (Section 4: "GRU turned out
    to be slightly more effective than LSTM").
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x, hidden=None, return_hidden=False):
        """return_hidden=True also gives back h_n of the last step."""
        output, hidden = self.gru(flatten_obs(x), hidden)
        return (output, hidden) if return_hidden else output


# ----------------------------------------------------------------------
# Transformer encoder
#
# A fourth way to carry information across timesteps, and a different KIND of
# memory from the two above. LSTM and GRU compress the past into a fixed
# hidden_size vector and pass it forward -- what is not written into h at step
# t is gone at t+1. Attention keeps every past step and lets step t look up
# whichever ones it needs, so the cue does not have to survive a squeeze
# through 64 numbers. The price is that it needs the sequence, not a state.
#
# The blocks below are the standard encoder half of "Attention Is All You
# Need", written out rather than taken from nn.TransformerEncoder so every
# tensor shape is visible.
# ----------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Adds fixed sin/cos position codes to (batch, seq_len, d_model).

        PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))     even columns
        PE(pos, 2i + 1) = cos(pos / 10000^(2i/d_model))     odd columns

    NOT optional, and not the same kind of detail as it is in a language
    model. Self-attention is PERMUTATION INVARIANT: shuffle the timesteps and
    the outputs shuffle with them, unchanged. Without these codes the encoder
    could tell that a key was seen but not WHEN, so "cue, then junction" and
    "junction, then cue" would look identical -- and on MemoryEnv the order is
    the whole task.

    d_model must be even; pe[:, 1::2] has floor(d_model/2) columns while the
    denominator has ceil(d_model/2), so an odd width raises here. It is even
    in practice anyway, since d_model must divide by n_heads.
    """

    def __init__(self, d_model, max_seq_length):
        super().__init__()

        # (max_seq_length, d_model), filled column-pair by column-pair
        pe = torch.zeros(size=(max_seq_length, d_model))

        # position 0 .. max_seq_length-1, as a column so it broadcasts
        position = torch.arange(0, max_seq_length, dtype=torch.float32).unsqueeze(1)

        denominator = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * denominator)
        pe[:, 1::2] = torch.cos(position * denominator)

        # a buffer, not a parameter: it is a constant, so it must move with
        # .to(device) but must never receive a gradient. persistent=False
        # keeps it OUT of state_dict -- it is fully determined by d_model and
        # max_seq_length, so saving it would only bloat every checkpoint and
        # make max_seq_length impossible to change without breaking load.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, S, d)

    def forward(self, x):
        """x (batch, seq_len, d_model) + pe (1, max_seq_length, d_model)."""
        seq_length = x.shape[1]
        return x + self.pe[:, :seq_length]


class MultiHeadAttention(nn.Module):
    """h independent attention heads over d_model, run as one batched matmul.

    Each head works in d_k = d_model // num_heads dimensions, so all heads
    together cost the same as one head of full width. Different heads can
    learn to look for different things -- one for the cue, one for the wall
    layout -- instead of averaging both into a single lookup.
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # width of ONE head

        # X (batch, seq_len, d_model) @ W (d_model, d_model) -> same shape
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_head(self, x):
        """(batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k).

        The transpose is what puts num_heads next to batch, so the attention
        matmuls below treat every head as just another item in the batch.
        """
        batch_size, seq_length, d_model = x.shape
        return x.reshape(batch_size, seq_length, self.num_heads, self.d_k).transpose(
            1, 2
        )

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """softmax(Q K^T / sqrt(d_k)) V, with positions where mask == 0 killed."""
        # Q, K, V (batch, num_heads, seq_len, d_k)
        # -> scores (batch, num_heads, seq_len, seq_len): row t, column s is
        #    "how much does step t want to read step s"
        # self.d_k ** 0.5 rather than np.sqrt: this file deliberately imports
        # nothing but torch, so it also runs standalone
        attention_score = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k**0.5)

        if mask is not None:
            # -1e9 rather than -inf: softmax of an all -inf row is NaN, and a
            # fully masked row is exactly what a causal mask produces if the
            # slicing is ever off by one. This fails visibly instead.
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(attention_score, dim=-1)

        return torch.matmul(attention_weights, V)  # (b, h, seq_len, d_k)

    def combine_heads(self, x):
        """(batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)."""
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).reshape(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_head(self.W_q(Q))
        K = self.split_head(self.W_k(K))
        V = self.split_head(self.W_v(V))

        attention_output = self.scaled_dot_product_attention(Q, K, V, mask=mask)

        return self.W_o(self.combine_heads(attention_output))


class PositionWiseFeedForward(nn.Module):
    """Linear -> ReLU -> Linear, applied to each position independently.

    Attention mixes ACROSS positions but is linear in V; this is where the
    per-step nonlinearity lives. d_ff is normally 4 * d_model.
    """

    def __init__(self, d_model, d_ff):
        super().__init__()

        self.fc1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.fc2 = nn.Linear(in_features=d_ff, out_features=d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    """Self-attention then feed-forward, each wrapped in residual + LayerNorm."""

    def __init__(self, d_model, num_heads, d_ff, p_drop):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads
        )
        self.feed_forward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, x, mask=None):
        # Q = K = V = x is what makes it SELF-attention
        attention_output = self.multi_head_attention(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout(attention_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class Encoder(nn.Module):
    """n_layers stacked EncoderLayers, all at the same width."""

    def __init__(self, d_model, num_heads, d_ff, p_drop, n_layers):
        super().__init__()

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model, num_heads=num_heads, d_ff=d_ff, p_drop=p_drop
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, mask=None):
        """The mask is FORWARDED, not dropped -- see Transformer.forward."""
        for layer in self.encoder_layers:
            x = layer(x, mask=mask)

        return x


class Transformer(nn.Module):
    """Causally masked self-attention encoder. Same contract as MLP/LSTM/GRU.

        (batch, seq_len, 7, 7, 3)  ->  (batch, seq_len, hidden_size)

    THE TWO LINEAR LAYERS. The stack runs at its own width d_model, which is
    neither of the sizes the contract names, so it is bracketed by one
    projection at each end:

        flatten_obs   (b, t, 7, 7, 3) -> (b, t, 980)      one-hot, see above
        fc_in         (b, t, 980)     -> (b, t, d_model)   into attention width
        + positional encoding, encoder layers
        fc_out        (b, t, d_model) -> (b, t, hidden_size)  back to contract

    fc_out is what lets d_model be chosen for the attention (bigger is a
    better lookup) while hidden_size stays whatever the actor and critic heads
    were built for. Setting d_model = hidden_size does not remove it; the
    layer is still there, just square.

    THE CAUSAL MASK, which is not cosmetic. Plain encoder self-attention is
    bidirectional: the output at step t attends to t+1, t+2, ... In a language
    model that is the point. Here it is the agent reading observations it has
    not made yet, and it breaks PPO twice over --

        1. the policy would be scored on information it could not have had,
           so a "solved" run proves nothing about memory;
        2. worse, it is INCONSISTENT. Rollouts are collected one step at a
           time, where no future exists, while the update sees whole
           sequences, where it does. The same weights would then give two
           different log_probs for the same state, so the ratio
           pi_new / pi_old is wrong from the very first gradient step and the
           clip fires on noise.

    So attention is restricted to a lower-triangular mask: step t sees steps
    0..t and nothing after. Verified by causality_test in main().

    DROPOUT DEFAULTS TO ZERO, on purpose. PPO compares log_probs computed at
    rollout time against log_probs recomputed at update time. Dropout makes
    the same observation give a different answer on two passes, so that
    comparison stops meaning anything -- the ratio would be noise even before
    any learning has happened. Nothing in this project calls model.eval(), so
    a nonzero p_drop would be active during BOTH passes. Turn it on only with
    an eval()/train() discipline in place.

    hidden is accepted only to match the LSTM/GRU signature. A transformer
    holds no state between calls: it remembers by re-reading earlier
    POSITIONS, which means it needs the sequence itself. Passing a real state
    raises rather than being ignored, because being ignored is precisely the
    silent failure that would turn this into an expensive MLP -- see the
    module docstring on why it is not in build_extractor yet.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=None,
        p_drop=0.0,
        max_seq_length=1024,
    ):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model  # the ratio the original paper uses

        self.d_model = d_model
        self.max_seq_length = max_seq_length

        self.fc_in = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(p=p_drop)

        self.encoder = Encoder(
            d_model=d_model,
            num_heads=n_heads,
            d_ff=d_ff,
            p_drop=p_drop,
            n_layers=n_layers,
        )

        # the layer the contract needs: d_model -> hidden_size
        self.fc_out = nn.Linear(d_model, hidden_size)

        # lower triangular ones, built once at full size and sliced per call.
        # bool rather than float is 4x smaller, and mask == 0 still works.
        # persistent=False for the same reason as pe: a constant, derivable
        # from max_seq_length, that has no business inside a checkpoint.
        causal = torch.ones(max_seq_length, max_seq_length, dtype=torch.bool).tril()
        self.register_buffer(
            "causal_mask", causal.view(1, 1, max_seq_length, max_seq_length),
            persistent=False,
        )

    def forward(self, x, hidden=None, return_hidden=False):
        """return_hidden=True gives back None -- there is no state to give.

        The None is not a placeholder to be filled in later. Carrying memory
        across calls would mean caching the last K observations per worker and
        prepending them here, which changes what the rollout buffer stores,
        not what this returns.
        """
        if hidden is not None:
            raise ValueError(
                "Transformer carries no state between calls, so a non-None "
                "hidden would be silently dropped and the encoder would "
                "behave as an MLP. Feed whole sequences instead."
            )

        seq_len = x.shape[1]
        if seq_len > self.max_seq_length:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_length "
                f"{self.max_seq_length}: the positional codes and the causal "
                f"mask only reach that far. Raise max_seq_length to at least "
                f"the env's max_steps."
            )

        x = flatten_obs(x)  # (b, t, 980), one-hot -- see flatten_obs
        x = self.fc_in(x)  # (b, t, d_model)
        x = self.dropout(self.positional_encoding(x))

        # slice the mask to this sequence: (1, 1, t, t) broadcasts over both
        # the batch and the head axis of the (b, h, t, t) attention scores
        x = self.encoder(x, mask=self.causal_mask[:, :, :seq_len, :seq_len])

        output = self.fc_out(x)  # (b, t, hidden_size)
        return (output, None) if return_hidden else output


OBS_SHAPE = (7, 7, 3)  # MiniGrid-MemoryS11-v0 partial observation
INPUT_SIZE = 7 * 7 * CELL_SIZE  # 980 AFTER one-hot, not 147. See flatten_obs.
HIDDEN_SIZE = 64
N_LAYERS = 3

# Transformer only. d_model is the attention width and is independent of
# hidden_size -- fc_out bridges the two. n_heads must divide d_model.
D_MODEL = 128
N_HEADS = 4
N_LAYERS_TRANSFORMER = 2  # 2, not 3: attention layers are far heavier than
#                           the MLP's, and this task is not a deep one
D_FF = 4 * D_MODEL
P_DROP = 0.0  # see Transformer's docstring -- nonzero breaks the PPO ratio

N_WORKERS = 2  # W how many games are played at the same time
WORKER_STEPS = 10  # T,	how many steps each game plays before training
N_TOTAL_STEPS = N_WORKERS * WORKER_STEPS  # total steps collected this iteration

SEQ_LEN = 5  # max(#T), the truncation length for truncated BPTT
NUM_SEQUENCES = N_TOTAL_STEPS // SEQ_LEN  # sequences the batch splits into


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def memory_test(model, x, name):
    """Change the observation at t=0 and report which outputs move.

    A memoryless encoder only changes at t=0. A recurrent one changes at
    every later timestep too, because the altered observation is still
    present in its hidden state. This is what the MiniGrid Memory task needs.
    """
    x_changed = x.clone()

    # perturb the first observation only. Modulo per channel, so every index
    # stays inside its own range -- +1 alone could push object 10 to 11, which
    # flatten_obs would reject. The (3,) tensor broadcasts over the last axis.
    ranges = torch.tensor(OBS_CHANNEL_SIZES, dtype=x.dtype)
    x_changed[:, 0] = (x[:, 0] + 1) % ranges

    with torch.no_grad():
        delta = (model(x) - model(x_changed)).abs().sum(dim=-1)  # (batch, seq_len)

    moved = (delta > 1e-6).any(dim=0)  # per timestep, over the batch
    marks = "  ".join("CHANGED" if m else "  same " for m in moved)
    print(f"  {name:<12} {marks}")


def causality_test(model, x, name):
    """Change the observation at the LAST step and report which outputs move.

    The mirror image of memory_test, and the check that only matters for
    attention. Memory must flow FORWARD only: perturbing the last timestep may
    change the last output, and must leave every earlier one alone.

    An unmasked encoder fails this at every timestep -- and that failure is
    invisible in memory_test, which an encoder reading the future passes just
    as happily as one that remembers. MLP, LSTM and GRU cannot fail it: they
    have no mechanism to look forward in the first place.
    """
    x_changed = x.clone()

    ranges = torch.tensor(OBS_CHANNEL_SIZES, dtype=x.dtype)
    x_changed[:, -1] = (x[:, -1] + 1) % ranges

    with torch.no_grad():
        delta = (model(x) - model(x_changed)).abs().sum(dim=-1)  # (batch, seq_len)

    moved = (delta > 1e-6).any(dim=0)
    marks = "  ".join("LEAKED " if m else "  same " for m in moved[:-1])
    ok = "OK" if not moved[:-1].any() else "FUTURE LEAK"
    print(f"  {name:<12} {marks}  (last step ignored)   {ok}")


def main():
    print(f"obs_shape={OBS_SHAPE}  input_size={INPUT_SIZE}  hidden_size={HIDDEN_SIZE}")
    print(
        f"workers={N_WORKERS} x worker_steps={WORKER_STEPS}  ->  n_total_steps={N_TOTAL_STEPS}"
    )
    print(f"split into num_sequences={NUM_SEQUENCES} x seq_len={SEQ_LEN}\n")

    # one batch of sequences, exactly as MiniGrid delivers it
    x = random_obs(NUM_SEQUENCES, SEQ_LEN)
    flatten_x = flatten_obs(x)
    print(f"flatten_x shape: {tuple(flatten_x.shape)}  (one-hot, was 147 raw)")
    print(f"input: {tuple(x.shape)}  dtype={x.dtype}\n")

    mlp = MLP(INPUT_SIZE, HIDDEN_SIZE, N_LAYERS)
    lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE)
    gru = GRU(INPUT_SIZE, HIDDEN_SIZE)
    transformer = Transformer(
        INPUT_SIZE,
        HIDDEN_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS_TRANSFORMER,
        d_ff=D_FF,
        p_drop=P_DROP,
    )

    models = [("MLP", mlp), ("LSTM", lstm), ("GRU", gru), ("Transformer", transformer)]

    print("FORWARD PASS")
    for name, model in models:
        out = model(x)
        print(
            f"  {name:<12} {tuple(x.shape)} -> {tuple(out.shape)}   {n_params(model):>8,} params"
        )

    # an initial hidden state can be handed to the recurrent encoders, which
    # is how one sequence continues the one before it (truncated BPTT)
    h0 = torch.zeros(1, NUM_SEQUENCES, HIDDEN_SIZE)
    c0 = torch.zeros(1, NUM_SEQUENCES, HIDDEN_SIZE)
    print(f"\n  LSTM with (h0, c0) -> {tuple(lstm(x, (h0, c0)).shape)}")
    print(f"  GRU  with h0       -> {tuple(gru(x, h0).shape)}")

    # the transformer has no such state, and says so instead of ignoring it
    try:
        transformer(x, h0)
    except ValueError as e:
        print(f"  Transformer with h0 -> ValueError: {str(e).split(',')[0]}...")

    # the two projections that bracket the attention stack
    print(f"\n  Transformer widths: {INPUT_SIZE} -(fc_in)-> {D_MODEL} "
          f"-(x{N_LAYERS_TRANSFORMER} layers)-> {D_MODEL} -(fc_out)-> {HIDDEN_SIZE}")

    print(f"\nMEMORY TEST (observation at t=0 perturbed)")
    print("               " + "  ".join(f"   t={t}  " for t in range(SEQ_LEN)))
    for name, model in models:
        memory_test(model, x, name)

    print(f"\nCAUSALITY TEST (observation at t={SEQ_LEN - 1} perturbed)")
    print("               " + "  ".join(f"   t={t}  " for t in range(SEQ_LEN - 1)))
    for name, model in models:
        causality_test(model, x, name)


if __name__ == "__main__":
    main()
