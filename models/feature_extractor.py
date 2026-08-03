"""
Four interchangeable encoders (MLP, LSTM, GRU, Transformer) for MiniGrid's
7x7x3 partial observation: (batch, seq_len, 7, 7, 3) uint8 -> (batch, seq_len,
hidden_size). Run with: python models/feature_extractor.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# MiniGrid cells: object (11), color (6), state (3). Hardcoded so this file is self-contained.
N_OBJECT, N_COLOR, N_STATE = 11, 6, 3
OBS_CHANNEL_SIZES = (N_OBJECT, N_COLOR, N_STATE)
CELL_SIZE = sum(OBS_CHANNEL_SIZES)  # 20 per cell


def flatten_obs(x):
    """One-hot each cell's channels: (batch, seq_len, 7, 7, 3) -> (batch, seq_len, 980)."""
    x = x.long()
    channels = [F.one_hot(x[..., c], n) for c, n in enumerate(OBS_CHANNEL_SIZES)]
    onehot = torch.cat(channels, dim=-1)
    return onehot.reshape(x.shape[0], x.shape[1], -1).float()


def random_obs(*shape):
    """A random but valid observation, (*shape, 7, 7, 3) uint8, each channel drawn from its own valid range."""
    channels = [torch.randint(0, n, (*shape, 7, 7)) for n in OBS_CHANNEL_SIZES]
    return torch.stack(channels, dim=-1).to(torch.uint8)


class MLP(nn.Module):
    """Memoryless: n_layers x (Linear -> ReLU)."""
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
    """Single-layer LSTM. Hidden: optional (h_0, c_0) pair."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x, hidden=None, return_hidden=False):
        output, hidden = self.lstm(flatten_obs(x), hidden)
        return (output, hidden) if return_hidden else output


class GRU(nn.Module):
    """Single-layer GRU. Hidden: optional h_0."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x, hidden=None, return_hidden=False):
        output, hidden = self.gru(flatten_obs(x), hidden)
        return (output, hidden) if return_hidden else output


# Standard "Attention Is All You Need" encoder half, written out rather than
# taken from nn.TransformerEncoder so every tensor shape is visible.


class PositionalEncoding(nn.Module):
    """Fixed sin/cos position codes (d_model must be even)."""
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        pe = torch.zeros(size=(max_seq_length, d_model))
        position = torch.arange(0, max_seq_length, dtype=torch.float32).unsqueeze(1)
        denominator = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * denominator)
        pe[:, 1::2] = torch.cos(position * denominator)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        seq_length = x.shape[1]
        return x + self.pe[:, :seq_length]


class MultiHeadAttention(nn.Module):
    """Multi-head attention (d_k = d_model // num_heads per head)."""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_head(self, x):
        """Reshape to (batch, num_heads, seq_len, d_k)."""
        batch_size, seq_length, d_model = x.shape
        return x.reshape(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Scaled dot-product attention."""
        attention_score = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k**0.5)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(attention_score, dim=-1)
        return torch.matmul(attention_weights, V)

    def combine_heads(self, x):
        """Reshape to (batch, seq_len, d_model)."""
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).reshape(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_head(self.W_q(Q))
        K = self.split_head(self.W_k(K))
        V = self.split_head(self.W_v(V))

        attention_output = self.scaled_dot_product_attention(Q, K, V, mask=mask)

        return self.W_o(self.combine_heads(attention_output))


class PositionWiseFeedForward(nn.Module):
    """Position-wise linear -> ReLU -> linear."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.fc2 = nn.Linear(in_features=d_ff, out_features=d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    """Attention + Feed-forward with residual + LayerNorm."""
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
        attention_output = self.multi_head_attention(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class Encoder(nn.Module):
    """Stacked encoder layers."""
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
        for layer in self.encoder_layers:
            x = layer(x, mask=mask)

        return x


class Transformer(nn.Module):
    """Causally masked self-attention encoder. No state between calls (hidden must be None)."""
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
            d_ff = 4 * d_model
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
        self.fc_out = nn.Linear(d_model, hidden_size)
        causal = torch.ones(max_seq_length, max_seq_length, dtype=torch.bool).tril()
        self.register_buffer(
            "causal_mask", causal.view(1, 1, max_seq_length, max_seq_length),
            persistent=False,
        )

    def forward(self, x, hidden=None, return_hidden=False):
        """Encode x; return_hidden=True returns (output, None)."""
        if hidden is not None:
            raise ValueError(
                "Transformer carries no state between calls; feed whole sequences instead."
            )
        seq_len = x.shape[1]
        if seq_len > self.max_seq_length:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_length {self.max_seq_length}"
            )
        x = flatten_obs(x)
        x = self.fc_in(x)
        x = self.dropout(self.positional_encoding(x))
        x = self.encoder(x, mask=self.causal_mask[:, :, :seq_len, :seq_len])
        output = self.fc_out(x)
        return (output, None) if return_hidden else output


OBS_SHAPE = (7, 7, 3)  # MiniGrid-MemoryS11-v0 partial observation
INPUT_SIZE = 7 * 7 * CELL_SIZE  # 980 AFTER one-hot, not 147. See flatten_obs.
HIDDEN_SIZE = 64
N_LAYERS = 3

# transformer only: d_model is the attention width, independent of
# hidden_size (fc_out bridges the two). n_heads must divide d_model.
D_MODEL = 128
N_HEADS = 4
N_LAYERS_TRANSFORMER = 2
D_FF = 4 * D_MODEL
P_DROP = 0.0  # must stay 0, see Transformer's docstring

N_WORKERS = 2  # W: games played in parallel
WORKER_STEPS = 10  # T: steps per game before training
N_TOTAL_STEPS = N_WORKERS * WORKER_STEPS

SEQ_LEN = 5  # truncated-BPTT length
NUM_SEQUENCES = N_TOTAL_STEPS // SEQ_LEN


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def memory_test(model, x, name):
    """Perturbs t=0 and prints which output timesteps change, to check the encoder carries memory forward."""
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
    """Perturbs the last step and prints which earlier outputs change, to check the encoder never leaks future information."""
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
