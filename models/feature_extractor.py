"""
Two interchangeable encoders (MLP, GRU) for MiniGrid's 7x7x3 partial
observation: (batch, seq_len, 7, 7, 3) uint8 -> (batch, seq_len, hidden_size).
Run with: python models/feature_extractor.py
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


class GRU(nn.Module):
    """Single-layer GRU. Hidden: optional h_0."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x, hidden=None, return_hidden=False):
        output, hidden = self.gru(flatten_obs(x), hidden)
        return (output, hidden) if return_hidden else output


OBS_SHAPE = (7, 7, 3)  # MiniGrid-MemoryS11-v0 partial observation
INPUT_SIZE = 7 * 7 * CELL_SIZE  # 980 AFTER one-hot, not 147. See flatten_obs.
HIDDEN_SIZE = 64
N_LAYERS = 3


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

    # perturb the first observation only. Modulo per channel keeps every index
    # in range -- +1 alone could push object 10 to 11, which flatten_obs rejects.
    ranges = torch.tensor(OBS_CHANNEL_SIZES, dtype=x.dtype)
    x_changed[:, 0] = (x[:, 0] + 1) % ranges

    with torch.no_grad():
        delta = (model(x) - model(x_changed)).abs().sum(dim=-1)  # (batch, seq_len)

    moved = (delta > 1e-6).any(dim=0)  # per timestep, over the batch
    marks = "  ".join("CHANGED" if m else "  same " for m in moved)
    print(f"  {name:<12} {marks}")


def causality_test(model, x, name):
    """Perturbs the last step and prints which earlier outputs change: checks for future leakage."""
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
    gru = GRU(INPUT_SIZE, HIDDEN_SIZE)

    models = [("MLP", mlp), ("GRU", gru)]

    print("FORWARD PASS")
    for name, model in models:
        out = model(x)
        print(
            f"  {name:<12} {tuple(x.shape)} -> {tuple(out.shape)}   {n_params(model):>8,} params"
        )

    # an initial hidden state can be handed to the GRU, which
    # is how one sequence continues the one before it (truncated BPTT)
    h0 = torch.zeros(1, NUM_SEQUENCES, HIDDEN_SIZE)
    print(f"\n  GRU  with h0       -> {tuple(gru(x, h0).shape)}")

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
