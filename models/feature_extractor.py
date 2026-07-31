"""
Feature extractors for comparing memory against no memory under a POMDP.

MiniGrid-MemoryS11-v0 gives the agent a 7x7x3 egocentric view, so the cue
shown at the start of the episode is out of sight by the time the agent has
to choose a branch. Only an encoder that carries information across timesteps
can solve it. This is the ablation of the RPPO paper (Figure 6, "With
Recurrence" vs "No Recurrence"), where Section 5 concludes that recurrence is
essential for the memory-dependent task.

Three independent, interchangeable encoders:

    MLP    no memory  -- each timestep is encoded on its own
    LSTM   memory     -- carries hidden state h and cell state c
    GRU    memory     -- carries hidden state h

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


OBS_SHAPE = (7, 7, 3)  # MiniGrid-MemoryS11-v0 partial observation
INPUT_SIZE = 7 * 7 * CELL_SIZE  # 980 AFTER one-hot, not 147. See flatten_obs.
HIDDEN_SIZE = 64
N_LAYERS = 3
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
    print(f"  {name:<5} {marks}")


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

    print("FORWARD PASS")
    for name, model in [("MLP", mlp), ("LSTM", lstm), ("GRU", gru)]:
        out = model(x)
        print(
            f"  {name:<5} {tuple(x.shape)} -> {tuple(out.shape)}   {n_params(model):>7,} params"
        )

    # an initial hidden state can be handed to the recurrent encoders, which
    # is how one sequence continues the one before it (truncated BPTT)
    h0 = torch.zeros(1, NUM_SEQUENCES, HIDDEN_SIZE)
    c0 = torch.zeros(1, NUM_SEQUENCES, HIDDEN_SIZE)
    print(f"\n  LSTM with (h0, c0) -> {tuple(lstm(x, (h0, c0)).shape)}")
    print(f"  GRU  with h0       -> {tuple(gru(x, h0).shape)}")

    print(f"\nMEMORY TEST (observation at t=0 perturbed)")
    print("        " + "  ".join(f"   t={t}  " for t in range(SEQ_LEN)))
    for name, model in [("MLP", mlp), ("LSTM", lstm), ("GRU", gru)]:
        memory_test(model, x, name)


if __name__ == "__main__":
    main()
