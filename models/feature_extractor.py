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

    input : (batch, seq_len, *obs_shape)   e.g. (B, T, 7, 7, 3) or (B, T, 147)
    output: (batch, seq_len, hidden_size)

batch is the number of sequences, seq_len the truncation length used for
truncated BPTT (paper Section 3.1). While collecting rollouts, seq_len = 1.

Run with:
    python models/feature_extractor.py
"""

import torch
import torch.nn as nn


def flatten_obs(x):
    """(batch, seq_len, *obs_shape) -> (batch, seq_len, input_size), as float.

    MiniGrid returns the observation as a uint8 array of shape (7, 7, 3), so
    the trailing dimensions are flattened into one feature vector of 147.
    """
    return x.reshape(x.shape[0], x.shape[1], -1).float()


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
INPUT_SIZE = 7 * 7 * 3
HIDDEN_SIZE = 64
N_LAYERS = 3
N_WORKERS = 2  # W how many games are played at the same time
WORKER_STEPS = 10  # T,	how many steps each game plays before training
BATCH_SIZE = N_WORKERS * WORKER_STEPS  # total steps collected this iteration

SEQ_LEN = 5  # max(#T), the truncation length for truncated BPTT
NUM_SEQUENCES = BATCH_SIZE // SEQ_LEN  # sequences the batch splits into


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def memory_test(model, x, name):
    """Change the observation at t=0 and report which outputs move.

    A memoryless encoder only changes at t=0. A recurrent one changes at
    every later timestep too, because the altered observation is still
    present in its hidden state. This is what the MiniGrid Memory task needs.
    """
    x_changed = x.clone()
    x_changed[:, 0] += 1  # perturb the first observation only

    with torch.no_grad():
        delta = (model(x) - model(x_changed)).abs().sum(dim=-1)  # (batch, seq_len)

    moved = (delta > 1e-6).any(dim=0)  # per timestep, over the batch
    marks = "  ".join("CHANGED" if m else "  same " for m in moved)
    print(f"  {name:<5} {marks}")


def main():
    print(f"obs_shape={OBS_SHAPE}  input_size={INPUT_SIZE}  hidden_size={HIDDEN_SIZE}")
    print(
        f"workers={N_WORKERS} x worker_steps={WORKER_STEPS}  ->  batch_size={BATCH_SIZE}"
    )
    print(f"split into num_sequences={NUM_SEQUENCES} x seq_len={SEQ_LEN}\n")

    # one batch of sequences, exactly as MiniGrid delivers it
    x = torch.randint(0, 11, (NUM_SEQUENCES, SEQ_LEN, *OBS_SHAPE), dtype=torch.uint8)
    flatten_x = flatten_obs(x)
    print("flatten_x shape:", flatten_x.shape)
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
