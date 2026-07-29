"""
Feature extractors for Recurrent PPO.

MLP encodes a flat observation into features (RPPO paper Figure 3: the
non-recurrent layers run on the entire batch at once). LSTM and GRU are the
recurrent layer that follows it, and their forward pass returns the output at
every timestep -- not just the last hidden state -- because the policy and
value heads need one prediction per timestep.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Stack of n_layers x (Linear -> ReLU).

    The first layer maps input_size -> hidden_size, every following layer
    maps hidden_size -> hidden_size.

    Input : (batch, input_size)
    Output: (batch, hidden_size)
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
        return self.layers(x)


class LSTM(nn.Module):
    """Single-layer LSTM returning only its output.

    Input : x (batch, seq_len, input_size), optional hidden state
    Output: (batch, seq_len, hidden_size)
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x, hidden=None):
        output, _ = self.lstm(x, hidden)
        return output


class GRU(nn.Module):
    """Single-layer GRU returning only its output.

    Input : x (batch, seq_len, input_size), optional hidden state
    Output: (batch, seq_len, hidden_size)
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x, hidden=None):
        output, _ = self.gru(x, hidden)
        return output


INPUT_SIZE = 7 * 7 * 3  # flattened MiniGrid observation
HIDDEN_SIZE = 64
N_LAYERS = 3
NUM_SEQ = 4  # sequences per batch
SEQ_LEN = 5  # truncation length
BATCH_SIZE = NUM_SEQ * SEQ_LEN


def main():
    """Run one forward pass through the whole encoder, the way RPPO does it:
    the MLP sees the flat batch, then the batch is reshaped to sequences for
    the recurrent layer (paper Section 3.1).

    Run with:
        python models/feature_extractor.py
    """
    print(f"input_size={INPUT_SIZE}  hidden_size={HIDDEN_SIZE}  n_layers={N_LAYERS}")
    print(f"num_seq={NUM_SEQ}  seq_len={SEQ_LEN}  batch_size={BATCH_SIZE}\n")

    # MLP: the entire batch at once, no sequence dimension
    mlp = MLP(INPUT_SIZE, HIDDEN_SIZE, N_LAYERS)
    x = torch.randn(BATCH_SIZE, INPUT_SIZE)
    features = mlp(x)
    print(mlp)
    print(f"  MLP      {tuple(x.shape)} -> {tuple(features.shape)}\n")

    # reshape to sequences before the recurrent layer
    sequences = features.reshape(NUM_SEQ, SEQ_LEN, HIDDEN_SIZE)
    print(f"  reshape  {tuple(features.shape)} -> {tuple(sequences.shape)}\n")

    # h_in of each sequence, taken from the buffer during optimization
    h0 = torch.zeros(1, NUM_SEQ, HIDDEN_SIZE)
    c0 = torch.zeros(1, NUM_SEQ, HIDDEN_SIZE)

    lstm = LSTM(HIDDEN_SIZE, HIDDEN_SIZE)
    print(
        f"  LSTM     {tuple(sequences.shape)} -> {tuple(lstm(sequences, (h0, c0)).shape)}"
    )

    gru = GRU(HIDDEN_SIZE, HIDDEN_SIZE)
    out = gru(sequences, h0)
    print(f"  GRU      {tuple(sequences.shape)} -> {tuple(out.shape)}")

    # back to the flat batch shape, ready for the value and policy heads
    print(
        f"  reshape  {tuple(out.shape)} -> {tuple(out.reshape(BATCH_SIZE, HIDDEN_SIZE).shape)}"
    )


if __name__ == "__main__":
    main()
