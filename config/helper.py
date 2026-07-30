"""
Helper: the things that are DECIDED BY the config but are not PPO itself.

Config inherits from this class, so every one of these is reachable as
config.something and can read config's own attributes directly:

    config.device              cuda if there is one, else cpu
    config.is_recurrent        True for LSTM and GRU, False for MLP
    config.is_lstm             True only for LSTM (it has a cell state too)
    config.build_extractor()   the encoder named by config.recurrent_model
    config.zero_hidden()       h_0 (and c_0) full of zeros
    config.reset_hidden_of()   zero the hidden state of ONE worker

None of this knows what a rollout, an advantage or a ratio is. Swapping GRU
for LSTM is a change here and in the config, never in the agent.
"""

import torch

from models.feature_extractor import MLP, LSTM, GRU


class Helper:
    """Builders and small utilities shared by anything that reads the config."""

    # ------------------------------------------------------------------
    # what the config implies
    # ------------------------------------------------------------------
    @property
    def device(self):
        """cuda when a GPU exists, cpu otherwise. This machine has no GPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_recurrent(self):
        """Does the encoder carry a hidden state between timesteps?"""
        return self.recurrent_model.upper() in ("LSTM", "GRU")

    @property
    def is_lstm(self):
        """An LSTM needs (h, c). A GRU needs only h. An MLP needs neither."""
        return self.recurrent_model.upper() == "LSTM"

    # ------------------------------------------------------------------
    # builders
    # ------------------------------------------------------------------
    def build_extractor(self):
        """MLP / LSTM / GRU, picked by self.recurrent_model.

        All three take (batch, seq_len, 7, 7, 3) and give back
        (batch, seq_len, hidden_size), so the agent never has to care
        which one it got.
        """
        name = self.recurrent_model.upper()

        if name == "MLP":
            return MLP(self.input_size, self.hidden_size, self.n_layers_mlp)
        if name == "LSTM":
            return LSTM(self.input_size, self.hidden_size)
        if name == "GRU":
            return GRU(self.input_size, self.hidden_size)

        raise ValueError(f"unknown recurrent_model {self.recurrent_model!r}")

    # ------------------------------------------------------------------
    # hidden state
    # ------------------------------------------------------------------
    def zero_hidden(self):
        """h_0 (and c_0) of shape (1, batch_size, hidden_size), None for MLP.

        The leading 1 is num_layers * num_directions -- NOT the batch. That
        stays 1 because both encoders are single-layer and one-directional.
        batch_size defaults to n_workers, which is what the rollout needs.

        Zeros are what the paper starts every episode from (Section 6.4 calls
        this "naive" but uses it anyway; learnable initial states are hard
        because the gradient is truncated).
        """
        if not self.is_recurrent:
            return None

        h = torch.zeros(1, self.n_workers, self.hidden_size, device=self.device)
        return (h, h.clone()) if self.is_lstm else h

    def reset_hidden_of(self, hidden, w):
        """Zero the hidden state of worker w only, in place.

        Called when worker w's game ends. The other workers are still in the
        middle of their own games and must keep remembering. Column w is the
        worker axis of (1, n_workers, hidden_size).

        Returns hidden so the caller can write  h = config.reset_hidden_of(h, w).
        """
        if not self.is_recurrent:
            return hidden

        if self.is_lstm:
            hidden[0][:, w] = 0.0  # h
            hidden[1][:, w] = 0.0  # c
        else:
            hidden[:, w] = 0.0

        return hidden
