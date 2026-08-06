"""
Actor-critic network: one shared feature extractor, two linear heads (actor
logits, critic value) reading the same features. Run: python models/model.py
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

try:  # when imported from the repo root as models.model
    from models.feature_extractor import MLP, LSTM, GRU, CELL_SIZE, random_obs
except ImportError:  # when run directly: python models/model.py
    from feature_extractor import MLP, LSTM, GRU, CELL_SIZE, random_obs


class Network(nn.Module):
    """Shared encoder feeding an actor head and a critic head, trained jointly by one optimizer."""

    def __init__(self, feature_extractor, hidden_size, n_actions):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.is_recurrent = isinstance(feature_extractor, (LSTM, GRU))
        self.fc_actor = nn.Linear(hidden_size, n_actions)
        self.fc_critic = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        """(batch, seq_len, ...) -> (dist, value, hidden)."""
        if self.is_recurrent:
            features, hidden = self.feature_extractor(x, hidden, return_hidden=True)
        else:
            features = self.feature_extractor(x)
            hidden = None

        logits = self.fc_actor(features)
        value = self.fc_critic(features).squeeze(-1)
        # validate_args=False: avoids expensive GPU -> CPU round trip
        return Categorical(logits=logits, validate_args=False), value, hidden


OBS_SHAPE = (7, 7, 3)  # MiniGrid partial observation
INPUT_SIZE = 7 * 7 * CELL_SIZE  # 980 AFTER one-hot, not 147. See flatten_obs.
HIDDEN_SIZE = 64
N_LAYERS = 3
N_ACTIONS = 7  # MiniGrid action space is Discrete(7)

NUM_SEQUENCES = 4
SEQ_LEN = 5


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    print(f"obs_shape={OBS_SHAPE}  hidden_size={HIDDEN_SIZE}  n_actions={N_ACTIONS}")
    print(f"input: ({NUM_SEQUENCES}, {SEQ_LEN}, {OBS_SHAPE})\n")

    x = random_obs(NUM_SEQUENCES, SEQ_LEN)
    extractors = [
        ("MLP", MLP(INPUT_SIZE, HIDDEN_SIZE, N_LAYERS)),
        ("LSTM", LSTM(INPUT_SIZE, HIDDEN_SIZE)),
        ("GRU", GRU(INPUT_SIZE, HIDDEN_SIZE)),
    ]

    print("FORWARD PASS")
    for name, extractor in extractors:
        net = Network(extractor, HIDDEN_SIZE, N_ACTIONS)
        dist, value, hidden = net(x)
        action = dist.sample()

        print(
            f"  {name:<5} recurrent={str(net.is_recurrent):<5} {n_params(net):>7,} params"
        )
        print(f"        logits   {tuple(dist.logits.shape)}")
        print(f"        value    {tuple(value.shape)}")
        print(
            f"        action   {tuple(action.shape)}  log_prob {tuple(dist.log_prob(action).shape)}"
        )
        if hidden is None:
            print(f"        hidden   None")
        elif isinstance(hidden, tuple):
            print(f"        hidden   (h {tuple(hidden[0].shape)}, c {tuple(hidden[1].shape)})")
        else:
            print(f"        hidden   {tuple(hidden.shape)}")

    print("\nWITH AN INITIAL HIDDEN STATE")
    h0 = torch.zeros(1, NUM_SEQUENCES, HIDDEN_SIZE)
    c0 = torch.zeros(1, NUM_SEQUENCES, HIDDEN_SIZE)

    lstm_net = Network(LSTM(INPUT_SIZE, HIDDEN_SIZE), HIDDEN_SIZE, N_ACTIONS)
    gru_net = Network(GRU(INPUT_SIZE, HIDDEN_SIZE), HIDDEN_SIZE, N_ACTIONS)

    print(f"  LSTM  value {tuple(lstm_net(x, (h0, c0))[1].shape)}")
    print(f"  GRU   value {tuple(gru_net(x, h0)[1].shape)}")

    print("\nSHARED ENCODER CHECK (GRU)")
    dist, value, _ = gru_net(x)
    loss = -dist.log_prob(dist.sample()).mean() + value.pow(2).mean()
    loss.backward()

    got_grad = [n for n, p in gru_net.named_parameters() if p.grad is not None]
    print(f"  parameters receiving a gradient: {len(got_grad)}")
    for n in got_grad:
        print(f"    {n}")


if __name__ == "__main__":
    main()
