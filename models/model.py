"""
Actor-critic network: one shared feature extractor, two linear heads.

The feature extractor is handed in already built, so the same Network works
with MLP, LSTM or GRU from feature_extractor.py without a single change. That
is what makes the "With Recurrence" vs "No Recurrence" ablation of the RPPO
paper (Figure 6) a one-line swap.

    obs  (batch, seq_len, 7, 7, 3)
              |
      feature_extractor            MLP / LSTM / GRU
              |
       (batch, seq_len, hidden_size)
          /              \\
    fc_actor           fc_critic
    (n_actions)           (1)
          |                 |
    Categorical          value

Both heads read the SAME features, which is what "shared" means: one encoder
is trained by the sum of the policy loss and the value loss.

Run with:
    python models/model.py
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

try:  # when imported from the repo root as models.model
    from models.feature_extractor import MLP, LSTM, GRU
except ImportError:  # when run directly: python models/model.py
    from feature_extractor import MLP, LSTM, GRU


class Network(nn.Module):
    """Shared-encoder actor-critic.

    feature_extractor : an already initialized MLP, LSTM or GRU
    hidden_size       : the size that extractor outputs, so the heads match it
    n_actions         : size of the discrete action space (7 for MiniGrid)

    The extractor is stored as a submodule, so its parameters show up in
    self.parameters() and one optimizer trains the encoder and both heads.
    """

    def __init__(self, feature_extractor, hidden_size, n_actions):
        super().__init__()

        self.feature_extractor = feature_extractor

        # MLP.forward takes only x, LSTM/GRU.forward also take a hidden state
        self.is_recurrent = isinstance(feature_extractor, (LSTM, GRU))

        self.fc_actor = nn.Linear(hidden_size, n_actions)
        self.fc_critic = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        """(batch, seq_len, *obs_shape) -> (Categorical, value).

        hidden is h_0 for a GRU, (h_0, c_0) for an LSTM, and ignored by the
        MLP. value is squeezed from (batch, seq_len, 1) to (batch, seq_len)
        so it lines up with the rewards.
        """
        if self.is_recurrent:
            features = self.feature_extractor(x, hidden)
        else:
            features = self.feature_extractor(x)

        logits = self.fc_actor(features)  # (batch, seq_len, n_actions)
        value = self.fc_critic(features).squeeze(-1)  # (batch, seq_len)

        # a distribution, not raw logits: PPO needs log_prob() and entropy(),
        # and dist.logits gives the raw numbers back if they are ever wanted
        return Categorical(logits=logits), value


OBS_SHAPE = (7, 7, 3)  # MiniGrid-MemoryS11-v0 partial observation
INPUT_SIZE = 7 * 7 * 3
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

    x = torch.randint(0, 11, (NUM_SEQUENCES, SEQ_LEN, *OBS_SHAPE), dtype=torch.uint8)

    extractors = [
        ("MLP", MLP(INPUT_SIZE, HIDDEN_SIZE, N_LAYERS)),
        ("LSTM", LSTM(INPUT_SIZE, HIDDEN_SIZE)),
        ("GRU", GRU(INPUT_SIZE, HIDDEN_SIZE)),
    ]

    print("FORWARD PASS")
    for name, extractor in extractors:
        net = Network(extractor, HIDDEN_SIZE, N_ACTIONS)
        dist, value = net(x)
        action = dist.sample()
        print("action:", action)

        print(
            f"  {name:<5} recurrent={str(net.is_recurrent):<5} {n_params(net):>7,} params"
        )
        print(f"        logits   {tuple(dist.logits.shape)}")
        print(f"        value    {tuple(value.shape)}")
        print(
            f"        action   {tuple(action.shape)}  log_prob {tuple(dist.log_prob(action).shape)}"
        )

    # the recurrent ones can be continued from a hidden state (truncated BPTT)
    print("\nWITH AN INITIAL HIDDEN STATE")
    h0 = torch.zeros(1, NUM_SEQUENCES, HIDDEN_SIZE)
    c0 = torch.zeros(1, NUM_SEQUENCES, HIDDEN_SIZE)

    lstm_net = Network(LSTM(INPUT_SIZE, HIDDEN_SIZE), HIDDEN_SIZE, N_ACTIONS)
    gru_net = Network(GRU(INPUT_SIZE, HIDDEN_SIZE), HIDDEN_SIZE, N_ACTIONS)

    print(f"  LSTM  value {tuple(lstm_net(x, (h0, c0))[1].shape)}")
    print(f"  GRU   value {tuple(gru_net(x, h0)[1].shape)}")

    # both heads read the same features, so one backward pass trains everything
    print("\nSHARED ENCODER CHECK (GRU)")
    dist, value = gru_net(x)
    loss = -dist.log_prob(dist.sample()).mean() + value.pow(2).mean()
    loss.backward()

    got_grad = [n for n, p in gru_net.named_parameters() if p.grad is not None]
    print(f"  parameters receiving a gradient: {len(got_grad)}")
    for n in got_grad:
        print(f"    {n}")


if __name__ == "__main__":
    main()
