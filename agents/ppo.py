"""
PPO agent -- sampling stage only.

This file covers one thing: filling the rollout buffer. The agent plays
W games in parallel for T steps each and writes down what happened. No
advantage, no loss, no optimizer yet.

    for t in 0 .. T-1:
        dist, value, hidden = network(obs, hidden)   one step, seq_len = 1
        action = dist.sample()                       sampled, never argmax
        obs, reward, done = env.step(action)
        if done: env.reset() and set that worker's hidden state to zero

Everything is stored as (n_workers, worker_steps, ...), which is the raw
shape the paper calls the batch: batch_size = W * T.

Two rules the loop must obey:

  1. seq_len is 1 while acting. The agent only knows the present step, so
     it feeds one observation and carries the hidden state forward by hand.
  2. done resets the hidden state OF THAT WORKER ONLY. A new game must not
     remember the maze of the old one.

The entry point is main.py in the repo root: it builds the Config and hands
it to PPOAgent.
"""

import gymnasium as gym
import minigrid  # noqa: F401  -- this import is what registers the MiniGrid envs
import numpy as np
import torch
from torch.distributions import Categorical

from models.feature_extractor import MLP, LSTM, GRU
from models.model import Network


class PPOAgent:
    """Collects rollouts from n_workers parallel MiniGrid games.

    config : a Config instance, built in main.py. Uses n_workers,
             worker_steps, name_env, input_size, hidden_size,
             recurrent_model, n_layers_mlp.
    """

    def __init__(self, config, seed=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = self.config.set_seed(seed=seed)

        self.n_workers = self.config.n_workers  # W
        self.worker_steps = self.config.worker_steps  # T

        # one independent game per worker, each with its own layout and cue
        self.envs = [gym.make(self.config.name_env) for _ in range(self.n_workers)]
        self.n_actions = self.envs[0].action_space.n  # 7
        self.obs_shape = self.envs[0].observation_space["image"].shape  # (7, 7, 3)

        extractor = self._build_extractor()
        self.model = Network(extractor, self.config.hidden_size, self.n_actions)
        self.model.to(self.device)

        self.is_recurrent = self.model.is_recurrent
        self.is_lstm = isinstance(extractor, LSTM)

        # the observation each worker is currently looking at. It survives
        # between rollouts, because a game usually is not finished when the
        # T steps run out -- the next rollout continues it.
        self.obs = np.zeros((self.n_workers, *self.obs_shape), dtype=np.uint8)
        for w, env in enumerate(self.envs):
            # a different seed per worker, otherwise all W games are identical
            obs, _ = env.reset(seed=self.seed + w)
            self.obs[w] = obs["image"]

        # the hidden state carries over between rollouts for the same reason
        self.hidden = self._zero_hidden()

        # running return and length of the episode each worker is inside
        self.ep_return = np.zeros(self.n_workers, dtype=np.float64)
        self.ep_length = np.zeros(self.n_workers, dtype=np.int64)

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------
    def _build_extractor(self):
        """MLP / LSTM / GRU, picked by config.recurrent_model."""
        name = self.config.recurrent_model.upper()
        if name == "MLP":
            return MLP(
                self.config.input_size, self.config.hidden_size, self.config.n_layers_mlp
            )
        if name == "LSTM":
            return LSTM(self.config.input_size, self.config.hidden_size)
        if name == "GRU":
            return GRU(self.config.input_size, self.config.hidden_size)
        raise ValueError(f"unknown recurrent_model {self.config.recurrent_model!r}")

    def _zero_hidden(self):
        """h_0 (and c_0) of shape (1, n_workers, hidden_size), or None for MLP.

        The leading 1 is num_layers * num_directions, NOT the batch. Zeros are
        what the paper uses at the start of an episode (Section 6.4).
        """
        if not self.is_recurrent:
            return None
        h = torch.zeros(1, self.n_workers, self.config.hidden_size, device=self.device)
        return (h, h.clone()) if self.is_lstm else h

    def _reset_hidden_of(self, w):
        """Zero the hidden state of ONE worker, because its game just ended."""
        if not self.is_recurrent:
            return
        if self.is_lstm:
            self.hidden[0][:, w] = 0.0
            self.hidden[1][:, w] = 0.0
        else:
            self.hidden[:, w] = 0.0

    # ------------------------------------------------------------------
    # one forward step
    # ------------------------------------------------------------------
    def _forward(self, obs, hidden):
        """(W, 1, 7, 7, 3) -> (Categorical, value (W, 1), next hidden).

        Network.forward drops the hidden state, so the two heads are applied
        here directly. That keeps model.py untouched.
        """
        if self.is_recurrent:
            features, hidden = self.model.feature_extractor(
                obs, hidden, return_hidden=True
            )
        else:
            features = self.model.feature_extractor(obs)
            hidden = None

        dist = Categorical(logits=self.model.fc_actor(features))
        value = self.model.fc_critic(features).squeeze(-1)
        return dist, value, hidden

    # ------------------------------------------------------------------
    # the rollout
    # ------------------------------------------------------------------
    def sample(self):
        """Play W x T steps and return the filled buffer.

        Every tensor is (n_workers, worker_steps, ...):

            obs        (W, T, 7, 7, 3) uint8   what the agent saw
            actions    (W, T)          long    what it did
            log_probs  (W, T)                  log pi_old(a), for the PPO ratio
            values     (W, T)                  the critic's guess
            rewards    (W, T)                  what the env paid
            dones      (W, T)                  1.0 where the episode ENDED
            hxs        (W, T, hidden_size)     h entering that step
            cxs        (W, T, hidden_size)     c entering that step (LSTM only)
        """
        W, T, H = self.n_workers, self.worker_steps, self.config.hidden_size

        buf = {
            "obs": torch.zeros(W, T, *self.obs_shape, dtype=torch.uint8),
            "actions": torch.zeros(W, T, dtype=torch.long),
            "log_probs": torch.zeros(W, T),
            "values": torch.zeros(W, T),
            "rewards": torch.zeros(W, T),
            "dones": torch.zeros(W, T),
        }
        if self.is_recurrent:
            buf["hxs"] = torch.zeros(W, T, H)
            if self.is_lstm:
                buf["cxs"] = torch.zeros(W, T, H)

        finished_returns, finished_lengths = [], []

        for t in range(T):
            buf["obs"][:, t] = torch.from_numpy(self.obs)

            # store the hidden state BEFORE the step. This is the state that
            # step t is computed from, so a sequence starting at t can later
            # be replayed from exactly here.
            if self.is_recurrent:
                h = self.hidden[0] if self.is_lstm else self.hidden
                buf["hxs"][:, t] = h[0].cpu()  # (1, W, H) -> (W, H)
                if self.is_lstm:
                    buf["cxs"][:, t] = self.hidden[1][0].cpu()

            # (W, 7, 7, 3) -> (W, 1, 7, 7, 3): seq_len = 1 while acting
            obs_t = torch.from_numpy(self.obs).to(self.device).unsqueeze(1)

            with torch.no_grad():  # sampling never needs gradients
                dist, value, self.hidden = self._forward(obs_t, self.hidden)
                action = dist.sample()  # (W, 1), sampled -> the policy explores
                log_prob = dist.log_prob(action)  # (W, 1)

            buf["actions"][:, t] = action[:, 0].cpu()
            buf["log_probs"][:, t] = log_prob[:, 0].cpu()
            buf["values"][:, t] = value[:, 0].cpu()

            for w, env in enumerate(self.envs):
                obs, reward, terminated, truncated, _ = env.step(action[w, 0].item())
                done = terminated or truncated

                buf["rewards"][w, t] = reward
                buf["dones"][w, t] = float(done)

                self.ep_return[w] += reward
                self.ep_length[w] += 1

                if done:
                    finished_returns.append(self.ep_return[w])
                    finished_lengths.append(self.ep_length[w])
                    self.ep_return[w] = 0.0
                    self.ep_length[w] = 0

                    obs, _ = env.reset()  # a fresh game with a fresh cue
                    self._reset_hidden_of(w)  # so it cannot remember the old one

                self.obs[w] = obs["image"]

        stats = {
            "episodes": len(finished_returns),
            "return_mean": float(np.mean(finished_returns)) if finished_returns else 0.0,
            "length_mean": float(np.mean(finished_lengths)) if finished_lengths else 0.0,
        }
        return buf, stats

    def close(self):
        for env in self.envs:
            env.close()
