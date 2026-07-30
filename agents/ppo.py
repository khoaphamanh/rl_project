"""
PPO agent -- sampling stage only.

This file covers one thing: filling the rollout buffer. The agent plays
W games in parallel for T steps each and writes down what happened. No
advantage, no loss, no optimizer yet.

    for t in 0 .. T-1:
        dist, value, hidden = self.model(obs, hidden)  one step, seq_len = 1
        action = dist.sample()                        sampled, never argmax
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
import numpy as np
import torch

# looks unused, but importing it is what registers MiniGrid-* with gymnasium.
# Delete it and gym.make raises NameNotFound. Do not let a linter remove it.
import minigrid  # noqa: F401

from models.model import Network


class PPOAgent:
    """Collects rollouts from n_workers parallel MiniGrid games.

    config : a Config instance, built in main.py. Uses n_workers,
             worker_steps, name_env, input_size, hidden_size,
             recurrent_model, n_layers_mlp.
    """

    def __init__(self, config, seed=None):
        self.config = config
        self.seed = config.set_seed(seed=seed)

        self.n_workers = config.n_workers  # W
        self.worker_steps = config.worker_steps  # T
        self.hidden_size = config.hidden_size

        # one independent game per worker, each with its own layout and cue
        self.envs = [gym.make(config.name_env) for _ in range(self.n_workers)]
        self.n_actions = self.envs[0].action_space.n  # 7
        self.obs_shape = self.envs[0].observation_space["image"].shape  # (7, 7, 3)

        # the config decides WHICH encoder, the agent only uses it
        self.model = Network(
            config.build_extractor(), self.hidden_size, self.n_actions
        )
        self.model.to(config.device)

        # the observation each worker is currently looking at. It survives
        # between rollouts, because a game usually is not finished when the
        # T steps run out -- the next rollout continues it.
        self.obs = np.zeros((self.n_workers, *self.obs_shape), dtype=np.uint8)
        for w, env in enumerate(self.envs):
            # a different seed per worker, otherwise all W games are identical
            obs, _ = env.reset(seed=self.seed + w)
            self.obs[w] = obs["image"]

        # the hidden state carries over between rollouts for the same reason
        self.hidden = config.zero_hidden()

        # running return and length of the episode each worker is inside
        self.ep_return = np.zeros(self.n_workers, dtype=np.float64)
        self.ep_length = np.zeros(self.n_workers, dtype=np.int64)

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
        W, T, H = self.n_workers, self.worker_steps, self.hidden_size

        buf = {
            "obs": torch.zeros(W, T, *self.obs_shape, dtype=torch.uint8),
            "actions": torch.zeros(W, T, dtype=torch.long),
            "log_probs": torch.zeros(W, T),
            "values": torch.zeros(W, T),
            "rewards": torch.zeros(W, T),
            "dones": torch.zeros(W, T),
        }
        if self.config.is_recurrent:
            buf["hxs"] = torch.zeros(W, T, H)
            if self.config.is_lstm:
                buf["cxs"] = torch.zeros(W, T, H)

        finished_returns, finished_lengths = [], []

        for t in range(T):
            buf["obs"][:, t] = torch.from_numpy(self.obs)

            # store the hidden state BEFORE the step. This is the state that
            # step t is computed from, so a sequence starting at t can later
            # be replayed from exactly here.
            if self.config.is_recurrent:
                h = self.hidden[0] if self.config.is_lstm else self.hidden
                buf["hxs"][:, t] = h[0].cpu()  # (1, W, H) -> (W, H)
                if self.config.is_lstm:
                    buf["cxs"][:, t] = self.hidden[1][0].cpu()

            # (W, 7, 7, 3) -> (W, 1, 7, 7, 3): seq_len = 1 while acting
            obs_t = torch.from_numpy(self.obs).to(self.config.device).unsqueeze(1)

            with torch.no_grad():  # sampling never needs gradients
                # the returned hidden is fed straight back in on the next
                # iteration: that is the agent's memory while it plays
                dist, value, self.hidden = self.model(obs_t, self.hidden)
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
                    # so the new game cannot remember the old one
                    self.hidden = self.config.reset_hidden_of(self.hidden, w)

                self.obs[w] = obs["image"]

        stats = {
            "episodes": len(finished_returns),
            "return_mean": (
                float(np.mean(finished_returns)) if finished_returns else 0.0
            ),
            "length_mean": (
                float(np.mean(finished_lengths)) if finished_lengths else 0.0
            ),
        }
        return buf, stats

    def close(self):
        for env in self.envs:
            env.close()
