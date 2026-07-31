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
shape the paper calls the batch: batch_size = W * T. The one exception is
values, which gets a T+1-th column: the bootstrap V(s_T) that GAE needs for
the very last step.

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
             recurrent_model, n_layers_mlp, gamma, gae_lambda.
    """

    def __init__(self, config, seed=None):
        self.config = config
        self.seed = config.set_seed(seed=seed)

        self.n_workers = config.n_workers  # W
        self.worker_steps = config.worker_steps  # T
        self.hidden_size = config.hidden_size

        self.gamma = config.gamma  # discount
        self.gae_lambda = config.gae_lambda  # GAE bias/variance knob

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
            values     (W, T+1)                the critic's guess, one column
                                               LONGER: index t is V(s_t) for
                                               t = 0..T-1, and index T is the
                                               bootstrap V(s_T)
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
            # T + 1: delta_t needs V(s_t) AND V(s_(t+1)). For t < T-1 the next
            # value is the one the next loop iteration writes, but the last
            # step has no next iteration -- column T is filled after the loop.
            "values": torch.zeros(W, T + 1),
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

        # the bootstrap. The loop ran T times, so it produced V(s_0)..V(s_(T-1))
        # -- T values for T+1 states. self.obs is s_T and no forward pass has
        # ever seen it. It has to be evaluated NOW, with the weights that
        # collected this rollout: the next sample() would compute it too, but
        # only after the optimizer has moved them, and GAE needs V_old.
        #
        # Where a worker finished exactly at t = T-1, self.obs is already the
        # reset game and self.hidden was zeroed above, so this value belongs to
        # the wrong episode -- harmless, because (1 - dones[w, T-1]) is 0 and
        # kills the whole term.
        # obs_t1 = s_(t+1) of the last step, i.e. s_T. NOT the loop's obs_t:
        # that one was s_t, read at the top of every iteration. self.obs has
        # been overwritten T times since (line 170), so what it holds now is
        # one past the end of the loop. Same expression, different content.
        obs_t1 = torch.from_numpy(self.obs).to(self.config.device).unsqueeze(1)
        with torch.no_grad():
            # dist is dropped: no action is ever taken from s_T, the rollout is
            # over. This call asks what the state is WORTH, it does not step.
            #
            # the returned hidden is dropped too. self.hidden must stay the
            # state that has consumed up to s_(T-1), because the next rollout
            # feeds s_T through the model at its t = 0. Store it here and that
            # forward would run s_T with a hidden that already consumed s_T --
            # the recurrence double-counts one step at every rollout boundary.
            _, last_value, _ = self.model(obs_t1, self.hidden)
        buf["values"][:, T] = last_value[:, 0].cpu()

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

    # ------------------------------------------------------------------
    # from rollout to advantages
    # ------------------------------------------------------------------
    def gae(self, buf):
        """Generalized Advantage Estimation. (W, T+1) values -> (W, T) targets.

        Run ONCE per rollout, before any update: the advantage says how much
        better an action turned out than the critic expected, measured under
        the policy that took it. Recomputing it mid-training would measure it
        under weights that did not collect the data.

        No gradient anywhere. Every input was written under torch.no_grad in
        sample(), so these are plain numbers and both outputs are constants --
        which is what the PPO objective needs them to be: advantages weight the
        ratio, returns are the critic's regression target. Neither may drag a
        graph into the loss.

            delta_t = r_t + gamma * V(s_(t+1)) * (1 - done_t) - V(s_t)
            adv_t   = delta_t + gamma * lam * (1 - done_t) * adv_(t+1)
            ret_t   = adv_t + V(s_t)

        delta_t is the one-step surprise. GAE is the discounted sum of all the
        surprises from t onwards, lam deciding how far it looks: lam = 0 keeps
        delta_t alone, lam = 1 sums them all and becomes the Monte Carlo return
        minus the baseline.

        Returns (advantages, returns), both (W, T), NOT normalized -- that is a
        per-minibatch step later. Put them into buf before split_pad_mask:

            buf["advantages"], buf["returns"] = agent.gae(buf)
        """
        W, T = self.n_workers, self.worker_steps

        V = buf["values"]  # (W, T+1), column T is the bootstrap
        rewards = buf["rewards"]  # (W, T)
        dones = buf["dones"]  # (W, T)

        advantages = torch.zeros(W, T)

        # adv_(T), the term the last step inherits: 0. Nothing is known past
        # the end of the buffer, so the sum simply stops there.
        last_adv = torch.zeros(W)

        # backwards, because adv_t is built from adv_(t+1). Forwards would need
        # the future before it exists.
        for t in reversed(range(T)):
            # one factor, two jobs, and both are needed at an episode boundary:
            #   - it drops V(s_(t+1)), which belongs to the NEXT episode after
            #     the env reset, not to this one. A finished episode is worth
            #     exactly its own last reward and nothing beyond.
            #   - it drops adv_(t+1), so the recursion cannot leak an advantage
            #     backwards across the boundary into a different game.
            not_done = 1.0 - dones[:, t]

            delta = rewards[:, t] + self.gamma * V[:, t + 1] * not_done - V[:, t]
            last_adv = delta + self.gamma * self.gae_lambda * not_done * last_adv
            advantages[:, t] = last_adv

        # V[:, :T] and not V: the bootstrap column is a state, never a step,
        # and there is no advantage to pair it with.
        returns = advantages + V[:, :T]

        return advantages, returns

    # ------------------------------------------------------------------
    # from rollout to training sequences
    # ------------------------------------------------------------------
    def split_pad_mask(self, buf):
        """(W, T, ...) -> (n_seq, L, ...): cut at every done, then zero pad.

        A sequence is one stretch of steps the encoder may run through in a
        single call. sample() wipes a worker's hidden state on done, so a
        done has to end the sequence too -- otherwise backprop would carry
        the old maze into the new one and undo that reset.

        So a worker with T = 7 and a done at t = 3 gives TWO sequences, of
        length 4 and 3, not one of length 7.

        Sequences therefore have different lengths. They are padded with
        zeros up to the longest one, L, and mask says which slots are real:

            mask (n_seq, L)   1.0 = real step, 0.0 = padding

        Every loss must be reduced as (loss * mask).sum() / mask.sum() and
        never .mean(), or the padding is trained on. mask.sum() is always
        W * T, because padding adds slots but never steps.

        Returns a dict with every buffer key reshaped to (n_seq, L, ...),
        plus:

            mask   (n_seq, L)
            hxs    (1, n_seq, H)   h_0 of each sequence
            cxs    (1, n_seq, H)   c_0 as well, LSTM only

        Note hxs changes meaning here: in buf it is one state PER STEP, here
        it is one state PER SEQUENCE -- the state that sequence starts from.
        """
        W, T, H = self.n_workers, self.worker_steps, self.hidden_size

        # (worker, first step, last step) of every unbroken stretch
        segments = []
        for w in range(W):
            start = 0
            for t in range(T):
                # a done ends the stretch, and so does running out of buffer
                if buf["dones"][w, t] > 0.5 or t == T - 1:
                    segments.append((w, start, t))
                    start = t + 1

        lengths = [stop - start + 1 for _, start, stop in segments]
        n_seq, L = len(segments), max(lengths)

        # each key keeps its trailing dims and its dtype: obs stays uint8
        # and (7, 7, 3), actions stay long, the rest stay float
        keys = [k for k in buf if k not in ("hxs", "cxs")]
        out = {
            k: torch.zeros(n_seq, L, *buf[k].shape[2:], dtype=buf[k].dtype)
            for k in keys
        }
        out["mask"] = torch.zeros(n_seq, L)

        if self.config.is_recurrent:
            out["hxs"] = torch.zeros(1, n_seq, H)
            if self.config.is_lstm:
                out["cxs"] = torch.zeros(1, n_seq, H)

        for i, (w, start, stop) in enumerate(segments):
            n = stop - start + 1

            for k in keys:
                out[k][i, :n] = buf[k][w, start : stop + 1]
            out["mask"][i, :n] = 1.0  # the remaining L - n slots stay 0.0

            if self.config.is_recurrent:
                # the state stored BEFORE the first step of this sequence.
                # right after a done that is zeros, because reset_hidden_of
                # wiped it -- the reset reaches training for free.
                out["hxs"][0, i] = buf["hxs"][w, start]
                if self.config.is_lstm:
                    out["cxs"][0, i] = buf["cxs"][w, start]

        return out

    def close(self):
        for env in self.envs:
            env.close()
