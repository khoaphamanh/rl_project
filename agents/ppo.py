"""
PPO agent -- rollout collection, GAE, the clipped-surrogate update, evaluation
and the training loop. main.py builds a Config and calls train_agent(); every
other method here is one stage of that pipeline.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.helper import SequenceDataset, Timing, format_clock
from models.model import Network


class PPOAgent:
    """Runs PPO over n_workers parallel MiniGrid envs: rollout collection, GAE, the update, evaluation and the training loop."""

    def __init__(self, config, seed=None):
        self.config = config
        self.seed = config.set_seed(seed=seed)

        self.device = config.device
        self.name_env = config.name_env

        self.n_workers = config.n_workers  # W
        self.worker_steps = config.worker_steps  # T
        self.hidden_size = config.hidden_size
        self.is_recurrent = config.is_recurrent
        self.is_lstm = config.is_lstm
        self.zero_hidden = config.zero_hidden
        self.reset_hidden_of = config.reset_hidden_of
        self.build_env = config.build_env
        self.build_vector_env = config.build_vector_env

        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_eps = config.clip_eps
        self.value_coef = config.value_coef
        self.entropy_coef = config.entropy_coef
        self.max_grad_norm = config.max_grad_norm
        self.n_epochs = config.n_epochs
        self.target_kl = config.target_kl
        self.mini_batch_size = config.mini_batch_size
        self.run_with_batch_size_fallback = config.run_with_batch_size_fallback
        self.logger = None
        self.n_iterations = config.n_iterations
        self.n_iterations_report = config.n_iterations_report
        self.n_eval_episodes = config.n_eval_episodes
        self.eval_seed = config.eval_seed
        self.eval_deterministic = config.eval_deterministic
        self.save_model = config.save_model
        self.eval_history = []  # Learning curve from train_agent()
        self.envs = self.build_vector_env(self.n_workers)
        self.n_actions = self.envs.single_action_space.n
        self.obs_shape = self.envs.single_observation_space.shape
        self.eval_envs = None  # Separate eval environment (built on first use)
        self.model = Network(config.build_extractor(), self.hidden_size, self.n_actions)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr, weight_decay=config.wd
        )
        self.obs, _ = self.envs.reset(
            seed=[self.seed + w for w in range(self.n_workers)]
        )
        self.hidden = self.zero_hidden()
        self.ep_return = np.zeros(self.n_workers, dtype=np.float64)
        self.ep_length = np.zeros(self.n_workers, dtype=np.int64)
        # every clock this run keeps; train_agent restarts it once the setup
        # is done. config.calculate_time=False makes every phase() below a
        # no-op and silences the TIME tables, so the `with` blocks can stay
        # where they are. See Timing in config/helper.py
        self.timing = Timing(self.device, enabled=config.calculate_time)

    # ---- rollout ----
    def sample(self):
        """Play n_workers x worker_steps env steps and return the filled rollout buffer plus episode stats."""
        W, T, H = self.n_workers, self.worker_steps, self.hidden_size

        buf = {
            "obs": torch.zeros(W, T, *self.obs_shape, dtype=torch.uint8),
            "actions": torch.zeros(W, T, dtype=torch.long),
            "log_probs": torch.zeros(W, T),
            # T + 1: GAE needs V(s_t) and V(s_t+1); the last column has no
            # next loop iteration to fill it, so it's set after the loop.
            "values": torch.zeros(W, T + 1),
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

            # store hidden state before the step, so a sequence starting at
            # t can be replayed from here.
            if self.is_recurrent:
                h = self.hidden[0] if self.is_lstm else self.hidden
                buf["hxs"][:, t] = h[0].cpu()  # (1, W, H) -> (W, H)
                if self.is_lstm:
                    buf["cxs"][:, t] = self.hidden[1][0].cpu()

            # (W, 7, 7, 3) -> (W, 1, 7, 7, 3): seq_len = 1 while acting
            obs_t = torch.from_numpy(self.obs).to(self.device).unsqueeze(1)

            with self.timing.phase("act_forward"):
                with torch.no_grad():
                    # returned hidden feeds back in next iteration: this is
                    # the agent's memory while it plays
                    dist, value, self.hidden = self.model(obs_t, self.hidden)
                    action = dist.sample()  # (W, 1)
                    log_prob = dist.log_prob(action)  # (W, 1)

            with self.timing.phase("act_to_host"):
                # one device->host copy here, then the loop below indexes it.
                # Not action[w, 0].item() per worker: each .item() on a GPU is
                # its own sync/round-trip, and there are W*T of them per
                # rollout.
                actions = action[:, 0].cpu()
                buf["actions"][:, t] = actions
                actions = actions.numpy()

                buf["log_probs"][:, t] = log_prob[:, 0].cpu()
                buf["values"][:, t] = value[:, 0].cpu()

            # Two clocks: "env_step" is just envs.step() (incl. autoreset of
            # finished workers); "env_loop" adds the python around it (buffer
            # writes, bookkeeping). If env_loop >> env_step, the fix belongs
            # in this file rather than in how the envs are stepped. Neither
            # syncs: nothing below queues GPU work.
            with self.timing.phase("env_loop", W, sync=False):
                # One call steps all W envs; shapes are (W,) regardless of
                # whether the vector env is sync or async.
                with self.timing.phase("env_step", W, sync=False):
                    obs, rewards, terminations, truncations, _ = self.envs.step(actions)

                dones = np.logical_or(terminations, truncations)

                buf["rewards"][:, t] = torch.from_numpy(rewards).float()
                buf["dones"][:, t] = torch.from_numpy(dones).float()

                self.ep_return += rewards
                self.ep_length += 1

                for w in np.flatnonzero(dones):
                    finished_returns.append(self.ep_return[w])
                    finished_lengths.append(self.ep_length[w])
                    self.ep_return[w] = 0.0
                    self.ep_length[w] = 0
                    self.hidden = self.reset_hidden_of(self.hidden, w)

                self.obs = obs

        # bootstrap value for the final state s_T
        obs_t1 = torch.from_numpy(self.obs).to(self.device).unsqueeze(1)
        with torch.no_grad():
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

    def gae(self, buf):
        """Generalized Advantage Estimation: values -> standardized advantages and returns."""
        W, T = self.n_workers, self.worker_steps
        V = buf["values"]
        rewards = buf["rewards"]
        dones = buf["dones"]
        advantages = torch.zeros(W, T)
        last_adv = torch.zeros(W)

        for t in reversed(range(T)):
            not_done = 1.0 - dones[:, t]
            delta = rewards[:, t] + self.gamma * V[:, t + 1] * not_done - V[:, t]
            last_adv = delta + self.gamma * self.gae_lambda * not_done * last_adv
            advantages[:, t] = last_adv

        returns = advantages + V[:, :T]
        # normalize over the whole rollout, not per minibatch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def split_pad_mask(self, buf):
        """Cut rollout at done, pad to rectangle, build mask: (W, T, ...) -> (n_seq, L, ...)."""
        W, T, H = self.n_workers, self.worker_steps, self.hidden_size
        segments = []
        for w in range(W):
            start = 0
            for t in range(T):
                if buf["dones"][w, t] > 0.5 or t == T - 1:
                    segments.append((w, start, t))
                    start = t + 1

        lengths = [stop - start + 1 for _, start, stop in segments]
        n_seq, L = len(segments), max(lengths)
        keys = [k for k in buf if k not in ("hxs", "cxs")]
        out = {
            k: torch.zeros(n_seq, L, *buf[k].shape[2:], dtype=buf[k].dtype)
            for k in keys
        }
        out["mask"] = torch.zeros(n_seq, L)

        if self.is_recurrent:
            out["hxs"] = torch.zeros(1, n_seq, H)
            if self.is_lstm:
                out["cxs"] = torch.zeros(1, n_seq, H)

        for i, (w, start, stop) in enumerate(segments):
            n = stop - start + 1
            for k in keys:
                out[k][i, :n] = buf[k][w, start : stop + 1]
            out["mask"][i, :n] = 1.0

            if self.is_recurrent:
                out["hxs"][0, i] = buf["hxs"][w, start]
                if self.is_lstm:
                    out["cxs"][0, i] = buf["cxs"][w, start]

        return out

    @staticmethod
    def masked_mean(x, mask):
        """Mean over unmasked slots only."""
        return (x * mask).sum() / mask.sum().clamp(min=1.0)

    def clip_loss(self, new_log_probs, old_log_probs, advantages, mask):
        """PPO clipped surrogate loss. Returns (loss, {clip_fraction, approx_kl})."""
        eps = self.clip_eps
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
        loss = -self.masked_mean(torch.min(unclipped, clipped), mask)

        with torch.no_grad():
            clip_fraction = self.masked_mean((ratio - 1.0).abs().gt(eps).float(), mask)
            # Schulman's k3 KL estimator
            approx_kl = self.masked_mean((ratio - 1.0) - log_ratio, mask)

        info = {
            "clip_fraction": float(clip_fraction),
            "approx_kl": float(approx_kl),
        }
        return loss, info

    def minibatch_loss(self, mb, model=None):
        """Replay one minibatch with grad on: policy + value_coef*value +
        entropy terms combined into one scalar. Returns (loss, info). `model`
        defaults to this agent's; Helper.probe_batch_size passes a throwaway
        copy so its probe never touches the real weights."""
        if model is None:
            model = self.model

        mask = mb["mask"].to(self.device)

        # h_0 per sequence (split_pad_mask changed hxs from per-step to
        # per-sequence), already detached since sample() wrote it under
        # no_grad. unsqueeze(0) restores the (num_layers, mb, H) shape nn.GRU
        # wants, which SequenceDataset stripped for collation.
        if self.is_lstm:
            hidden = (
                mb["hxs"].to(self.device).unsqueeze(0),
                mb["cxs"].to(self.device).unsqueeze(0),
            )
        elif self.is_recurrent:
            hidden = mb["hxs"].to(self.device).unsqueeze(0)
        else:
            hidden = None

        # returned hidden is dropped (meaningless past the padding, and the
        # next rollout carries its own self.hidden). Safe because padding
        # sits at the end of each sequence, so no real output ever depends on
        # a padded step.
        dist, values, _ = model(mb["obs"].to(self.device), hidden)

        # log pi_new of the action actually taken; only the weights differ
        # from the log_probs stored in the buffer
        new_log_probs = dist.log_prob(mb["actions"].to(self.device))  # (n, L)

        # 1. policy
        policy_loss, info = self.clip_loss(
            new_log_probs,
            mb["log_probs"].to(self.device),
            mb["advantages"].to(self.device),
            mask,
        )

        # 2. critic. Only term reaching fc_critic -- drop it and those params
        # never get a gradient.
        value_loss = self.masked_mean((values - mb["returns"].to(self.device)).pow(2), mask)

        # 3. entropy bonus. log 7 = 1.95 for a uniform policy over MiniGrid's
        # 7 actions, 0 for deterministic.
        entropy = self.masked_mean(dist.entropy(), mask)

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # detach before float(): only loss may keep the graph alive
        info.update(
            {
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                "loss": float(loss.detach()),
            }
        )
        return loss, info

    # ---- training loop ----
    def train(self):
        """One PPO iteration: sample() -> gae() -> split_pad_mask() -> learn() for n_epochs. Returns a flat stats dict."""
        # "iteration" is the outer clock; the phases inside it should sum to
        # just under it, and the gap is untimed python.
        with self.timing.phase("iteration"):
            # ---- collect ----
            with self.timing.phase("sample"):
                buf, stats = self.sample()
            with self.timing.phase("gae"):
                buf["advantages"], buf["returns"] = self.gae(buf)
            with self.timing.phase("split_pad_mask"):
                batch = self.split_pad_mask(buf)

            # ---- learn, at the largest minibatch that fits ----
            # Update is a callable so run_with_batch_size_fallback can retry at
            # a smaller size on OOM. mini_batch_size is already a single number
            # by now (train_agent probed it before iteration 0), so this only
            # has something to fall back to if a rollout turns out bigger than
            # the probe.
            self.mini_batch_size, (epochs_run, logs) = self.run_with_batch_size_fallback(
                lambda mini_batch_size: self.learn(batch, mini_batch_size),
                self.mini_batch_size,
                self.logger,
                what="update minibatch size",
            )

            # recorded because it's a hyperparameter the machine chose, which
            # can differ across machines for the same config
            stats["mini_batch_size"] = self.mini_batch_size
            stats["epochs_run"] = epochs_run
            stats["updates"] = len(logs)
            for k in logs[0]:
                stats[k] = float(np.mean([d[k] for d in logs]))

        return stats

    def learn(self, batch, mini_batch_size):
        """Run n_epochs of minibatch updates over one rollout, with an optional early stop on target_kl. Returns (epochs_run, logs)."""
        # rebuilt every iteration since each rollout has a different number
        # of sequences. num_workers=0: data is already in-memory tensors.
        loader = DataLoader(
            SequenceDataset(batch),
            batch_size=mini_batch_size,
            shuffle=True,  # reshuffles once per epoch below
            drop_last=False,
        )

        logs = []

        for epoch in range(self.n_epochs):
            for mb in loader:
                # host->device copy is included in "update_fwd" on purpose --
                # it's the same tensors every epoch, so if it dominates the
                # fix is to move the batch once in train(), not per minibatch.
                with self.timing.phase("update_fwd"):
                    loss, info = self.minibatch_loss(mb)

                with self.timing.phase("update_bwd"):
                    self.optimizer.zero_grad()  # torch accumulates otherwise
                    loss.backward()  # one backward for all three terms

                    # clip the gradient norm so one lucky sparse-reward
                    # episode can't wreck the policy in a single step
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                info["grad_norm"] = float(grad_norm)
                logs.append(info)

            # early stop if pi_new drifted too far. Checked between epochs,
            # not mid-epoch, so sequences aren't updated unevenly.
            if self.target_kl is not None and logs[-1]["approx_kl"] > self.target_kl:
                break

        return epoch + 1, logs

    # ---- evaluation ----
    def evaluate(self, n_episodes=None, deterministic=None):
        """Play n_eval_episodes complete episodes with no gradients or training, on private envs; returns success rate, timeout rate, return and length stats."""
        if n_episodes is None:
            n_episodes = self.n_eval_episodes
        if deterministic is None:
            deterministic = self.eval_deterministic
        dev = self.device

        # private envs, one slot per episode in flight. Same builder as the
        # rollout's so wrappers match. Built once and reused (respawning is
        # not free); each wave below reseeds and zeroes hidden state
        # explicitly, so reuse doesn't leak state across calls.
        n_slots = min(n_episodes, self.n_workers)
        if self.eval_envs is None or self.eval_envs.num_envs != n_slots:
            if self.eval_envs is not None:
                self.eval_envs.close()
            self.eval_envs = self.build_vector_env(n_slots)
        envs = self.eval_envs

        returns, lengths, successes, timeouts = [], [], [], []

        # no-op for these encoders (no dropout/batchnorm), but standard
        # contract to honour
        self.model.eval()

        # every phase below is sync=False: this is the innermost loop, and a
        # cuda synchronize() per step would cost more than what it measures.
        # "eval_forward" is counted per call (comparable to act_forward),
        # "eval_step" per env step including dead slots, which really are
        # stepped (comparable to env_step).
        for first in range(0, n_episodes, n_slots):
            # fixed seed per episode index, so episode i is reproducible
            # across calls. Clip pins a short final wave's surplus slots to
            # the last episode; `counts` drops the duplicate.
            episode = np.minimum(first + np.arange(n_slots), n_episodes - 1)

            # units=0: a reset is not an env step, but its cost belongs to the
            # env rather than to the model
            with self.timing.phase("eval_step", 0, sync=False):
                obs, _ = envs.reset(seed=[self.eval_seed + int(i) for i in episode])

            # zeroed here so an episode never starts with prior memory
            hidden = self.zero_hidden(batch_size=n_slots)

            # a slot is live until its episode ends; output from a dead slot
            # (incl. the autoreset episode it's bounced into) is ignored
            live = np.ones(n_slots, dtype=bool)
            counts = first + np.arange(n_slots) < n_episodes

            ep_return = np.zeros(n_slots)
            ep_length = np.zeros(n_slots, dtype=np.int64)

            while live.any():
                with self.timing.phase("eval_forward", sync=False):
                    # (S, 7, 7, 3) -> (S, 1, 7, 7, 3): seq_len 1, same shape as
                    # the rollout uses
                    obs_t = torch.from_numpy(obs).to(dev).unsqueeze(1)

                    with torch.no_grad():
                        dist, _, hidden = self.model(obs_t, hidden)
                        action = (
                            dist.probs.argmax(dim=-1) if deterministic else dist.sample()
                        )

                    # .numpy() reads the tensor, forcing a GPU sync -- kept
                    # inside the timed region so the clock is honest without
                    # paying for an explicit synchronize()
                    actions = action[:, 0].cpu().numpy()

                with self.timing.phase("eval_step", n_slots, sync=False):
                    obs, rewards, terminations, truncations, _ = envs.step(actions)

                # a dead slot's reward and step must not be counted
                ep_return += rewards * live
                ep_length += live

                ending = np.logical_or(terminations, truncations) & live
                for s in np.flatnonzero(ending):
                    if counts[s]:
                        returns.append(float(ep_return[s]))
                        lengths.append(int(ep_length[s]))
                        # MemoryS7 pays 0 for a wrong door
                        successes.append(ep_return[s] > 0.0)
                        timeouts.append(bool(truncations[s] and not terminations[s]))
                    live[s] = False

        self.model.train()

        return {
            "eval_episodes": n_episodes,
            "success_rate": float(np.mean(successes)),
            "timeout_rate": float(np.mean(timeouts)),
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "length_mean": float(np.mean(lengths)),
        }

    # ---- whole run ----
    def train_agent(self, n_iterations=None, n_iterations_report=None, logger=None):
        """Run n_iterations of train(), evaluating and logging every n_iterations_report. Saves a checkpoint at the end and returns the per-iteration history."""
        if n_iterations is None:
            n_iterations = self.n_iterations
        if n_iterations_report is None:
            n_iterations_report = self.n_iterations_report

        # a swap not an addition: logger's StreamHandler already echoes to
        # the terminal
        log = print if logger is None else logger.info

        # so run_with_batch_size_fallback logs to file too, not just stdout
        self.logger = logger

        # settled once, here, against a worst-case minibatch rather than
        # iteration 0's real data -- see Helper.probe_batch_size
        self.mini_batch_size = self.config.probe_batch_size(
            self.model,
            self.minibatch_loss,
            self.obs_shape,
            self.mini_batch_size,
            logger,
        )

        header = (
            f"{'iter':>5} {'eps':>4} {'return':>7} {'len':>6} "
            f"{'entropy':>8} {'v_loss':>9} {'kl':>8} {'clip':>6}"
            f"   |  EVAL  success  timeout   return    std"
        )

        history = []
        eval_history = []  # one entry per report iteration

        # the clocks start here, so building the envs and probing the
        # minibatch size are not billed to the training loop
        self.timing.start()

        for i in range(n_iterations):
            stats = self.train()
            stats["iteration"] = i

            if i == 0:
                # delayed one train() call so the header prints after the
                # model summary / minibatch-size search, not above it.
                # Separate log("") rather than "\n" in the header, so the
                # blank line doesn't inherit the header's timestamp prefix.
                log("")
                log(header)

            # last iteration always reports, so iteration n_iterations-1 is
            # always in eval_history regardless of n_iterations_report
            report = i % n_iterations_report == 0 or i == n_iterations - 1

            if report:
                # one evaluate() call, reused below -- calling it twice could
                # give different numbers (sampling is stochastic unless
                # eval_deterministic) and desync the printed vs stored row.
                # Timed but outside the "iteration" clock: it's the cost of
                # measuring, not part of training.
                with self.timing.phase("evaluate"):
                    evaluation = self.evaluate()

                # startswith guard: eval_episodes is already prefixed
                stats.update(
                    {
                        k if k.startswith("eval_") else f"eval_{k}": v
                        for k, v in evaluation.items()
                    }
                )

                # unprefixed keys here (unlike stats): every entry is an
                # evaluation, no rollout number to collide with
                eval_history.append({"iteration": i, **evaluation})

                # training columns are one noisy rollout; eval_return is the
                # trustworthy number. std is the spread over n_eval_episodes,
                # not an error bar on the mean: returns are bimodal (0 or
                # ~0.9), so std stays pinned near 0.45 whenever success rate
                # is mid-range, regardless of training progress. The actual
                # error bar on the mean is std/sqrt(n_eval_episodes) (~0.06
                # at n=50).
                log(
                    f"{i:>5} {stats['episodes']:>4} {stats['return_mean']:>7.3f} "
                    f"{stats['length_mean']:>6.1f} {stats['entropy']:>8.4f} "
                    f"{stats['value_loss']:>9.4f} {stats['approx_kl']:>8.5f} "
                    f"{stats['clip_fraction']:>6.3f}"
                    f"   |        {stats['eval_success_rate']:>7.2f} "
                    f"{stats['eval_timeout_rate']:>8.2f} "
                    f"{stats['eval_return_mean']:>8.3f}"
                    f" +- {stats['eval_return_std']:<5.3f}"
                )

                # timing since the last report; this empties the window so
                # the next one starts fresh
                self.timing.report(log)

            history.append(stats)

        # after the loop, so the last report() has already folded its window
        # into the whole-run totals
        self.timing.summary(log, self.seed)

        # ---- checkpoint ----
        self.eval_history = eval_history

        if eval_history:
            last = eval_history[-1]

            path = self.save_model(
                self.model,
                self.optimizer,
                iteration=last["iteration"],
                # curve travels with the weights, plain ints/floats so
                # torch.load(weights_only=True) can read it -- see helper.save_model
                eval_history=eval_history,
                # False means measured by sampling, not argmax -- not
                # reproducible to the digit and not comparable to a
                # deterministic curve, see evaluate()
                eval_deterministic=self.eval_deterministic,
                # flat headline numbers for watch_agent's sidebar
                eval_success_rate=last["success_rate"],
                eval_return_mean=last["return_mean"],
                # size that actually fit on this machine, not the config's
                # candidate list
                mini_batch_size=self.mini_batch_size,
            )

            # peak is reported but not saved: the checkpoint must hold the
            # policy the run actually produced, even if a KL spike left it
            # below its best
            peak = max(eval_history, key=lambda e: e["success_rate"])
            log("")
            log(
                f"kept {len(eval_history)} evaluations at iterations "
                f"{[e['iteration'] for e in eval_history]}"
            )
            log(
                f"final   iter {last['iteration']}  "
                f"eval success {last['success_rate']:.2f}   "
                f"(peak was {peak['success_rate']:.2f} at "
                f"iter {peak['iteration']})"
            )
            log(f"saved  {path}")
            log(f"took   {format_clock(self.timing.elapsed)}")

        return history

    def close(self):
        """Close the rollout and eval environments, releasing their subprocesses."""
        self.envs.close()
        if self.eval_envs is not None:
            self.eval_envs.close()
            self.eval_envs = None
