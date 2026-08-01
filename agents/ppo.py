"""
PPO agent -- rollout, advantages, loss and update. The whole algorithm.

main.py only ever calls train_agent(). Everything else is a stage of it, one
method each:

  train_agent()       the run: n_iterations of the below, and every
                      n_iterations_report of them an evaluate()

    train()           collect once, then learn from it n_epochs times

      sample()          play W games for T steps, write down what happened
      gae()             turn (rewards, values) into (advantages, returns)
      split_pad_mask()  cut at every done, pad to a rectangle, build the mask
      SequenceDataset   (config/helper.py) wrap that for a torch DataLoader --
                        ONE SAMPLE IS ONE SEQUENCE, so batch_size counts
                        episode fragments, not timesteps
      minibatch_loss()  replay one batch with grad ON, sum the three terms
        clip_loss()       the clipped surrogate, term 1 of 3
        masked_mean()     the only legal reduction once padding exists

    evaluate()        no grad, no training, COMPLETE episodes on a private
                      env. Never share sample()'s envs: those are mid-episode

The rollout itself:

    for t in 0 .. T-1:
        dist, value, hidden = self.model(obs, hidden)  one step, seq_len = 1
        action = dist.sample()                        sampled, never argmax
        obs, reward, done = env.step(action)
        if done: env.reset() and set that worker's hidden state to zero

Everything is stored as (n_workers, worker_steps, ...), which is the raw
shape the paper calls the batch: n_total_steps = W * T. The one exception is
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

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.helper import SequenceDataset
from models.model import Network


class PPOAgent:
    """Collects rollouts from n_workers parallel MiniGrid games.

    config : a Config instance, built in main.py. Every hyperparameter it
             needs is copied into an attribute in __init__ and read from
             there afterwards, so no method ever reaches back through
             self.config -- one place to look for what this agent depends on.
    """

    def __init__(self, config, seed=None):
        self.config = config
        self.seed = config.set_seed(seed=seed)

        self.device = config.device
        self.name_env = config.name_env

        self.n_workers = config.n_workers  # W
        self.worker_steps = config.worker_steps  # T
        self.hidden_size = config.hidden_size

        # which encoder was built, asked once here instead of at every use.
        # These are Helper properties, so reading them re-derives a string
        # comparison -- pointless inside a loop over T timesteps.
        self.is_recurrent = config.is_recurrent
        self.is_lstm = config.is_lstm

        # bound methods, not values: the config still owns the logic of what a
        # hidden state looks like, and of which wrappers an env gets. The agent
        # just gets a short name for each.
        self.zero_hidden = config.zero_hidden
        self.reset_hidden_of = config.reset_hidden_of
        self.build_env = config.build_env

        self.gamma = config.gamma  # discount
        self.gae_lambda = config.gae_lambda  # GAE bias/variance knob

        self.clip_eps = config.clip_eps  # PPO trust region half-width
        self.value_coef = config.value_coef  # weight of the critic term
        self.entropy_coef = config.entropy_coef  # weight of the exploration term
        self.max_grad_norm = config.max_grad_norm

        self.n_epochs = config.n_epochs  # passes over one rollout
        self.mini_batch_size = config.mini_batch_size  # sequences per minibatch
        self.target_kl = config.target_kl  # None = never stop early

        # the run. Defaults only -- train_agent() takes overrides
        self.n_iterations = config.n_iterations  # train() calls
        self.n_iterations_report = config.n_iterations_report  # evaluate() every

        # evaluation. Defaults only -- evaluate() takes overrides
        self.n_eval_episodes = config.n_eval_episodes
        self.eval_seed = config.eval_seed
        self.eval_deterministic = config.eval_deterministic

        # checkpointing. The config owns the path and the file format, the
        # same way it owns the env and the encoder -- this is a bound method,
        # not a path string, so nothing here has to know where it writes.
        self.save_model = config.save_model

        # one independent game per worker, each with its own layout and cue.
        # config.build_env(), never gym.make: the wrappers are part of what the
        # env IS, and evaluate() must build the same thing.
        self.envs = [self.build_env() for _ in range(self.n_workers)]
        self.n_actions = self.envs[0].action_space.n  # 7
        self.obs_shape = self.envs[0].observation_space["image"].shape  # (7, 7, 3)

        # the config decides WHICH encoder, the agent only uses it
        self.model = Network(config.build_extractor(), self.hidden_size, self.n_actions)
        self.model.to(self.device)

        # ONE optimizer over model.parameters(): the encoder and both heads.
        # That is what makes the shared encoder shared -- it is stepped by the
        # sum of all three loss terms, not by any one of them.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        # the observation each worker is currently looking at. It survives
        # between rollouts, because a game usually is not finished when the
        # T steps run out -- the next rollout continues it.
        self.obs = np.zeros((self.n_workers, *self.obs_shape), dtype=np.uint8)
        for w, env in enumerate(self.envs):
            # a different seed per worker, otherwise all W games are identical
            obs, _ = env.reset(seed=self.seed + w)
            self.obs[w] = obs["image"]

        # the hidden state carries over between rollouts for the same reason
        self.hidden = self.zero_hidden()

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
                    self.hidden = self.reset_hidden_of(self.hidden, w)

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
        # been overwritten T times since (the last line of the worker loop,
        # self.obs[w] = obs["image"]), so what it holds now is one past the
        # end of the loop. Same expression, different content.
        obs_t1 = torch.from_numpy(self.obs).to(self.device).unsqueeze(1)
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

        Returns (advantages, returns), both (W, T). advantages come out
        STANDARDIZED (mean 0, std 1 over the whole rollout), returns come out
        raw -- see the two comments at the bottom, the order matters. Put both
        into buf before split_pad_mask:

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
        #
        # Computed FIRST, from the raw advantages, and that order is not
        # optional. returns is the critic's regression target, so it has to
        # stay on the scale of actual discounted reward. Build it out of a
        # rescaled advantage instead and V is being asked to predict a number
        # that no longer means anything.
        returns = advantages + V[:, :T]

        # normalize the ADVANTAGES ONLY.
        #
        # The surrogate multiplies the ratio by A, so A is what sets the size
        # of the policy step. Raw GAE values drift with the reward scale of the
        # task and with how wrong the critic currently is -- early on they are
        # large, later they shrink towards 0 as V catches up. One fixed lr
        # cannot suit both. Standardizing pins the step size, and it changes
        # nothing the policy gradient actually reads: every SIGN and every
        # relative ORDER survives an affine rescale. "Better than average"
        # stays better than average.
        #
        # Over the whole (W, T) rollout, NOT per minibatch. Two reasons:
        #   - here every slot is a real step. After split_pad_mask there is
        #     padding, and a plain .mean()/.std() would average the fake zeros
        #     in -- it would have to be a masked mean and a masked std.
        #   - a recurrent minibatch is a handful of SEQUENCES, some of them
        #     very short. Its std is a noisy estimate of the same quantity,
        #     and dividing each minibatch by a different noisy number makes
        #     the effective step size jitter between minibatches of one epoch.
        #
        # 1e-8 covers the degenerate rollout where every advantage is equal
        # and std is 0 -- on a sparse-reward task that scored nothing at all,
        # which for MemoryS7 is exactly what the first iterations look like.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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

        if self.is_recurrent:
            out["hxs"] = torch.zeros(1, n_seq, H)
            if self.is_lstm:
                out["cxs"] = torch.zeros(1, n_seq, H)

        for i, (w, start, stop) in enumerate(segments):
            n = stop - start + 1

            for k in keys:
                out[k][i, :n] = buf[k][w, start : stop + 1]
            out["mask"][i, :n] = 1.0  # the remaining L - n slots stay 0.0

            if self.is_recurrent:
                # the state stored BEFORE the first step of this sequence.
                # right after a done that is zeros, because reset_hidden_of
                # wiped it -- the reset reaches training for free.
                out["hxs"][0, i] = buf["hxs"][w, start]
                if self.is_lstm:
                    out["cxs"][0, i] = buf["cxs"][w, start]

        return out

    # ------------------------------------------------------------------
    # the loss
    # ------------------------------------------------------------------
    @staticmethod
    def masked_mean(x, mask):
        """Average x over the REAL slots only. (n_seq, L) -> scalar.

        split_pad_mask pads every sequence out to L, so some of the slots are
        zeros that never happened. x.mean() divides by all n_seq * L of them,
        which shrinks every gradient by (real / total) -- and worse, any slot
        whose per-step loss is not zero gets trained on. Padding is not data.

            (x * mask).sum() / mask.sum()

        The multiply deletes the fake entries, the denominator counts only the
        real ones. For a full batch mask.sum() is W * T, never n_seq * L:
        padding adds slots, it never adds steps.

        Every term of the PPO loss goes through here -- policy, value,
        entropy. clamp(min=1) only stops an all-padding input from returning
        nan, which cannot happen for a minibatch with at least one sequence.
        """
        return (x * mask).sum() / mask.sum().clamp(min=1.0)

    def clip_loss(self, new_log_probs, old_log_probs, advantages, mask):
        """PPO's clipped surrogate -- the policy loss. (n_seq, L) -> scalar.

        Called once per minibatch, inside the grad-on region. new_log_probs is
        the ONLY argument with a graph behind it: old_log_probs came from the
        rollout buffer, advantages from gae(), mask from split_pad_mask(), and
        the objective is only the PPO objective if all three are constants.

            ratio_t = pi_new(a_t | s_t) / pi_old(a_t | s_t)
                    = exp(new_log_prob_t - old_log_prob_t)

            L_t     = -min( ratio_t * A_t,
                            clip(ratio_t, 1-eps, 1+eps) * A_t )

        Both log-probs are evaluated at the SAME a_t -- the action the rollout
        actually took, replayed out of the buffer. The ratio asks "how much
        more likely is the current policy to do that again?". 1.0 means not
        changed at all, which is exactly what it is on the first minibatch of
        the first epoch, before the optimizer has stepped once. Useful sanity
        check: there, ratio must be 1.0 and loss must equal -masked_mean(A).

        The min is a PESSIMISTIC bound -- of the two estimates it always takes
        the smaller, which after the minus sign is the larger loss. What it
        does depends on which way the advantage points:

                            A > 0 (do it more)   A < 0 (do it less)
            ratio > 1+eps   clipped              not clipped
            ratio < 1-eps   not clipped          clipped

        Read the pattern down the diagonal: the gradient is cut only when the
        policy has ALREADY moved past eps in the direction the advantage asked
        for. Moving the wrong way is never clipped, so a correction always
        gets through. That is what keeps n_epochs of reuse on one batch from
        riding a single good-looking action all the way to a collapsed policy.

        And a clipped step contributes a CONSTANT to the loss, so its gradient
        is exactly 0. Clipping does not shrink an update, it deletes it.

        Returns (loss, info):
            loss  scalar, with a graph, already negated -- torch.optim
                  descends and the surrogate is something to maximize. Add the
                  value and entropy terms to it, then call backward once.
            info  plain floats for logging, all computed under no_grad:
                  clip_fraction  fraction of real steps that hit a clip. It
                                 climbing during an iteration is normal; ~0
                                 means the lr is too small to move anything,
                                 ~1 means pi_new has run away from pi_old.
                  approx_kl      how far the policy has drifted this
                                 iteration. The usual early-stop signal.
        """
        eps = self.clip_eps

        # subtract in log space, then exp. The buffer stores logs, and this is
        # both cheaper and far more stable than dividing two tiny probabilities.
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages

        loss = -self.masked_mean(torch.min(unclipped, clipped), mask)

        with torch.no_grad():
            # |ratio - 1| > eps is exactly the condition under which the clamp
            # bit, so this is the clipped fraction without re-deriving the min
            clip_fraction = self.masked_mean((ratio - 1.0).abs().gt(eps).float(), mask)

            # Schulman's k3 estimator of KL(pi_old || pi_new): (r - 1) - log r.
            # The naive -mean(log_ratio) is unbiased but can come out negative,
            # which a KL never is; this one cannot, and has far less variance.
            approx_kl = self.masked_mean((ratio - 1.0) - log_ratio, mask)

        info = {
            "clip_fraction": float(clip_fraction),
            "approx_kl": float(approx_kl),
        }
        return loss, info

    def minibatch_loss(self, mb):
        """Replay one minibatch with grad ON and build the total loss.

        mb is a slice of split_pad_mask's output: every tensor is
        (n, L, ...) with n sequences, and hxs/cxs are (1, n, H).

        ONE forward call, seq_len = L -- not a loop over timesteps. The whole
        sequence goes through the encoder at once, which is what lets the
        gradient flow backwards through the recurrence. The truncation in
        "truncated BPTT" is h_0: it comes out of the buffer as a plain number,
        so the graph begins at the first step of this sequence and stops dead
        there. Keep the value, drop the history.

            loss = policy_loss
                 + value_coef   * value_loss
                 - entropy_coef * entropy

        Three terms, one shared encoder, one backward:

            policy   -min(ratio*A, clip(ratio)*A)   do more of what beat V
            value    (V(s) - returns)^2             make V accurate -- A was
                                                    computed FROM V, so a bad
                                                    critic poisons term 1
            entropy  H(pi)                          MINUS in front: entropy is
                                                    wanted HIGH and optim
                                                    descends. Without it the
                                                    policy can collapse onto
                                                    noise before it ever scores

        Returns (loss, info). Only loss carries a graph; info is plain floats.
        """
        mask = mb["mask"].to(self.device)

        # h_0 of each sequence. Already detached by construction -- sample()
        # wrote it under no_grad. Note this is one state PER SEQUENCE here,
        # not one per step: split_pad_mask changed what hxs means.
        #
        # unsqueeze(0): the DataLoader collated these to (mb, H), but nn.GRU
        # wants (num_layers, mb, H). SequenceDataset stripped that leading
        # axis so the collate could work; this puts it back.
        if self.is_lstm:
            hidden = (
                mb["hxs"].to(self.device).unsqueeze(0),
                mb["cxs"].to(self.device).unsqueeze(0),
            )
        elif self.is_recurrent:
            hidden = mb["hxs"].to(self.device).unsqueeze(0)
        else:
            hidden = None

        # the returned hidden is DROPPED: for a padded sequence it is the state
        # after the padding, which means nothing. Nothing continues from here
        # either -- the next rollout carries its own self.hidden.
        #
        # Padding sits at the END of every sequence, which is what makes this
        # safe: the recurrence runs over the real steps first, so no real
        # output was ever computed from a fake one.
        dist, values, _ = self.model(mb["obs"].to(self.device), hidden)

        # log pi_NEW of the action the rollout actually took. Same states, same
        # actions, CURRENT weights -- that is the only difference from the
        # log_probs sitting in the buffer.
        new_log_probs = dist.log_prob(mb["actions"].to(self.device))  # (n, L)

        # 1. the policy
        policy_loss, info = self.clip_loss(
            new_log_probs,
            mb["log_probs"].to(self.device),
            mb["advantages"].to(self.device),
            mask,
        )

        # 2. the critic. returns came from gae() and is a fixed target. This is
        # the ONLY term that reaches fc_critic -- drop it and those two
        # parameters never receive a gradient at all.
        value_loss = self.masked_mean((values - mb["returns"].to(self.device)).pow(2), mask)

        # 3. exploration. log 7 = 1.95 for a uniform policy over MiniGrid's 7
        # actions, 0 for a deterministic one. Watch it fall as training runs.
        entropy = self.masked_mean(dist.entropy(), mask)

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # .detach() before float(): these four are still attached to the graph,
        # and only loss is allowed to keep it. Logging must never hold a graph
        # alive across the training loop.
        info.update(
            {
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                "loss": float(loss.detach()),
            }
        )
        return loss, info

    # ------------------------------------------------------------------
    # the training loop
    # ------------------------------------------------------------------
    def train(self):
        """One full PPO iteration: collect a rollout, then learn from it.

            sample -> gae -> split_pad_mask -> DataLoader -> n_epochs passes

        The rollout is collected ONCE and reused n_epochs times. From the
        second pass onwards the data is off-policy -- pi has moved since it was
        collected -- and surviving that is the entire job of the clipped ratio.

        The DataLoader is the ordinary supervised one, with two differences
        worth knowing:

            batch_size counts SEQUENCES, not timesteps -- see SequenceDataset
            drop_last=False, because n_seq is small and a dropped tail is a
            dropped episode, not a rounding error

        Returns one flat stats dict: sample()'s episode numbers, plus every
        loss term and diagnostic averaged over the updates that actually ran.

        Called once per iteration by train_agent(), which is what main.py
        actually drives. Call it directly only to run a single iteration by
        hand.
        """
        # ---- collect ---------------------------------------------------
        buf, stats = self.sample()
        buf["advantages"], buf["returns"] = self.gae(buf)
        batch = self.split_pad_mask(buf)

        # rebuilt every iteration: a new rollout means a new number of
        # sequences, so this dataset is genuinely a different one each time.
        # num_workers stays 0 -- the data is already tensors in memory, and
        # subprocesses would only add overhead and break the seeding.
        loader = DataLoader(
            SequenceDataset(batch),
            batch_size=self.mini_batch_size,
            shuffle=True,  # reshuffles on every new iteration of the loader,
            #                which is exactly once per epoch below
            drop_last=False,
        )

        logs = []

        # ---- learn -----------------------------------------------------
        for epoch in range(self.n_epochs):
            for mb in loader:
                loss, info = self.minibatch_loss(mb)

                self.optimizer.zero_grad()  # torch accumulates otherwise
                loss.backward()  # ONE backward for all three terms

                # rescale the whole gradient down to max_grad_norm if it is
                # longer. On a sparse-reward task one lucky episode can produce
                # a huge gradient; this stops a single step wrecking a policy
                # that took thousands of iterations to build.
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                info["grad_norm"] = float(grad_norm)
                logs.append(info)

            # optional early stop: pi_new has drifted too far to keep reusing
            # this batch. Checked BETWEEN epochs -- abandoning a pass halfway
            # would update some sequences more often than others.
            if self.target_kl is not None and logs[-1]["approx_kl"] > self.target_kl:
                break

        stats["epochs_run"] = epoch + 1
        stats["updates"] = len(logs)
        for k in logs[0]:
            stats[k] = float(np.mean([d[k] for d in logs]))

        return stats

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def evaluate(self, n_episodes=None, deterministic=None):
        """Play COMPLETE episodes with no gradients and no training.

        Both arguments default to None, meaning "use the config":
        n_eval_episodes and eval_deterministic. Pass either one to override it
        for a single call, e.g. evaluate(deterministic=True) for the final
        number after training.

        Why this cannot be sample(): sample() stops dead after T steps,
        wherever each worker happens to be. Its return_mean therefore counts
        only the episodes that HAPPENED to finish inside that window, which
        selects for short ones -- and on MemoryS7 short means lucky, because
        the reward decays with length. Here every episode runs to its own end,
        so nothing is cut off and nothing is selected for.

        Runs on its OWN env, so self.envs, self.obs and self.hidden are never
        touched: training resumes from exactly where it was. Evaluating
        changes nothing, which is the whole point.

        No manual step cap is needed. MiniGrid wraps the env in a TimeLimit,
        so max_steps (245 for MemoryS7) arrives on its own as truncated=True.

        deterministic=False samples from the policy, exactly as training does,
        which makes these numbers directly comparable to train()'s.
        deterministic=True takes the argmax: the cleaner final number, but
        early in training it deadlocks -- a barely-trained policy repeats one
        action until the time limit and scores 0.

        Every episode index gets a fixed seed, so the same n_episodes mazes
        are replayed at every call. That pins down the maze draw, but with
        deterministic=False the ACTION SAMPLING still varies, so two calls on
        the same weights will not agree exactly. Only deterministic=True is
        fully reproducible.

        Returns:
            success_rate  fraction that reached the CORRECT object. THE number
                          for this project, because unlike return it does not
                          mix in how fast the agent walked. Read it WITH
                          timeout_rate: the 0.5 coin-flip baseline only
                          applies to episodes that got as far as the split
            timeout_rate  fraction that ran out of steps without touching
                          anything. A different failure from picking the wrong
                          door -- that one is bad memory, this one is bad
                          navigation, and early on it dominates
            return_mean   what the env actually paid, 1 - 0.9*(steps/245) on
                          success and 0 otherwise
            return_std    spread of those returns ACROSS episodes. Not an
                          error bar -- the return is bimodal (0 on failure,
                          ~0.9 on success), so this is roughly
                          0.9*sqrt(p(1-p)) and stays near 0.45 for any p in
                          the middle. It only shrinks as success_rate
                          approaches 0 or 1. To compare two encoders use the
                          standard error, return_std / sqrt(n_episodes)
            length_mean   steps taken, averaged over all episodes
        """
        if n_episodes is None:
            n_episodes = self.n_eval_episodes
        if deterministic is None:
            deterministic = self.eval_deterministic
        dev = self.device

        # a private env. Built and closed here so evaluation owns no state
        # between calls -- there is nothing to carry over, every episode
        # starts from a reset and a zeroed hidden state. Same builder as the
        # rollout's, so it carries the same wrappers: evaluating a different
        # game from the one being trained on would measure nothing.
        env = self.build_env()

        returns, lengths, successes, timeouts = [], [], [], []

        # a no-op for these three encoders (no dropout, no batchnorm), but it
        # is the standard contract and costs nothing to honour
        self.model.eval()

        for i in range(n_episodes):
            # a fixed seed PER EPISODE INDEX, so episode i is the same maze
            # and the same cue at every call to evaluate()
            obs, _ = env.reset(seed=self.eval_seed + i)

            # batch of 1, and zeroed HERE: an episode must never begin with
            # the memory of the one before it
            hidden = self.zero_hidden(batch_size=1)

            ep_return, ep_length, terminated, truncated = 0.0, 0, False, False

            while not (terminated or truncated):
                # (7, 7, 3) -> (1, 1, 7, 7, 3): batch 1, seq_len 1, the same
                # one-step-at-a-time shape the rollout uses
                obs_t = torch.from_numpy(obs["image"]).to(dev)[None, None]

                with torch.no_grad():  # nothing here may build a graph
                    dist, _, hidden = self.model(obs_t, hidden)
                    action = (
                        dist.probs.argmax(dim=-1) if deterministic else dist.sample()
                    )

                obs, reward, terminated, truncated, _ = env.step(int(action[0, 0]))
                ep_return += reward
                ep_length += 1

            returns.append(ep_return)
            lengths.append(ep_length)
            successes.append(ep_return > 0.0)  # MemoryS7 pays 0 for a wrong door
            timeouts.append(truncated and not terminated)

        self.model.train()
        env.close()

        return {
            "eval_episodes": n_episodes,
            "success_rate": float(np.mean(successes)),
            "timeout_rate": float(np.mean(timeouts)),
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "length_mean": float(np.mean(lengths)),
        }

    # ------------------------------------------------------------------
    # the whole run
    # ------------------------------------------------------------------
    def train_agent(self, n_iterations=None, n_iterations_report=None, logger=None):
        """The full run: train() in a loop, evaluate() now and then.

        This is the only method main.py needs to call.

            for i in range(n_iterations):
                train()                     EVERY iteration
                if i % n_iterations_report == 0:
                    evaluate()                        COMPLETE episodes

        Both arguments default to None, meaning "use the config"
        (n_iterations, n_iterations_report). Pass either to override it for a
        single run, e.g. train_agent(5, 1) for a smoke test.

        WHY EVALUATION IS NOT EVERY ITERATION. evaluate() plays
        n_eval_episodes episodes to their own end, and an unsolved one runs to
        the env's time limit: up to 50 * 245 = 12,250 env steps on MemoryS7,
        against the 8 * 245 = 1,960 that train() collects. Every iteration it
        would be most of the runtime and would measure nothing new -- one
        update barely moves the policy.

        The report iterations are i = 0, r, 2r, ... plus the LAST one, so a run
        always ends on a fresh number however n_iterations and
        n_iterations_report happen to divide. i = 0 is the baseline: one update
        in, i.e. essentially the untrained policy.

        EVAL KEYS ARE PREFIXED eval_*. evaluate() and sample() both report
        return_mean and length_mean, and they are NOT the same quantity -- the
        rollout's is cut off after T steps and biased towards short episodes,
        which is the entire reason evaluate() exists. Merged raw, the eval
        number would silently overwrite the training one.

        CHECKPOINTING happens on report iterations only, and only when
        eval_success_rate BEATS every earlier one. config.save_model writes
        agents/pretrained_model/ppo_<encoder>.pth -- weights, optimizer state,
        and the four attributes that decide the architecture, so
        config.load_model can refuse a file that does not match the config.
        config.watch_agent() then plays it back in a window.

        logger comes from config.build_logger() and is optional: without one
        the report goes to stdout and nowhere else, with one it also lands in
        logs/log_<date>_<time>.log under the hyperparameters that produced it.

        Returns the history: one stats dict per iteration, in order. Report
        iterations carry the eval_* keys too, the others do not. The history
        itself is not written to disk -- that is still open.
        """
        if n_iterations is None:
            n_iterations = self.n_iterations
        if n_iterations_report is None:
            n_iterations_report = self.n_iterations_report

        # the logger's StreamHandler already echoes to the terminal, so this
        # is a swap, not an addition: exactly one copy either way
        log = print if logger is None else logger.info

        # a separate blank record, not "\n" inside the header: a newline in
        # the middle of a log record puts the file's timestamp prefix on the
        # blank line and leaves the header unprefixed
        log("")
        log(
            f"{'iter':>5} {'eps':>4} {'return':>7} {'len':>6} "
            f"{'entropy':>8} {'v_loss':>9} {'kl':>8} {'clip':>6}"
            f"   |  EVAL  success  timeout   return    std"
        )

        history = []
        best_success = -1.0  # -1, not 0: even a policy that never scores gets
        #                      written once, so a file always exists afterwards

        for i in range(n_iterations):
            stats = self.train()
            stats["iteration"] = i

            # the last iteration always reports, so the run cannot end on an
            # iteration whose numbers were never measured
            report = i % n_iterations_report == 0 or i == n_iterations - 1

            if report:
                # startswith guard: eval_episodes is already prefixed
                stats.update(
                    {
                        k if k.startswith("eval_") else f"eval_{k}": v
                        for k, v in self.evaluate().items()
                    }
                )

                # the training columns are ONE rollout, so they are noisy:
                # with episodes near 0, return_mean means nothing that
                # iteration. eval_return is the trustworthy one.
                #
                # std is the spread OVER the n_eval_episodes, printed as
                # "mean +- std". Read it as a description of the task, not as
                # an error bar: the return is bimodal (0 or ~0.9), so std is
                # pinned near 0.9*sqrt(p(1-p)) ~ 0.45 whenever the success
                # rate sits anywhere in the middle, and NO amount of training
                # shrinks it until p reaches 0 or 1. The error bar on the mean
                # is std / sqrt(n_eval_episodes) -- ~0.06 at n = 50, which is
                # the number to beat before calling two encoders different.
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

                # KEEP THE BEST, NOT THE LAST. The policy is not monotone --
                # a KL spike can take it from 1.00 back to 0.52 in one report
                # interval -- so whatever iteration n_iterations-1 happens to
                # be is a lottery ticket, not a result. Saving on improvement
                # means the file holds the best policy the run ever had, and
                # the FINAL printed row can be compared against it.
                if stats["eval_success_rate"] > best_success:
                    best_success = stats["eval_success_rate"]
                    path = self.save_model(
                        self.model,
                        self.optimizer,
                        iteration=i,
                        eval_success_rate=best_success,
                        eval_return_mean=stats["eval_return_mean"],
                    )
                    log(f"{'':>5} saved  {path}  (best so far {best_success:.2f})")

            history.append(stats)

        return history

    def close(self):
        for env in self.envs:
            env.close()
