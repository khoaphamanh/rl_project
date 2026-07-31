import random
import os
import numpy as np
import torch

from config.helper import Helper


class Config(Helper):
    """All hyperparameters. Built once, in main.py.

    Inheriting Helper means config.build_extractor(), config.zero_hidden()
    and config.is_recurrent read these attributes directly.
    """

    def __init__(self):

        # default seed
        self.seed_default = 42

        # fix hyperparameters
        self.input_size = 7 * 7 * 20  # 980, NOT 147: flatten_obs one-hots the
        #   observation's three CATEGORY channels (11 objects + 6 colours +
        #   3 door states = 20 numbers per cell). Handing the raw indices to
        #   the encoder instead makes key(5) vs ball(6) a difference of 1.0 in
        #   one of 147 inputs, along the same axis as empty(1) vs wall(2) --
        #   and then no encoder can read the cue, so all three score alike.
        #   See feature_extractor.flatten_obs for the measurements.

        # model hyperparameters
        self.hidden_size = 64
        self.recurrent_model = "GRU"  # or "LSTM"
        self.n_layers_mlp = 3
        self.lr = 1e-3

        self.tbptt_length = "max"

        # env
        self.name_env = "MiniGrid-MemoryS11-v0"
        self.force_cue_visible = True  # wrap the env in StartInCueView, which
        #   spawns the agent at (1, height//2) instead of a random x along the
        #   hallway. MiniGrid's own MemoryEnv shows the cue from x = 1 ONLY, so
        #   without this it is unobservable in 7 of 8 episodes -- and an
        #   unobservable cue makes GRU, LSTM and MLP the same agent. This is
        #   what makes the ablation an ablation. See helper.StartInCueView.

        # sampling: W games played in parallel, T steps each per iteration
        self.n_workers = 8  # W

        # T = the env's OWN time limit, 5 * size^2 (245 on S7, 605 on S11),
        # read off the env instead of typed in. NOT a free choice.
        #
        # WHY T = 256 SOLVED S7 AND NOT S11, on identical everything else. The
        # limit is what differs: 245 on S7, 605 on S11. An unsolved episode runs
        # to that limit, so one 8 x 256 = 2048-step rollout contains
        #
        #     S7    2048 / 245  ~  8 complete episodes  -> 8 reward signals
        #     S11   2048 / 605  ~  3 complete episodes  -> 3 reward signals
        #
        # per update. On a task that pays nothing until the right object is
        # touched, three outcomes is not enough to estimate an advantage from,
        # and S7 got nearly three times as many for the same compute. T = 245
        # happened to sit just under 256; that coincidence is the whole reason
        # one size worked and the other did not.
        #
        # Setting T = max_steps makes that structural rather than lucky: the
        # longest episode the env can produce still fits inside one rollout at
        # every size, so each worker contributes at least one complete episode
        # per update and split_pad_mask only ever cuts at a real done -- where
        # truncating BPTT is correct, because the hidden state was zeroed there
        # anyway. The cost is rollout size, which now grows with the maze:
        # 8 * 605 = 4840 steps per iteration on S11 against 1960 on S7.
        self.worker_steps = self.env_max_steps  # T

        # total env steps collected per iteration. NOT called batch_size on
        # purpose: mini_batch_size below is handed to a torch DataLoader, where
        # batch_size counts SEQUENCES, and two sibling attributes counting
        # different things under the same word is a trap. This one counts steps.
        self.n_total_steps = self.n_workers * self.worker_steps  # W * T

        # advantage estimation
        self.gamma = 0.99  # discount: how far ahead a reward still counts
        self.gae_lambda = 0.95  # 0 = TD(0), low variance / high bias
        #                         1 = Monte Carlo, high variance / no bias

        # the update
        self.clip_eps = 0.2  # how far pi_new may drift from pi_old before
        #                      the gradient is cut off. The paper's value.
        self.value_coef = 0.1  # weight of the critic's regression term. Damped,
        #                        or a squared error on returns would dominate
        #                        the shared encoder and the policy learns nothing.
        self.entropy_coef = 0.03  # weight of the exploration bonus. Small on
        #                           purpose: a brake against premature collapse,
        #                           not a goal.
        self.max_grad_norm = 0.5  # rescale the gradient if it is longer than this

        # how hard each rollout is reused
        self.n_epochs = 3  # passes over the SAME batch. The whole reason the
        #                    clip exists: after pass 1 the data is off-policy.
        self.mini_batch_size = 4  # DataLoader batch size, counted in SEQUENCES
        #                           (not timesteps): one sample is one whole
        #                           padded episode fragment of up to L steps.
        self.target_kl = None  # e.g. 0.015 to abandon the remaining epochs once
        #                        pi_new has drifted too far. None = off.

        # training length
        self.n_iterations = 1000  # sample -> update, this many times
        self.n_iterations_report = 100  # every this many iterations, run a full
        #                                evaluate() and print a line. NOT every
        #                                iteration: 50 unsolved episodes cost up
        #                                to 50 * max_steps env steps, which is
        #                                more than the n_total_steps one
        #                                iteration collects, and the policy
        #                                barely moves in a single update anyway.

        # evaluation: no gradients, no training, COMPLETE episodes only
        self.n_eval_episodes = 50  # played to their own end, never truncated
        self.eval_seed = 10_000  # fixed and far from seed_default, so the same
        #                          50 mazes are replayed at every evaluation.
        #                          That removes the maze draw as a source of
        #                          noise -- but NOT the action sampling, which
        #                          still varies unless deterministic=True.
        self.eval_deterministic = False  # False = sample from pi, the same way
        #                                  training does, so the curve is
        #                                  comparable to train()'s.
        #                                  True = argmax: the cleaner final
        #                                  number, but it deadlocks on a
        #                                  half-trained policy (one action
        #                                  repeated until the time limit).
        #                                  evaluate(deterministic=True)
        #                                  overrides this for one call.

        # where a trained model is written
        self.dir_pretrained_model = os.path.join("agents", "pretrained_model")

        # ppo_GRU.pth, ppo_LSTM.pth, ppo_MLP.pth -- the ENCODER IS IN THE NAME
        # rather than a single fixed ppo_feature_extractor.pth, because the
        # whole experiment is running all three and one fixed name would have
        # each run silently overwrite the previous one's weights. Relative, like
        # logs/, so everything is written under the repo root -- which means
        # python must be run FROM the repo root.
        self.name_model = f"ppo_{self.recurrent_model.upper()}.pth"
        self.path_model = os.path.join(self.dir_pretrained_model, self.name_model)
        # the directory itself is made by helper.build_model_path(), at save
        # time, not here

    def set_seed(self, env=None, seed=None):
        """Seed every source of randomness and return the seed that was used.

        seed=None falls back to self.seed_default, so set_seed() with no
        argument is always reproducible.

        The CUDA calls are no-ops on a machine without a GPU. They stay in so
        the exact same config also reproduces on a machine that has one.

        env is optional. In Gymnasium the env RNG is seeded through
        reset(seed=...), so this consumes one reset and throws its
        observation away. Call set_seed first, then reset() again without a
        seed for the real first observation -- that keeps the stream going
        from the seeded state instead of restarting it.
        """
        if seed is None:
            seed = self.seed_default
        self.seed = seed

        # affects subprocesses only (the parallel workers), not this process
        os.environ["PYTHONHASHSEED"] = str(seed)

        random.seed(seed)  # python's own random
        np.random.seed(seed)  # numpy, used by gymnasium internally

        torch.manual_seed(seed)  # cpu tensors, weight init, dropout
        torch.cuda.manual_seed(seed)  # one gpu
        torch.cuda.manual_seed_all(seed)  # all gpus

        if torch.cuda.is_available():
            # without these two, cudnn picks algorithms by benchmarking and
            # the same seed can still give different numbers
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if env is not None:
            env.reset(seed=seed)  # seeds the env RNG (layout, cue position)
            env.action_space.seed(seed)  # seeds env.action_space.sample()
            env.observation_space.seed(seed)

        return seed
