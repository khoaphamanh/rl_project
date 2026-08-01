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
        self.recurrent_model = "TRANSFORMER"  # or "LSTM", "MLP", "TRANSFORMER", "GRU"
        self.n_layers_mlp = 3
        self.lr = 1e-3

        self.tbptt_length = "max"

        # env
        self.name_env = (
            "MiniGrid-MemoryS11-v0"  # "MiniGrid-MemoryS11-v0""MiniGrid-DoorKey-8x8-v0"
        )
        self.force_cue_visible = False  # wrap the env in StartInCueView, which
        #   spawns the agent at (1, height//2) instead of a random x along the
        #   hallway. MiniGrid's own MemoryEnv shows the cue from x = 1 ONLY, so
        #   without this it is unobservable in 7 of 8 episodes -- and an
        #   unobservable cue makes GRU, LSTM and MLP the same agent. This is
        #   what makes the ablation an ablation. See helper.StartInCueView.

        # sampling: W games played in parallel, T steps each per iteration
        self.n_workers = 16  # W

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

        # ------------------------------------------------------------------
        # transformer only -- read by build_extractor when recurrent_model is
        # "TRANSFORMER", ignored otherwise. Down here rather than up beside
        # hidden_size because max_seq_length is derived from worker_steps,
        # which is only known above.
        #
        # SMALL ON PURPOSE: a first test of the encoder, not a tuned run.
        # Measured on DoorKey-8x8, one fwd+bwd over mini_batch_size=4
        # sequences of L=640, it is CHEAPER than the GRU beside it:
        #
        #     GRU                                200,832 params     97 ms
        #     Transformer d_model=64, 2 layers   166,912 params     74 ms
        #     Transformer d_model=128, 4 layers  926,912 params    279 ms
        #
        # A GRU walks 640 steps strictly one after another; attention does all
        # 640 in one matmul. What does bite is that attention is quadratic in
        # L, so the third line above is where it stops being free.
        # ------------------------------------------------------------------
        self.d_model = 64  # the width the attention stack runs at. Independent
        #   of hidden_size -- fc_out projects d_model -> hidden_size -- but set
        #   equal to it here so the test adds no width anywhere and the
        #   comparison against the GRU stays honest.
        self.n_heads = 4  # must DIVIDE d_model. 64 / 4 = 16 numbers per head.
        self.n_layers_transformer = 1  # one layer is a single lookup and cannot
        #   compose two of them ("find the cue" then "relate it to here"), so
        #   2 is the smallest stack that is still a transformer.
        self.d_ff = 4 * self.d_model  # 256, the ratio the original paper uses

        self.p_drop = 0.0  # NOT a free choice. PPO compares log_probs recorded
        #   during the rollout against log_probs recomputed during the update.
        #   Dropout makes the same observation give two different answers, so
        #   the ratio pi_new / pi_old is noise before any learning happens --
        #   and nothing in this project calls model.eval(), so a nonzero value
        #   would be live during both passes. Leave at 0.0.

        self.max_seq_length = self.worker_steps  # also NOT free. The positional
        #   codes and the causal mask are built once, this long, and forward
        #   raises on anything longer. split_pad_mask pads to L = the longest
        #   unbroken stretch of steps, which is at most T -- so tying this to
        #   worker_steps makes it follow the env instead of being one more
        #   number to remember to change per maze size.

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
        self.n_iterations = 500  # sample -> update, this many times
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
