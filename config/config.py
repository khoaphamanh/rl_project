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
        self.input_size = 7 * 7 * 3

        # model hyperparameters
        self.hidden_size = 64
        self.recurrent_model = "GRU"  # or "LSTM"
        self.n_layers_mlp = 3
        self.lr = 1e-3

        self.tbptt_length = "max"

        # env
        self.name_env = "MiniGrid-MemoryS7-v0"

        # sampling: W games played in parallel, T steps each per iteration
        # the paper uses W=16, T=512; smaller here so it runs on a laptop
        self.n_workers = 4  # W
        self.worker_steps = 128  # T
        self.batch_size = self.n_workers * self.worker_steps  # W * T

        # advantage estimation
        self.gamma = 0.99  # discount: how far ahead a reward still counts
        self.gae_lambda = 0.95  # 0 = TD(0), low variance / high bias
        #                         1 = Monte Carlo, high variance / no bias

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
