"""
Inspect the input (observation) and output (action) size of the MiniGrid
Memory environment used in this project.

Run with:
    conda activate rl_project
    python test_enviroment/test_input_output.py
"""

import gymnasium as gym
import minigrid
import numpy as np

ENV_ID = "MiniGrid-MemoryS11-v0"


def main():
    env = gym.make(ENV_ID)
    obs, info = env.reset(seed=0)
    
    print("info:", info)

    print(f"Environment: {ENV_ID}")
    print("=" * 60)

    # -----------------------------------------------------------------
    # INPUT (observation space) — what the policy network receives
    # -----------------------------------------------------------------
    print("\nOBSERVATION SPACE (network INPUT)")
    print("-" * 60)
    print(env.observation_space)
    print()
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"  obs['{key}']: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  obs['{key}']: {value!r}")

    # -----------------------------------------------------------------
    # OUTPUT (action space) — what the policy network produces
    # -----------------------------------------------------------------
    print("\nACTION SPACE (network OUTPUT)")
    print("-" * 60)
    print(env.action_space)
    print(f"  Number of discrete actions: {env.action_space.n}")

    env.close()


if __name__ == "__main__":
    main()
