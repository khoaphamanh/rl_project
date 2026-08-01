"""
Entry point. Trains PPO with ONE encoder, named on the command line.

    python main.py MLP
    python main.py GRU
    python main.py LSTM
    python main.py TRANSFORMER

The encoder is the only thing that changes between runs of an ablation, so it
is the only argument. make_config(name) builds the matching Config subclass
(config/__init__.py); everything else -- env, rollout size, PPO knobs -- is the
same across all four and lives in config.py.

No PPO happens here. main.py builds the config, hands it to the agent, calls
agent.train_agent() once, prints the final number, and closes the envs.
"""

import argparse

from agents.ppo import PPOAgent
from config import make_config, MODEL_CHOICES


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO with one feature extractor on a MiniGrid env."
    )
    parser.add_argument(
        "model",
        type=str.upper,  # so "mlp" and "MLP" both work; choices stay uppercase
        choices=MODEL_CHOICES,
        help="which feature extractor to train (%(choices)s)",
    )
    args = parser.parse_args()

    # the encoder decides which Config subclass, and the subclass decides which
    # architecture knobs exist -- see config/__init__.py
    config = make_config(args.model)

    agent = PPOAgent(config)

    # AFTER the agent, not before: PPOAgent.__init__ calls config.set_seed(),
    # and that is what puts the seed actually used into the dump. One file per
    # run, logs/log_<date>_<time>.log, hyperparameters at the top.
    logger = config.build_logger()

    # what was actually built, straight under the hyperparameters that asked
    # for it: layer by layer, total and trainable parameters, size in MB
    config.log_model_summary(agent.model, logger)

    try:
        # the whole run. One stats dict per iteration comes back; the ones on
        # report iterations carry the eval_* keys too
        history = agent.train_agent(logger=logger)

        # the clean final number: argmax instead of sampling, so it is fully
        # reproducible. Early in training this deadlocks (one action repeated
        # until the time limit) -- at the END of a run it is the honest score.
        final = agent.evaluate(deterministic=True)

        logger.info("")
        logger.info(f"ran {len(history)} iterations")
        logger.info("FINAL (deterministic)")
        logger.info(f"  success_rate  {final['success_rate']:.3f}")
        logger.info(f"  timeout_rate  {final['timeout_rate']:.3f}")
        logger.info(
            f"  return_mean   {final['return_mean']:.3f} "
            f"+- {final['return_std']:.3f}"
        )
        logger.info(f"  length_mean   {final['length_mean']:.1f}")
    finally:
        # the W envs are released even if the run is interrupted with ctrl-c
        agent.close()


if __name__ == "__main__":
    main()
