"""
Entry point.

The Config is built here and handed to the agent, so every hyperparameter
lives in one place and nothing below imports Config on its own.

No PPO happens here. main.py prints what it is about to run, calls
agent.train_agent() once, prints the final number, and closes the envs.

Run with:
    python main.py
"""

from agents.ppo import PPOAgent
from config.config import Config


def main():
    config = Config()

    agent = PPOAgent(config)

    # AFTER the agent, not before: PPOAgent.__init__ calls config.set_seed(),
    # and that is what puts the seed actually used into the dump. One file per
    # run, logs/log_<date>_<time>.log, hyperparameters at the top.
    logger = config.build_logger()

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
