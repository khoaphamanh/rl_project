"""
Entry point. Trains PPO with ONE encoder, named on the command line, once per
seed.

    python main.py MLP
    python main.py GRU
    python main.py LSTM
    python main.py TRANSFORMER

The encoder is the only thing that changes between runs of an ablation, so it
is the only argument. make_config(name) builds the matching Config subclass
(config/__init__.py); everything else -- env, rollout size, PPO knobs, and the
seeds -- is the same across all four and lives in config.py.

WHY A LIST OF SEEDS AND NOT ONE. A single PPO run on a sparse-reward MiniGrid
task can land anywhere; two seeds of the SAME config routinely differ by more
than two different encoders do. One run is an anecdote, so the unit of a result
here is config.base_seed_list as a whole: run_one() per seed, then aggregate
across them at the bottom. Each seed gets its own config, its own agent (fresh
weight init, fresh optimizer, fresh envs), its own log file and its own
checkpoint -- the seed is in the filename, so they cannot overwrite each other.

The seed reaches the config through PPOAgent(config, seed=s), which calls
config.set_seed(): one path in, so config.seed, the checkpoint name and the
logged hyperparameters can never disagree.

No PPO happens here. main.py builds the config, hands it to the agent, calls
agent.train_agent(), prints the numbers, and closes the envs.
"""

import argparse

import numpy as np

from agents.ppo import PPOAgent
from config import make_config, MODEL_CHOICES


def run_one(model_name, seed):
    """One split: train this encoder from this seed. Returns its scores."""
    # the encoder decides which Config subclass, and the subclass decides which
    # architecture knobs exist -- see config/__init__.py
    config = make_config(model_name)

    # seed= goes in HERE and nowhere else: __init__ calls config.set_seed()
    # before it builds the Network, so it seeds the weight init too
    agent = PPOAgent(config, seed=seed)

    # AFTER the agent, not before: set_seed() has now run, so the seed actually
    # used is what lands in the dump. One file per run, handlers cleared each
    # call, so three seeds give three logs and no duplicated lines.
    logger = config.build_logger()

    # what was actually built, straight under the hyperparameters that asked
    # for it: layer by layer, total and trainable parameters, size in MB
    config.log_model_summary(agent.model, logger)

    try:
        history = agent.train_agent(logger=logger)

        # the clean final number: argmax instead of sampling, so it is fully
        # reproducible. Early in training this deadlocks (one action repeated
        # until the time limit) -- at the END of a run it is the honest score.
        final = agent.evaluate(deterministic=True)

        # the learning CURVE, not just its last point: every report iteration
        # left an eval_success_rate behind. Index 0 is dropped because it is
        # the untrained policy, the same floor for every run.
        curve = [h["eval_success_rate"] for h in history if "eval_success_rate" in h]
        aulc = float(np.mean(curve[1:] or curve))

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
        logger.info(f"  aulc          {aulc:.3f}  over {len(curve)} evaluations")

        return {"seed": seed, "aulc": aulc, "success_rate": final["success_rate"]}
    finally:
        # the W envs are released even if the run is interrupted with ctrl-c
        agent.close()


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

    # a throwaway config, built only to read the seed list. run_one() builds
    # its own -- one per seed, never shared.
    seeds = make_config(args.model).base_seed_list

    results = [run_one(args.model, s) for s in seeds]

    # PRINTED PER SEED, not only as a mean. On these tasks the outcome tends to
    # be bimodal -- some seeds find the reward and some never do -- and the
    # mean of {0.5, 0.5, 0.95} describes no run that happened. The spread is
    # over SEEDS (n = len(seeds)), which is the real uncertainty about this
    # config; the spread over eval episodes inside one run is not.
    print(f"\n{args.model}  over {len(results)} seeds")
    print(f"{'seed':>8} {'aulc':>8} {'success':>9}")
    for r in results:
        print(f"{r['seed']:>8} {r['aulc']:>8.3f} {r['success_rate']:>9.3f}")

    for key in ("aulc", "success_rate"):
        vals = [r[key] for r in results]
        print(
            f"{key:>8}  mean {np.mean(vals):.3f}  std {np.std(vals):.3f}  "
            f"median {np.median(vals):.3f}"
        )


if __name__ == "__main__":
    main()
