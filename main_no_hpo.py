"""
Entry point for the HAND-PICKED run. No optuna, no study, no trials.

    python main_no_hpo.py               the encoder named in config_no_hpo.py
    python main_no_hpo.py GRU           override it for one run

Everything it trains with is written out by hand in config/config_no_hpo.py --
that file is the single place to change a hyperparameter, and it lists all four
encoders' knobs side by side so it is obvious what is being held equal.

HOW THIS DIFFERS FROM main.py. Only in which config it builds, and in that it
runs the whole seed list rather than one seed. main.py uses make_config(name),
i.e. the four search-ready configs whose values are whatever the last HPO edit
left them at; this one uses ConfigNoHPO, whose values are fixed and readable.
Same agent, same env, same seeds, same checkpoint format -- so the two are
directly comparable, which is the point:

    python main_no_hpo.py GRU       ->  the hand-picked baseline
    python -m agents.hpo_ppo GRU    ->  what a search can do better

The checkpoints land in agents/pretrained_model_GRU/no_hpo/, parallel to the
search's hpo/, so neither run can overwrite the other.

WHY A LIST OF SEEDS AND NOT ONE. A single PPO run on a sparse-reward MiniGrid
task can land anywhere, and two seeds of the SAME config routinely differ by
more than two different encoders do. The unit of a result is config.seed_list
as a whole -- one seed is an anecdote.
"""

import argparse

import numpy as np

from agents.ppo import PPOAgent
from config import MODEL_CHOICES
from config.config_no_hpo import ConfigNoHPO, FEATURE_EXTRACTOR


def run_one(model_name, seed):
    """One split: train this encoder from this seed. Returns its scores."""
    # a fresh config per run. PPOAgent copies every value out of it in
    # __init__, and train() writes the resolved mini_batch_size back into it,
    # so sharing one across seeds would carry state between runs.
    config = ConfigNoHPO(model_name)

    # seed= goes in HERE and nowhere else: __init__ calls config.set_seed()
    # before it builds the Network, so it seeds the weight init too -- and it
    # is what puts the seed into the checkpoint's filename
    agent = PPOAgent(config, seed=seed)

    # AFTER the agent, not before: set_seed() has now run, so the seed actually
    # used is what lands in the dump. One file per run, handlers cleared each
    # call, so three seeds give three logs and no duplicated lines.
    logger = config.build_logger()
    config.log_model_summary(agent.model, logger)

    try:
        history = agent.train_agent(logger=logger)

        # the clean final number: argmax instead of sampling, so it is fully
        # reproducible. Early in training this deadlocks -- at the END of a
        # run it is the honest score.
        final = agent.evaluate(deterministic=True)

        # the learning CURVE, not just its last point. Index 0 is dropped
        # because it is the untrained policy, the same floor for every run.
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
        description="Train PPO with hand-picked hyperparameters (config/config_no_hpo.py)."
    )
    parser.add_argument(
        "model",
        nargs="?",  # OPTIONAL, unlike main.py: the config already names one,
        #              and editing that file is the intended way to choose
        default=FEATURE_EXTRACTOR,
        type=str.upper,
        choices=MODEL_CHOICES,
        help=f"which feature extractor to train (default {FEATURE_EXTRACTOR})",
    )
    args = parser.parse_args()

    # a throwaway config, built only to read the seed list and echo the
    # settings. run_one() builds its own -- one per seed, never shared.
    config = ConfigNoHPO(args.model)
    seeds = config.seed_list

    print(f"NO-HPO run: {args.model} on {config.name_env}")
    print(f"  seeds        {seeds}")
    print(f"  iterations   {config.n_iterations}")
    print(f"  checkpoints  {config.dir_pretrained_model}")

    results = [run_one(args.model, s) for s in seeds]

    # PRINTED PER SEED, not only as a mean. On these tasks the outcome tends to
    # be bimodal -- some seeds find the reward and some never do -- and the
    # mean of {0.5, 0.5, 0.95} describes no run that happened. The spread is
    # over SEEDS, which is the real uncertainty about this config; the spread
    # over the eval episodes inside one run is not.
    print(f"\n{args.model}  over {len(results)} seeds")
    print(f"{'seed':>8} {'aulc':>8} {'success':>9}")
    for r in results:
        print(f"{r['seed']:>8} {r['aulc']:>8.3f} {r['success_rate']:>9.3f}")

    for key in ("aulc", "success_rate"):
        vals = [r[key] for r in results]
        print(
            f"{key:>12}  mean {np.mean(vals):.3f}  std {np.std(vals):.3f}  "
            f"median {np.median(vals):.3f}"
        )


if __name__ == "__main__":
    main()
