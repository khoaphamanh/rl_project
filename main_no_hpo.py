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
    """One split: train this encoder from this seed. Returns its scores.

    Reports the finished policy TWICE -- sampled and argmax -- see below.
    """
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

        # ---- the two FINAL blocks ------------------------------------
        #
        # THE SAME WEIGHTS, MEASURED TWO WAYS, and both are reported because
        # they routinely disagree -- a policy can solve every eval maze when
        # sampled and none of them under argmax, which is a greedy deadlock
        # (one action repeated into the time limit) and a real property of the
        # policy, not a measurement error.
        #
        #   NON-DETERMINISTIC  actions drawn from pi, the way training and
        #                      every HPO trial measure. This is NOT re-measured:
        #                      it is the last entry of the run's own curve, so
        #                      the number printed here is one that appears in
        #                      the table above. Re-running a sampled evaluation
        #                      would give a different answer for identical
        #                      weights, and then the two would quietly disagree.
        #   DETERMINISTIC      argmax. Fully reproducible -- fixed mazes AND no
        #                      action sampling -- which is what a results table
        #                      wants. Early in training it deadlocks; at the END
        #                      of a run it is an honest second view.
        #
        # ONE evaluate() call either way: whichever mode the curve was NOT
        # measured in is the one measured here. With the default
        # (eval_deterministic=False) the curve gives sampled and this adds
        # argmax; flip the config and the roles swap.
        last = agent.eval_history[-1]
        fresh = agent.evaluate(deterministic=not config.eval_deterministic)
        if config.eval_deterministic:
            argmax, sampled = last, fresh
        else:
            sampled, argmax = last, fresh

        # the learning CURVE, one entry per report iteration -- 0, 100, 200,
        # 300, 400, 499 at n_iterations=500 and n_iterations_report=100.
        # Printed in full rather than summarised into a single number: the
        # SHAPE is the information, and any average over it throws away the
        # order of the points -- 1, 1, 0 and 0, 1, 1 have the same mean and are
        # not the same run.
        logger.info("")
        logger.info(f"ran {len(history)} iterations")
        logger.info(f"{'iter':>7} {'success':>9} {'timeout':>9} {'return':>9}")
        for c in agent.eval_history:
            logger.info(
                f"{c['iteration']:>7} {c['success_rate']:>9.3f} "
                f"{c['timeout_rate']:>9.3f} {c['return_mean']:>9.3f}"
            )

        def report(title, evaluation, note):
            logger.info("")
            logger.info(f"FINAL ({title})   {note}")
            logger.info(f"  success_rate  {evaluation['success_rate']:.3f}")
            logger.info(f"  timeout_rate  {evaluation['timeout_rate']:.3f}")
            logger.info(
                f"  return_mean   {evaluation['return_mean']:.3f} "
                f"+- {evaluation['return_std']:.3f}"
            )
            logger.info(f"  length_mean   {evaluation['length_mean']:.1f}")

        report(
            "non-deterministic",
            sampled,
            f"iteration {last['iteration']}, straight off the curve above"
            if not config.eval_deterministic
            else "re-evaluated, sampled from pi",
        )
        report(
            "deterministic",
            argmax,
            "re-evaluated, argmax"
            if not config.eval_deterministic
            else f"iteration {last['iteration']}, straight off the curve above",
        )

        return {
            "seed": seed,
            # BOTH, and named for how they were taken. "success_rate" alone
            # would have to mean one of them, and which one is exactly the
            # thing worth not guessing about.
            "success_sampled": sampled["success_rate"],
            "success_argmax": argmax["success_rate"],
        }
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
    print(f"{'seed':>8} {'sampled':>9} {'argmax':>9}")
    for r in results:
        print(
            f"{r['seed']:>8} {r['success_sampled']:>9.3f} "
            f"{r['success_argmax']:>9.3f}"
        )

    # median and IQR beside mean and std, because the same seeds feed both and
    # they answer different questions -- "the average run" against "the typical
    # run". On a bimodal task one dead seed of three moves the mean by a third
    # and the median not at all, and seeing the gap is the point. These are the
    # same two summaries config.hpo_objective chooses between for a study.
    for key in ("success_sampled", "success_argmax"):
        vals = [r[key] for r in results]
        print(
            f"{key:>16}  mean {np.mean(vals):.3f}  std {np.std(vals):.3f}  "
            f"median {np.median(vals):.3f}  "
            f"iqr {np.percentile(vals, 75) - np.percentile(vals, 25):.3f}"
        )


if __name__ == "__main__":
    main()
