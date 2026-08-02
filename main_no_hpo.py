"""
Entry point for the HAND-PICKED run. No optuna, no study, no trials.

    python main_no_hpo.py               the encoder named in config_no_hpo.py
    python main_no_hpo.py GRU           override it for one run
    python main_no_hpo.py GRU --report-only    report the runs already on disk

Everything it trains with is written out by hand in config/config_no_hpo.py --
that file is the single place to change a hyperparameter, and it lists all four
encoders' knobs side by side so it is obvious what is being held equal.

RUNNING THIS TWICE RETRAINS AND OVERWRITES. There is no resume here and no
"already done" check: run_one() builds a fresh PPOAgent and calls train_agent(),
which trains from random weights every time and writes its checkpoint at the
end, over whatever ppo_<seed>_<ENC>_<env>.pth was already there. That is the
opposite of main.py, where the study is kept in a database and re-running
CONTINUES it -- a difference worth knowing before spending a night on a run:

    python main.py GRU            resumes; finished trials are not repaid
    python main_no_hpo.py GRU     retrains all of seed_list, from scratch

--report-only is the way to get the closing numbers back WITHOUT retraining.
It loads each seed's saved checkpoint, evaluates it in both modes and prints
the same two FINAL blocks a training run ends with. Use it when the run has
already happened and you only want the table again -- or want it in argmax
after reading it in sampled.

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
import os

import numpy as np

from agents.ppo import PPOAgent
from config import MODEL_CHOICES
from config.config_no_hpo import ConfigNoHPO, FEATURE_EXTRACTOR


def report_run(logger, eval_history, fresh, curve_deterministic):
    """Print the learning curve and the two FINAL blocks. Returns (sampled, argmax).

    Shared by the two paths that produce a report -- run_one() after training
    and report_saved() after loading -- so a retrained run and a re-read one
    cannot end up printing different tables of the same quantities.

    THE SAME WEIGHTS, MEASURED TWO WAYS, and both are reported because they
    routinely disagree: a policy can solve every eval maze when sampled and
    none of them under argmax, which is a greedy deadlock (one action repeated
    into the time limit) and a real property of the policy, not a measurement
    error.

        NON-DETERMINISTIC  actions drawn from pi, the way training measures.
        DETERMINISTIC      argmax. Fully reproducible -- fixed mazes AND no
                           action sampling -- which is what a results table
                           wants.

    ONE OF THE TWO IS NOT RE-MEASURED. eval_history already holds whichever
    mode the run was evaluated in, so that one is read straight off the curve
    -- re-running a sampled evaluation would give a different answer for
    identical weights, and then the FINAL block and the table above it would
    quietly disagree. `fresh` is the other mode, measured once by the caller.

    curve_deterministic says which is which, and it is the RUN's setting, not
    today's config: for a loaded checkpoint it comes out of the file.
    """
    last = eval_history[-1]
    if curve_deterministic:
        argmax, sampled = last, fresh
    else:
        sampled, argmax = last, fresh

    # the learning CURVE, one entry per report iteration -- 0, 100, 200, 300,
    # 400, 499 at n_iterations=500 and n_iterations_report=100. Printed in full
    # rather than summarised into a single number: the SHAPE is the
    # information, and any average over it throws away the order of the points
    # -- 1, 1, 0 and 0, 1, 1 have the same mean and are not the same run.
    logger.info(f"{'iter':>7} {'success':>9} {'timeout':>9} {'return':>9}")
    for c in eval_history:
        logger.info(
            f"{c['iteration']:>7} {c['success_rate']:>9.3f} "
            f"{c['timeout_rate']:>9.3f} {c['return_mean']:>9.3f}"
        )

    def block(title, evaluation, note):
        logger.info("")
        logger.info(f"FINAL ({title})   {note}")
        logger.info(f"  success_rate  {evaluation['success_rate']:.3f}")
        logger.info(f"  timeout_rate  {evaluation['timeout_rate']:.3f}")
        logger.info(
            f"  return_mean   {evaluation['return_mean']:.3f} "
            f"+- {evaluation['return_std']:.3f}"
        )
        logger.info(f"  length_mean   {evaluation['length_mean']:.1f}")

    off_curve = f"iteration {last['iteration']}, straight off the curve above"
    block(
        "non-deterministic",
        sampled,
        "re-evaluated, sampled from pi" if curve_deterministic else off_curve,
    )
    block(
        "deterministic",
        argmax,
        off_curve if curve_deterministic else "re-evaluated, argmax",
    )
    return sampled, argmax


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

        # ONE evaluate() call: whichever mode the curve was NOT measured in is
        # the one measured here. With the default (eval_deterministic=False)
        # the curve gives sampled and this adds argmax; flip the config in
        # config_no_hpo.py and the roles swap. See report_run.
        fresh = agent.evaluate(deterministic=not config.eval_deterministic)

        logger.info("")
        logger.info(f"ran {len(history)} iterations")
        sampled, argmax = report_run(
            logger, agent.eval_history, fresh, config.eval_deterministic
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


def report_saved(model_name, seed_index, seed):
    """Report ONE already-trained run. TRAINS NOTHING, WRITES NOTHING.

    The --report-only path. run_one() trains from random weights and overwrites
    its checkpoint every time it is called; this reads that checkpoint back and
    reproduces the closing table from it, which is what you want when the run
    has already happened.

    WHAT IT COSTS. One evaluate() -- n_eval_episodes episodes of the mode the
    curve does NOT hold -- against a full training run. The curve itself is
    free: train_agent stores eval_history inside the .pth, so the table of
    report iterations is read, not recomputed.

    WHICH MODE WAS RE-MEASURED COMES OUT OF THE FILE, not out of the config.
    The checkpoint records the eval_deterministic it was trained under, so
    editing config_no_hpo.py after the fact relabels nothing: an old run stays
    described the way it actually happened.

    Returns the same dict run_one does, or None if this seed has no usable
    checkpoint -- an interrupted run, or a seed_list that has grown since. The
    caller reports what it got rather than failing on the whole set.
    """
    # a fresh config per seed, for the same reason run_one builds one
    config = ConfigNoHPO(model_name)

    # trial=None leaves dir_pretrained_model where ConfigNoHPO put it, i.e.
    # no_hpo/. Nothing is read yet -- the file may not exist.
    path = config.select_run(seed_index=seed_index)
    if not os.path.exists(path):
        print(f"  seed {seed}: no checkpoint at {path}, skipped")
        return None

    # {} for a ConfigNoHPO checkpoint, whose search_space is empty by design --
    # its values are hand-written constants, so rebuilding the config already
    # reproduces the architecture. Kept anyway so this still works on a file
    # written by a tuned run that was dropped into no_hpo/ by hand.
    config.apply_params(config.checkpoint_params(path))

    agent = PPOAgent(config, seed=seed)
    logger = config.build_logger()
    try:
        # guards the encoder, the widths, the env AND force_cue_visible against
        # this config, and raises rather than loading a file that does not
        # belong to it
        checkpoint = config.load_model(agent.model, path)

        curve = list(checkpoint.get("eval_history") or [])
        if not curve:
            print(f"  seed {seed}: {path} holds no eval_history, skipped")
            return None

        curve_deterministic = bool(
            checkpoint.get("eval_deterministic", config.eval_deterministic)
        )
        fresh = agent.evaluate(deterministic=not curve_deterministic)

        logger.info("")
        logger.info(f"REPORT ONLY  seed {seed}  --  loaded {path}, nothing trained")
        sampled, argmax = report_run(logger, curve, fresh, curve_deterministic)

        return {
            "seed": seed,
            "success_sampled": sampled["success_rate"],
            "success_argmax": argmax["success_rate"],
        }
    finally:
        agent.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO with hand-picked hyperparameters (config/config_no_hpo.py)."
    )
    # WHICH ENCODER. Positional and OPTIONAL, unlike main.py's: this file's
    # whole point is that config_no_hpo.py already names one, so a bare
    # `python main_no_hpo.py` is a complete command and editing FEATURE_EXTRACTOR
    # is the intended way to choose. The argument is the override for one run,
    # which is what makes four encoders comparable without four edits:
    #
    #     python main_no_hpo.py           whatever the file says
    #     python main_no_hpo.py GRU       the GRU, this once
    #
    # type=str.upper so "gru" works and choices stay uppercase; MODEL_CHOICES
    # is the same list make_config validates against, so the two entry points
    # cannot disagree about what an encoder is called.
    parser.add_argument(
        "model",
        nargs="?",
        default=FEATURE_EXTRACTOR,
        type=str.upper,
        choices=MODEL_CHOICES,
        help=f"which feature extractor to train (%(choices)s, default {FEATURE_EXTRACTOR})",
    )
    # THE FLAG THAT SAYS "DO NOT RETRAIN". Without it this command always
    # trains from random weights and overwrites the checkpoints -- there is no
    # resume in the no-HPO path, and no check for work already done.
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="skip training; load the checkpoints already in no_hpo/ and print "
        "their FINAL blocks (one evaluate() per seed, nothing is written)",
    )
    args = parser.parse_args()

    # a throwaway config, built only to read the seed list and echo the
    # settings. run_one() and report_saved() build their own -- one per seed,
    # never shared.
    config = ConfigNoHPO(args.model)
    seeds = config.seed_list

    mode = "REPORT ONLY (no training)" if args.report_only else "run"
    print(f"NO-HPO {mode}: {args.model} on {config.name_env}")
    print(f"  seeds        {seeds}")
    if not args.report_only:
        print(f"  iterations   {config.n_iterations}")
    print(f"  checkpoints  {config.dir_pretrained_model}")

    if args.report_only:
        # a missing seed is skipped, not fatal -- see report_saved
        results = [
            result
            for index, seed in enumerate(seeds)
            if (result := report_saved(args.model, index, seed)) is not None
        ]
        if not results:
            raise SystemExit(
                f"\nno checkpoint could be read under {config.dir_pretrained_model}. "
                f"Run `python main_no_hpo.py {args.model}` first -- --report-only "
                f"reads runs, it does not make them."
            )
    else:
        # RETRAINS AND OVERWRITES, every time. train_agent() starts from random
        # weights and saves over whatever was in no_hpo/ already.
        results = [run_one(args.model, s) for s in seeds]

    # PRINTED PER SEED, not only as a mean. On these tasks the outcome tends to
    # be bimodal -- some seeds find the reward and some never do -- and the
    # mean of {0.5, 0.5, 0.95} describes no run that happened. The spread is
    # over SEEDS, which is the real uncertainty about this config; the spread
    # over the eval episodes inside one run is not.
    print(f"\n{args.model}  over {len(results)} seed(s)")
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
