"""Entry point for the hand-picked run: no optuna, values come straight from
config/config_no_hpo.py. `python main_no_hpo.py [ENCODER] [--report-only]`
trains (or, with --report-only, just reports) every seed in seed_list and
writes to no_hpo/, parallel to the search's hpo/. Always retrains from
scratch and overwrites -- there is no resume here, unlike main.py."""

import argparse
import os

from agents.ppo import PPOAgent
from config import MODEL_CHOICES
from config.config_no_hpo import ConfigNoHPO, FEATURE_EXTRACTOR


def report_run(logger, eval_history, fresh, curve_deterministic):
    """Log learning curve and FINAL blocks (sampled/argmax). Returns (sampled, argmax)."""
    last = eval_history[-1]
    if curve_deterministic:
        argmax, sampled = last, fresh
    else:
        sampled, argmax = last, fresh

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


def run_one(model_name, seed, logger, tbptt=None):
    """Train encoder from seed. Returns a result_row (seed + SUMMARY_METRICS)."""
    config = ConfigNoHPO(model_name, tbptt_length=tbptt)
    agent = PPOAgent(config, seed=seed)

    logger.info("")
    logger.info("-" * 78)
    logger.info(f"SEED {seed}  --  training {model_name} on {config.name_env}")
    logger.info("-" * 78)

    config.log_model_summary(agent.model, logger)

    try:
        history = agent.train_agent(logger=logger)
        # Evaluate in the mode not used for the learning curve
        fresh = agent.evaluate(deterministic=not config.eval_deterministic)

        logger.info("")
        logger.info(f"ran {len(history)} iterations")
        sampled, argmax = report_run(
            logger, agent.eval_history, fresh, config.eval_deterministic
        )

        return config.seed_result_row(seed, sampled, argmax)
    finally:
        agent.close()


def report_saved(model_name, seed_index, seed, logger, tbptt=None):
    """Load checkpoint and report it (no training). Returns dict or None if missing."""
    config = ConfigNoHPO(model_name, tbptt_length=tbptt)
    path = config.select_run(seed_index=seed_index)
    if not os.path.exists(path):
        logger.info(f"  seed {seed}: no checkpoint at {path}, skipped")
        return None

    config.apply_params(config.checkpoint_params(path))
    agent = PPOAgent(config, seed=seed)
    try:
        checkpoint = config.load_model(agent.model, path)
        curve = list(checkpoint.get("eval_history") or [])
        if not curve:
            logger.info(f"  seed {seed}: {path} holds no eval_history, skipped")
            return None

        curve_deterministic = bool(
            checkpoint.get("eval_deterministic", config.eval_deterministic)
        )
        fresh = agent.evaluate(deterministic=not curve_deterministic)

        logger.info("")
        logger.info(f"REPORT ONLY  seed {seed}  --  loaded {path}, nothing trained")
        sampled, argmax = report_run(logger, curve, fresh, curve_deterministic)

        return config.seed_result_row(seed, sampled, argmax)
    finally:
        agent.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO with hand-picked hyperparameters (config/config_no_hpo.py)."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=FEATURE_EXTRACTOR,
        type=str.upper,
        choices=MODEL_CHOICES,
        help=f"which feature extractor to train (%(choices)s, default {FEATURE_EXTRACTOR})",
    )
    parser.add_argument(
        "--tbptt",
        type=int,
        default=None,
        metavar="L",
        help="GRU only: truncate the gradient's backward reach to L steps "
        "(default: full BPTT). Writes to pretrained_model_GRU_tbptt<L>/no_hpo/ "
        "so it cannot overwrite the full-BPTT baseline in "
        "pretrained_model_GRU/no_hpo/.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="skip training; load and report checkpoints in no_hpo/ (no retraining)",
    )
    args = parser.parse_args()

    if args.tbptt is not None and args.tbptt < 1:
        parser.error(f"--tbptt {args.tbptt} must be at least 1 step")

    # ConfigNoHPO raises for --tbptt on an MLP; a command-line mistake deserves
    # a usage error, not a traceback
    try:
        config = ConfigNoHPO(args.model, tbptt_length=args.tbptt)
    except ValueError as error:
        parser.error(str(error))

    seeds = config.seed_list
    logger = config.build_logger()

    mode = "REPORT ONLY (no training)" if args.report_only else "run"
    logger.info("")
    logger.info(f"NO-HPO {mode}: {args.model} on {config.name_env}")
    logger.info(f"  seeds        {seeds}")
    logger.info(f"  tbptt_length {config.tbptt_length} (worker_steps {config.worker_steps})")
    if not args.report_only:
        logger.info(f"  iterations   {config.n_iterations}")
    logger.info(f"  checkpoints  {config.dir_pretrained_model}")

    if args.report_only:
        results = [
            result
            for index, seed in enumerate(seeds)
            if (result := report_saved(args.model, index, seed, logger, args.tbptt))
            is not None
        ]
        if not results:
            raise SystemExit(
                f"\nno checkpoint could be read under {config.dir_pretrained_model}. "
                f"Run `python main_no_hpo.py {args.model}"
                f"{'' if args.tbptt is None else f' --tbptt {args.tbptt}'}` first -- --report-only "
                f"reads runs, it does not make them."
            )
    else:
        results = [run_one(args.model, s, logger, args.tbptt) for s in seeds]

    log = logger.info

    # the same table HPOPPO.final ends with, so the two are read the same way
    config.log_seed_summary(
        logger, results, header=f"{args.model}  over {len(results)} seed(s)"
    )

    # the same number the study maximizes, so hand-picked and tuned runs are
    # read off one scale. Scored on the sampled column, like HPO.
    scored = {
        "return_mean": "sampled_return",
        "success_rate": "sampled_success_rate",
    }[config.hpo_metric]
    score = config.aggregate_scores([r[scored] for r in results])
    log("")
    log(f"  SCORE  {config.score_name} = {score:.4f}")

    # what plot_hpo draws for the search, drawn here for the hand-picked run.
    # Reads eval_history off the checkpoints, so --report-only plots too.
    config.plot_eval_curves(
        config.dir_pretrained_model,
        name="no_hpo",
        logger=logger,
    )


if __name__ == "__main__":
    main()
