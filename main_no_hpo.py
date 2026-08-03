"""Entry point for the hand-picked run: no optuna, values come straight from
config/config_no_hpo.py. `python main_no_hpo.py [ENCODER] [--report-only]`
trains (or, with --report-only, just reports) every seed in seed_list and
writes to no_hpo/, parallel to the search's hpo/. Always retrains from
scratch and overwrites -- there is no resume here, unlike main.py."""

import argparse
import os

import numpy as np

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


def run_one(model_name, seed, logger):
    """Train encoder from seed. Returns {"seed", "success_sampled", "success_argmax"}."""
    config = ConfigNoHPO(model_name)
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

        return {
            "seed": seed,
            "success_sampled": sampled["success_rate"],
            "success_argmax": argmax["success_rate"],
        }
    finally:
        agent.close()


def report_saved(model_name, seed_index, seed, logger):
    """Load checkpoint and report it (no training). Returns dict or None if missing."""
    config = ConfigNoHPO(model_name)
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

        return {
            "seed": seed,
            "success_sampled": sampled["success_rate"],
            "success_argmax": argmax["success_rate"],
            "logger": logger,
        }
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
        "--report-only",
        action="store_true",
        help="skip training; load and report checkpoints in no_hpo/ (no retraining)",
    )
    args = parser.parse_args()

    config = ConfigNoHPO(args.model)
    seeds = config.seed_list
    logger = config.build_logger()

    mode = "REPORT ONLY (no training)" if args.report_only else "run"
    logger.info("")
    logger.info(f"NO-HPO {mode}: {args.model} on {config.name_env}")
    logger.info(f"  seeds        {seeds}")
    if not args.report_only:
        logger.info(f"  iterations   {config.n_iterations}")
    logger.info(f"  checkpoints  {config.dir_pretrained_model}")

    if args.report_only:
        results = [
            result
            for index, seed in enumerate(seeds)
            if (result := report_saved(args.model, index, seed, logger)) is not None
        ]
        if not results:
            raise SystemExit(
                f"\nno checkpoint could be read under {config.dir_pretrained_model}. "
                f"Run `python main_no_hpo.py {args.model}` first -- --report-only "
                f"reads runs, it does not make them."
            )
    else:
        results = [run_one(args.model, s, logger) for s in seeds]

    log = logger.info
    log("")
    log(f"{args.model}  over {len(results)} seed(s)")
    log(f"{'seed':>8} {'sampled':>9} {'argmax':>9}")
    for r in results:
        log(f"{r['seed']:>8} {r['success_sampled']:>9.3f} {r['success_argmax']:>9.3f}")

    for key in ("success_sampled", "success_argmax"):
        vals = [r[key] for r in results]
        log(
            f"{key:>16}  mean {np.mean(vals):.3f}  std {np.std(vals):.3f}  "
            f"median {np.median(vals):.3f}  "
            f"iqr {np.percentile(vals, 75) - np.percentile(vals, 25):.3f}"
        )


if __name__ == "__main__":
    main()
