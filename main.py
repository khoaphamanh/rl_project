"""Entry point for the hyperparameter search: `python main.py MLP|GRU
[--tbptt L]` runs hpo() (the optuna study) then final() (reports the winning
trial's saved runs; trains nothing). Without --tbptt the GRU gets full BPTT;
with it, the gradient's backward reach is fixed at L for the whole study and
the results land in their own directory, so several lengths can be searched
independently and compared afterwards. --trials/--final-only/--search-only run
one phase at a time. Resumable: the study and sampler are checkpointed to disk,
so re-running continues an interrupted search."""

import argparse

from agents.hpo_ppo import HPOPPO
from config import make_config, MODEL_CHOICES


def main():
    parser = argparse.ArgumentParser(
        description="Search hyperparameters for PPO with one feature extractor."
    )
    parser.add_argument(
        "model",
        type=str.upper,
        choices=MODEL_CHOICES,
        help="which feature extractor to tune (%(choices)s)",
    )
    parser.add_argument(
        "--tbptt",
        type=int,
        default=None,
        metavar="L",
        help="GRU only: fix the gradient's backward reach at L steps for this "
        "whole study, writing to pretrained_model_GRU_tbptt<L>/. Omit for full "
        "BPTT (pretrained_model_GRU/). Each length is its own study, so nothing "
        "inside a study ever has to rank one length against another.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="override config.n_trials for this run",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="skip the search; just report the winning trial's saved runs",
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="run the search but skip the final report",
    )
    args = parser.parse_args()

    if args.tbptt is not None and args.tbptt < 1:
        parser.error(f"--tbptt {args.tbptt} must be at least 1 step")

    # make_config raises for --tbptt on an MLP; turn it into a usage error
    # rather than a traceback, since that is a command-line mistake
    try:
        config = make_config(args.model, tbptt_length=args.tbptt)
    except ValueError as error:
        parser.error(str(error))

    if args.trials is not None:
        config.n_trials = args.trials

    logger = config.build_logger()
    hpo = HPOPPO(config, logger=logger)

    if not args.final_only:
        hpo.hpo()

    if not args.search_only:
        hpo.final()


if __name__ == "__main__":
    main()
