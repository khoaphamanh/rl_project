"""Entry point for the hyperparameter search: `python main.py MLP|LSTM|GRU|
TRANSFORMER` runs hpo() (the optuna study) then final() (reports the winning
trial's saved runs; trains nothing). --trials/--final-only/--search-only run
one phase at a time. Resumable: the study and sampler are checkpointed to
disk, so re-running continues an interrupted search."""

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

    config = make_config(args.model)
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
