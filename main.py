"""
Entry point for the SEARCH. Tunes PPO with ONE encoder, named on the command line.

    python main.py MLP
    python main.py GRU
    python main.py LSTM
    python main.py TRANSFORMER

WITH NO FLAGS THAT IS THE WHOLE JOB: the search, and then the final retrain at
the params it found. Two phases, one command --

    1. hpo()     n_trials draws, each trained once per seed in seed_list,
                 pruned and resumable       -> hpo/trial_<n>/
                 the winner copied out of it -> hpo/best_trial/
    2. final()   ONE retrain at the winning params, which is the number that
                 goes in the writeup        -> hpo/final/

Each of those three directories holds one checkpoint per seed plus the two
learning-curve figures drawn from them, in .html and .svg. final/ also holds
final_<ENC>_<env>.json, the report. See the agents/hpo_ppo.py docstring for
the whole tree.

The flags exist only to run one phase without the other, which is what you
want when a study is already finished or when you mean to inspect it first:

    python main.py GRU --trials 10     a shorter study than config.n_trials
    python main.py GRU --final-only    skip the search; retrain at the best
                                       params already in the study
    python main.py GRU --search-only   search, but do not retrain

The encoder is the only argument because it is the only thing that changes
between runs of the ablation. make_config(name) builds the matching Config
subclass (config/__init__.py); everything else -- env, rollout size, which
knobs are searched -- is decided there.

NO PPO AND NO OPTUNA HAPPEN HERE. main.py builds the config, hands it to
HPOPPO (agents/hpo_ppo.py) and calls two methods. The search itself -- trials,
pruning, resuming -- lives in that class, and the training it drives lives in
agents/ppo.py.

    python main.py GRU              tuned      -> pretrained_model_GRU/hpo/
    python main_no_hpo.py GRU       hand-picked -> pretrained_model_GRU/no_hpo/

Those two are the pair the writeup compares, and they cannot overwrite each
other: separate directories, by construction.

RE-RUNNING IS RESUMING. The study is kept in a sqlite file and the sampler is
pickled beside it, so the same command picks up where an interrupted one
stopped -- a crashed trial is retried rather than skipped, and finished trials
are not paid for twice. Run it again after a ctrl-c and it continues.
"""

import argparse

from agents.hpo_ppo import HPOPPO
from config import make_config, MODEL_CHOICES


def main():
    parser = argparse.ArgumentParser(
        description="Search hyperparameters for PPO with one feature extractor."
    )
    parser.add_argument(
        "model",
        type=str.upper,  # so "mlp" and "MLP" both work; choices stay uppercase
        choices=MODEL_CHOICES,
        help="which feature extractor to tune (%(choices)s)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="override config.n_trials for this run",
    )
    # NAMED "-only" BOTH, so that neither reads as "do the final step" -- which
    # is what the default already does. With no flags this runs BOTH phases;
    # each flag suppresses the other phase.
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="skip the search; retrain at the best params already in the study",
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="run the search but skip the final retrain",
    )
    args = parser.parse_args()

    # the encoder decides which Config subclass, and the subclass decides both
    # which architecture knobs exist and which of them are searched over
    config = make_config(args.model)
    if args.trials is not None:
        config.n_trials = args.trials

    # ONE log file for the whole study, not one per trial: built here and
    # handed down, so thirty trials and the final retrain all land in the same
    # file and can be read in order
    logger = config.build_logger()

    hpo = HPOPPO(config, logger=logger)

    # PHASE 1: the search. Runs unless explicitly skipped.
    if not args.final_only:
        hpo.hpo()

    # PHASE 2: the number to report. Runs unless explicitly skipped.
    # NOT study.best_value, which is the maximum of n_trials noisy
    # measurements and biased upward by the selection itself -- this retrains
    # at the winning params and scores that. See HPOPPO.final.
    if not args.search_only:
        hpo.final()


if __name__ == "__main__":
    main()
