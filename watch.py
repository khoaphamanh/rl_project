"""Watch a trained policy play in a pygame window: reads one checkpoint, trains
nothing, writes nothing. `watch.py GRU` opens the GRU study's winning trial,
`watch.py GRU --tbptt 8` the winner of the --tbptt 8 study, and `--no-hpo` the
hand-picked run instead; --help lists every flag. The viewer itself is
Helper.watch_agent in config/helper.py.
"""

import argparse

from config import make_config, MODEL_CHOICES
from config.config_no_hpo import ConfigNoHPO


def main():
    """Parse the command line, resolve it to exactly one checkpoint, and play
    it. Takes no arguments -- everything comes from argv: model (str, MLP|GRU),
    steps_per_sec (float), --hpo/--no-hpo and --fullscreen (flags), --tbptt
    (int, GRU only), --seed (int, an INDEX into seed_list), --trial (str,
    best|N|final, tuned runs only). Blocks until the window closes."""
    parser = argparse.ArgumentParser(
        description="Watch a trained PPO policy play in a pygame window."
    )
    parser.add_argument(
        "model",
        nargs="?",  # OPTIONAL, so a bare `python watch.py` still does something
        default="MLP",
        type=str.upper,  # "gru" and "GRU" both work; choices stay uppercase
        help="which feature extractor's checkpoint to load (%(choices)s, default MLP)",
        choices=MODEL_CHOICES,
    )
    parser.add_argument(
        "steps_per_sec",
        nargs="?",
        type=float,
        default=None,  # None -> use watch_agent's own default, not duplicated here
        help="agent steps per second (default 2.5)",
    )
    # Default ON: the tuned runs are what the results are reported from, and
    # the only ones a fresh clone ships. --no-hpo opts out of them.
    parser.add_argument(
        "--hpo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="watch the TUNED run, pretrained_model_<ENC>/hpo/ (the default); "
        "--no-hpo watches the hand-picked one in no_hpo/ instead",
    )
    parser.add_argument(
        "--tbptt",
        type=int,
        default=None,
        metavar="L",
        help="GRU only: watch a run trained with --tbptt L "
        "(pretrained_model_GRU_tbptt<L>/) instead of the full-BPTT one. Works "
        "with and without --hpo -- the length names the directory either way.",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="open filling the screen instead of in a 1000x880 window. The "
        "window is resizable either way and F11 toggles it while running -- "
        "the layout is drawn at a fixed size and scaled to fit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="INDEX",
        help="which seed to watch, as an INDEX into config.seed_list (default 0)",
    )
    # default=None, not "best", so "not given" and "given as best" stay
    # distinguishable -- that is what makes --trial with --no-hpo an error.
    parser.add_argument(
        "--trial",
        type=str,
        default=None,
        metavar="WHICH",
        help="which tuned run: 'best' (the winning trial, default), a trial "
        "number, or 'final' (an accepted alias for 'best'). Not valid with "
        "--no-hpo.",
    )
    args = parser.parse_args()

    if args.trial is not None and not args.hpo:
        parser.error(
            f"--trial {args.trial} cannot be combined with --no-hpo. A "
            f"hand-picked run has no trials -- it is one run, in no_hpo/, and "
            f"--seed alone picks the file."
        )

    # The flag picks the config class, and the class knows where its own
    # checkpoints live, so the viewer cannot drift from the trainer.
    try:
        config = (
            make_config(args.model, tbptt_length=args.tbptt)
            if args.hpo
            else ConfigNoHPO(args.model, tbptt_length=args.tbptt)
        )
    except ValueError as error:
        parser.error(str(error))
    trial = (args.trial or "best") if args.hpo else None

    # sets dir_pretrained_model and config.seed, returns the path they imply.
    # Raises on an out-of-range seed index; the file is checked by load_model.
    path = config.select_run(trial=trial, seed_index=args.seed)

    where = f"hpo / {trial}" if args.hpo else "no_hpo"
    print(f"{args.model} on {config.name_env}   [{where}]")
    print(f"  seed    {config.seed}   (index {args.seed} of {config.seed_list})")
    print(f"  loading {path}")
    print(
        "buttons: STEP -1   | PAUSE/PLAY | STEP +1     within one episode\n"
        "         LAST GAME | REPLAY     | NEW GAME    between eval mazes\n"
        "         AUTO NEW GAME (walks the whole eval set unattended)\n"
        "keys:    SPACE pause   <- -> step   P/N maze   R replay   A auto\n"
        "         F11 fullscreen   Q quit"
    )

    # Blocks until the window closes. A missing file is not a bug (an untrained
    # run is the ordinary mistype), so it gets the path, not a traceback.
    try:
        if args.steps_per_sec is not None:
            config.watch_agent(
                path_model=path,
                steps_per_sec=args.steps_per_sec,
                fullscreen=args.fullscreen,
            )
        else:
            config.watch_agent(path_model=path, fullscreen=args.fullscreen)
    except FileNotFoundError as error:
        raise SystemExit(f"\n{error}") from None


if __name__ == "__main__":
    main()
