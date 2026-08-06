"""Watch a trained policy play in a pygame window: reads one checkpoint, trains
nothing, writes nothing. `watch.py GRU --hpo` opens the GRU's winning trial,
`watch.py GRU` the hand-picked run; --help lists every flag. The viewer itself
is Helper.watch_agent in config/helper.py.
"""

import argparse

from config import make_config, MODEL_CHOICES
from config.config_no_hpo import ConfigNoHPO


def main():
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
    parser.add_argument(
        "--hpo",
        action="store_true",
        help="watch the TUNED run (pretrained_model_<ENC>/hpo/) instead of the "
        "hand-picked one (no_hpo/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="INDEX",
        help="which seed to watch, as an INDEX into config.seed_list (default 0)",
    )
    # default=None, not "best", so "not given" and "given as best" stay
    # distinguishable -- that is what makes --trial without --hpo an error.
    parser.add_argument(
        "--trial",
        type=str,
        default=None,
        metavar="WHICH",
        help="with --hpo: 'best' (the winning trial, default), a trial number, "
        "or 'final' (an accepted alias for 'best')",
    )
    args = parser.parse_args()

    if args.trial is not None and not args.hpo:
        parser.error(
            f"--trial {args.trial} needs --hpo. A hand-picked run has no trials "
            f"-- it is one run, in no_hpo/, and --seed alone picks the file."
        )

    # the flag picks the config class, and the class knows where its own
    # checkpoints live -- nothing here spells a directory, so the viewer
    # cannot drift from the trainer.
    config = make_config(args.model) if args.hpo else ConfigNoHPO(args.model)
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
        "keys:    SPACE pause   <- -> step   P/N maze   R replay   A auto   Q quit"
    )

    # Blocks until the window closes. A missing file is not a bug -- naming a
    # run that was never trained is the ordinary way to mistype this command --
    # so it gets the path, not a traceback.
    try:
        if args.steps_per_sec is not None:
            config.watch_agent(path_model=path, steps_per_sec=args.steps_per_sec)
        else:
            config.watch_agent(path_model=path)
    except FileNotFoundError as error:
        raise SystemExit(f"\n{error}") from None


if __name__ == "__main__":
    main()
