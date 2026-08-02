"""
Watch a trained policy play. The viewer half of main.py and main_no_hpo.py.

Training writes one checkpoint per seed; this opens ONE of them in a pygame
window and plays it on the eval mazes. The controls come in two rows that mean
the same thing at two scales: the top one moves within an episode (a step at a
time), the bottom one moves between episodes (a maze at a time), and in both,
left goes back and right forward.

No training happens here and nothing is written -- it only reads the
checkpoint. See Helper.watch_agent in config/helper.py.

WHICH CHECKPOINT. Four things pick it out, and they are the four flags:

    model      MLP / GRU / LSTM / TRANSFORMER   which encoder      (default MLP)
    --hpo      tuned run or hand-picked run     which directory    (default off)
    --seed     an INDEX into config.seed_list   which file         (default 0)
    --trial    best / <n>                       which run, if --hpo (default best)

    python watch.py                       MLP, hand-picked, first seed
    python watch.py GRU                   the GRU, hand-picked
    python watch.py GRU --hpo             the GRU's WINNING TRIAL -- the run
                                          the writeup reports
    python watch.py GRU --hpo --trial 7         one particular draw
    python watch.py GRU --hpo --seed 1          the second seed of that run
    python watch.py LSTM --hpo 1                ... at 1 agent step per second

--trial ONLY MEANS SOMETHING WITH --hpo, and passing it without is an error
rather than a silent no-op: a hand-picked run is one run, so there is no trial
to choose and a request for one is a misunderstanding worth surfacing.

WHY --seed IS AN INDEX AND NOT A SEED. seed_list is [0, 26, 98] for a study and
whatever config_no_hpo.py says for a hand-picked run, so the same index means a
different seed in the two modes -- which is the point. `--seed 1` is "the second
run", and out-of-range says so with the list printed.

--trial final IS ACCEPTED AND MEANS best. There used to be a separate hpo/final/
holding a retrain at the winning params; there is no retrain any more, and the
alias is only here so an old command does not fail on a directory that no longer
exists. See HPOPPO.final.

The env comes from the config, so watch an encoder on the env it was trained on;
load_model checks the checkpoint's env, encoder, widths and force_cue_visible
against the live config and refuses a mismatch rather than showing you nonsense.
The tuned hyperparameters ride along INSIDE the checkpoint, so a trial that was
drawn at hidden_size=384 rebuilds at 384 without being told.
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
    # default=None, not "best", so that "not given" and "given as best" are
    # distinguishable -- which is what lets --trial without --hpo be an error
    # instead of being quietly ignored. The default is applied below.
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

    # THE FLAG PICKS THE CONFIG CLASS, and the class already knows where its own
    # checkpoints live: ConfigNoHPO points dir_pretrained_model at no_hpo/, and
    # select_run repoints it inside hpo/ for the tuned case. Nothing here spells
    # a directory, so the viewer cannot drift from the trainer.
    config = make_config(args.model) if args.hpo else ConfigNoHPO(args.model)
    trial = (args.trial or "best") if args.hpo else None

    # sets dir_pretrained_model and config.seed, and hands back the path they
    # imply. Raises on an out-of-range seed index; the file itself is checked
    # by load_model, further in.
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

    # blocks until the window is closed.
    #
    # THE MISSING FILE IS NOT A BUG, so it does not get a traceback: naming a
    # run that has not been trained yet -- --hpo before any study has been run,
    # a --seed that exists in seed_list but whose run was interrupted, --trial
    # for a number that was pruned -- is the ordinary way to mistype this
    # command, and the path already in the message is the whole answer.
    try:
        if args.steps_per_sec is not None:
            config.watch_agent(path_model=path, steps_per_sec=args.steps_per_sec)
        else:
            config.watch_agent(path_model=path)
    except FileNotFoundError as error:
        raise SystemExit(f"\n{error}") from None


if __name__ == "__main__":
    main()
