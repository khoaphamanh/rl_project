"""
Watch a trained policy play. The viewer half of main.py.

main.py trains and writes agents/pretrained_model/ppo_<encoder>_<env>.pth. This
opens that file in a pygame window and plays it on the eval mazes. The controls
come in two rows that mean the same thing at two scales: the top one moves
within an episode (a step at a time), the bottom one moves between episodes (a
maze at a time), and in both, left goes back and right forward.

No training happens here and nothing is written -- it only reads the
checkpoint. See Helper.watch_agent in config/helper.py.

Run with:
    python watch.py MLP            the MLP checkpoint, at the default 2.5/sec
    python watch.py GRU            the GRU checkpoint
    python watch.py LSTM 1         ... at 1 agent step per second
    python watch.py TRANSFORMER

The encoder is required and picks the Config subclass exactly the way main.py
does, so watcher and trainer always agree on the filename. The env comes from
config.py, so watch the encoder on the same env it was trained on -- a DoorKey
policy has no checkpoint under a MemoryS11 name and load_model will say so.
"""

import argparse

from config import make_config, MODEL_CHOICES


def main():
    parser = argparse.ArgumentParser(
        description="Watch a trained PPO policy play in a pygame window."
    )
    parser.add_argument(
        "model",
        type=str.upper,  # "gru" and "GRU" both work; choices stay uppercase
        choices=MODEL_CHOICES,
        help="which feature extractor's checkpoint to load (%(choices)s)",
    )
    parser.add_argument(
        "steps_per_sec",
        nargs="?",
        type=float,
        default=None,  # None -> use watch_agent's own default, not duplicated here
        help="agent steps per second (default 2.5)",
    )
    args = parser.parse_args()

    # make_config sets name_model / path_model from the encoder AND the env, so
    # unlike the old manual rebuild there is nothing to keep in sync by hand
    config = make_config(args.model)

    print(f"loading {config.path_model}")
    print(
        "buttons: STEP -1   | PAUSE/PLAY | STEP +1     within one episode\n"
        "         LAST GAME | REPLAY     | NEW GAME    between eval mazes\n"
        "         AUTO NEW GAME (walks the whole eval set unattended)\n"
        "keys:    SPACE pause   <- -> step   P/N maze   R replay   A auto   Q quit"
    )

    # blocks until the window is closed
    if args.steps_per_sec is not None:
        config.watch_agent(steps_per_sec=args.steps_per_sec)
    else:
        config.watch_agent()


if __name__ == "__main__":
    main()
