"""
Watch a trained policy play. The viewer half of main.py.

main.py trains and writes agents/pretrained_model/ppo_<encoder>.pth. This
opens that file in a pygame window and plays it on the eval mazes, with two
buttons: NEW GAME (the next eval maze) and REPLAY (the same one again).

No training happens here and nothing is written -- it only reads the
checkpoint. See Helper.watch_agent in config/helper.py.

Run with:
    python watch.py            the encoder config.py is currently set to
    python watch.py LSTM       a specific one, whatever config.py says
    python watch.py GRU 1      ... at 1 agent step per second (default 2.5)
"""

import os
import sys

from config.config import Config


def main():
    config = Config()

    # optional: pick the encoder from the command line, so all three saved
    # runs can be watched without editing config.py between them
    if len(sys.argv) > 1:
        config.recurrent_model = sys.argv[1].upper()

        # REBUILD THE PATH. name_model and path_model were computed once in
        # Config.__init__, from the encoder name as it stood THEN -- assigning
        # recurrent_model here does not reach back and update them, so without
        # these two lines "python watch.py LSTM" would build an LSTM and then
        # try to load ppo_GRU.pth into it. (build_extractor() and is_lstm are
        # methods, so those do follow the change on their own.)
        config.name_model = f"ppo_{config.recurrent_model}.pth"
        config.path_model = os.path.join(config.dir_pretrained_model, config.name_model)

    print(f"loading {config.path_model}")
    print(
        "buttons: STEP -1 | PAUSE/PLAY | STEP +1,  NEW GAME | REPLAY,\n"
        "         AUTO NEW GAME (walks the whole eval set unattended)\n"
        "keys:    SPACE pause   <- -> step   N new   R replay   A auto   Q quit"
    )

    # blocks until the window is closed. No steps_per_sec unless one was asked
    # for, so the default lives in watch_agent and is not duplicated here.
    if len(sys.argv) > 2:
        config.watch_agent(steps_per_sec=float(sys.argv[2]))
    else:
        config.watch_agent()


if __name__ == "__main__":
    main()
