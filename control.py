"""Play the project's MiniGrid env yourself, from the keyboard, to see what the
agent's 7x7 observation actually contains -- and, with --detail, every number
in it. Loads no model and writes nothing.

    python control.py            # maze + observation + what is in it
    python control.py --detail   # ... plus every channel value, raw and decoded

The env is whatever config/config.py names: name_env, force_cue_visible and the
worker_steps time limit all come from the shared Config, so this always plays
the same game the agents are trained on. The viewer itself is Helper.play_env in
config/helper.py.
"""

import argparse

from config import make_config


def main():
    parser = argparse.ArgumentParser(
        description="Play the configured MiniGrid env by hand in a pygame window."
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="add a third column showing every value of obs['image'], raw and "
        "decoded (default: off)",
    )
    args = parser.parse_args()

    # Config is abstract, so this needs an encoder to build -- but nothing the
    # explorer reads is per-encoder: name_env, force_cue_visible and
    # worker_steps all live on the shared base. MLP is just the cheapest build.
    config = make_config("MLP")

    print(f"{config.name_env}   (max_steps {config.worker_steps})")
    print(f"  force_cue_visible {config.force_cue_visible}")
    print(f"  detail panel      {'on' if args.detail else 'off (--detail adds it)'}")

    config.play_env(detail=args.detail)


if __name__ == "__main__":
    main()
