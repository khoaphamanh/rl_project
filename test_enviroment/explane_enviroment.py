"""
Inspect the input (observation) and output (action) size of the MiniGrid
Memory environment used in this project, and document the full step-by-step
interaction protocol (reset -> action -> step -> ... -> close) with the exact
type / shape / size of everything that crosses the agent-environment boundary.

Run with:
    conda activate rl_project
    python test_enviroment/test_input_output.py
"""

import gymnasium as gym
import minigrid
import numpy as np
from minigrid.core.actions import Actions
from minigrid.core.constants import (
    COLOR_TO_IDX,
    DIR_TO_VEC,
    OBJECT_TO_IDX,
    STATE_TO_IDX,
)

ENV_ID = "MiniGrid-MemoryS11-v0"

# obs['direction'] value -> (compass name, movement vector (dx, dy))
# vector comes straight from minigrid's own DIR_TO_VEC table.
# Grid coords: x grows right, y grows DOWN (image/array convention).
DIRECTION_MEANING = {
    i: (name, tuple(int(v) for v in DIR_TO_VEC[i]))
    for i, name in enumerate(["right (east)", "down (south)", "left (west)", "up (north)"])
}

# obs['image'] is (7, 7, 3): the last axis is NOT RGB, it is 3 id channels.
# channel index -> (name, id table it is encoded with)
IMAGE_CHANNEL_MEANING = {
    0: ("object id", OBJECT_TO_IDX),
    1: ("color id", COLOR_TO_IDX),
    2: ("state id", STATE_TO_IDX),
}

# What each of the 3 axes of obs['image'] indexes.
# Read one number as: obs['image'][x, y, channel] -> a uint8 id.
# axis index -> (short name, list of explanation lines)
IMAGE_AXIS_MEANING = {
    0: ("x", [
        "column in the AGENT's OWN frame (not world coords):",
        "x=0 = far left of the view ... x=6 = far right,",
        "x=3 = the column straight in front of the agent.",
    ]),
    1: ("y", [
        "row in the AGENT's OWN frame, measured FORWARD:",
        "y=0 = farthest cell ahead ... y=6 = the agent's own row.",
        "=> the agent always sits at [x=3, y=6] and looks towards y=0.",
    ]),
    2: ("channel", [
        "which id table the number belongs to (this axis is NOT RGB):",
        "c=0 object id, c=1 color id, c=2 state id.",
    ]),
}


def decode_table(table):
    """Invert a minigrid *_TO_IDX table into idx -> name."""
    return {idx: name for name, idx in table.items()}

# action value -> short description of what it does in this env
ACTION_MEANING = {
    Actions.left: "turn 90 deg left (in place, doesn't move)",
    Actions.right: "turn 90 deg right (in place, doesn't move)",
    Actions.forward: "move one cell forward (blocked by walls)",
    Actions.pickup: "pickup (MemoryEnv remaps this to 'toggle')",
    Actions.drop: "drop (no effect, nothing to carry in this env)",
    Actions.toggle: "toggle (no effect, no doors/objects to toggle here)",
    Actions.done: "no-op (episode end is decided by agent position, not this)",
}


def show_direction_meanings():
    print("\nDIRECTION VALUES (meaning of obs['direction'])")
    print("-" * 60)
    for value, (name, vec) in DIRECTION_MEANING.items():
        print(f"  {value} -> {name:<12} movement vector (dx,dy)={vec}")

    # Empirical demo: turning right 4x should cycle through all 4 values
    # and land back where it started.
    print("\n  demo: reset, then apply action 'right' 4 times in a row")
    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=0)
    name, _ = DIRECTION_MEANING[obs["direction"]]
    print(f"    reset          -> direction={obs['direction']} ({name})")
    for _ in range(4):
        obs, *_ = env.step(Actions.right)
        name, _ = DIRECTION_MEANING[obs["direction"]]
        print(f"    action=right   -> direction={obs['direction']} ({name})")
    env.close()


def show_action_meanings():
    print("\nACTION VALUES (meaning of env.step(action))")
    print("-" * 60)
    for action, meaning in ACTION_MEANING.items():
        print(f"  {action.value} -> {action.name:<8} : {meaning}")


def describe(value):
    """One-line 'what exactly is this object': python type + shape + size."""
    if isinstance(value, np.ndarray):
        return (
            f"np.ndarray  shape={value.shape}  dtype={value.dtype}  "
            f"size={value.size} numbers"
        )
    if isinstance(value, dict):
        return f"dict  len={len(value)}  keys={list(value)}"
    if isinstance(value, str):
        return f"str  len={len(value)} chars (scalar, no shape)"
    if isinstance(value, (bool, np.bool_)):
        return f"{type(value).__name__}  scalar  shape=()  value={value}"
    return f"{type(value).__name__}  scalar  shape=()  value={value}"


def print_channel_grid(image, channel, decode, indent="    "):
    """
    Print every value of one channel of obs['image'] as a 7x7 grid.

    The array is indexed image[x, y, c], but it is printed with rows = y and
    columns = x, so the grid on screen is laid out the way the agent sees it:
    top row = farthest ahead, bottom row = the agent's own row.
    """
    n_x, n_y = image.shape[0], image.shape[1]
    idx_to_name = decode_table(IMAGE_CHANNEL_MEANING[channel][1])
    width = max(len(name) for name in idx_to_name.values()) if decode else 3

    print(f"{indent}      " + " ".join(f"{'x=' + str(x):>{width}}" for x in range(n_x)))
    for y in range(n_y):
        cells = []
        for x in range(n_x):
            value = int(image[x, y, channel])
            cells.append(f"{idx_to_name.get(value, '?') if decode else value:>{width}}")
        tag = "   <- agent's own row" if y == n_y - 1 else ""
        print(f"{indent}y={y}   " + " ".join(cells) + tag)


def show_image_encoding():
    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=0)
    image = obs["image"]

    print(f"\nWHAT IS INSIDE obs['image'] (shape {image.shape})")
    print("-" * 60)

    # -------------------------------------------------------------
    # The 3 axes: what an index along each axis actually selects
    # -------------------------------------------------------------
    print("\n  THE 3 AXES OF obs['image'][x, y, channel]")
    for axis, (name, lines) in IMAGE_AXIS_MEANING.items():
        length = image.shape[axis]
        print(f"    axis {axis}  '{name}'  length={length}  "
              f"valid indices 0..{length - 1}")
        for line in lines:
            print(f"              {line}")
    print(f"    => {image.shape[0]} * {image.shape[1]} * {image.shape[2]} "
          f"= {image.size} uint8 numbers per timestep")

    # -------------------------------------------------------------
    # Axis 2 in detail: the value each channel can take
    # -------------------------------------------------------------
    print("\n  ALL POSSIBLE VALUES PER CHANNEL (axis 2)")
    for channel, (name, table) in IMAGE_CHANNEL_MEANING.items():
        ids = ", ".join(f"{idx}={key}" for key, idx in table.items())
        print(f"    [:, :, {channel}] {name:<10}: {ids}")
    print("    careful: color/state are 0-padding wherever the object id is")
    print("    'unseen'(0) or 'empty'(1). A 0 there decodes to 'red'/'open'")
    print("    but means nothing -> only read them on real objects.")

    print("\n  the view is EGOCENTRIC: it rotates with the agent, and cells")
    print("  behind walls are 'unseen' (object id 0) -> this is what makes the")
    print("  task partially observable and needs memory (recurrence).")

    # -------------------------------------------------------------
    # Every value, for one concrete observation
    # -------------------------------------------------------------
    print(f"\n  ALL {image.size} VALUES AFTER reset(seed=0)")
    print("  (rows = y = front..back, cols = x = left..right, as seen)")
    for channel, (name, _) in IMAGE_CHANNEL_MEANING.items():
        present = ", ".join(str(v) for v in np.unique(image[:, :, channel]))
        print(f"\n    channel {channel} - {name}   (values present: {present})")
        print("      raw ids:")
        print_channel_grid(image, channel, decode=False, indent="      ")
        print("      decoded:")
        print_channel_grid(image, channel, decode=True, indent="      ")

    # -------------------------------------------------------------
    # The other two observation keys, in full
    # -------------------------------------------------------------
    print("\n  THE OTHER OBSERVATION KEYS AFTER reset(seed=0)")
    name, vec = DIRECTION_MEANING[obs["direction"]]
    print(f"    obs['direction'] = {obs['direction']}  -> {name}, "
          f"world-frame vector (dx,dy)={vec}")
    print(f"    obs['mission']   = {obs['mission']!r}")
    env.close()


def show_protocol_summary():
    """One-page summary table of the full agent-environment protocol.
    Values (grid size, view size, max_steps, n actions) are read from a
    live env, not hardcoded, so this stays correct if ENV_ID changes."""
    env = gym.make(ENV_ID)
    u = env.unwrapped
    n = env.action_space.n

    rows = [
        ("1 CREATE", "env = gym.make(ENV_ID)",
         f"grid {u.width}x{u.height}, view {u.agent_view_size}x{u.agent_view_size}, "
         f"max_steps={u.max_steps}"),
        ("2 RESET", "obs, info = env.reset(seed=0)", "2-tuple"),
        ("3 ACT", "action = policy(obs)",
         f"net emits ({n},) logits -> one int in [0,{n - 1}]"),
        ("4 STEP", "env.step(action)", "5-tuple"),
        ("5 LOOP", "repeat 3-4", "while not (terminated or truncated)"),
        ("6 CLOSE", "env.close()", "-"),
    ]
    env.close()

    print("\nPROTOCOL SUMMARY")
    print("-" * 90)
    stage_w = max(len("Stage"), *(len(r[0]) for r in rows))
    call_w = max(len("Call"), *(len(r[1]) for r in rows))
    print(f"  {'Stage':<{stage_w}}  {'Call':<{call_w}}  Returns")
    print(f"  {'-' * stage_w}  {'-' * call_w}  {'-' * 40}")
    for stage, call, returns in rows:
        print(f"  {stage:<{stage_w}}  {call:<{call_w}}  {returns}")


def show_interaction_loop():
    """
    The actual protocol: what you call, in which order, and what comes back.
    Every shape below is printed from a live env, not hardcoded.
    """
    print("\nSTEP-BY-STEP INTERACTION WITH THE ENVIRONMENT")
    print("-" * 60)

    # --- [1] create --------------------------------------------------
    print("\n[1] CREATE   env = gym.make(ENV_ID)")
    env = gym.make(ENV_ID)
    u = env.unwrapped
    print(f"      env.observation_space : {env.observation_space}")
    print(f"      env.action_space      : {env.action_space}")
    print(f"      full grid   : {u.width} x {u.height} cells (not visible to the agent)")
    print(f"      agent view  : {u.agent_view_size} x {u.agent_view_size} cells")
    print(f"      max_steps   : {u.max_steps} (episode truncated after this)")

    # --- [2] reset ---------------------------------------------------
    print("\n[2] RESET    obs, info = env.reset(seed=0)")
    obs, info = env.reset(seed=0)
    print("      returns a 2-tuple (obs, info)")
    print(f"      obs                 : {describe(obs)}")
    for key, value in obs.items():
        print(f"        obs['{key}']{'':<{max(0, 10 - len(key))}}: {describe(value)}")
    print(f"      info                : {describe(info)}  (always empty here)")
    print("      -> the network INPUT per timestep is the (7,7,3)=147 image")
    print("         (+ optionally the direction int; the mission string is")
    print("          constant in this env, so it carries no information)")

    # --- [3] pick an action ------------------------------------------
    action = int(Actions.forward)
    print("\n[3] ACT      action = policy(obs)")
    print(f"      the network OUTPUT is {env.action_space.n} logits -> shape=({env.action_space.n},)")
    print("      argmax/sample over them gives ONE integer:")
    print(f"      action              : {describe(action)}  "
          f"({Actions(action).name}), must be in [0, {env.action_space.n - 1}]")

    # --- [4] step ----------------------------------------------------
    print("\n[4] STEP     obs, reward, terminated, truncated, info = env.step(action)")
    obs, reward, terminated, truncated, info = env.step(action)
    print("      returns a 5-tuple")
    print(f"      obs                 : {describe(obs)}   <- same layout as reset")
    for key, value in obs.items():
        print(f"        obs['{key}']{'':<{max(0, 10 - len(key))}}: {describe(value)}")
    print(f"      reward              : {describe(reward)}")
    print("        careful: it is a python int 0 on ordinary steps and a float")
    print("        only on success -> cast with float(reward) before buffering")
    print(f"      terminated          : {describe(terminated)}  (goal/failure reached)")
    print(f"      truncated           : {describe(truncated)}  (step limit hit)")
    print(f"      info                : {describe(info)}")
    print("      careful: obs['image'] is EGOCENTRIC (rotates with the agent)")
    print("      but obs['direction'] is in WORLD frame (absolute compass).")

    # --- [5] loop ----------------------------------------------------
    print("\n[5] LOOP     repeat [3] and [4] while not (terminated or truncated)")
    print("      live demo, 5 fixed actions from the state above:")
    header = f"      {'t':>3}  {'action':<14} {'obs image':<12} {'dir':>3} " \
             f"{'reward':>8}  {'term':<5} {'trunc':<5}"
    print(header)
    for t, a in enumerate([Actions.forward, Actions.right, Actions.forward,
                           Actions.left, Actions.forward]):
        obs, reward, terminated, truncated, info = env.step(a)
        print(f"      {t:>3}  {a.value} {a.name:<12} {str(obs['image'].shape):<12} "
              f"{obs['direction']:>3} {reward:>8.4f}  {str(terminated):<5} "
              f"{str(truncated):<5}")
        if terminated or truncated:
            break

    # --- [6] close ---------------------------------------------------
    print("\n[6] CLOSE    env.close()")
    env.close()

    # --- reward / episode-end rules -----------------------------------
    print("\n      REWARD AND EPISODE END (MemoryEnv rules)")
    print("      reward = 0.0 on every ordinary step (sparse!)")
    print("      reward = 1 - 0.9 * (step_count / max_steps)  -> terminated=True,")
    print("               when the agent steps onto the MATCHING object")
    print("      reward = 0.0                                 -> terminated=True,")
    print("               when the agent steps onto the WRONG object")
    print("      truncated=True when step_count >= max_steps (no extra reward)")

    # one full episode with a seeded random policy, to show the loop in practice
    env = gym.make(ENV_ID)
    obs, info = env.reset(seed=0)
    rng = np.random.default_rng(0)
    total_reward, steps = 0.0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = int(rng.integers(env.action_space.n))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
    env.close()
    print("\n      demo: one full episode with a uniform random policy (seed 0)")
    print(f"        steps taken   : {steps}")
    print(f"        return (sum r): {total_reward:.4f}")
    print(f"        terminated={terminated}, truncated={truncated}")


def main():
    print(f"Environment: {ENV_ID}")
    print("=" * 60)

    # -----------------------------------------------------------------
    # Meaning of the numerical values themselves
    # -----------------------------------------------------------------
    show_direction_meanings()
    show_action_meanings()
    show_image_encoding()

    # -----------------------------------------------------------------
    # How to actually talk to the environment, step by step
    # -----------------------------------------------------------------
    show_protocol_summary()
    show_interaction_loop()


if __name__ == "__main__":
    main()
