"""Render the task figure at the top of the README: the full 17x17 maze with the
agent in the corridor, next to the 7x7 patch it actually sees at that moment.

Run from the repo root -- it writes figures/task_memory_s17.png.
"""

import gymnasium as gym
import minigrid  # noqa: F401  -- registers the MiniGrid envs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
CELL = 32  # pixels per grid cell in the 544x544 render

env = gym.make("MiniGrid-MemoryS17Random-v0", render_mode="rgb_array", highlight=True)
env.reset(seed=SEED)
u = env.unwrapped
full = env.render()
pov = u.get_frame(agent_pov=True, highlight=False, tile_size=48)
print("agent", u.agent_pos, "dir", u.agent_dir)
env.close()

BG = "#15171a"
FG = "#e6e6e6"
BLUE = "#8ab4f8"
RED = "#ff8a8a"

fig = plt.figure(figsize=(13.5, 7.4), dpi=150, facecolor=BG)
grid = fig.add_gridspec(
    1, 2, width_ratios=[1.55, 1], wspace=0.06, left=0.03, right=0.97,
    top=0.91, bottom=0.14,
)

ax = fig.add_subplot(grid[0, 0])
ax.imshow(full)
ax.set_axis_off()
ax.set_title("MiniGrid-MemoryS17Random-v0   (17 x 17)", color=FG, fontsize=13, pad=14)


def cell(x, y):
    """Centre of grid cell (x, y), in render pixels."""
    return (x + 0.5) * CELL, (y + 0.5) * CELL


def label(text, target, where, color=FG):
    ax.annotate(
        text,
        xy=cell(*target),
        xytext=cell(*where),
        color=color,
        fontsize=11,
        ha="center",
        va="center",
        linespacing=1.45,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, shrinkA=4, shrinkB=6),
    )


label("cue: one key or one ball,\nseen only if the agent turns back", (1, 7), (6.2, 3.2), BLUE)
label("agent, facing east", (9, 8), (3.6, 11.6), RED)
label("fork: one key, one ball.\nOnly the prong that matches the cue pays", (14, 10), (7.4, 14.0), BLUE)

ax2 = fig.add_subplot(grid[0, 1])
ax2.imshow(pov)
ax2.set_axis_off()
ax2.set_title("everything the agent observes: 7 x 7, egocentric",
              color=FG, fontsize=13, pad=14)

fig.text(
    0.5,
    0.045,
    "The pale band is the agent's current 7 x 7 view: the cue is already behind it.\n"
    "Both prongs look the same whichever object was the cue, so the answer can only "
    "come from memory.",
    color=FG,
    fontsize=11.5,
    ha="center",
    va="center",
    linespacing=1.5,
)

fig.savefig("figures/task_memory_s17.png", facecolor=BG)
print("written figures/task_memory_s17.png")
