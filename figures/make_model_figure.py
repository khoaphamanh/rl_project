"""Render the network figure in the README's Training section: one shared
encoder feeding a linear actor head and a linear critic head.

Run from the repo root -- it writes figures/model_architecture.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BG = "#15171a"
FG = "#e6e6e6"
MUTED = "#9aa0a6"
BLUE = "#8ab4f8"
RED = "#ff8a8a"
GREEN = "#8fd19e"

fig = plt.figure(figsize=(13.5, 5.0), dpi=150, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 37)
ax.set_axis_off()
ax.set_facecolor(BG)


def box(x, y, w, h, title, sub, edge, fill="#1e2126"):
    """Rounded box centred on (x, y), titled above its subtitle."""
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.2",
            linewidth=1.6,
            edgecolor=edge,
            facecolor=fill,
        )
    )
    ax.text(x, y + 1.5, title, color=FG, fontsize=12, ha="center", va="center")
    ax.text(x, y - 1.9, sub, color=MUTED, fontsize=9.5, ha="center", va="center",
            linespacing=1.4)


def arrow(x0, y0, x1, y1, color=MUTED, style="-|>", conn="arc3,rad=0"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle=style, color=color, linewidth=1.5,
            mutation_scale=14, connectionstyle=conn, shrinkA=0, shrinkB=0,
        )
    )


MID, TOP, BOT = 17.0, 26.5, 7.5

box(9, MID, 16, 11, "observation", "(batch, seq, 7, 7, 3)\nuint8, egocentric", BLUE)
box(29, MID, 16, 11, "one-hot + flatten", "7 x 7 x 20 = 980\nper timestep", MUTED)
box(50, MID, 18, 14, "encoder", "MLP: Linear+ReLU blocks\nGRU: recurrent, carries h", FG)
box(72, TOP, 16, 9, "actor head", "Linear(hidden, 7)", RED)
box(72, BOT, 16, 9, "critic head", "Linear(hidden, 1)", GREEN)
box(92, TOP, 12, 9, "action", "7 logits", RED)
box(92, BOT, 12, 9, "value", "1 scalar", GREEN)

arrow(17, MID, 21, MID)
arrow(37, MID, 41, MID)
arrow(59, MID, 64, TOP, RED, conn="arc3,rad=-0.18")
arrow(59, MID, 64, BOT, GREEN, conn="arc3,rad=0.18")
arrow(80, TOP, 86, TOP, RED)
arrow(80, BOT, 86, BOT, GREEN)

# the GRU's hidden state, looping out of the encoder and back into it
ax.add_patch(
    FancyArrowPatch(
        (45, 24), (55, 24), arrowstyle="-|>", color=BLUE, linewidth=1.5,
        mutation_scale=14, connectionstyle="arc3,rad=-1.5", linestyle="--",
    )
)
ax.text(50, 33.5, "hidden state h, passed to the next timestep (GRU only)",
        color=BLUE, fontsize=10, ha="center", va="center")

ax.text(
    50, 1.6,
    "The encoder is the only part that differs between arms; both heads always read the same features.",
    color=MUTED, fontsize=10.5, ha="center", va="center",
)

fig.savefig("figures/model_architecture.png", facecolor=BG)
print("written figures/model_architecture.png")
