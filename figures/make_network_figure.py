#!/usr/bin/env python3
"""Figures for the actor-critic skeleton and the GRU encoder.

Exports three PNGs (all sized in inches at the size they occupy on the A0 poster,
so a font size given here equals points on the printed poster):

    network_combined.png   both panels side by side   18.93 x 4.28 in
    model_architecture.png actor-critic skeleton only  11.20 x 4.28 in
    gru_architecture.png   GRU unrolled + equations     7.80 x 4.28 in

Usage
-----
    python make_network_figure.py                 # all three, into ./
    python make_network_figure.py --out figures   # into figures/
    python make_network_figure.py --dpi 300

Fonts: IBM Plex Sans if installed, otherwise matplotlib's default sans.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --------------------------------------------------------------------------- style
BLUE = "#0B5394"
GREY = "#5F6368"
DARK = "#202124"
RED = "#b3261e"
GREEN = "#1e7d4f"
DIVIDER = "#dfe4ea"

PLEX_DIRS = [
    "/usr/share/fonts/truetype/plex",
    os.path.expanduser("~/Library/Fonts"),
    "/Library/Fonts",
]


def use_plex():
    """Register IBM Plex Sans if it is on the system; fall back silently."""
    for d in PLEX_DIRS:
        if os.path.isdir(d):
            for f in font_manager.findSystemFonts(d):
                if "Plex" in os.path.basename(f):
                    try:
                        font_manager.fontManager.addfont(f)
                    except Exception:
                        pass
    names = {f.name for f in font_manager.fontManager.ttflist}
    if "IBM Plex Sans" in names:
        plt.rcParams["font.family"] = "IBM Plex Sans"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"


# --------------------------------------------------------------------- primitives
def box(ax, cx, cy, w, h, title, sub=None, fc="#F4F6F9", ec=BLUE, fs=26, subfs=21):
    """Rounded box with a bold title and an optional grey sub-line."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.15",
        fc=fc, ec=ec, lw=2.6, zorder=3))
    if sub:
        ax.text(cx, cy + 0.19, title, ha="center", va="center",
                fontsize=fs, color=DARK, fontweight="bold", zorder=4)
        ax.text(cx, cy - 0.24, sub, ha="center", va="center",
                fontsize=subfs, color=GREY, zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=fs, color=DARK, fontweight="bold", zorder=4)


def arrow(ax, x0, y0, x1, y1, color=GREY, lw=2.6, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=28, lw=lw,
        color=color, connectionstyle="arc3,rad=%s" % rad,
        shrinkA=2, shrinkB=2, zorder=2))


def canvas(width, height, dpi):
    fig = plt.figure(figsize=(width, height), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


# ------------------------------------------------------------------------ panel A
def draw_skeleton(ax, x0=0.0, heading=True):
    """observation -> encoder -> actor / critic heads. Occupies ~11.2 x 4.28 in."""
    if heading:
        ax.text(x0 + 0.30, 3.95, "actor-critic skeleton", ha="left", va="center",
                fontsize=24, color=BLUE, fontweight="bold")
    yc = 2.05
    box(ax, x0 + 2.00, yc, 3.55, 1.15,
        "observation", "7 × 7 × 3", subfs=20)
    arrow(ax, x0 + 3.80, yc, x0 + 4.35, yc)
    box(ax, x0 + 6.05, yc, 3.05, 1.15, "encoder", "MLP  or  GRU", ec="#333333")
    arrow(ax, x0 + 7.65, yc + 0.16, x0 + 8.20, yc + 0.78, color=RED, rad=-0.18)
    arrow(ax, x0 + 7.65, yc - 0.16, x0 + 8.20, yc - 0.78, color=GREEN, rad=0.18)
    box(ax, x0 + 9.62, yc + 1.02, 2.55, 0.78, "actor head",
        fc="#FDF2F1", ec=RED, fs=24)
    box(ax, x0 + 9.62, yc - 1.02, 2.55, 0.78, "critic head",
        fc="#F0F7F2", ec=GREEN, fs=24)
    ax.text(x0 + 9.62, yc + 0.40, "Linear(H, 7)",
            ha="center", va="center", fontsize=21, color=GREY)
    ax.text(x0 + 9.62, yc - 0.42, "Linear(H, 1)",
            ha="center", va="center", fontsize=21, color=GREY)
    ax.text(x0 + 0.30, 0.36, "the encoder is the only thing that changes between arms",
            ha="left", va="center", fontsize=21, color=GREY, style="italic")


# ------------------------------------------------------------------------ panel B
GRU_EQS = [
    (r"$r_t=\sigma\,(W_r\,x_t+U_r\,h_{t-1})$", "reset gate"),
    (r"$z_t=\sigma\,(W_z\,x_t+U_z\,h_{t-1})$", "update gate"),
    (r"$n_t=\tanh\,(W_n\,x_t+r_t\odot U_n\,h_{t-1})$", "candidate state"),
    (r"$h_t=(1-z_t)\odot n_t+z_t\odot h_{t-1}$", "new hidden state"),
]


def draw_gru(ax, x0, x1, heading=True):
    """Unrolled GRU chain plus the forward-pass equations, inside [x0, x1]."""
    if heading:
        ax.text(x0 + 0.28, 3.95, "GRU encoder, unrolled", ha="left", va="center",
                fontsize=24, color=BLUE, fontweight="bold")

    yb, bw, bh = 3.10, 1.35, 0.62
    cx = [x0 + 1.53, x0 + 3.58, x0 + 5.63]
    for x in cx:
        box(ax, x, yb, bw, bh, "GRU", fc="#EEF3FA", ec=BLUE, fs=23)

    h_lbl = [r"$h_{t-2}$", r"$h_{t-1}$", r"$h_{t}$", r"$h_{t+1}$"]
    arrow(ax, x0 + 0.33, yb, cx[0] - bw / 2, yb)
    ax.text(x0 + 0.60, yb + 0.47, h_lbl[0], fontsize=22, color=BLUE, ha="center")
    for i in range(2):
        arrow(ax, cx[i] + bw / 2, yb, cx[i + 1] - bw / 2, yb)
        ax.text((cx[i] + cx[i + 1]) / 2, yb + 0.47, h_lbl[i + 1],
                fontsize=22, color=BLUE, ha="center")
    arrow(ax, cx[2] + bw / 2, yb, x1 - 0.58, yb)
    ax.text(x1 - 0.98, yb + 0.47, h_lbl[3], fontsize=22, color=BLUE, ha="center")

    x_lbl = [r"$x_{t-1}$", r"$x_{t}$", r"$x_{t+1}$"]
    for i, x in enumerate(cx):
        arrow(ax, x, 2.32, x, yb - bh / 2, color="#3c7fd0")
        ax.text(x, 2.14, x_lbl[i], fontsize=22, color=DARK,
                ha="center", va="center")

    y = 1.74
    for eq, note in GRU_EQS:
        ax.text(x0 + 0.33, y, eq, fontsize=24, color=DARK, ha="left", va="center")
        ax.text(x1 - 0.25, y, note, fontsize=19, color=GREY, ha="right", va="center")
        y -= 0.43


# ------------------------------------------------------------------------ exports
H = 4.28  # every figure shares this height so panels stay comparable


def export_combined(path, dpi):
    w = 18.93
    fig, ax = canvas(w, H, dpi)
    draw_skeleton(ax, x0=0.0)
    ax.plot([11.12, 11.12], [0.22, 4.06], color=DIVIDER, lw=2)
    draw_gru(ax, x0=11.12, x1=w)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def export_skeleton(path, dpi):
    w = 11.20
    fig, ax = canvas(w, H, dpi)
    draw_skeleton(ax, x0=0.0)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def export_gru(path, dpi):
    w = 7.80
    fig, ax = canvas(w, H, dpi)
    draw_gru(ax, x0=0.0, x1=w)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--dpi", type=int, default=200, help="output resolution")
    args = ap.parse_args()

    use_plex()
    os.makedirs(args.out, exist_ok=True)
    for name, fn in (("network_combined.png", export_combined),
                     ("model_architecture.png", export_skeleton),
                     ("gru_architecture.png", export_gru)):
        p = fn(os.path.join(args.out, name), args.dpi)
        print("wrote", p)


if __name__ == "__main__":
    main()
