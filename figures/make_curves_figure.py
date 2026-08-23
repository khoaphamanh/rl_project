#!/usr/bin/env python3
"""The evaluation-return learning curves, for the Results block of the poster.

Reads the per-iteration curves straight out of the plotly export written by
compare.py (agents/comparison/compare_curve_return_mean.html), so the figure
can be re-rendered at any size without re-running the studies.

Sized in inches at the size it occupies on the A0 poster, so a font size given
here equals points on the printed poster.

Usage
-----
    python make_curves_figure.py --out figures --width 18.45 --height 6.10
"""
import argparse
import base64
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_network_figure import DARK, GREY, use_plex

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                   "agents", "comparison", "compare_curve_return_mean.html")

# study name in the export -> label on the poster, in legend order
ARMS = [
    ("MLP", "MLP (memoryless)"),
    ("GRU (tbptt 1)", "GRU, L = 1"),
    ("GRU (tbptt 4)", "GRU, L = 4"),
    ("GRU (tbptt 8)", "GRU, L = 8"),
    ("GRU (full BPTT)", "GRU, full BPTT"),
]
CHANCE = 0.5

# margins in inches, so the plotting area keeps its proportions at any size
M_LEFT, M_RIGHT, M_TOP, M_BOTTOM = 1.60, 0.68, 0.90, 1.30


def load(path=SRC):
    """Pull the trace list out of the Plotly.newPlot(...) call in the export."""
    html = open(path).read()
    tail = re.search(r'Plotly\.newPlot\(\s*"[^"]+"\s*,\s*(\[.*)', html, re.S).group(1)
    depth = 0
    for i, c in enumerate(tail):
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return json.loads(tail[:i + 1])
    raise ValueError("no trace array in " + path)


def arr(v):
    """Plotly writes long arrays as base64; short ones stay plain lists."""
    if isinstance(v, dict) and "bdata" in v:
        import numpy as np
        return np.frombuffer(base64.b64decode(v["bdata"]), dtype=v["dtype"])
    return v


def export(path, width, height, dpi):
    traces = load()
    lines = {t["legendgroup"]: t for t in traces if t.get("showlegend") is not False}
    bands = {t["legendgroup"]: t for t in traces if t.get("showlegend") is False}

    fig = plt.figure(figsize=(width, height), dpi=dpi, facecolor="white")
    ax = fig.add_axes([M_LEFT / width,
                       M_BOTTOM / height,
                       (width - M_LEFT - M_RIGHT) / width,
                       (height - M_TOP - M_BOTTOM) / height])

    for key, label in ARMS:
        colour = lines[key]["line"]["color"]
        b = bands.get(key)
        if b:
            ax.fill(arr(b["x"]), arr(b["y"]), color=colour, alpha=0.12, lw=0, zorder=1)
        ax.plot(arr(lines[key]["x"]), arr(lines[key]["y"]), color=colour, lw=3.4,
                label=label, solid_capstyle="round", zorder=3)

    ax.axhline(CHANCE, color=GREY, lw=2.0, ls=(0, (6, 5)), zorder=2)
    ax.text(18, CHANCE + 0.045, "chance", fontsize=26, color="#9AA0A6",
            ha="left", va="bottom", zorder=4)

    ax.set_xlim(0, 1000)
    ax.set_ylim(-0.04, 1.06)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("PPO iteration", fontsize=34, color=DARK, labelpad=8)
    ax.set_ylabel("evaluation return", fontsize=34, color=DARK, labelpad=8)
    ax.tick_params(labelsize=30, colors=DARK, length=6, width=1.6, pad=6)
    ax.grid(True, color="#DFE4EA", lw=1.4, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9AA0A6")
        ax.spines[side].set_linewidth(1.6)

    # anchored to the figure, not the axes, so the row of five stays centred
    # on the panel however wide the y-label margin is
    ax.legend(loc="lower center", ncol=5, frameon=False, fontsize=25,
              handlelength=1.5, handletextpad=0.5, columnspacing=1.6,
              labelcolor=DARK, bbox_transform=fig.transFigure,
              bbox_to_anchor=(0.5, 1.0 - M_TOP / height))

    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--width", type=float, default=18.45,
                    help="width in inches on the poster")
    ap.add_argument("--height", type=float, default=6.10,
                    help="height in inches on the poster")
    ap.add_argument("--dpi", type=int, default=200, help="output resolution")
    args = ap.parse_args()

    use_plex()
    os.makedirs(args.out, exist_ok=True)
    p = export(os.path.join(args.out, "compare_curves.png"),
               args.width, args.height, args.dpi)
    print(f"wrote {p}  ({args.width:.2f} x {args.height:.2f} in)")


if __name__ == "__main__":
    main()
