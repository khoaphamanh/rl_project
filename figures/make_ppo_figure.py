#!/usr/bin/env python3
"""The PPO objective, typeset as mathtext, for the "PPO objective" block.

Exports ppo_losses.png sized in inches at the size it occupies on the A0
poster, so a font size given here equals points on the printed poster. The
typography deliberately copies the equation list in the GRU panel of
make_network_figure.py: equation on the left, grey annotation on the right.

Usage
-----
    python make_ppo_figure.py --out figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_network_figure import DARK, GREY, use_plex

# the fill of the block behind it, so the panel and the figure are seamless
CARD = "#F7F9FC"

EQS = [
    (r"$\rho_t=\pi_\theta(a_t\mid s_t)\ /\ \pi_{\theta_{old}}(a_t\mid s_t)$",
     "probability ratio"),
    (r"$\mathcal{L}^{\pi}=-\,\mathbb{E}_t\left[\min\left(\rho_t\hat{A}_t,\ "
     r"\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\,\hat{A}_t\right)\right]$",
     "clipped policy loss"),
    (r"$\mathcal{L}^{V}=\mathbb{E}_t\left[\left(V_\theta(s_t)-"
     r"\hat{R}_t\right)^{2}\right]$",
     "value loss"),
    (r"$\mathcal{L}=\mathcal{L}^{\pi}+c_v\,\mathcal{L}^{V}-c_e\,"
     r"\mathcal{H}\left[\pi_\theta\right]$",
     "what is backpropagated"),
]

EQ_FS = 36        # points on the printed poster
NOTE_FS = 26
PITCH = 0.76      # inches between equation baselines
PAD_X = 0.34      # inset from the figure edge
PAD_Y = 0.30


def export(path, width, dpi):
    h = len(EQS) * PITCH + 2 * PAD_Y - (PITCH - EQ_FS * 1.3 / 72)
    fig = plt.figure(figsize=(width, h), dpi=dpi, facecolor=CARD)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.set_facecolor(CARD)

    y = h - PAD_Y - EQ_FS * 1.3 / 144
    for eq, note in EQS:
        ax.text(PAD_X, y, eq, fontsize=EQ_FS, color=DARK,
                ha="left", va="center")
        ax.text(width - PAD_X, y, note, fontsize=NOTE_FS, color=GREY,
                ha="right", va="center")
        y -= PITCH

    fig.savefig(path, dpi=dpi, facecolor=CARD)
    plt.close(fig)
    return path, h


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--width", type=float, default=18.01,
                    help="width in inches on the poster")
    ap.add_argument("--dpi", type=int, default=200, help="output resolution")
    args = ap.parse_args()

    use_plex()
    os.makedirs(args.out, exist_ok=True)
    p, h = export(os.path.join(args.out, "ppo_losses.png"), args.width, args.dpi)
    print(f"wrote {p}  ({args.width:.2f} x {h:.2f} in)")


if __name__ == "__main__":
    main()
