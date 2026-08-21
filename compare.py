"""Compare the winning trial of every tuned study against the others, in one
figure per metric. Trains nothing and evaluates nothing -- it reads the curves
already stored in each hpo/best_trial/ checkpoint.

    python compare.py                          # every study found under agents/
    python compare.py --models GRU GRU_tbptt8  # just these two
    python compare.py --out figures/           # write somewhere else

Each study contributes one mean +- 1 std band over its seeds, so the encoder
comparison (MLP vs GRU) and the tbptt ablation (GRU vs GRU_tbptt<L>) are read
off the same axes. Two figures per metric are written: the learning curves over
iterations, and the final value as a bar with the same mean +- std as an error
bar. Runs measured in different eval modes, or on different envs, are dropped
rather than drawn -- those numbers are not comparable, see CLAUDE.md.
"""

import argparse
import csv
import json
import os
import re

import numpy as np
import plotly.graph_objects as go
import torch

from config import make_config, MODEL_CHOICES
from config.helper import _HPO_METRICS, Helper, log_with

# Directories a tuned study writes to: pretrained_model_GRU_tbptt8 and friends.
_DIR_PATTERN = re.compile(r"^pretrained_model_([A-Za-z]+)(?:_tbptt(\d+))?$")

# Categorical slots in fixed order, never cycled -- a 9th study would need its
# own figure rather than a ninth hue. Validated for CVD on a white surface.
_PALETTE = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#4a3aa7",  # violet
    "#008300",  # green
    "#12a4c9",  # cyan
    "#e34948",  # red
)


def studies_root():
    """Where the pretrained_model_* directories live (str). Taken off a built
    config rather than spelled here, so discovery and the trainer cannot
    disagree about the directory."""
    return os.path.dirname(make_config("MLP").dir_model)


def discover_runs(root=None):
    """Find every tuned study on disk, newest naming scheme only.

    Args:
        root (str | None): where to look; None asks studies_root().

    Returns:
        list[tuple]: (encoder, tbptt_length) pairs, encoder upper-case and
        tbptt_length an int or None for full BPTT. MLP first, then each
        encoder's full-BPTT baseline ahead of its truncated lengths.
    """
    if root is None:
        root = studies_root()

    runs = []
    if not os.path.isdir(root):
        return runs

    for entry in sorted(os.listdir(root)):
        match = _DIR_PATTERN.match(entry)
        if not match or not os.path.isdir(os.path.join(root, entry)):
            continue

        encoder = match.group(1).upper()
        if encoder not in MODEL_CHOICES:
            continue

        runs.append((encoder, None if match.group(2) is None else int(match.group(2))))

    # full BPTT ahead of the truncated lengths: it is the baseline they are
    # being compared against, so it should lead the legend and the palette
    return sorted(
        runs,
        key=lambda run: (MODEL_CHOICES.index(run[0]), run[1] is not None, run[1] or 0),
    )


def run_key(encoder, tbptt_length):
    """What --models spells a study as: the directory's own suffix, upper-case.

    Args:
        encoder (str): "MLP" or "GRU".
        tbptt_length (int | None): backward reach; None = full BPTT.

    Returns:
        str: e.g. "MLP", "GRU" or "GRU_TBPTT8".
    """
    return encoder if tbptt_length is None else f"{encoder}_TBPTT{tbptt_length}"


def run_label(encoder, tbptt_length):
    """The legend/axis name for one study.

    Args:
        encoder (str): "MLP" or "GRU".
        tbptt_length (int | None): backward reach; None = full BPTT.

    Returns:
        str: e.g. "MLP", "GRU (full BPTT)" or "GRU (tbptt 8)".
    """
    if encoder != "GRU":
        return encoder
    return "GRU (full BPTT)" if tbptt_length is None else f"GRU (tbptt {tbptt_length})"


def style_map(runs):
    """Pin one colour per study, keyed by name rather than position.

    Assigned from every study on disk, so narrowing the figure with --models
    leaves the survivors the style they already had.

    Args:
        runs (list[tuple]): (encoder, tbptt_length) pairs, in palette order.

    Returns:
        dict: {run_key: {"colour": hex}}.

    Raises:
        ValueError: more studies than the palette defines slots for.
    """
    if len(runs) > len(_PALETTE):
        raise ValueError(
            f"{len(runs)} studies is more than the {len(_PALETTE)} colours "
            f"this palette defines, and a cycled hue would make two of them "
            f"look like one. Narrow it with --models."
        )
    return {
        run_key(*run): {"colour": _PALETTE[index]} for index, run in enumerate(runs)
    }


def read_meta(directory):
    """What a directory of checkpoints was measured with, off the first file
    that loads. Cheap sanity data, not the curves themselves.

    Args:
        directory (str): a best_trial/ directory.

    Returns:
        dict: name_env, eval_deterministic and tbptt_length as stored, or {}
        when nothing in the directory could be read.
    """
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".pth"):
            continue
        try:
            checkpoint = torch.load(
                os.path.join(directory, name), map_location="cpu", weights_only=True
            )
        except Exception:
            continue
        if isinstance(checkpoint, dict) and "name_env" in checkpoint:
            return {
                key: checkpoint.get(key)
                for key in ("name_env", "eval_deterministic", "tbptt_length")
            }
    return {}


def read_study_score(config):
    """The headline number final() wrote for a study, if it got that far.

    Args:
        config (Config): the study's config, which knows its own paths.

    Returns:
        tuple: (score_name, score), both None when there is no final_*.json.
    """
    path = os.path.join(config.dir_hpo_best_trial, f"final_{config.name_hpo}.json")
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None, None
    return data.get("score_name"), data.get("score")


def collect(runs, styles=None, logger=None):
    """Load every study's winning-trial curves and check they can be compared.

    Args:
        runs (list[tuple]): (encoder, tbptt_length) pairs from discover_runs.
        styles (dict | None): from style_map; None styles these runs alone,
            which only matches the unfiltered figure when `runs` is the whole
            set.
        logger (logging.Logger | None): where the notes go; None prints.

    Returns:
        list[dict]: one entry per usable study -- label, histories, colour,
        score_name, score and the eval mode its curves were measured in.
        A study with no best_trial/, no curves, or measured on another env or
        in the other eval mode is skipped with a note.
    """
    say = log_with(logger)
    if styles is None:
        styles = style_map(runs)

    collected, envs, modes = [], set(), set()
    for encoder, tbptt_length in runs:
        label = run_label(encoder, tbptt_length)

        try:
            config = make_config(encoder, tbptt_length=tbptt_length)
        except ValueError as error:
            say(f"{label}: {error}, skipped")
            continue

        # the config knows where its own checkpoints live, so nothing here
        # spells a directory and the comparison cannot drift from the trainer
        directory = config.dir_hpo_best_trial
        if not os.path.isdir(directory):
            say(f"{label}: no {directory}, skipped (has the study run final()?)")
            continue

        histories = config.load_eval_histories(directory)
        if not histories:
            say(f"{label}: no eval_history under {directory}, skipped")
            continue

        meta = read_meta(directory)
        score_name, score = read_study_score(config)

        envs.add(meta.get("name_env", config.name_env))
        modes.add(bool(meta.get("eval_deterministic", config.eval_deterministic)))

        collected.append(
            {
                "label": label,
                **styles[run_key(encoder, tbptt_length)],
                "directory": directory,
                "histories": histories,
                "seeds": list(histories),
                "score_name": score_name,
                "score": score,
                "name_env": meta.get("name_env", config.name_env),
                "deterministic": bool(
                    meta.get("eval_deterministic", config.eval_deterministic)
                ),
            }
        )

    # A sampled curve and an argmax one are not comparable, and neither are two
    # envs -- drawing them together invents a difference that is the measurement.
    if len(envs) > 1 or len(modes) > 1:
        keep_env, keep_mode = collected[0]["name_env"], collected[0]["deterministic"]
        dropped = [
            entry["label"]
            for entry in collected
            if entry["name_env"] != keep_env or entry["deterministic"] != keep_mode
        ]
        say(
            f"not comparable, keeping env {keep_env} / "
            f"{'argmax' if keep_mode else 'sampled'} eval and dropping "
            f"{', '.join(dropped)}"
        )
        collected = [
            entry
            for entry in collected
            if entry["name_env"] == keep_env and entry["deterministic"] == keep_mode
        ]

    return collected


def curves(entry, metric):
    """One study's mean +- std curve for one metric.

    Args:
        entry (dict): an element of collect()'s list.
        metric (str): "return_mean" or "success_rate".

    Returns:
        tuple: (iterations, mean, std) as lists/arrays, or (None, None, None)
        when this study never recorded the metric.
    """
    iterations, _, values = Helper.curve_table(entry["histories"], metric)
    if not iterations:
        return None, None, None

    # nan-aware: a seed missing one iteration must not blank the whole row
    return iterations, np.nanmean(values, axis=1), np.nanstd(values, axis=1)


def _header(collected, metric):
    """The subtitle both figures share: env, eval mode and how many seeds."""
    seeds = sorted({len(entry["seeds"]) for entry in collected})
    mode = "argmax" if collected[0]["deterministic"] else "sampled"
    return (
        f"{collected[0]['name_env']}  --  winning trial of each study  --  "
        f"{'/'.join(str(n) for n in seeds)} seed(s), {mode} eval"
    )


def plot_curves(collected, metric, out_dir, logger=None, include_plotlyjs="cdn"):
    """Overlay every study's mean +- std learning curve for one metric.

    Args:
        collected (list[dict]): from collect().
        metric (str): which metric to draw.
        out_dir (str): directory to write into; created if missing.
        logger (logging.Logger | None): where the notes go; None prints.
        include_plotlyjs (str | bool): plotly's write_html switch -- "cdn"
            keeps the file small, True inlines the library for offline use.

    Returns:
        list[str]: the paths written; [] when no study has the metric.
    """
    say = log_with(logger)
    fig = go.Figure()
    drawn = 0

    # every band first, so no study's band paints over another's mean line
    for entry in collected:
        iterations, mean, std = curves(entry, metric)
        if iterations is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=list(iterations) + list(iterations)[::-1],
                y=list(mean + std) + list(mean - std)[::-1],
                fill="toself",
                fillcolor=Helper._rgba(entry["colour"], 0.12),
                # spelled out: plotly defaults to "lines+markers" under 20
                # points, which would make the band's edges dotted
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                legendgroup=entry["label"],
                showlegend=False,
            )
        )

    for entry in collected:
        iterations, mean, std = curves(entry, metric)
        if iterations is None:
            say(f"{entry['label']}: no {metric!r} in its curves, left out")
            continue
        drawn += 1
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=mean,
                mode="lines",
                line=dict(color=entry["colour"], width=2.5),
                name=entry["label"],
                # the bands are hoverinfo="skip", so the spread has to ride
                # along on the mean line to be readable
                customdata=np.stack([std, mean - std, mean + std], axis=-1),
                legendgroup=entry["label"],
                hovertemplate=(
                    f"{entry['label']}<br>" + metric + " %{y:.3f} +- "
                    "%{customdata[0]:.3f}"
                    "<br>band [%{customdata[1]:.3f}, %{customdata[2]:.3f}]"
                    "<extra></extra>"
                ),
            )
        )

    if not drawn:
        say(f"no study recorded {metric!r}, no curve figure written")
        return []

    fig.update_layout(
        title=dict(
            text=f"{metric}  --  mean over seeds and +- 1 std"
            f"<br><sub>{_header(collected, metric)}</sub>"
        ),
        xaxis_title="iteration",
        yaxis_title=metric,
        template="plotly_white",
        hovermode="x unified",
        width=1000,
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        margin=dict(t=110, r=30, b=60, l=70),
    )

    return _write(
        fig, os.path.join(out_dir, f"compare_curve_{metric}"), say, include_plotlyjs
    )


def plot_final(collected, metric, out_dir, logger=None, include_plotlyjs="cdn"):
    """The end of each study's curve as a bar, mean +- std across its seeds.

    Args:
        collected (list[dict]): from collect().
        metric (str): which metric to draw.
        out_dir (str): directory to write into; created if missing.
        logger (logging.Logger | None): where the notes go; None prints.
        include_plotlyjs (str | bool): plotly's write_html switch.

    Returns:
        list[str]: the paths written; [] when no study has the metric.
    """
    say = log_with(logger)

    labels, means, stds, colours, last = [], [], [], [], []
    for entry in collected:
        iterations, mean, std = curves(entry, metric)
        if iterations is None:
            continue
        labels.append(entry["label"])
        means.append(float(mean[-1]))
        stds.append(float(std[-1]))
        colours.append(entry["colour"])
        last.append(iterations[-1])

    if not labels:
        say(f"no study recorded {metric!r}, no final figure written")
        return []

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=means,
            marker=dict(color=colours, cornerradius=4),
            error_y=dict(type="data", array=stds, thickness=1.5, width=8),
            customdata=np.array(last),
            hovertemplate=(
                "%{x}<br>iteration %{customdata}<br>"
                + metric
                + " %{y:.3f}<extra></extra>"
            ),
        )
    )

    # The value above each bar, anchored above the error bar's cap -- plotly's
    # own textposition="outside" would draw the label straight through it.
    for label, mean, std in zip(labels, means, stds):
        fig.add_annotation(
            x=label,
            y=mean + std,
            text=f"{mean:.3f} +- {std:.3f}",
            yshift=14,
            showarrow=False,
        )

    fig.update_layout(
        title=dict(
            text=f"{metric} at the last evaluated iteration  --  mean +- 1 std"
            f"<br><sub>{_header(collected, metric)}</sub>"
        ),
        yaxis_title=metric,
        template="plotly_white",
        showlegend=False,  # the x axis already names every bar
        width=1000,
        height=560,
        margin=dict(t=110, r=30, b=60, l=70),
    )
    # bars are read against zero; the headroom is for the labels above the caps
    top = max(mean + std for mean, std in zip(means, stds))
    fig.update_yaxes(range=[0, (top or 1) * 1.12])

    return _write(
        fig, os.path.join(out_dir, f"compare_final_{metric}"), say, include_plotlyjs
    )


def _write(fig, stem, say, include_plotlyjs):
    """Write one figure as .html and .svg.

    Args:
        fig (go.Figure): the figure.
        stem (str): output path without an extension.
        say (callable): where the notes go.
        include_plotlyjs (str | bool): plotly's write_html switch.

    Returns:
        list[str]: the paths written. A missing kaleido costs the .svg only.
    """
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)

    fig.write_html(f"{stem}.html", include_plotlyjs=include_plotlyjs)
    paths = [f"{stem}.html"]

    try:
        fig.write_image(f"{stem}.svg")
        paths.append(f"{stem}.svg")
    except Exception as error:
        say(f"could not write {stem}.svg ({type(error).__name__}: {error})")

    return paths


def write_table(collected, metrics, out_dir, logger=None):
    """The same numbers the figures show, as a csv and on the terminal -- the
    table view the colour-blind and greyscale cases fall back to.

    Args:
        collected (list[dict]): from collect().
        metrics (tuple[str]): which metrics to tabulate.
        out_dir (str): directory to write into; created if missing.
        logger (logging.Logger | None): where the notes go; None prints.

    Returns:
        str | None: the csv path written, or None when not one study recorded
        any of the metrics asked for.
    """
    say = log_with(logger)

    rows = []
    for entry in collected:
        for metric in metrics:
            iterations, mean, std = curves(entry, metric)
            if iterations is None:
                continue
            rows.append(
                {
                    "run": entry["label"],
                    "metric": metric,
                    "iteration": iterations[-1],
                    "mean": round(float(mean[-1]), 4),
                    "std_across_seeds": round(float(std[-1]), 4),
                    "seeds": len(entry["seeds"]),
                    "study_score_name": entry["score_name"] or "",
                    "study_score": (
                        "" if entry["score"] is None else round(entry["score"], 4)
                    ),
                }
            )

    # every metric asked for was missing from every study, so there is no table
    # to write -- the figures said the same thing already
    if not rows:
        say(f"no study recorded any of {', '.join(metrics)}, no csv written")
        return None

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "compare_summary.csv")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    say("")
    say(f"{'run':>18}  {'metric':>14}  {'iter':>5}  {'mean +- std':>18}  {'seeds':>5}")
    for row in rows:
        say(
            f"{row['run']:>18}  {row['metric']:>14}  {row['iteration']:>5}  "
            f"{row['mean']:>8.3f} +- {row['std_across_seeds']:<7.3f}  "
            f"{row['seeds']:>5}"
        )

    return path


def main():
    """Parse the command line, load every study it names and write the figures.
    Takes no arguments -- everything comes from argv: --models, --metrics,
    --out and --offline."""
    parser = argparse.ArgumentParser(
        description="Compare the winning trial of every tuned study, in one "
        "mean +- std figure per metric."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="NAME",
        default=None,
        help="which studies to draw, named as their directories are suffixed: "
        "MLP, GRU, GRU_tbptt8. Default: every one found. Narrowing the set "
        "does not change any study's colour.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        metavar="NAME",
        default=list(_HPO_METRICS),
        help=f"which metrics to draw (default: {' '.join(_HPO_METRICS)})",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("agents", "comparison"),
        help="directory for the figures and the csv (default: agents/comparison)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="inline plotly.js into every .html instead of linking the CDN, so "
        "the figures open without a network (bigger files)",
    )
    args = parser.parse_args()

    root = studies_root()
    found = discover_runs(root)
    if not found:
        parser.error(f"no pretrained_model_* directory under {root}")

    # styled from every study on disk, not just the ones asked for, so
    # --models narrows the figure without repainting what survives
    styles = style_map(found)

    runs = found
    if args.models is not None:
        wanted = {name.upper() for name in args.models}
        runs = [run for run in found if run_key(*run) in wanted]
        if not runs:
            parser.error(
                f"none of {', '.join(sorted(wanted))} is a study under {root}. "
                f"Found: {', '.join(run_key(*run) for run in found)}"
            )

    collected = collect(runs, styles=styles)
    if not collected:
        parser.error("no study had a winning trial with curves to compare")

    print(f"comparing {len(collected)}: {', '.join(e['label'] for e in collected)}")

    paths = []
    for metric in args.metrics:
        include = True if args.offline else "cdn"
        paths += plot_curves(collected, metric, args.out, include_plotlyjs=include)
        paths += plot_final(collected, metric, args.out, include_plotlyjs=include)

    table = write_table(collected, tuple(args.metrics), args.out)
    if table is not None:
        paths.append(table)

    if not paths:
        parser.error(
            f"nothing to write: no study recorded any of "
            f"{', '.join(args.metrics)}"
        )

    print("")
    for path in paths:
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
