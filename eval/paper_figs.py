#!/usr/bin/env python3
"""
Paper-sized render of fig_guard_sensitivity (2026-08-20 figure regeneration).

Why this file exists. `figures.py` renders `fig_guard_sensitivity.png` at figsize 11.5x4.2 in,
10 pt fonts, 150 dpi. Included at 0.68\\linewidth in the NeurIPS manuscript that prints at
2.8-3.3 pt text (visual-story-editor storyboard 2026-08-20, section 1). The fix the author
approved is "regenerate same content, larger fonts": same data, same panels, same series,
markers, colours and annotations, at a figsize that matches the 5.5 in NeurIPS text width so
fonts render at their nominal size when the PNG is included at width=\\linewidth.

What this wrapper does, and does not do.
  - Imports `fig_guard_sensitivity` from `figures.py` unchanged; the shipped script and its
    default output are untouched.
  - Rebuilds the `cls` frame exactly as `figures.main()` does (same six lines) but WITHOUT
    going through `common.parse_args`, which stamps `eval/out/_DATA_ROOT.txt` as a side effect.
    Nothing under eval/out/ is read or written by this script; the only input is
    `<data-root>/golden_solutions.csv`.
  - `--mode paper` (default): sets paper rcParams (font sizes below) and monkeypatches
    `plt.subplots` so the hard-coded `figsize=(11.5, 4.2)` becomes (5.5, 2.0); renders at
    300 dpi with a tight bbox. The legend entries for the never-proposed population are
    renamed to "never-proposed ..." (storyboard section 2.2, terminology [major]) and the
    left panel's two-line title is replaced by a one-line neutral one (page budget, same
    day); those are the only text changes. Explicit `fontsize=` kwargs in the original
    (legend 8.5, tau 10, tail note 8, 68% annotation 9) are left as they are; at scale 1.0
    they print at those sizes.
  - `--mode paper27`: the first paper render of 2026-08-20 (5.5 x 2.7 in canvas, original
    two-line title), kept so that it can be reproduced byte-for-byte (md5 d35d142c...).
  - `--mode v4`: renders with the original style, untouched, to reproduce the previous PNG
    byte-for-byte (md5 check that the data have not drifted since the v4 figure was made).
  - `--check`: after rendering, test every text/legend box against every data artist in
    display coordinates and print the overlaps (none expected).
  - `--dump <json>`: writes every number the figure draws (histogram bin edges and densities,
    both sweep curves, the tau line, the 68% marker, all text) so a v4 vs paper dump can be
    diffed.

Usage (from the repo root, in the repo venv):
  .venv/bin/python eval/paper_figs.py --mode v4    --out /tmp/x/guard_v4.png    --dump /tmp/x/guard_v4.json
  .venv/bin/python eval/paper_figs.py --mode paper --out <paper>/figures/fig_guard_sensitivity.png --dump /tmp/x/guard_paper.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.path import Path as MPath
from matplotlib.transforms import Affine2D

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))

import common as C  # noqa: E402
import figures as F  # noqa: E402

# The data root the a2 artifacts and the v4 figure were built from (eval/out/_DATA_ROOT.txt).
DEFAULT_DATA_ROOT = REPO_ROOT / "grouping" / "out_v2_public"
DEFAULT_OUT = Path(__file__).resolve().parent / "out" / "fig_guard_sensitivity.png"

# "paper27" is the first paper render (2026-08-20 14:43); "paper" is the page-budget render of
# the same afternoon (canvas height 2.7 -> 2.0 in, one-line left title). Same data, panels,
# series, fonts; only the layout constants below differ.
PAPER_FIGSIZE = {"paper": (5.5, 2.0), "paper27": (5.5, 2.7)}
# Left-panel title. The original is two lines ("The guard's threshold sits inside the bulk /
# of ordinary within-repair dispersion"); at 2.0 in that second line is the whole height
# budget, so "paper" uses a one-line neutral title (<= 6 words). The right title is unchanged.
PAPER_LEFT_TITLE = {"paper": "Cost-separation ratio by population", "paper27": None}
# Right panel: the y-axis top is extended (ticks stay 0..100) so the wrapped legend has a
# band of its own above both curves, the tau line and the 68% marker. At 2.0 in the axes are
# ~1.3 in tall and the 4-line legend takes ~0.6 in of it, so the band must be taller.
PAPER_RIGHT_YTOP = {"paper": 140, "paper27": 150}
# Left panel: axes-y of the (right-aligned, va=top) tail note; it sits under the legend.
PAPER_TAIL_NOTE_Y = {"paper": 0.47, "paper27": 0.52}
# Left panel y-axis top (auto is ~1.42 = 1.05 x the tallest bin). At 2.0 in the legend takes
# ~43% of the axes height and the tail note another ~27%; 1.6 lifts both clear of the bins
# under them (orange step <= 0.35 at x >= 2.75). Ticks 0.0 .. 1.5. The tau label is re-placed
# at 0.93 x the new top, exactly as figures.py places it relative to the auto top.
PAPER_LEFT_YTOP = {"paper": 1.6, "paper27": None}
# Right panel "68%" annotation: (offset pt, ha, va). The original (8, 4) puts it right of the
# marker, where the legend's last line now is; lower-left of the marker is free (both curves
# are at 90-100% there and the text ends 5 pt left of the tau line).
PAPER_PCT_LABEL = {"paper": ((-5, -4), "right", "top"), "paper27": None}
PAPER_SAVE_KW = {"paper": {"pad_inches": 0.02}, "paper27": {}}
# Right panel y-label: one line is ~2.0 in long, the whole height of the 2.0 in canvas; wrapped
# to two lines (same words) for "paper". None = leave as is.
PAPER_RIGHT_YLABEL = {"paper": "% of signatures\nat or above $\\tau$", "paper27": None}
# Legend geometry (same entries, same order): shorter handles and tighter spacing at 2.0 in so
# each legend box is ~1.2 in wide and ~0.55 in tall.
PAPER_LEGEND_KW = {
    "paper":   dict(handlelength=1.0, handletextpad=0.5, labelspacing=0.25, borderpad=0.15, borderaxespad=0.15),
    "paper27": dict(handlelength=1.5, labelspacing=0.3, borderpad=0.2, borderaxespad=0.3),
}
PAPER_TIGHT_PAD = {"paper": 0.3, "paper27": 1.08}
# Original width_ratios are [1.35, 1]; at 5.5 in the right panel is then ~1.9 in wide and its
# wrapped legend (~1.4 in) cannot sit clear of the tau line. [1.15, 1] gives it ~2.2 in.
PAPER_WIDTH_RATIOS = {"paper": [1.25, 1], "paper27": [1.15, 1]}
PAPER_RC = {
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300, "savefig.dpi": 300,
}
# Legend-string rename only (never-proposed population). Keys are the original labels
# with the n-count stripped; the count is re-attached from the original string.
LEGEND_RENAME = {
    "analyst proposed no split": "never-proposed",
    "un-proposed signatures clearing": "never-proposed signatures clearing",
}
# Line wraps only (same words): at 5.5 in total width a one-line legend entry is ~1.9 in and the
# left x-label is 3.65 in, wider than its 2.7 in panel. Applied after the rename.
LEGEND_WRAP = {
    "analyst proposed a split (n=": "analyst proposed\na split (n=",
    "never-proposed (n=": "never-proposed\n(n=",
    "proposed splits honored": "proposed splits\nhonored",
    "never-proposed signatures clearing": "never-proposed\nsignatures clearing",
}
XLABEL_WRAP = {
    "cost separation ratio  (high tier median / low tier median)":
    "cost separation ratio\n(high tier median / low tier median)",
}


def build_cls(data_root: Path):
    """Identical to the data-prep lines of figures.main()."""
    g = C.load_golden(data_root)
    cls = C.classify_signatures(g)
    single = g[g["n_repairs_detected"] == 1].set_index("signature_id")
    q3q1 = (single["cost_q75"] / single["cost_q25"]).replace([np.inf, -np.inf], np.nan)
    cls = cls.merge(q3q1.rename("q3q1"), left_on="signature_id", right_index=True, how="left")
    cls["guard_ratio"] = np.where(cls["klass"] == "single", cls["q3q1"], cls["tier_ratio"])
    return cls


def rename_legends(fig, mode: str = "paper27"):
    """Rename the never-proposed entries, wrap long entries, wrap the long x-label. Same location,
    same handles, same order; legend font 8 (was 8.5), handle length 1.5 (default 2.0)."""
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        new = []
        for lab in labels:
            out = lab
            for old, repl in LEGEND_RENAME.items():
                if lab.startswith(old):
                    out = repl + lab[len(old):]
            for old, repl in LEGEND_WRAP.items():
                if out.startswith(old):
                    out = repl + out[len(old):]
            new.append(out)
        leg = ax.get_legend()
        ax.legend(handles, new, loc=leg._loc, fontsize=8, frameon=leg.get_frame_on(),
                  **PAPER_LEGEND_KW[mode])
        if ax.get_xlabel() in XLABEL_WRAP:
            ax.set_xlabel(XLABEL_WRAP[ax.get_xlabel()])
    # Right panel: at this size the wrapped legend is ~1.35 in wide and ~0.5 in tall. Nowhere
    # inside the original y-range is free of the two curves, the tau line or the 68% marker,
    # so the y-axis is extended upward to give the legend its own band above the curves.
    # Ticks stay 0..100; no series, marker or text is moved.
    ax2 = fig.axes[1]
    ax2.set_ylim(ax2.get_ylim()[0], PAPER_RIGHT_YTOP[mode])
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    # The left panel's tail note ("tail beyond 4x not shown (a% / b%)") is at axes y=0.62 in
    # the original, two lines; the wrapped legend now reaches down to ~0.45, so the note is
    # re-wrapped to three lines (same words) and moved to y=0.52, where the only bins under it
    # (x >= 3) are below 0.1 density. Same right alignment, colour and size.
    for t in fig.axes[0].texts:
        if t.get_text().startswith("tail beyond"):
            t.set_text(t.get_text().replace("tail beyond 4x not shown\n", "tail beyond 4x\nnot shown\n"))
            t.set_position((0.985, PAPER_TAIL_NOTE_Y[mode]))
    if PAPER_LEFT_TITLE[mode]:
        fig.axes[0].set_title(PAPER_LEFT_TITLE[mode])
    if PAPER_RIGHT_YLABEL[mode]:
        ax2.set_ylabel(PAPER_RIGHT_YLABEL[mode])
    if PAPER_LEFT_YTOP[mode]:
        ax0 = fig.axes[0]
        top = PAPER_LEFT_YTOP[mode]
        ax0.set_ylim(ax0.get_ylim()[0], top)
        ax0.set_yticks([v for v in np.arange(0, top + 1e-9, 0.5)])
        for t in ax0.texts:
            if t.get_text().startswith(r"$\tau="):
                t.set_position((t.get_position()[0], top * 0.93))
    if PAPER_PCT_LABEL[mode]:
        (dx, dy), ha, va = PAPER_PCT_LABEL[mode]
        for t in ax2.texts:
            if t.get_text().endswith("%") and hasattr(t, "xyann"):
                t.xyann = (dx, dy)
                t.set_ha(ha)
                t.set_va(va)


def _data_paths(ax):
    """(name, Path in display coords, filled) for every data artist on `ax`: lines (incl. axvline),
    patches (histogram bars, step polygon), collections (scatter markers, padded by their radius)."""
    out = []
    for ln in ax.get_lines():
        out.append((f"line:{ln.get_label()}", ln.get_path().transformed(ln.get_transform()), False))
    for p in ax.patches:
        filled = p.get_fill()
        out.append((f"patch:{p.__class__.__name__}", p.get_path().transformed(p.get_transform()), filled))
    for coll in ax.collections:
        offs = np.asarray(coll.get_offsets())
        if isinstance(coll, PathCollection) and len(offs):   # scatter markers
            pts = coll.get_offset_transform().transform(offs)
            sizes = coll.get_sizes()
            r_pt = float(np.sqrt(sizes.max()) / 2) if len(sizes) else 3.0   # s is area in pt^2
            r = r_pt * ax.figure.dpi / 72
            for x, y in pts:
                out.append((f"marker:({x:.0f},{y:.0f})",
                            MPath.unit_circle().transformed(Affine2D().scale(r).translate(x, y)), True))
        else:
            for pth in coll.get_paths():
                out.append((f"coll:{coll.__class__.__name__}", pth.transformed(coll.get_transform()), False))
    return out


def check_overlaps(fig) -> list[str]:
    """Every text/legend box of every axes (and figure legends) against every data artist of the
    same axes, and text boxes against one another, in display coordinates. Returns the overlaps."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    problems = []
    for i, ax in enumerate(fig.axes):
        boxes = []
        for leg in ([ax.get_legend()] if ax.get_legend() is not None else []):
            # texts and handles separately: the legend's padding box may touch the full-height
            # tau line, which is not an overlap anyone can see
            for t in leg.get_texts():
                boxes.append((f"legend:{t.get_text().replace(chr(10), '|')[:24]!r}", t.get_window_extent(r)))
            for h in leg.legend_handles:
                boxes.append(("legend-handle", h.get_window_extent(r)))
        for t in ax.texts:
            boxes.append((repr(t.get_text().replace("\n", "|")[:28]), t.get_window_extent(r)))
        boxes.append(("title", ax.title.get_window_extent(r)))
        boxes.append(("xlabel", ax.xaxis.label.get_window_extent(r)))
        boxes.append(("ylabel", ax.yaxis.label.get_window_extent(r)))
        for axis, lo, hi in ((ax.xaxis, *sorted(ax.get_xlim())), (ax.yaxis, *sorted(ax.get_ylim()))):
            for tick in axis.get_major_ticks():
                tl = tick.label1
                if tl.get_visible() and tl.get_text() and lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    boxes.append((f"tick:{tl.get_text()}", tl.get_window_extent(r)))
        data = _data_paths(ax)
        texty = [b for b in boxes if not b[0].startswith("tick:") and b[0] not in ("xlabel", "ylabel", "title")]
        for name, bb in texty:
            for dname, pth, filled in data:
                if pth.intersects_bbox(bb, filled=filled):
                    problems.append(f"ax{i}: {name} overlaps {dname}")
        boxes = [b for b in boxes if b[1].width > 0 and b[1].height > 0]   # drop empty labels
        rotated_ticks = {f"tick:{tl.get_text()}" for tl in ax.get_xticklabels() + ax.get_yticklabels()
                         if tl.get_rotation() % 180 != 0}
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                if boxes[a][0] in rotated_ticks and boxes[b][0] in rotated_ticks:
                    continue   # axis-aligned boxes of rotated tick labels overlap even when glyphs do not
                if boxes[a][1].overlaps(boxes[b][1]):
                    problems.append(f"ax{i}: {boxes[a][0]} overlaps {boxes[b][0]}")
        # clipped? every box must lie inside the figure canvas
        W, H = fig.bbox.width, fig.bbox.height
        for name, bb in boxes:
            if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
                problems.append(f"ax{i}: {name} leaves the canvas {bb}")
    for fl in fig.legends:
        bb = fl.get_window_extent(r)
        for i, ax in enumerate(fig.axes):
            if bb.overlaps(ax.get_tightbbox(r)):
                problems.append(f"figure legend overlaps ax{i} tight bbox")
        W, H = fig.bbox.width, fig.bbox.height
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
            problems.append(f"figure legend leaves the canvas {bb}")
    return problems


def dump_numbers(fig) -> dict:
    """Everything drawn, as plain numbers/strings, for a before/after diff."""
    out = {}
    for i, ax in enumerate(fig.axes):
        d = {"title": ax.get_title(), "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
             "xlim": [float(v) for v in ax.get_xlim()],
             "legend": [t.get_text() for t in ax.get_legend().get_texts()],
             "texts": [[t.get_text(), [float(v) for v in t.get_position()]] for t in ax.texts],
             "lines": [], "hists": [], "scatter": []}
        for ln in ax.get_lines():
            x, y = ln.get_xdata(), ln.get_ydata()
            d["lines"].append({"label": ln.get_label(), "ls": ln.get_linestyle(),
                               "color": ln.get_color(), "n": len(x),
                               "x": [round(float(v), 10) for v in np.atleast_1d(x)],
                               "y": [round(float(v), 10) for v in np.atleast_1d(y)]})
        # filled hist -> Rectangle patches; step hist -> Polygon
        rects = [p for p in ax.patches if p.__class__.__name__ == "Rectangle"]
        if rects:
            d["hists"].append({"kind": "bar", "n_bins": len(rects),
                               "edges": [round(float(p.get_x()), 10) for p in rects],
                               "density": [round(float(p.get_height()), 10) for p in rects]})
        polys = [p for p in ax.patches if p.__class__.__name__ == "Polygon"]
        for p in polys:
            xy = p.get_xy()
            d["hists"].append({"kind": "step", "n_vertices": len(xy),
                               "xy": [[round(float(a), 10), round(float(b), 10)] for a, b in xy]})
        for coll in ax.collections:
            offs = coll.get_offsets()
            d["scatter"].append([[round(float(a), 10), round(float(b), 10)] for a, b in np.asarray(offs)])
        out[f"ax{i}"] = d
    out["figsize_in"] = [float(v) for v in fig.get_size_inches()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--mode", choices=("paper", "paper27", "v4"), default="paper")
    ap.add_argument("--dump", default=None, help="write the drawn numbers as JSON here")
    ap.add_argument("--check", action="store_true", help="print text/data overlaps after rendering")
    args = ap.parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        ap.error(f"--data-root does not exist: {data_root}")

    plt = C.setup_plot_style()
    cls = build_cls(data_root)

    if args.mode in ("paper", "paper27"):
        plt.rcParams.update(PAPER_RC)
        _orig_subplots = plt.subplots

        def _paper_subplots(*a, **kw):
            kw["figsize"] = PAPER_FIGSIZE[args.mode]
            if "gridspec_kw" in kw and "width_ratios" in kw["gridspec_kw"]:
                kw["gridspec_kw"] = {**kw["gridspec_kw"], "width_ratios": PAPER_WIDTH_RATIOS[args.mode]}
            return _orig_subplots(*a, **kw)
        plt.subplots = _paper_subplots
        try:
            fig = F.fig_guard_sensitivity(plt, cls, C.SPLIT_SEPARATION_MIN)
        finally:
            plt.subplots = _orig_subplots
        rename_legends(fig, args.mode)
        fig.tight_layout(pad=PAPER_TIGHT_PAD[args.mode])
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight", **PAPER_SAVE_KW[args.mode])
    else:
        fig = F.fig_guard_sensitivity(plt, cls, C.SPLIT_SEPARATION_MIN)
        # same call sequence as common.save_fig, minus its directory handling
        fig.tight_layout()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}  ({args.mode}; data_root={data_root})")

    if args.dump:
        Path(args.dump).write_text(json.dumps(dump_numbers(fig), indent=1) + "\n")
        print(f"dumped {args.dump}")
    if args.check:
        problems = check_overlaps(fig)
        print("overlap check:", "none" if not problems else "")
        for pr in problems:
            print("  ", pr)


if __name__ == "__main__":
    main()
