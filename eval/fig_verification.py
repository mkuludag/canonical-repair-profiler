import argparse, json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = str(Path(__file__).resolve().parent.parent)
OUT  = str(Path(__file__).resolve().parent / "out" / "fig_verification.png")

# --paper (2026-08-20): same data, panels, series, colours and annotations, rendered at the
# NeurIPS text width (5.5 in) with 8-9 pt fonts for inclusion at width=\linewidth. Without the
# flag the script renders exactly as before (7.0 x 2.5 in, 5.8-7 pt fonts). --out overrides the
# destination (default: the paper's figures/fig_verification.png, as before).
# --paper28 is the first paper layout of 2026-08-20 (5.5 x 2.8 in canvas, legend inside the left
# panel; md5 df0464ed...), kept reproducible. --paper is the page-budget layout of the same day:
# 5.5 x 2.0 in canvas, the 4-entry legend as a 2 x 2 figure-level legend above both panels
# (a two-column legend is ~4.5 in wide at 8 pt, wider than the left panel), left y-range back
# to 60..102, right y-range 2.6 (all points are <= 0.1%). Same fonts as --paper28.
_ap = argparse.ArgumentParser()
_ap.add_argument("--paper", action="store_true", help="paper-sized render (5.5 x 2.0 in, 8-9 pt)")
_ap.add_argument("--paper28", action="store_true", help="the 5.5 x 2.8 in paper render of 2026-08-20 14:49")
_ap.add_argument("--check", action="store_true", help="print text/data overlaps after rendering")
_ap.add_argument("--out", default=OUT)
_args = _ap.parse_args()
OUT = _args.out
if _args.paper and _args.paper28:
    _ap.error("--paper and --paper28 are exclusive")
FIG_LEGEND = False
if _args.paper:
    FIGSIZE = (5.5, 2.0)
    WIDTH_RATIOS = [1.5, 1]   # legend no longer inside the left panel; the right panel gets a bit more
    FS = {"tick": 8, "label": 9, "legend": 8, "annot": 8, "note": 8}
    YLIM1 = (60, 102)   # as the v4 render (101) plus room for the rep1 x rep2 error-bar cap
    YLIM2 = (-2.0, 3.5) # top 4 -> 3.5: every point is <= 0.1%; the range is set by the two-line
                        # labels above (to ~1.8) and below (to ~-1.9) the points and the two-line note
    YLABEL1 = "split-decision\nagreement (%)"     # same words, wrapped: one line is 1.7 in, taller than the axes
    YLABEL2 = "critic gate\nfailure rate (%)"
    # offsets 5-6 pt (were 8-9): the markers' radius is sqrt(28)/2 = 2.6 pt
    ANNOT = {"B2: 8 comments":  ("B2: 8\ncomments",    (0, -6), "center", "top"),
             "B2: 40 comments": ("B2: 40\ncomments",   (-5, 5), "right",  "bottom"),
             "B3: single call": ("B3: single\ncall",   (5, 5),  "left",   "bottom")}
    CHAIN_XY = (4, -11)
    FIG_LEGEND = True   # 2 x 2, upper centre of the figure, rows read B1, B2 / B3, revision
    LEGEND_KW = {"ncol": 2, "columnspacing": 1.5, "handlelength": 1.5, "handletextpad": 0.5,
                 "labelspacing": 0.25, "borderaxespad": 0.0, "borderpad": 0.1}
    LEGEND_TOP = 0.865  # fraction of the canvas height left to the axes (tight_layout rect)
    EXTEND_RIGHT_DOWN = True   # see below: the right axes gets the height under the left panel's rotated ticks
    NOTE_XY = (7.2, 2.2)
    XLABEL2 = "median tier-cost\nrelative error (%)"
    LABEL85 = "85% bar,\nfixed before the rerun"
    SAVE_KW = {"bbox_inches": "tight", "pad_inches": 0.02}
    TIGHT_PAD = 0.3
elif _args.paper28:
    FIGSIZE = (5.5, 2.8)
    WIDTH_RATIOS = [1.6, 1]   # was [1.45, 1]; the one-column legend needs ~2.3 in of left panel
    FS = {"tick": 8, "label": 9, "legend": 8, "annot": 8, "note": 8}
    YLIM1 = (60, 132)   # top extended from 101 so the 4-entry legend sits above the bars
    YLIM2 = (-1.3, 4)   # bottom extended from -1.1 for the two-line label under the 11.9 point
    # Right-panel point labels: same words, wrapped to two lines and re-placed so that none
    # overlaps another or leaves the axes at the larger font. (name -> text, offset pt, ha)
    ANNOT = {"B2: 8 comments":  ("B2: 8\ncomments",    (0, -9), "center", "top"),
             "B2: 40 comments": ("B2: 40\ncomments",   (-5, 8), "right",  "bottom"),
             "B3: single call": ("B3: single\ncall",   (5, 8),  "left",   "bottom")}
    CHAIN_XY = (4, -14)   # "chain, in-loop" moves below its point (was above, (4, 5))
    LEGEND_KW = {"labelspacing": 0.35, "borderaxespad": 0.2}
    XLABEL2 = "median tier-cost\nrelative error (%)"   # same words, wrapped: 2.2 in wide at 9 pt
    LABEL85 = "85% bar,\nfixed before the rerun"        # wrapped: one line would run into the B1 bars
    SAVE_KW = {"bbox_inches": "tight", "pad_inches": 0.02}
    NOTE_XY = (7.2, 2.6)
    TIGHT_PAD = 1.08
    YLABEL1 = "split-decision agreement (%)"
    YLABEL2 = "critic gate failure rate (%)"
else:
    FIGSIZE = (7.0, 2.5)
    WIDTH_RATIOS = [1.45, 1]
    FS = {"tick": 6.5, "label": 7, "legend": 5.8, "annot": 6.5, "note": 7}
    YLIM1 = (60, 101)
    YLIM2 = (-1.1, 4)
    ANNOT = {"B2: 8 comments":  ("B2: 8 comments",  (0, 10),   "center", "baseline"),
             "B2: 40 comments": ("B2: 40 comments", (-4, 14),  "right",  "baseline"),
             "B3: single call": ("B3: single call", (5, -16),  "left",   "baseline")}
    CHAIN_XY = (4, 5)
    LEGEND_KW = {}
    XLABEL2 = "median tier-cost relative error (%)"
    LABEL85 = "85% bar, fixed before the rerun"
    SAVE_KW = {}
    NOTE_XY = (7.2, 2.6)
    TIGHT_PAD = 1.08
    YLABEL1 = "split-decision agreement (%)"
    YLABEL2 = "critic gate failure rate (%)"
W = {"single": 2262/3623, "honored": 1073/3623, "collapsed": 288/3623}
N = {"single": 250, "honored": 100, "collapsed": 50}

def strat_se(per):  # per: {stratum: p}
    return math.sqrt(sum(W[s]**2 * per[s]*(1-per[s]) / N[s] for s in per))

b1 = json.load(open(f"{ROOT}/eval/out/b1_stability.json"))
pairs = {p["pair"]: p for p in b1["pairs"]}
b2 = {a["arm"]: a for a in json.load(open(f"{ROOT}/eval/out/b2_llm_numbers.json"))["arms"]}

# B2 per-stratum agreement for SEs
import pandas as pd
panel = pd.read_csv(f"{ROOT}/eval/panel_400.csv")
def per_stratum_from(arm):
    if "per_stratum_agreement" in b2[arm]:
        return {k: v for k, v in b2[arm]["per_stratum_agreement"].items()}
    csv = f"{ROOT}/eval/out/b2_llm_numbers_per_signature.csv"
    df = pd.read_csv(csv)
    df = df[df.arm == arm] if "arm" in df.columns else df
    if "stratum" not in df.columns:
        df = df.merge(panel[["signature_id","stratum"]], on="signature_id")
    col = [c for c in df.columns if "agree" in c][0]
    return df.groupby("stratum")[col].mean().to_dict()

per_b3 = {"single": 0.688, "honored": 0.680, "collapsed": 0.740}
try:
    b2se = {a: strat_se(per_stratum_from(a)) for a in ("s8","s40")}
except Exception as e:
    print("B2 per-stratum fallback (binomial):", e)
    b2se = {a: math.sqrt(b2[a]["proposal_agreement_with_v2_weighted"]
                         * (1-b2[a]["proposal_agreement_with_v2_weighted"]) / 400)
            for a in ("s8","s40")}
churn_se = math.sqrt(0.6898*0.3102/3623)

bars = [
    ("B1: rep1$\\times$rep2",  100*pairs["rep1 x rep2"]["n_repairs_agreement_population_weighted"], 100*pairs["rep1 x rep2"]["population_weighted_se"], "#2b6cb0"),
    ("B1: v2$\\times$rep1",    100*pairs["v2 x rep1"]["n_repairs_agreement_population_weighted"],  100*pairs["v2 x rep1"]["population_weighted_se"],  "#2b6cb0"),
    ("B1: v2$\\times$rep2",    100*pairs["v2 x rep2"]["n_repairs_agreement_population_weighted"],  100*pairs["v2 x rep2"]["population_weighted_se"],  "#2b6cb0"),
    ("B2: 8 comments",        100*b2["s8"]["proposal_agreement_with_v2_weighted"],  100*b2se["s8"],  "#dd6b20"),
    ("B2: 40 comments",       100*b2["s40"]["proposal_agreement_with_v2_weighted"], 100*b2se["s40"], "#dd6b20"),
    ("B3: single call",   68.98, 100*strat_se(per_b3), "#805ad5"),
    ("revision: v1$\\times$v2",      68.98, 100*churn_se, "#c53030"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE,
                               gridspec_kw={"width_ratios": WIDTH_RATIOS})
xs = range(len(bars))
ax1.bar(xs, [b[1] for b in bars], yerr=[b[2] for b in bars],
        color=[b[3] for b in bars], width=0.62, capsize=2,
        error_kw={"lw": 0.8})
ax1.axhline(85, ls="--", lw=0.9, color="0.35")
# 2026-08-20: was "pre-registered 85% bar"; the bar exists only as a plan comment
# (a7_rerun_churn.py:10), which supports "fixed before the rerun", not "pre-registered".
ax1.text(len(bars)-0.4, 85.8, LABEL85, ha="right", va="bottom", fontsize=FS["annot"], color="0.25")
ax1.set_xticks(list(xs))
ax1.set_xticklabels([b[0] for b in bars], rotation=32, ha="right", fontsize=FS["tick"])
ax1.set_ylim(*YLIM1)
if _args.paper or _args.paper28:
    ax1.set_yticks([60, 70, 80, 90, 100])
ax1.set_ylabel(YLABEL1, fontsize=FS["label"])
ax1.tick_params(axis="y", labelsize=FS["tick"])
from matplotlib.patches import Patch
LEGEND_HANDLES = [Patch(color="#2b6cb0", label="B1: same prompt, sampler only"),
                  Patch(color="#dd6b20", label="B2: statistics stripped from prompt"),
                  Patch(color="#805ad5", label="B3: chain replaced by one call"),
                  Patch(color="#c53030", label="pipeline revision (v1 to v2)")]
if FIG_LEGEND:
    # matplotlib fills a multi-column legend column by column; reorder so the rows read
    # B1, B2 / B3, revision (the same order as the bars, left to right).
    h = LEGEND_HANDLES
    fig.legend(handles=[h[0], h[2], h[1], h[3]], fontsize=FS["legend"], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=False, **LEGEND_KW)
else:
    ax1.legend(handles=LEGEND_HANDLES,
               fontsize=FS["legend"], loc="upper right", frameon=False, **LEGEND_KW)

pts = [("B2: 8 comments", 11.9), ("B2: 40 comments", 6.4), ("B3: single call", 7.4)]
ax2.scatter([p[1] for p in pts], [0, 0, 0], s=28, color="#dd6b20", zorder=3)
ax2.scatter([0.0], [0.1], s=28, color="#2b6cb0", zorder=3)
ax2.annotate("chain, in-loop", (0.0, 0.1), textcoords="offset points",
             xytext=CHAIN_XY, fontsize=FS["annot"], color="#2b6cb0")
for name, x in pts:
    text, xy, ha, va = ANNOT[name]
    ax2.annotate(text, (x, 0), textcoords="offset points", xytext=xy,
                 fontsize=FS["annot"], ha=ha, va=va)
ax2.axhline(0, lw=0.7, color="0.6")
ax2.set_xlim(-0.8, 14.5)
ax2.set_ylim(*YLIM2)
if _args.paper:
    ax2.set_yticks([0, 1, 2, 3])   # the space below 0 holds the point labels; no negative ticks
ax2.set_xlabel(XLABEL2, fontsize=FS["label"])
ax2.set_ylabel(YLABEL2, fontsize=FS["label"])
ax2.tick_params(labelsize=FS["tick"])
ax2.text(*NOTE_XY, "error rises,\ngates never fire", fontsize=FS["note"], ha="center", color="0.2")
for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
if FIG_LEGEND:
    fig.tight_layout(pad=TIGHT_PAD, rect=[0, 0, 1, LEGEND_TOP])   # tight_layout ignores figure legends
else:
    fig.tight_layout(pad=TIGHT_PAD)
if _args.paper and EXTEND_RIGHT_DOWN:
    # The left panel's 32-degree tick labels need ~0.6 in under its axes; the right panel's
    # two-line x-label needs ~0.45 in. With a shared row both axes stop at the taller margin and
    # the right panel is left with ~0.93 in for three bands of two-line labels. Lower the right
    # axes' bottom so that its decorations end level with the left panel's, i.e. the two
    # panels share a bottom *content* edge instead of a bottom *axes* edge.
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    _tb1 = ax1.get_tightbbox(_r).transformed(fig.transFigure.inverted())
    _tb2 = ax2.get_tightbbox(_r).transformed(fig.transFigure.inverted())
    _pos2 = ax2.get_position()
    _offset = _pos2.y0 - _tb2.y0            # x-label + tick labels below the right axes, fig fraction
    _new_y0 = _tb1.y0 + _offset
    ax2.set_position([_pos2.x0, _new_y0, _pos2.width, _pos2.y1 - _new_y0])
fig.savefig(OUT, dpi=300, **SAVE_KW)
print("saved", OUT)
if _args.check:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paper_figs import check_overlaps
    problems = check_overlaps(fig)
    print("overlap check:", "none" if not problems else "")
    for pr in problems:
        print("  ", pr)
# Every number drawn, for before/after diffs of the figure.
print("bars:", json.dumps([[b[0], round(b[1], 6), round(b[2], 6), b[3]] for b in bars]))
print("points:", json.dumps(pts + [("chain, in-loop", 0.0, 0.1)]))
print("figsize:", FIGSIZE, "fonts:", FS, "ylim1:", YLIM1, "ylim2:", YLIM2)
