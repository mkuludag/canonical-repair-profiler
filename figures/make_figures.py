#!/usr/bin/env python3
"""
Generate submission figures from the Canonical Repair Library artifacts.
Outputs PNGs into figures/. No external calls (reads grouping/out/*.csv only).

Override the input and output directories with CRP_FIG_DATA / CRP_FIG_OUT to render the figures from
a different artifact set - e.g. the public release, whose dollar values are rescaled.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get("CRP_FIG_OUT", "figures"); os.makedirs(OUT, exist_ok=True)
G = os.environ.get("CRP_FIG_DATA", "grouping/out")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


def _corrected_cost_by_sig():
    """Per-signature corrected cost (median across its golden-solution tiers)."""
    gs = pd.read_csv(f"{G}/golden_solutions.csv")
    return gs.groupby("signature_id")["cost_med"].median()


def fig_support_vs_consensus():
    # built entirely from the corrected golden_solutions.csv (support, consensus, cost all post-fix)
    gs = pd.read_csv(f"{G}/golden_solutions.csv")
    gs["support"] = pd.to_numeric(gs["support"], errors="coerce").fillna(gs["tier_n_claims"])
    gs["consensus"] = pd.to_numeric(gs["consensus"], errors="coerce")
    gs["cost_med"] = pd.to_numeric(gs["cost_med"], errors="coerce")
    d = gs.groupby("signature_id").apply(lambda x: pd.Series({
        "n": x["support"].sum(),
        "gt_score": np.average(x["consensus"].fillna(0), weights=x["support"].clip(lower=1)),
        "cost_corr": x["cost_med"].median(),
    })).reset_index()
    d = d[d["n"] >= 5]
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(d["n"], d["gt_score"], s=10, alpha=0.3,
                    c=d["cost_corr"].clip(upper=20000), cmap="viridis")
    ax.set_xscale("log")
    ax.axhline(0.6, ls="--", c="green", lw=1); ax.axhline(0.4, ls="--", c="red", lw=1)
    ax.axvline(10, ls=":", c="grey", lw=1)
    ax.set_xlabel("support  (claims per repair signature, log)")
    ax.set_ylabel("consensus  (ground-truth score)")
    ax.set_title("Canonical Repair prioritization quadrants\n(top-right = GOLD ground truth; bottom-right = discrepancy hotspot)")
    plt.colorbar(sc, label="median approved $ (capped 20k)")
    ax.text(d["n"].max() * 0.25, 0.92, "GOLD\nuse as ground truth", color="green", ha="center", fontsize=9)
    ax.text(d["n"].max() * 0.25, 0.15, "HOTSPOT\nclean / investigate", color="red", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_support_vs_consensus.png"); plt.close(fig)


def fig_status_breakdown():
    d = pd.read_csv(f"{G}/canonical_repairs.csv")
    order = ["GOLD", "SILVER", "HOTSPOT", "SPARSE"]
    counts = d["status"].value_counts().reindex(order).dropna()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(counts.index, counts.values, color=["#2a9d8f", "#e9c46a", "#e76f51", "#bbbbbb"][:len(counts)])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{int(v):,}", ha="center", va="bottom")
    ax.set_ylabel("# repair signatures")
    ax.set_title("Canonical Repair Library — signature status")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_status_breakdown.png"); plt.close(fig)


def fig_group_size_and_cost():
    d = pd.read_csv(f"{G}/canonical_repairs.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.hist(d["n"].clip(upper=500), bins=50, color="#264653", log=True)
    a1.set_xlabel("claims per signature (capped 500)"); a1.set_ylabel("# signatures (log)")
    a1.set_title("Support distribution")
    cost = pd.to_numeric(_corrected_cost_by_sig(), errors="coerce").dropna()
    cost = cost[(cost > 0) & (cost < 50000)]
    a2.hist(cost, bins=60, color="#2a9d8f")
    a2.set_xlabel("median repair cost ($, gsar_tot_cost_gross)"); a2.set_ylabel("# signatures")
    a2.set_title("Canonical repair cost distribution")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_distributions.png"); plt.close(fig)


def fig_split_and_confidence():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    try:
        gs = pd.read_csv(f"{G}/golden_solutions.csv")
        sp = gs.groupby("signature_id")["n_repairs_detected"].first().value_counts().sort_index()
        bars = a1.bar([f"{int(k)} repair(s)" for k in sp.index], sp.values, color="#e76f51")
        for b, v in zip(bars, sp.values):
            a1.text(b.get_x() + b.get_width() / 2, v, f"{int(v):,}", ha="center", va="bottom")
        a1.set_title("Agentic split decision (repairs per signature)")
        a1.set_ylabel("# signatures")
        conf = pd.to_numeric(gs["confidence"], errors="coerce").dropna()
        a2.hist(conf, bins=40, color="#457b9d")
        # kappa = consensus * min(1, log10(n+1)/log10(31))  (stats_tool.py:54-62) - NOT support x consensus
        a2.set_xlabel(r"confidence $\kappa$ = consensus $\times$ support weight")
        a2.set_ylabel("# solutions")
        a2.set_title("Golden solution confidence distribution")
    except FileNotFoundError:
        pass
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_split_and_confidence.png"); plt.close(fig)


def fig_region():
    try:
        d = pd.read_csv(f"{G}/region_overall.csv").sort_values("med_cost", ascending=False)
    except FileNotFoundError:
        return
    top = pd.concat([d.head(8), d.tail(8)])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#2a9d8f"] * 8 + ["#e76f51"] * 8
    ax.bar(top["state"].astype(str), top["med_cost"], color=colors[:len(top)])
    ax.set_ylabel("median claim cost ($)")
    # NOT a same-repair contrast: this pools every repair type in each state (region_overall.csv).
    spread = float(d["med_cost"].max() / d["med_cost"].min())
    ax.set_title("Pooled state median warranty cost, all repairs aggregated "
                 f"(top 8 vs bottom 8) — {spread:.2f}x spread")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_region_cost.png"); plt.close(fig)


# The architecture diagram is authored as a hand-crafted SVG (figures/architecture.svg) and rendered
# to figures/architecture.png + the root architecture_diagram.png. To re-render after editing the SVG:
#   chrome --headless=new --screenshot=figures/architecture.png --window-size=1660,1060 \
#          --force-device-scale-factor=2 figures/architecture.svg


def main():
    for fn in [fig_support_vs_consensus, fig_status_breakdown,
               fig_group_size_and_cost, fig_split_and_confidence, fig_region]:
        try:
            fn(); print("ok:", fn.__name__)
        except Exception as e:  # keep going; figures are independent
            print("FAIL:", fn.__name__, str(e)[:160])


if __name__ == "__main__":
    main()
