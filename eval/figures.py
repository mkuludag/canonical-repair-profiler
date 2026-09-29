#!/usr/bin/env python3
"""
Figures for the re-derived results (a2, a3). Reads the same artifacts as the analyses; writes PNGs.

  fig_guard_sensitivity.png - the separation guard's decision boundary against the distribution it
      is drawn on, plus the tau sweep. Shows that the boundary sits inside the bulk of ordinary
      within-signature dispersion rather than between two separated modes.
  fig_confidence_decomposition.png - the support weight across the emitted library, showing that it
      saturates for only a minority of repairs and therefore is not inert.

    python eval/figures.py --data-root ~/canonical-repair-profiler-public/grouping/out
"""
import numpy as np
import pandas as pd

import common as C

TAUS = np.arange(1.0, 3.01, 0.05)
SATURATION_N = 30


def fig_guard_sensitivity(plt, cls: pd.DataFrame, tau: float):
    never = cls.loc[cls["klass"] == "single", "guard_ratio"].dropna()
    honored = cls.loc[cls["klass"] == "honored", "guard_ratio"].dropna()
    collapsed = cls.loc[cls["klass"] == "collapsed", "guard_ratio"].dropna()
    proposed = pd.concat([honored, collapsed])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2),
                                  gridspec_kw={"width_ratios": [1.35, 1]})

    # Truncate rather than clip: clipping piles the whole tail onto the last bin and reads as a mode.
    xmax = 4.0
    bins = np.linspace(1.0, xmax, 61)
    ax.hist(never[never <= xmax], bins=bins, density=True, alpha=0.55,
            color=C.PALETTE["single"], label=f"analyst proposed no split (n={len(never):,})")
    ax.hist(proposed[proposed <= xmax], bins=bins, density=True, histtype="step", lw=2.0,
            color=C.PALETTE["accent"], label=f"analyst proposed a split (n={len(proposed):,})")
    ax.axvline(tau, color="k", ls="--", lw=1.4)
    ax.set_xlim(1.0, xmax)
    ax.text(tau + 0.04, ax.get_ylim()[1] * 0.93, rf"$\tau={tau}$", fontsize=10, fontweight="bold")
    ax.set_xlabel("cost separation ratio  (high tier median / low tier median)")
    ax.set_ylabel("density")
    ax.set_title("The guard's threshold sits inside the bulk\nof ordinary within-repair dispersion")
    ax.legend(loc="upper right", fontsize=8.5)
    beyond_n = float((never > xmax).mean() * 100)
    beyond_p = float((proposed > xmax).mean() * 100)
    ax.text(0.985, 0.62, f"tail beyond {xmax:g}x not shown\n({beyond_n:.1f}% / {beyond_p:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=C.PALETTE["guide"])

    share_prop = [(proposed >= t).mean() for t in TAUS]
    share_never = [(never >= t).mean() for t in TAUS]
    ax2.plot(TAUS, np.array(share_prop) * 100, color=C.PALETTE["accent"], lw=2,
             label="proposed splits honored")
    ax2.plot(TAUS, np.array(share_never) * 100, color=C.PALETTE["single"], lw=2, ls="--",
             label="un-proposed signatures clearing")
    ax2.axvline(tau, color="k", ls="--", lw=1.2)
    at = float((never >= tau).mean() * 100)
    ax2.scatter([tau], [at], color=C.PALETTE["single"], zorder=5, s=28)
    ax2.annotate(f"{at:.0f}%", (tau, at), textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax2.set_xlabel(r"separation threshold $\tau$")
    ax2.set_ylabel("% of signatures at or above $\\tau$")
    ax2.set_title("Threshold sensitivity")
    ax2.legend(fontsize=8.5, loc="upper right")
    return fig


def fig_confidence_decomposition(plt, g: pd.DataFrame):
    support = pd.to_numeric(g["support"], errors="coerce").fillna(0)
    w = C.support_weight(support)
    sat = support >= SATURATION_N

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # The weight is a deterministic function of n, so the informative overlay is where the LIBRARY's
    # mass actually sits relative to the saturation point - not a scatter that retraces the curve.
    n_grid = np.logspace(0, np.log10(max(support.max(), 100)), 300)
    ax.plot(n_grid, C.support_weight(n_grid), color=C.PALETTE["accent"], lw=2.2, zorder=4)
    ax.axvspan(1, SATURATION_N, color=C.PALETTE["single"], alpha=0.07, zorder=0)
    ax.axvline(SATURATION_N, color=C.PALETTE["guide"], ls=":", lw=1.2)
    ax.axhline(1.0, color=C.PALETTE["guide"], ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("support  (claims backing the repair, log)")
    ax.set_ylabel("support weight  min(1, log$_{10}$(n+1)/log$_{10}$31)")
    ax.set_ylim(0, 1.08)

    hist = ax.twinx()
    edges = np.logspace(0, np.log10(max(support.max(), 100)), 45)
    hist.hist(support.clip(lower=1), bins=edges, color=C.PALETTE["single"], alpha=0.30, zorder=1)
    hist.set_ylabel("# canonical repairs", color=C.PALETTE["guide"], fontsize=9)
    hist.tick_params(axis="y", labelcolor=C.PALETTE["guide"], labelsize=8)
    hist.grid(False)
    ax.set_zorder(hist.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_title(f"Saturates at n={SATURATION_N} (shaded = discounted),\n"
                 f"reached by only {sat.mean() * 100:.1f}% of repairs")

    ax2.hist(w, bins=40, color=C.PALETTE["honored"])
    ax2.axvline(float(w.mean()), color="k", ls="--", lw=1.3)
    ax2.text(float(w.mean()) - 0.02, ax2.get_ylim()[1] * 0.9, f"mean {w.mean():.2f}",
             ha="right", fontsize=9, fontweight="bold")
    ax2.set_xlabel("support weight applied to the emitted repair")
    ax2.set_ylabel("# canonical repairs")
    ax2.set_title("The support term is not inert:\nit discounts three quarters of the library")
    return fig


def main() -> None:
    args = C.parse_args(__doc__)
    plt = C.setup_plot_style()

    g = C.load_golden(args.data_root)
    cls = C.classify_signatures(g)
    single = g[g["n_repairs_detected"] == 1].set_index("signature_id")
    q3q1 = (single["cost_q75"] / single["cost_q25"]).replace([np.inf, -np.inf], np.nan)
    cls = cls.merge(q3q1.rename("q3q1"), left_on="signature_id", right_index=True, how="left")
    cls["guard_ratio"] = np.where(cls["klass"] == "single", cls["q3q1"], cls["tier_ratio"])

    C.save_fig(fig_guard_sensitivity(plt, cls, C.SPLIT_SEPARATION_MIN), "fig_guard_sensitivity.png")
    C.save_fig(fig_confidence_decomposition(plt, g), "fig_confidence_decomposition.png")


if __name__ == "__main__":
    main()
