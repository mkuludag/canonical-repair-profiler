#!/usr/bin/env python3
"""
a6 - What drives the cross-state cost spread, and is a repair-independent index defensible?

RegionDelegatorAgent localizes a specific repair's national cost by multiplying it by a per-state
index derived from *all-repairs* pooled state medians (region_delegator_agent.py:21). That assumes
the regional multiplier is repair-independent - an assumption the draft flags as unvalidated.

The direct test (does the pooled index predict within-signature state variation?) is NOT possible
from the released artifacts: region_cost_by_signature.csv holds only 200 rows over 33 signatures and
reports cost_min/cost_max extremes across states, not a per-signature x state matrix. Ratios there
run past 100x, which is outlier structure, not price structure.

What IS testable is the composition of the spread. region_overall.csv carries per-state median labor
cost and median material cost alongside the overall median. A spread driven by labor *rate* is far
more plausibly repair-independent than one driven by material, since a state's labor rate applies
across repairs while material content is intrinsic to the repair. Decomposing the spread therefore
bounds how defensible the assumption is, without settling it.

Note medians do not add: med_cost is not med_labor_cost + med_material. Each column is compared
against its own cross-state spread, which is the meaningful comparison.

    python eval/a6_region_decomposition.py --data-root ~/canonical-repair-profiler-public/grouping/out
"""
import numpy as np
import pandas as pd

import common as C

COLS = {"med_cost": "total claim cost", "med_labor_cost": "labor cost",
        "med_material": "material cost", "med_labor_hrs": "labor hours"}


def main() -> None:
    args = C.parse_args(__doc__)
    r = C.load_region(args.data_root).copy()
    for c in list(COLS) + ["claims"]:
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r = r.dropna(subset=["med_cost"])

    # Claims-weighted national median anchor, as RegionIndexTool derives it.
    anchor = float(np.average(r["med_cost"], weights=r["claims"]))
    r["cost_index"] = r["med_cost"] / anchor

    rows = []
    for col, label in COLS.items():
        v = r[col].dropna()
        v = v[v > 0]
        hi, lo = r.loc[v.idxmax()], r.loc[v.idxmin()]
        rows.append({
            "component": label, "column": col,
            "max_state": hi["state"], "max_value": float(v.max()),
            "min_state": lo["state"], "min_value": float(v.min()),
            "spread_ratio": float(v.max() / v.min()),
            "cv": float(v.std() / v.mean()),
        })
    comp = pd.DataFrame(rows)
    C.write_csv(comp, "a6_region_components.csv")
    C.write_csv(r[["state", "claims", "med_cost", "cost_index", "med_labor_cost",
                   "med_material", "med_labor_hrs"]].sort_values("cost_index", ascending=False),
                "a6_region_index.csv")

    lab = comp.loc[comp["column"] == "med_labor_cost"].iloc[0]
    mat = comp.loc[comp["column"] == "med_material"].iloc[0]
    tot = comp.loc[comp["column"] == "med_cost"].iloc[0]
    hrs = comp.loc[comp["column"] == "med_labor_hrs"].iloc[0]

    corr_lab = float(r["med_cost"].corr(r["med_labor_cost"]))
    corr_mat = float(r["med_cost"].corr(r["med_material"]))
    # Do the components move together, or offset? And how much does cost COMPOSITION vary by state?
    corr_lab_mat = float(r["med_labor_cost"].corr(r["med_material"]))
    r["labor_share"] = r["med_labor_cost"] / (r["med_labor_cost"] + r["med_material"])
    ls = r["labor_share"].dropna()
    ls_lo, ls_hi = r.loc[ls.idxmin()], r.loc[ls.idxmax()]
    # Robustness: the single most extreme state can dominate a max/min statistic.
    ls_trimmed = ls.sort_values().iloc[1:-1]

    C.write_json({
        "n_states": int(len(r)), "national_anchor_med_cost": anchor,
        "cost_index_min": float(r["cost_index"].min()), "cost_index_max": float(r["cost_index"].max()),
        "spread_total": float(tot["spread_ratio"]), "spread_labor": float(lab["spread_ratio"]),
        "spread_material": float(mat["spread_ratio"]), "spread_labor_hours": float(hrs["spread_ratio"]),
        "corr_cost_labor": corr_lab, "corr_cost_material": corr_mat,
        "corr_labor_material": corr_lab_mat,
        "labor_share_min": float(ls.min()), "labor_share_min_state": str(ls_lo["state"]),
        "labor_share_max": float(ls.max()), "labor_share_max_state": str(ls_hi["state"]),
        "labor_share_mean": float(ls.mean()), "labor_share_std": float(ls.std()),
        "labor_share_spread": float(ls.max() / ls.min()),
        "labor_share_spread_trimmed": float(ls_trimmed.max() / ls_trimmed.min()),
    }, "a6_region_summary.json")
    body = f"""
Across {len(r)} states the pooled all-repairs median cost spans **{tot['spread_ratio']:.2f}x**
({tot['max_state']} to {tot['min_state']}), and the claims-weighted national anchor gives a cost index
spanning {r['cost_index'].min():.2f}-{r['cost_index'].max():.2f}.

Decomposing that spread by component:

| Component | Spread (max/min) | Extremes | CV |
|---|---|---|---|
| Total claim cost | {tot['spread_ratio']:.2f}x | {tot['max_state']} / {tot['min_state']} | {tot['cv']:.3f} |
| Labor cost | {lab['spread_ratio']:.2f}x | {lab['max_state']} / {lab['min_state']} | {lab['cv']:.3f} |
| Material cost | {mat['spread_ratio']:.2f}x | {mat['max_state']} / {mat['min_state']} | {mat['cv']:.3f} |
| Labor hours | {hrs['spread_ratio']:.2f}x | {hrs['max_state']} / {hrs['min_state']} | {hrs['cv']:.3f} |

Note both components spread **more** than the total ({lab['spread_ratio']:.2f}x and
{mat['spread_ratio']:.2f}x vs {tot['spread_ratio']:.2f}x). That is only possible if they partially
offset, and they do: labor and material medians correlate at **r = {corr_lab_mat:.3f}** across states.
Total cost tracks labor (r = {corr_lab:.3f}) more closely than material (r = {corr_mat:.3f}), but
neither component alone explains the spread.

**This weakens the repair-independent multiplier rather than supporting it.** The offsetting shows
states differ not just in price *level* but in cost *composition*: labor's share of labor-plus-material
runs from {ls.min():.3f} ({ls_lo['state']}) to {ls.max():.3f} ({ls_hi['state']}), a
{ls.max() / ls.min():.2f}x spread ({ls_trimmed.max() / ls_trimmed.min():.2f}x after trimming the two
extreme states, so this is not one outlier). A single pooled index applies one multiplier to every
repair in a state. If states vary in labor-vs-material mix, then a labor-heavy repair and a
material-heavy repair in the same state should scale by *different* factors, and the pooled index will
misprice both - overstating one and understating the other.

We flag this as evidence **against** treating the index as repair-independent, and note the assumption
is only settled by measuring within-signature state variation, which requires the claim-level table
(`region_cost_by_signature.csv` covers just 33 signatures as extremes) and remains future work.

Artifacts: `eval/out/a6_region_components.csv`, `eval/out/a6_region_index.csv`,
`eval/out/a6_region_summary.json`.
"""
    C.append_results("a6 - Regional decomposition", body, args.data_root, not args.no_append)

    print(f"\n  total {tot['spread_ratio']:.2f}x | labor {lab['spread_ratio']:.2f}x | "
          f"material {mat['spread_ratio']:.2f}x | corr(labor,material) {corr_lab_mat:.3f} | "
          f"labor-share spread {ls.max() / ls.min():.2f}x")


if __name__ == "__main__":
    main()
