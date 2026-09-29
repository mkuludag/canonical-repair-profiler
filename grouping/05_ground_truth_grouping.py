#!/usr/bin/env python3
"""
Ground-truth-oriented grouping (the hackathon core).

Reframe: a group is valuable if a SINGLE consensus solution is defensible AND it has enough
claims to define + police that solution. So we score each group on SUPPORT x CONSENSUS, where
consensus is measured on the dense solution dimensions:
    - cost   : det_approved_amt   (99.9% filled)
    - labor  : gsar_labor_hrs     (68%)
    - text   : repair archetype from the 3C comment trail (85%)
    (parts list is only 16% filled -> reported as a bonus, not in the core score)

We compare three grouping definitions to answer "how specific can we go before groups collapse":
    G1 = vehicle_line + causal_part                 (structural baseline)
    G2 = vehicle_line + causal_part + sub_category   (tighter coded)
    G3 = vehicle_line + causal_part + TEXT ARCHETYPE  (code x symptom, idea #3)

Outputs (grouping/out/):
    gt_grouping_summary.csv         per-grouping: sizes, singleton%, consensus, #gold, #hotspots
    gt_G3_gold.csv / gt_G3_hotspots.csv
    fig_support_vs_consensus.png    prioritization quadrants
    fig_group_size_dist.png         singleton/size tradeoff across G1/G2/G3
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

PARQUET = os.environ.get("REPAIR_PARQUET", "data/repair_profile.parquet")
OUT = "grouping/out"; os.makedirs(OUT, exist_ok=True)
ARCH_K = int(os.environ.get("ARCH_K", "60"))
FIT_SAMPLE = 150_000
TOL = 0.25                 # +/-25% band around the group median = "matches consensus"
COST, LAB, TXT = "det_approved_amt", "gsar_labor_hrs", "paws_comment_trail"


def add_archetypes(df):
    """Fit TF-IDF+KMeans on a sample, label every claim that has comment text."""
    has = df[TXT].notna() & (df[TXT].str.len() > 15)
    texts = df.loc[has, TXT]
    fit = texts.sample(min(FIT_SAMPLE, len(texts)), random_state=42)
    vec = TfidfVectorizer(max_features=4000, stop_words="english", ngram_range=(1, 2),
                          min_df=5, max_df=0.7, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
    km = MiniBatchKMeans(n_clusters=ARCH_K, random_state=42, n_init=3, batch_size=2048)
    km.fit(vec.fit_transform(fit))
    arch = pd.Series(np.nan, index=df.index, dtype="float")
    arch.loc[has] = km.predict(vec.transform(texts))
    df["arch"] = arch
    print(f"  text archetypes assigned to {has.sum():,} claims (K={ARCH_K})")
    return df


def consensus_frac(df, keys, col, tol=TOL):
    """Per-group fraction of claims within +/-tol of the group median (vectorized)."""
    med = df.groupby(keys, observed=True)[col].transform("median")
    valid = df[col].notna() & (med > 0)
    within = (df[col].sub(med).abs() <= tol * med) & valid
    tmp = pd.DataFrame({"_v": valid.astype(int), "_w": within.astype(int)})
    for i, k in enumerate(keys):
        tmp[k] = df[k].values
    agg = tmp.groupby(keys, observed=True)[["_v", "_w"]].sum()
    return (agg["_w"] / agg["_v"].where(agg["_v"] > 0)), med


def text_cohesion(df, keys):
    """Dominant-archetype share per group (how text-coherent a CODE group is)."""
    sub = df.dropna(subset=["arch"]).copy()
    sub["arch"] = sub["arch"].astype(int)
    cnt = sub.groupby(keys + ["arch"], observed=True).size().rename("c").reset_index()
    tot = cnt.groupby(keys, observed=True)["c"].sum().rename("tot")
    top = cnt.groupby(keys, observed=True)["c"].max().rename("top")
    return (top / tot)


def score_grouping(df, keys, tag):
    g = df.groupby(keys, observed=True)
    n = g.size().rename("n")
    cost_c, _ = consensus_frac(df, keys, COST)
    lab_c, _ = consensus_frac(df, keys, LAB)
    coh = text_cohesion(df, keys) if "arch" not in keys else pd.Series(1.0, index=n.index)
    out = pd.concat([n, cost_c.rename("cost_consensus"), lab_c.rename("labor_consensus"),
                     coh.rename("text_cohesion")], axis=1)
    # core ground-truth score = mean of available consensus dims (text, labor, cost)
    out["gt_score"] = out[["text_cohesion", "labor_consensus", "cost_consensus"]].mean(axis=1, skipna=True)
    out["cost_med"] = g[COST].median()
    out["labor_med"] = g[LAB].median()

    big = out[out["n"] >= 5]
    summary = {
        "grouping": tag, "keys": "+".join(keys),
        "n_groups": int(len(out)),
        "pct_singletons": round(100 * (out["n"] == 1).mean(), 1),
        "median_size": int(out["n"].median()),
        "pct_claims_in_grp_ge5": round(100 * out.loc[out["n"] >= 5, "n"].sum() / out["n"].sum(), 1),
        "median_gt_score_ge5": round(float(big["gt_score"].median()), 3),
        "gold_groups(n>=10,score>=.6)": int(((out["n"] >= 10) & (out["gt_score"] >= 0.6)).sum()),
        "hotspots(n>=20,score<=.4)": int(((out["n"] >= 20) & (out["gt_score"] <= 0.4)).sum()),
    }
    return out, summary


def main():
    cols = ["gsar_veh_line_desc", "paws_causal_part", "rep_sub_cat_label", COST, LAB, TXT, "part_numbers"]
    print(f"Loading {len(cols)} cols ...")
    df = pd.read_parquet(PARQUET, columns=cols)
    df[COST] = pd.to_numeric(df[COST], errors="coerce");  df.loc[(df[COST] <= 0) | (df[COST] >= 100000), COST] = np.nan
    df[LAB]  = pd.to_numeric(df[LAB],  errors="coerce");  df.loc[(df[LAB] <= 0) | (df[LAB] >= 200), LAB] = np.nan
    print(f"  {len(df):,} rows\n")
    df = add_archetypes(df)

    groupings = {
        "G1_code_vehicle":    ["gsar_veh_line_desc", "paws_causal_part"],
        "G2_code_veh_subcat": ["gsar_veh_line_desc", "paws_causal_part", "rep_sub_cat_label"],
        "G3_code_veh_archetype": ["gsar_veh_line_desc", "paws_causal_part", "arch"],
    }
    results, summaries = {}, []
    for tag, keys in groupings.items():
        print(f"scoring {tag} ...")
        out, summ = score_grouping(df, keys, tag)
        results[tag] = out; summaries.append(summ)

    summ_df = pd.DataFrame(summaries)
    summ_df.to_csv(f"{OUT}/gt_grouping_summary.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n=== Grouping comparison (support x consensus) ===")
    print(summ_df.to_string(index=False))

    # G3 gold + hotspots tables
    g3 = results["G3_code_veh_archetype"].reset_index()
    g3["arch"] = g3["arch"].astype(int)
    gold = g3[(g3["n"] >= 10) & (g3["gt_score"] >= 0.6)].sort_values(["n"], ascending=False)
    hot  = g3[(g3["n"] >= 20) & (g3["gt_score"] <= 0.4)].sort_values(["n"], ascending=False)
    gold.head(200).to_csv(f"{OUT}/gt_G3_gold.csv", index=False)
    hot.head(200).to_csv(f"{OUT}/gt_G3_hotspots.csv", index=False)
    print(f"\nG3: {len(gold)} GOLD ground-truth groups, {len(hot)} discrepancy HOTSPOTS")

    # ---- viz 1: support vs consensus quadrants (G3, groups n>=5) ----
    p = results["G3_code_veh_archetype"]; p = p[p["n"] >= 5]
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(p["n"], p["gt_score"], s=8, alpha=0.25,
                    c=p["cost_med"].clip(upper=20000), cmap="viridis")
    ax.set_xscale("log"); ax.axhline(0.6, ls="--", c="green", lw=1); ax.axhline(0.4, ls="--", c="red", lw=1)
    ax.axvline(10, ls=":", c="grey", lw=1)
    ax.set_xlabel("support  (claims per group, log)"); ax.set_ylabel("consensus  (ground-truth score)")
    ax.set_title("G3 groups: prioritization quadrants\n(top-right = gold ground truths; bottom-right = discrepancy hotspots)")
    plt.colorbar(sc, label="median approved $ (capped 20k)")
    ax.text(p["n"].max()*0.3, 0.9, "GOLD\n(use as ground truth)", color="green", ha="center", fontsize=9)
    ax.text(p["n"].max()*0.3, 0.15, "HOTSPOT\n(clean / investigate)", color="red", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_support_vs_consensus.png", dpi=120); plt.close(fig)

    # ---- viz 2: group-size distribution across G1/G2/G3 (singleton tradeoff) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    for tag in groupings:
        sizes = results[tag]["n"].clip(upper=100)
        ax.hist(sizes, bins=50, histtype="step", lw=1.8, label=tag, log=True)
    ax.set_xlabel("claims per group (capped 100)"); ax.set_ylabel("# groups (log)")
    ax.set_title("Group-size distribution: how specificity shrinks groups")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/fig_group_size_dist.png", dpi=120); plt.close(fig)

    print(f"\nWrote figs: {OUT}/fig_support_vs_consensus.png , {OUT}/fig_group_size_dist.png")


if __name__ == "__main__":
    main()
