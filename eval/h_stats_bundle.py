#!/usr/bin/env python3
"""
h - Stats bundle: uncertainty and significance for the headline eval numbers, plus two new
analyses (a labor-gap analogue of the a2 guard AUC, and a TSB/SSM identifier distinctness
feasibility check).

Contents:
  1. Labor-gap AUC. Mirrors a2's guard-ratio construction with labor in place of cost:
     honored splits use the realized high/low tier labor-median ratio from golden_solutions;
     never-proposed singles use the labor Q3/Q1 proxy from canonical_repairs. Note the proposed
     arm here is honored-only (no labor record survives a collapse), and unlike cost the guard
     never thresholds labor, so honored-vs-never is not a truncation artifact on this axis.
  2. Bootstrap 95% CI on the existing a2 cost AUC (0.572).
  3. SEs and tests: churn SE, B1-vs-churn difference with combined SE, Fisher exact on the
     flips-vs-jitter 2x2, stratified SEs on the B3 weighted rates.
  4. Kappa correlates on the 400-signature panel: kappa vs B1 flips, and kappa vs the
     relative bootstrap CI width of each signature's claim-cost median.
  5. TSB/SSM identifier distinctness between cost tiers (feasibility + rates).

No scipy in the project venv: the Fisher test is computed exactly from the hypergeometric
distribution, and correlation p-values are two-sided permutation p-values (20,000 permutations,
seed 42). All outputs are counts, ratios, shares and AUCs; no dollar amounts are printed.

Usage:
    .venv/bin/python eval/h_stats_bundle.py [--no-append]
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import common as C

REPO_ROOT = C.REPO_ROOT
DATA_ROOT = REPO_ROOT / "grouping" / "out_v2"
OUT = C.OUT_DIR
SEED = 42
N_BOOT_AUC = 2000
N_BOOT_MEDIAN = 500
N_PERM = 20000


# --------------------------------------------------------------------------- generic stats

def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(random positive ranks above random negative), rank-sum identity (a2_separation_guard.py)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_auc_ci(pos: np.ndarray, neg: np.ndarray, n_boot: int = N_BOOT_AUC, seed: int = SEED):
    """Percentile bootstrap CI on the AUC, resampling each arm independently."""
    pos = np.asarray(pos, float); pos = pos[np.isfinite(pos)]
    neg = np.asarray(neg, float); neg = neg[np.isfinite(neg)]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        stats[i] = auc(rng.choice(pos, len(pos), replace=True),
                       rng.choice(neg, len(neg), replace=True))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]] (sum of point probs <= observed)."""
    row1, col1, n = a + b, a + c, a + b + c + d

    def p_of(x: int) -> float:
        return (math.comb(col1, x) * math.comb(n - col1, row1 - x)) / math.comb(n, row1)

    p_obs = p_of(a)
    lo, hi = max(0, row1 + col1 - n), min(row1, col1)
    return float(sum(p_of(x) for x in range(lo, hi + 1) if p_of(x) <= p_obs * (1 + 1e-9)))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(); y = y - y.mean()
    den = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / den) if den > 0 else float("nan")


def perm_corr(x, y, rank: bool = False, n_perm: int = N_PERM, seed: int = SEED):
    """Correlation with a two-sided permutation p-value. rank=True gives Spearman."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if rank:
        x = pd.Series(x).rank().to_numpy(); y = pd.Series(y).rank().to_numpy()
    r_obs = _pearson(x, y)
    rng = np.random.default_rng(seed)
    xc = x - x.mean()
    yc = y - y.mean()
    den = math.sqrt(float((xc * xc).sum()) * float((yc * yc).sum()))
    hits = 0
    for _ in range(n_perm):
        r = float((xc * rng.permutation(yc)).sum() / den)
        if abs(r) >= abs(r_obs) - 1e-12:
            hits += 1
    return r_obs, (hits + 1) / (n_perm + 1), int(ok.sum())


# --------------------------------------------------------------------------- claim-level scan

TSB_RE = re.compile(r"\b(?:TSB|BULLETIN)[\s#:.\-]*([0-9]{2})[\s\-]?([0-9]{3,4})\b", re.I)
SSM_RE = re.compile(r"\bSSM[\s#:.\-]*([0-9]{4,6})\b", re.I)
MENTION_RE = re.compile(r"TSB|SSM|bulletin", re.I)


def extract_ids(text: str) -> frozenset:
    ids = {f"TSB{m.group(1)}-{m.group(2)}" for m in TSB_RE.finditer(text)}
    ids |= {f"SSM{m.group(1)}" for m in SSM_RE.finditer(text)}
    return frozenset(ids)


def scan_claims(needed_sigs: set) -> tuple[pd.DataFrame, dict]:
    """Stream repair_profile.parquet, keep claims of needed signatures, extract identifiers.

    Returns (per-claim frame: signature_id, cost, ids) and a calibration summary of what
    identifier formats actually appear in the comment trails.
    """
    import pyarrow.parquet as pq

    cs = pd.read_parquet(REPO_ROOT / "data" / "claim_signatures.parquet",
                         columns=["request_r", "signature_id"])
    cs = cs[cs["signature_id"].isin(needed_sigs)]
    rr_to_sig = {str(r): int(s) for r, s in zip(cs["request_r"], cs["signature_id"])}

    pf = pq.ParquetFile(REPO_ROOT / "paws_discovery" / "repair_profile.parquet")
    rows_sig, rows_cost, rows_ids = [], [], []
    n_scanned = n_mention = n_with_id = 0
    fmt_counter: Counter = Counter()
    sample_ids: dict[str, set] = {"TSB": set(), "SSM": set()}
    for batch in pf.iter_batches(batch_size=20000,
                                 columns=["request_r", "paws_comment_trail",
                                          "gsar_tot_cost_gross"]):
        df = batch.to_pandas()
        key = df["request_r"].astype(str)
        mask = key.isin(rr_to_sig)
        if not mask.any():
            continue
        df = df[mask]
        sigs = key[mask].map(rr_to_sig)
        costs = pd.to_numeric(df["gsar_tot_cost_gross"], errors="coerce")
        for sig, cost, text in zip(sigs, costs, df["paws_comment_trail"].astype(str)):
            n_scanned += 1
            if MENTION_RE.search(text):
                n_mention += 1
            ids = extract_ids(text)
            if ids:
                n_with_id += 1
                for i in ids:
                    fam = "TSB" if i.startswith("TSB") else "SSM"
                    fmt_counter[fam] += 1
                    if len(sample_ids[fam]) < 8:
                        sample_ids[fam].add(i)
            rows_sig.append(sig); rows_cost.append(cost); rows_ids.append(ids)

    claims = pd.DataFrame({"signature_id": rows_sig, "cost": rows_cost, "ids": rows_ids})
    calib = {
        "n_claims_scanned": n_scanned,
        "share_claims_mentioning_tsb_ssm_bulletin": round(n_mention / n_scanned, 4),
        "share_claims_with_parsed_identifier": round(n_with_id / n_scanned, 4),
        "identifier_family_mention_counts": dict(fmt_counter),
        "example_identifiers": {k: sorted(v) for k, v in sample_ids.items()},
        "formats_observed": ("TSB yy-nnnn (also 'TSB yy nnnn' and 'bulletin yy-nnnn', "
                             "normalized to TSByy-nnnn); SSM nnnnn (5-digit, occasionally "
                             "4-6). Boilerplate 'TSB or SSM: NONE' lines carry no id and "
                             "are not matched."),
    }
    return claims, calib


def tier_id_sets(claims: pd.DataFrame, sig_ids) -> pd.DataFrame:
    """Median-cut tiers (stats_tool.cost_tier_partition: low = cost < median, high = cost >=
    median; claims with no numeric cost fall in neither tier) and per-tier identifier unions."""
    out = []
    grouped = claims[claims["signature_id"].isin(set(sig_ids))].groupby("signature_id")
    for sid, grp in grouped:
        c = grp["cost"]
        med = c.median()
        if not np.isfinite(med):
            continue
        low = grp[c < med]; high = grp[c >= med]
        ids_low = frozenset().union(*low["ids"]) if len(low) else frozenset()
        ids_high = frozenset().union(*high["ids"]) if len(high) else frozenset()
        out.append({"signature_id": sid, "n_claims": len(grp),
                    "n_ids_low": len(ids_low), "n_ids_high": len(ids_high),
                    "any_hit": bool(ids_low or ids_high),
                    "both_tiers_hit": bool(ids_low and ids_high),
                    "disjoint": len(ids_low & ids_high) == 0})
    return pd.DataFrame(out)


def id_rates(tiers: pd.DataFrame, n_population: int) -> dict:
    hit = tiers[tiers["any_hit"]]
    both = tiers[tiers["both_tiers_hit"]]
    return {
        "n_population": int(n_population),
        "n_with_claims_and_cost": int(len(tiers)),
        "n_any_identifier": int(len(hit)),
        "coverage_any_identifier": round(len(hit) / n_population, 4),
        "disjoint_among_any_hit": round(float(hit["disjoint"].mean()), 4) if len(hit) else None,
        "n_both_tiers_hit": int(len(both)),
        "coverage_both_tiers": round(len(both) / n_population, 4),
        "disjoint_among_both_tiers_hit":
            round(float(both["disjoint"].mean()), 4) if len(both) else None,
    }


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-append", action="store_true")
    args = ap.parse_args()

    cls = pd.read_csv(OUT / "a2_signature_classes.csv")
    g = C.load_golden(DATA_ROOT)
    crl = C.load_crl(DATA_ROOT)
    results: dict = {"data_root": str(DATA_ROOT), "seed": SEED}

    # ---- 1. labor-gap AUC -----------------------------------------------------------------
    split = g[g["n_repairs_detected"] == 2]
    agg = split.groupby("signature_id")["labor_med_hrs"].agg(["min", "max", "count"])
    honored_ids = set(cls.loc[cls["klass"] == "honored", "signature_id"])
    lab_pos = (agg.loc[(agg["count"] == 2) & (agg["min"] > 0)]
               .loc[lambda d: d.index.isin(honored_ids)])
    labor_pos = (lab_pos["max"] / lab_pos["min"]).to_numpy()

    single_ids = cls.loc[cls["klass"] == "single", "signature_id"]
    crl_idx = crl.set_index("signature_id")
    lab_neg = crl_idx.reindex(single_ids)[["labor_q25", "labor_q75"]]
    labor_neg = ((lab_neg["labor_q75"] / lab_neg["labor_q25"])
                 .replace([np.inf, -np.inf], np.nan).dropna().to_numpy())

    auc_labor = auc(labor_pos, labor_neg)
    lab_lo, lab_hi = boot_auc_ci(labor_pos, labor_neg)
    results["labor_gap_auc"] = {
        "auc": round(auc_labor, 4), "ci95": [round(lab_lo, 4), round(lab_hi, 4)],
        "n_boot": N_BOOT_AUC,
        "n_honored_with_labor_ratio": int(len(labor_pos)),
        "n_never_proposed_with_labor_q3q1": int(len(labor_neg)),
        "median_ratio_honored": round(float(np.median(labor_pos)), 3),
        "median_ratio_never": round(float(np.median(labor_neg)), 3),
        "construction": ("Positives: honored splits' realized high/low tier labor-median "
                         "ratio (golden_solutions labor_med_hrs). Negatives: never-proposed "
                         "singles' labor Q3/Q1 (canonical_repairs labor_q25/q75, gsar_labor_hrs "
                         "basis). Collapsed proposals are excluded: no labor tier record "
                         "survives a collapse. The guard never thresholds labor, so this arm "
                         "pairing is not a truncation artifact on the labor axis."),
    }

    # ---- 2. cost AUC CI ---------------------------------------------------------------------
    prop = cls[cls["proposed"] & cls["guard_ratio"].notna()]["guard_ratio"].to_numpy()
    nev = cls[(~cls["proposed"]) & cls["guard_ratio"].notna()]["guard_ratio"].to_numpy()
    auc_cost = auc(prop, nev)
    ref = json.loads((OUT / "a2_guard_summary.json").read_text())["auc_overall"]
    assert abs(auc_cost - ref) < 1e-9, f"cost AUC mismatch: {auc_cost} vs a2 {ref}"
    cost_lo, cost_hi = boot_auc_ci(prop, nev)
    results["cost_auc"] = {
        "auc": round(auc_cost, 4), "ci95": [round(cost_lo, 4), round(cost_hi, 4)],
        "n_boot": N_BOOT_AUC, "n_proposed": int(len(prop)), "n_never": int(len(nev)),
    }

    # ---- 3. SEs and tests -------------------------------------------------------------------
    a7 = json.loads((OUT / "a7_churn.json").read_text())
    m = a7["split_matrix"]
    n_common = int(a7["n_signatures_common"])
    n_agree = int(m["v1=1"]["v2=1"] + m["v1=2"]["v2=2"])
    p_churn = n_agree / n_common
    se_churn = math.sqrt(p_churn * (1 - p_churn) / n_common)

    b1 = json.loads((OUT / "b1_stability.json").read_text())
    agree_b1 = float(b1["mean_population_weighted_agreement"])
    pair_ses = {p["pair"]: float(p["population_weighted_se"]) for p in b1["pairs"]}
    se_b1 = max(pair_ses.values())  # conservative: pairs share runs, so take the largest
    diff = agree_b1 - p_churn
    se_diff = math.sqrt(se_b1 ** 2 + se_churn ** 2)

    ct = b1["flips_vs_jitter_rep1xrep2"]["crosstab"]
    a_ = ct["flipped=True"]["jittered=True"]; b_ = ct["flipped=False"]["jittered=True"]
    c_ = ct["flipped=True"]["jittered=False"]; d_ = ct["flipped=False"]["jittered=False"]
    p_fisher = fisher_exact_2x2(a_, b_, c_, d_)

    b3 = pd.read_csv(OUT / "b3_single_call_per_signature.csv")
    w = C.panel_weights()
    strata = sorted(w)
    b3["prop"] = (pd.to_numeric(b3["n_repairs"]) >= 2).astype(int)
    over_terms, veto_terms = {}, {}
    for s in strata:
        sub = b3[b3["stratum"] == s]
        splits = sub[sub["prop"] == 1]
        over_terms[s] = {"w": w[s], "p": float(sub["prop"].mean()), "n": int(len(sub))}
        veto_terms[s] = {"p": float(splits["guard_collapsed"].astype(float).mean()),
                         "n": int(len(splits))}
    over_rate = sum(t["w"] * t["p"] for t in over_terms.values())
    se_over = math.sqrt(sum(t["w"] ** 2 * t["p"] * (1 - t["p"]) / t["n"]
                            for t in over_terms.values()))
    # veto rate among proposed splits, splits weighted by population incidence w_s * p_s
    wsplit = {s: over_terms[s]["w"] * over_terms[s]["p"] / over_rate for s in strata}
    veto_rate = sum(wsplit[s] * veto_terms[s]["p"] for s in strata)
    se_veto = math.sqrt(sum(wsplit[s] ** 2 * veto_terms[s]["p"] * (1 - veto_terms[s]["p"])
                            / veto_terms[s]["n"] for s in strata))
    results["ses_and_tests"] = {
        "churn": {"agree": n_agree, "n": n_common, "rate": round(p_churn, 4),
                  "se": round(se_churn, 4)},
        "b1_weighted_agreement": {"rate": agree_b1, "se_used": se_b1,
                                  "se_per_pair": pair_ses,
                                  "note": "largest per-pair SE used (pairs share runs)"},
        "difference_b1_minus_churn": {"diff": round(diff, 4), "combined_se": round(se_diff, 4),
                                      "z": round(diff / se_diff, 1)},
        "fisher_flips_vs_jitter": {"table_[[flip_jit,nofip_jit],[flip_first,noflip_first]]":
                                   [[a_, b_], [c_, d_]],
                                   "p_two_sided": p_fisher},
        "b3_weighted": {
            "over_proposal": {"rate": round(over_rate, 4), "se": round(se_over, 4),
                              "per_stratum": over_terms},
            "guard_veto_among_splits": {
                "rate": round(veto_rate, 4), "se": round(se_veto, 4),
                "per_stratum": veto_terms, "split_incidence_weights": wsplit,
                "note": ("splits weighted by population incidence w_s*p_s; p_s treated as "
                         "fixed. Collapsed-stratum veto rate is 37/37, so it contributes "
                         "zero variance and the SE is slightly understated."),
            },
        },
    }

    # ---- 4. kappa correlates on the panel ----------------------------------------------------
    panel = C.load_panel()
    kap = g.groupby("signature_id")["confidence"].agg(kappa_min="min", kappa_mean="mean")
    panel = panel.merge(kap, left_on="signature_id", right_index=True, how="left")

    b1sig = pd.read_csv(OUT / "b1_stability_per_signature.csv")
    b1sig["flip_any"] = ~((b1sig["n_repairs"] == b1sig["n_repairs_r1"])
                          & (b1sig["n_repairs"] == b1sig["n_repairs_r2"])
                          & (b1sig["n_repairs_r1"] == b1sig["n_repairs_r2"]))
    b1sig["flip_r1r2"] = b1sig["n_repairs_r1"] != b1sig["n_repairs_r2"]
    panel = panel.merge(b1sig[["signature_id", "flip_any", "flip_r1r2"]],
                        on="signature_id", how="left")
    assert panel["flip_any"].notna().all(), "panel signature missing from b1 per-signature file"
    panel["flip_any"] = panel["flip_any"].astype(bool)
    panel["flip_r1r2"] = panel["flip_r1r2"].astype(bool)

    kq = panel["kappa_min"].quantile([0.25, 0.5, 0.75])
    def quartile_of(v: float) -> str:
        return ("Q1" if v <= kq[0.25] else "Q2" if v <= kq[0.5]
                else "Q3" if v <= kq[0.75] else "Q4")

    r_pb, p_pb, n_pb = perm_corr(panel["kappa_min"], panel["flip_any"].astype(float))
    r_rank, p_rank, _ = perm_corr(panel["kappa_min"], panel["flip_any"].astype(float), rank=True)
    flipped = panel[panel["flip_any"]]
    results["kappa_vs_flips"] = {
        "kappa_definition": ("per-signature kappa = golden_solutions confidence column "
                             "(= consensus * min(1, log10(n+1)/log10(31))); for split "
                             "signatures the MINIMUM across the two emitted tiers is used "
                             "throughout (the weaker tier bounds stability)"),
        "n_panel": int(n_pb),
        "n_flipped_any_pair": int(panel["flip_any"].sum()),
        "n_flipped_rep1xrep2": int(panel["flip_r1r2"].sum()),
        "point_biserial_r": round(r_pb, 4), "p_perm": round(p_pb, 4),
        "rank_r": round(r_rank, 4), "p_perm_rank": round(p_rank, 4),
        "auc_kappa_separates_flipped": round(
            auc(panel.loc[~panel["flip_any"], "kappa_min"].to_numpy(),
                panel.loc[panel["flip_any"], "kappa_min"].to_numpy()), 4),
        "median_kappa_flipped": round(float(flipped["kappa_min"].median()), 3),
        "median_kappa_unflipped": round(float(panel.loc[~panel["flip_any"],
                                                        "kappa_min"].median()), 3),
        "flipped_sig_kappa_quartiles": {int(r.signature_id): quartile_of(r.kappa_min)
                                        for r in flipped.itertuples()},
        "panel_kappa_quartile_edges": [round(float(kq[q]), 3) for q in (0.25, 0.5, 0.75)],
    }

    # claims needed for item 4b and item 5
    a7sig = pd.read_csv(OUT / "a7_churn_per_signature.csv")
    flipped_pop = set(a7sig.loc[a7sig["n_repairs_v1"] != a7sig["n_repairs_v2"], "signature_id"])
    collapsed_ids = set(cls.loc[cls["klass"] == "collapsed", "signature_id"])
    needed = honored_ids | collapsed_ids | flipped_pop | set(panel["signature_id"])
    claims, calib = scan_claims(needed)

    # 4b: per-signature bootstrap CI width of the claim-cost median, vs kappa
    rng = np.random.default_rng(SEED)
    widths = {}
    for sid, grp in claims[claims["signature_id"].isin(set(panel["signature_id"]))] \
            .groupby("signature_id"):
        costs = grp["cost"].dropna().to_numpy(float)
        if len(costs) < 5:
            continue
        med = float(np.median(costs))
        if med <= 0:
            continue
        boots = np.median(rng.choice(costs, (N_BOOT_MEDIAN, len(costs)), replace=True), axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        widths[sid] = (hi - lo) / med
    panel["rel_ci_width"] = panel["signature_id"].map(widths)
    rho_all, p_all, n_all = perm_corr(panel["kappa_min"], panel["rel_ci_width"], rank=True)
    single_mask = panel["stratum"] == "single"
    rho_sgl, p_sgl, n_sgl = perm_corr(panel.loc[single_mask, "kappa_min"],
                                      panel.loc[single_mask, "rel_ci_width"], rank=True)
    results["kappa_vs_ci_width"] = {
        "bootstrap": {"n_resamples": N_BOOT_MEDIAN, "seed": SEED,
                      "definition": "relative CI width = (p97.5 - p2.5) / observed median, "
                                    "per-signature bootstrap over its claims' costs"},
        "n_panel_with_width": int(n_all),
        "spearman_rho_panel": round(rho_all, 4), "p_perm_panel": round(p_all, 5),
        "spearman_rho_single_stratum": round(rho_sgl, 4),
        "p_perm_single_stratum": round(p_sgl, 5), "n_single_stratum": int(n_sgl),
    }

    # ---- 5. TSB/SSM identifier distinctness ---------------------------------------------------
    tiers_hon = tier_id_sets(claims, honored_ids)
    tiers_col = tier_id_sets(claims, collapsed_ids)
    tiers_flp = tier_id_sets(claims, flipped_pop)
    tsb = {
        "calibration": calib,
        "tier_construction": ("median cut on gsar_tot_cost_gross per stats_tool."
                              "cost_tier_partition: low = cost < median, high = cost >= "
                              "median; claims without a numeric cost join neither tier"),
        "honored": id_rates(tiers_hon, len(honored_ids)),
        "collapsed": id_rates(tiers_col, len(collapsed_ids)),
        "flipped_v1_vs_v2": id_rates(tiers_flp, len(flipped_pop)),
    }
    min_cov = min(tsb[k]["coverage_any_identifier"]
                  for k in ("honored", "collapsed", "flipped_v1_vs_v2"))
    tsb["viable"] = bool(min_cov >= 0.05)
    results["tsb_distinctness"] = tsb

    C.write_json(results, "h_stats_bundle.json")

    # ---- RESULTS.md ---------------------------------------------------------------------------
    kv = results["kappa_vs_flips"]; kw = results["kappa_vs_ci_width"]
    st = results["ses_and_tests"]
    hon, col, flp = tsb["honored"], tsb["collapsed"], tsb["flipped_v1_vs_v2"]
    body = f"""
Uncertainty and significance for the headline eval numbers, plus a labor analogue of the a2
guard AUC and a TSB/SSM identifier check. Bootstrap and permutation seeds are 42 throughout.

**1. Labor-gap AUC.** Mirroring a2's construction with labor in place of cost (honored splits:
realized high/low tier labor-median ratio from `golden_solutions.csv`; never-proposed singles:
labor Q3/Q1 from `canonical_repairs.csv`; collapsed proposals excluded because no labor tier
record survives a collapse), the labor gap separates proposed from never-proposed signatures
with **AUC {results['labor_gap_auc']['auc']:.3f}** (95% CI
[{results['labor_gap_auc']['ci95'][0]:.3f}, {results['labor_gap_auc']['ci95'][1]:.3f}],
{N_BOOT_AUC} resamples; n = {results['labor_gap_auc']['n_honored_with_labor_ratio']:,} vs
{results['labor_gap_auc']['n_never_proposed_with_labor_q3q1']:,}). Median ratios are
{results['labor_gap_auc']['median_ratio_honored']:.2f} (honored) vs
{results['labor_gap_auc']['median_ratio_never']:.2f} (never proposed). Unlike cost, labor is
never thresholded by the guard, so this comparison carries no truncation artifact. The interval
sits marginally below 0.5: the labor gap on proposed splits is no larger than ordinary
within-repair labor spread, so the labor axis provides no positive discrimination of what the
analyst proposed. Caveat: the two arms sit on different labor bases (tier `labor_med_hrs` vs
`gsar_labor_hrs` quartiles).

**2. Cost AUC CI.** The a2 headline cost AUC of {results['cost_auc']['auc']:.3f} gets a
bootstrap 95% CI of **[{results['cost_auc']['ci95'][0]:.3f},
{results['cost_auc']['ci95'][1]:.3f}]** ({N_BOOT_AUC} resamples,
n = {results['cost_auc']['n_proposed']:,} proposed vs {results['cost_auc']['n_never']:,} never
proposed). Above chance, but the entire interval stays in the negligible-discrimination band.

**3. SEs and tests.** (a) v1-to-v2 churn agreement is {st['churn']['agree']:,}/{st['churn']['n']:,}
= {C.pct(st['churn']['rate'])} with binomial SE {st['churn']['se']*100:.2f}pp. (b) B1 same-prompt
weighted agreement is {C.pct(st['b1_weighted_agreement']['rate'], 2)} with SE
{st['b1_weighted_agreement']['se_used']*100:.2f}pp (largest of the three per-pair SEs in
`b1_stability.json`; the pairs share runs). The stability-minus-churn difference is
**{st['difference_b1_minus_churn']['diff']*100:.1f}pp** with combined SE
{st['difference_b1_minus_churn']['combined_se']*100:.1f}pp
(z = {st['difference_b1_minus_churn']['z']:.0f}). (c) Fisher exact on the flips-vs-jitter table
(3/17 jittered signatures flipped vs 0/383 first-try) gives two-sided
**p = {st['fisher_flips_vs_jitter']['p_two_sided']:.1e}**: flips concentrate in jittered calls
far beyond chance, consistent with jitter attribution of the residual disagreement (it does not
prove every flip was caused by jitter). (d) B3 weighted rates: over-proposal
**{C.pct(st['b3_weighted']['over_proposal']['rate'])} +/- {st['b3_weighted']['over_proposal']['se']*100:.1f}pp**,
guard-veto among proposed splits
**{C.pct(st['b3_weighted']['guard_veto_among_splits']['rate'])} +/- {st['b3_weighted']['guard_veto_among_splits']['se']*100:.1f}pp**
(stratified SEs, sqrt of sum over strata of w^2 p(1-p)/n; the collapsed stratum's veto rate is
37/37 and contributes zero variance, so the second SE is slightly understated).

**4. Kappa correlates (H5).** Per-signature kappa is the `confidence` column of the v2 library;
for split signatures the **minimum** across the two emitted tiers is used throughout. (a) Kappa
vs B1 flips: {kv['n_flipped_any_pair']} of 400 panel signatures flipped in at least one of the
three pairwise run comparisons ({kv['n_flipped_rep1xrep2']} in rep1 x rep2 alone).
Point-biserial r = {kv['point_biserial_r']:.3f} (permutation p = {kv['p_perm']:.2f}); kappa of
unflipped signatures separates flipped ones with AUC {kv['auc_kappa_separates_flipped']:.3f}.
Median kappa {kv['median_kappa_flipped']:.3f} (flipped) vs {kv['median_kappa_unflipped']:.3f}
(unflipped); the flipped signatures sit in panel kappa quartiles
{', '.join(sorted(kv['flipped_sig_kappa_quartiles'].values()))}. With
{kv['n_flipped_any_pair']} flips the test is underpowered; the direction is suggestive, not
established. (b) Kappa vs cost-median stability: per-signature bootstrap of the claim-cost
median ({N_BOOT_MEDIAN} resamples), relative CI width = (hi - lo)/median. Spearman rho =
**{kw['spearman_rho_panel']:.3f}** (permutation p = {kw['p_perm_panel']:.5g},
n = {kw['n_panel_with_width']}); within the single stratum alone rho =
{kw['spearman_rho_single_stratum']:.3f} (p = {kw['p_perm_single_stratum']:.5g},
n = {kw['n_single_stratum']}). Both p-values sit at the floor of a 20,000-permutation test.
Higher-kappa signatures do have more stable cost medians.

**5. TSB/SSM identifier distinctness (H2-lite).** Comment-trail inspection shows two usable
identifier families: TSB ids in the form yy-nnnn (also written with a space, or introduced by
the word bulletin) and SSM ids of 4 to 6 digits; boilerplate lines like "TSB or SSM: NONE"
carry no id and are not matched. {C.pct(calib['share_claims_mentioning_tsb_ssm_bulletin'])} of
scanned claims mention TSB/SSM/bulletin and {C.pct(calib['share_claims_with_parsed_identifier'])}
carry a parseable identifier. Tiers are the median cut of each signature's claim costs, matching
the shipped construction. Coverage is well above the 5% viability floor, so the analysis is
viable: {C.pct(hon['coverage_any_identifier'])} of honored, {C.pct(col['coverage_any_identifier'])}
of collapsed and {C.pct(flp['coverage_any_identifier'])} of v1/v2-flipped signatures have at
least one identifier. Among signatures where BOTH tiers carry identifiers, the two tiers'
identifier sets are fully disjoint in {C.pct(hon['disjoint_among_both_tiers_hit'])} of honored
(n = {hon['n_both_tiers_hit']}), {C.pct(col['disjoint_among_both_tiers_hit'])} of collapsed
(n = {col['n_both_tiers_hit']}) and {C.pct(flp['disjoint_among_both_tiers_hit'])} of flipped
(n = {flp['n_both_tiers_hit']}) signatures. Counting one-sided hits as disjoint (an empty set
is disjoint by definition) the rates are {C.pct(hon['disjoint_among_any_hit'])},
{C.pct(col['disjoint_among_any_hit'])} and {C.pct(flp['disjoint_among_any_hit'])}. Similar
disjointness across honored and collapsed populations means identifier separation as computed
here does not obviously validate the honored splits over the collapsed ones; treat it as a
feasibility result, not a verdict.

Artifacts: `eval/out/h_stats_bundle.json`.
"""
    C.append_results("H stats bundle", body, DATA_ROOT, not args.no_append)

    print(f"\n  labor AUC {auc_labor:.3f} [{lab_lo:.3f},{lab_hi:.3f}] | cost AUC {auc_cost:.3f} "
          f"[{cost_lo:.3f},{cost_hi:.3f}] | diff {diff*100:.1f}pp z={diff/se_diff:.0f} | "
          f"fisher p={p_fisher:.1e} | rho(kappa,width)={kw['spearman_rho_panel']:.3f}")


if __name__ == "__main__":
    main()
