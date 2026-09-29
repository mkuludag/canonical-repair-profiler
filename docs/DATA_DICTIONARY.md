# Data Dictionary - Canonical Repair Profiler

Schemas for the datasets the pipeline produces and consumes. Numeric fields are deterministic
(`StatsTool`); only `unified_diagnosis` / `suggested_correction` / `repair_name` text is LLM-generated.

---

## `repair_profile` (unified dataset, `repair_profile.parquet`)
One row per PAWS prior-approval request, joined across PAWS + GSAR + OWS on masked VIN + repair date.
938,307 rows x 112 columns. Key columns used downstream:

| Column | Type | Meaning |
|---|---|---|
| `request_r` | int | PAWS request id (the repair grain key; joins to `claim_signatures`) |
| `gsar_veh_line_desc` / `vehicle_line` | str | Vehicle line + model-year window, e.g. `P552N F-150 [15-20]` |
| `paws_causal_part` / `causal_part` | str | Causal part base code, e.g. `6148` |
| `paws_comment_trail` | str | Technician/customer narrative (used for symptom clustering + LLM context) |
| `gsar_tot_cost_gross` | float | **Realistic** total warranty claim cost (the cost basis used everywhere) |
| `gsar_labor_hrs` | float | Labor hours on the claim |
| `cond_cd` | str | Condition code (symptom) |

Cleaning rule applied on read: non-positive or implausibly large `gsar_tot_cost_gross` -> NaN (data errors dropped).

---

## `claim_signatures` (grouping output, `data/claim_signatures.parquet`)
Maps each claim to its Repair Signature.

| Column | Type | Meaning |
|---|---|---|
| `request_r` | int | claim id (joins to `repair_profile`) |
| `signature_id` | int | Repair Signature = vehicle_line x causal_part x symptom_archetype |

---

## `canonical_repairs.csv` (Canonical Repair Library, `grouping/out/`)
One row per signature (the CRL). 6,654 rows.

| Column | Type | Meaning |
|---|---|---|
| `signature_id` | int | signature key |
| `vehicle_line`, `causal_part` | str | signature dimensions |
| `arch`, `arch_theme` | int, str | symptom archetype id + its theme label |
| `n` | int | support (number of claims in the signature) |
| `status` | str | `GOLD` (high support x consensus) / `HOTSPOT` / other |
| `confidence`, `gt_score` | float | trust weight (support x consensus) and ground-truth score |
| `cost_med`, `cost_q25`, `cost_q75` | float | consensus cost band |
| `labor_med`, `labor_q25`, `labor_q75` | float | consensus labor band |
| `cost_consensus`, `labor_consensus` | float | agreement fractions (share within +/-25% of median) |
| `parts_modal_set` | str | most common parts |

---

## `golden_solutions.csv` (product output, `grouping/out/`)
One row per Canonical Repair (a signature may yield 1-2 after the LLM split). 5,070 rows.

| Column | Type | Meaning |
|---|---|---|
| `signature_id` | int | source signature |
| `vehicle_line`, `causal_part` | str | dimensions |
| `n_repairs_detected` | int | 1 or 2 (the agentic split decision, post separation-guard) |
| `repair_name` | str (LLM) | short repair label |
| `cost_tier` | str | `all` / `low` / `high` (which tier this row represents) |
| `tier_n_claims`, `support` | int | claims backing this repair |
| `split_rationale` | str (LLM) | why the group was/ wasn't split |
| `unified_diagnosis` | str (LLM) | dealer-selectable diagnosis |
| `suggested_correction` | str (LLM) | recommended correction |
| `typical_parts` | str (LLM) | typical parts |
| `labor_med_hrs` | float | consensus labor hours |
| `cost_med`, `cost_q25`, `cost_q75` | float | consensus cost band |
| `consensus` | float | tier agreement (cost & labor) |
| `confidence` | float | support-weighted consensus |
| `usable` | bool | has a valid cost median (safe to consume) |

---

## `region_overall.csv` (region study input, `grouping/out/`)
Per-US-state aggregates over 10.6M GSAR claims; powers the `RegionIndexTool`.

| Column | Type | Meaning |
|---|---|---|
| `state` | str | US state code |
| `claims` | int | claim count (weight for the national anchor) |
| `med_cost`, `mean_cost` | float | per-state cost central tendency |
| `med_labor_hrs`, `med_labor_cost`, `med_material` | float | labor/material medians |

Derived: `cost_index[state] = med_cost[state] / claims-weighted national median` (range 0.76-1.26, **1.67x** spread).

---

## `group_savings.csv` (savings output, `grouping/out/`, from `12_group_savings.py`)
Per single-repair canonical group, the avoidable overspend vs consensus.

| Column | Type | Meaning |
|---|---|---|
| `signature_id`, `vehicle_line`, `causal_part`, `repair_name` | - | group identity |
| `canonical_cost` | float | consensus cost the group should converge to |
| `n_claims`, `n_above` | int | claims analyzed / claims above consensus |
| `avoidable_overspend` | float | sum of `max(0, claim_cost - canonical_cost)` |
| `savings_per_claim` | float | overspend / n_claims |
| `pct_claims_above` | float | % of claims above consensus |
| `total_spend` | float | total historical spend in the group |

Aggregate headline: `grouping/out/group_savings_headline.txt` ($75.7M across 2,172 repairs; all dollar values rescaled for the public release).

---

## `data/sample/` (bundled offline sample)
Masked CSV slice for 8 signatures so the demo + tests run with no cloud (`CRP_USE_SAMPLE=1`).
Mirrors the schemas above: `canonical_repairs.csv`, `golden_solutions.csv`, `claim_signatures.csv`
(`request_r,signature_id`), and `sample_claims.csv` (`request_r, paws_comment_trail,
gsar_tot_cost_gross, gsar_labor_hrs, signature_id`; comment text truncated to 300 chars; no VINs).
