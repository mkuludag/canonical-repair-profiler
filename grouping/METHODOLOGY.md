# Methodology — Canonical Repair Profiling (CRP)

> This is the consolidated methodology for the `grouping/` work. It merges what were five
> separate working docs (`DEFINITIONS`, `PLAN`, `STRATEGY`, `FINDINGS`, `SUMMARY`) into one
> reference: the vocabulary, the grouping evaluation, the consensus recipe, the measured
> results, and the emitted Canonical Repair Library + honest limitations.

---

## 1. The method, in one paragraph

**Canonical Repair Profiling (CRP)** takes many real warranty/PAWS claims, clusters the *alike*
ones into a **Repair Signature**, and distills each cluster into one **canonical (consensus)
repair** — a defensible *correction + labor + cost + parts* with a **confidence** weight. The
per-group consensus solution is a **Canonical Repair (CR)** (a "golden record" for that repair);
the full collection is the **Canonical Repair Library (CRL)**, the dataset we emit. Deviations
from the consensus become **discrepancy** signals (data cleaning + cost-savings). We deliberately
retire the loose phrase "ground truth" in favor of **Canonical Repair**, because these are
*derived consensus* labels (with a confidence), not externally-verified truth.

Engine repairs and tail-light repairs must never share a canonical repair; "F-150 part 6148,
'long block' archetype, 16.8 hr, $8.9k" should.

---

## 2. Glossary (shared vocabulary)

| Term | Definition |
|---|---|
| **Claim / repair line** | One row of `repair_profile` = one PAWS request (`request_r`), ≈ one VIN+RO+repair-line. The atomic unit. |
| **Repair Signature** | The grouping key that defines "alike" claims. Default = **vehicle line × causal part × symptom archetype** (optionally × **cost tier**). What a Canonical Repair is built for. |
| **Symptom archetype** | A cluster id (0..K) from TF-IDF + KMeans over the 3 C's narrative (`paws_comment_trail`). Captures the *repair theme* (e.g. "transmission overhaul", "water pump / coolant") that codes miss. |
| **Causal part** | `paws_causal_part` — the part deemed to have caused the failure. The structural "what failed". |
| **Cost tier** | Optional sub-split of a signature into low/high cost bands when a single causal-part code hides two repair scopes (reseal vs replace). Turns one fuzzy signature into two clean ones. |
| **Support** | Number of claims in a signature (`n`). Bigger = more statistical confidence in the consensus. |
| **Consensus** | How much the claims in a signature *agree* on the solution. Measured per dimension as the fraction of claims within ±25% of the signature median (cost, labor) or the dominant-archetype share (text). |
| **gt_score** (consensus score) | Mean of the available consensus dimensions (text cohesion, labor consensus, cost consensus). 0–1. |
| **Confidence** | A single trust weight for a Canonical Repair = f(**support**, **consensus**). Lets downstream models/auditors down-weight shaky labels. |
| **Canonical Repair (CR)** | The emitted consensus solution for one signature: representative correction, labor hours (median ± band), cost (median ± band), modal parts, recall share, confidence, status. |
| **Discrepancy** | A claim's deviation from its signature's Canonical Repair (e.g. cost > 1.8× the CR median). The cleaning + savings signal. |
| **Hotspot** | A high-support, low-consensus signature — many claims that *don't* agree → investigate / clean (often a hidden multi-repair or a process problem). |
| **Status (per signature)** | `GOLD` (n≥10 & gt_score≥0.6 → emit as CR), `SILVER` (n≥10 & 0.4≤gt<0.6 → CR with low confidence / try tiering), `HOTSPOT` (n≥20 & gt<0.4 → discrepancy review), `SPARSE` (n<10 → not enough support). |

### Identity / cross-system keys
- **VIN**: masked only (`vin_masked`, first-11 + hash); identical scheme across PAWS/GSAR/OWS → cross-system join key.
- **Cross-system bridge**: masked VIN + repair date (RO# and dealer codes are NOT comparable across systems).
- **Spine key**: `request_r` (PAWS request id), one per claim row.

---

## 3. Grouping evaluation — the reframe that drives everything: SUPPORT × CONSENSUS

A grouping is good for a canonical repair only if groups are simultaneously:
- **high support** — enough claims to define a consensus and police outliers (≥10 is workable), and
- **high consensus** — claims actually agree on the solution (labor/cost/text), so one answer is valid.

We score each group on a **gt_score** = mean of the consensus it achieves on the *dense* solution
dimensions (text-archetype agreement, labor-hours agreement, cost agreement). Parts list is only
**16% populated**, so it is a *bonus* signal, not the backbone. Two complementary products fall out
of the same scoring:
- **Top-right (high support, high consensus) = GOLD** → emit as canonical-repair labels.
- **Bottom-right (high support, low consensus) = HOTSPOT** → clean / investigate / dealer-correct.

See `out/fig_support_vs_consensus.png`.

### 3.1 Structural grouping techniques benchmarked (`01_grouping_potential.py`)

Each technique groups the 938,307 claims by different keys. We measure **# groups**,
**claims/group**, and **within-group variability** = coefficient of variation (CV = std/mean) of
cost and labor. Lower CV = a more coherent, less contradictory group.

| Technique | Keys | Groups | Avg/grp | Median | Max grp | approved-$ CV | GSAR-$ CV | labor CV |
| --------- | ---- | ------ | ------- | ------ | ------- | ------------- | --------- | -------- |
| T1 | repair category | 16 | 58,644 | 40,432 | 188,568 | 3.02 | 1.67 | 1.03 |
| T2 | category + sub-category | 158 | 5,939 | 1,795 | 143,469 | 2.25 | 1.40 | 0.94 |
| T8 | WCC (warranty condition code) | 473 | 1,278 | 111 | 40,075 | 1.89 | 1.02 | 0.96 |
| T7 | CCC concern + condition | 4,652 | 113 | 4 | 47,375 | 1.19 | 0.74 | 0.75 |
| T3 | causal part | 8,519 | 110 | 3 | 44,716 | 1.40 | 0.79 | 0.78 |
| T4 | sub-category + vehicle line | 5,134 | 124 | 12 | 18,987 | 1.13 | 0.83 | 0.79 |
| T6 | model-year + line + sub-cat | 15,426 | 25 | 4 | 5,671 | 0.94 | 0.70 | 0.70 |
| **T5** | **causal part + vehicle line** | **26,943** | **24** | **2** | **28,362** | **0.88** | **0.66** | **0.65** |

**Reading it:** coarse keys (category, WCC) make a few huge but internally-contradictory buckets
(a "Gas Engine Performance" group mixes trivial and five-figure repairs, CV 3.0). **Adding vehicle line
tightens everything** — T5 (causal part + vehicle line) is the sweet spot: 27k groups, cost CV
0.88, labor CV 0.65. The residual 0.6–0.9 CV even in tight groups is exactly the *opportunity*
(same part + same vehicle, yet different dollars/labor → cost-standardization & outlier detection).

### 3.2 Semantic grouping from the 3 C's text (`04_text_clustering.py`)

TF-IDF + MiniBatchKMeans on the `paws_comment_trail` narrative recovers **repair archetypes that
codes miss**, each with its own cost profile:

| Archetype (top terms) | Claims | Median approved $ |
| --------------------- | ------ | ----------------- |
| long block / cylinder / coolant | 2,866 | **$30,734** |
| transmission replace / overhaul / gear | 3,646 | **$21,436** |
| trans / clutch / converter | 2,486 | $14,166 |
| valve body / main control | 2,332 | $8,421 |
| labor-hours amendment | 3,741 | $6,432 |
| cam phasers / rattle / VCT / TSB | 3,396 | $5,230 |
| sunroof / running board / motors | 3,155 | $4,831 |
| oil pan / oil leak | 3,484 | $3,629 |
| water pump / coolant | 3,553 | $2,891 |
| steering / tie rod / ball joint | 3,060 | $2,078 |

Text clusters are broader on cost (group by *symptom/repair theme* regardless of severity), so
they are best for **discovery/triage**; code+vehicle groups (T5) are best for **cost
benchmarking**. **The winner combines them:** text archetype *within* a causal-part+vehicle group.

### 3.3 The chosen grouping — G3

| Code | Keys | Use |
|---|---|---|
| **G1** | vehicle line + causal part | coarse; good for *discrepancy* detection |
| **G2** | + sub-category | tighter coded |
| **G3** | + symptom archetype | **primary CR grouping** (best support×consensus) |
| **G3+T** | G3 + cost tier | refine fuzzy G3 signatures |

G3 (vehicle line × causal part × symptom archetype) lifts median consensus **0.41 → 0.63** and
yields **14× more usable groups** than code+vehicle alone. 57% of groups are singletons, but
**83.7% of claims sit in groups ≥5** — the usable core is large.

---

## 4. The consensus recipe (`05_ground_truth_grouping.py`, `08_build_ground_truth.py`)

1. **Assign claims to signatures (full grouping).** Symptom archetype via TF-IDF (alphabetic 1–2
   grams) + KMeans(K=60) on `paws_comment_trail`, fit on a sample, label all text claims.
   `signature_id = stable_hash(vehicle_line, causal_part, archetype)`. Output:
   `data/claim_signatures.parquet` (request_r → signature keys + ids).
2. **Gate by support & decide tiering.** `n ≥ 10` to attempt a CR. If a signature is **fuzzy**
   (cost or labor consensus < 0.5) and `n ≥ 20`, split into 2 cost tiers and treat each as its own.
3. **Outlier-robust consensus per dimension** (the heart of it):
   - **Cost** (`gsar_tot_cost_gross`): drop placeholders (≤0, ≥100k); **trimmed median** as
     `cost_med`, IQR as the band; `cost_consensus` = share within ±25%.
   - **Labor** (`gsar_labor_hrs`): same trimmed-median ± IQR; `labor_consensus`.
   - **Correction** (`paws_comment_trail`): `correction_theme` = archetype top terms;
     `correction_example` = representative comment in the dominant archetype.
   - **Parts** (`part_numbers`, 16% — bonus): `parts_modal_set` + its support %.
   - **Recall**: `recall_share` = fraction flagged `Y`.
4. **Confidence & status.** `gt_score` = mean(text cohesion, labor consensus, cost consensus);
   `confidence = gt_score × support_weight`; `status ∈ {GOLD, SILVER, HOTSPOT, SPARSE}`.
5. **Discrepancy flags (per claim).** `cost_ratio = cost / signature.cost_med`; flag `|ratio−1| >
   0.8` (i.e. <0.55× or >1.8×) → exclude/relabel for training, surface as overpayment candidates.
6. **Regional / dealer normalization (serve time).** `cost_expected(state) = cost_med ×
   state_cost_index[state]`; dealer audit within (state, signature) flags deviators ≥ 1.8×. *(Now
   productized as the `RegionDelegatorAgent` + `RegionIndexTool` in `src/`.)*

> The LLM stage (`src/agents/`, `10_golden_solutions.py`) sits on top of this: it only writes the
> **prose** (unified diagnosis + correction) and the **split decision**; all numbers stay
> deterministic and come from the consensus math above.

---

## 5. Measured side-studies (the savings signal)

**Regional cost variation.** Across 50 US states (GSAR, real causal parts, implausible-value guards), the
**median claim cost spans 1.67×** (AK/ID/CA high, SD/WY/IA low). High-cost states skew
remote/high-labor-rate; MI is high *material* but low *labor* — a different cost mix.
→ `out/region_cost_by_signature.csv`, `out/region_overall.csv`.

**Dealer outliers (`06_dealer_outliers.py`) — strongest actionable signal.** Across **12,450**
(state × repair-signature) cells, within-state dealer median cost spreads **2.02× (median), 6.85×
(p90)** for the *same* repair. Concrete: a MI dealer bills **$2,128 vs the $288 state consensus
(7.4×)** on Escape part `7820125`. (Extreme 100×+ ratios are inflated by data-error low ends —
deductible-only/miscoded claims — so the 2× median is the headline.)
→ `out/dealer_spread_by_state_signature.csv`, `out/dealer_outliers_examples.csv`, `out/fig_dealer_spread.png`.

**Bottom-up avoidable overspend (`12_group_savings.py`).** Over 2,172 single-repair canonical
repairs / 127,465 claims with $406.3M historical spend (rescaled), **$75.7M (18.6%) sits above consensus**
($594/claim) — the addressable cleansing/recovery opportunity. → `out/group_savings.csv`,
`out/group_savings_headline.txt`.

**Cost-tier sub-split — real win.** Of **3,234** fuzzy groups (n≥20, cost consensus <0.5), a simple
2-tier median split lifts cost consensus **0.295 → 0.473** and turns **475 (14.7%)** into clean
canonical repairs with no new data. → `out/cost_tier_split_summary.csv`.

**Recall vertical (`09_recall_vertical.py`) — honest result.** 212 recall canonical repairs (n≥30)
from the broad GSAR recall population. Recalls are more *dealer*-uniform (1.50× vs 2.02× general)
but **not** more cost-uniform at the causal-part grain (within-signature cost CV 0.787 recall vs
0.622 non-recall) — because a recall causal-part code bundles the flat fix with
rentals/diagnostics/escalations. **Implication:** clean recall canonical repairs need
**campaign-code-level grouping + cost-tiering**, not causal-part alone. → `out/fig_recall_uniformity.png`.

**Labor-op codes — documented, deferred.** The standardized labor unit (Ford std times) is the
ideal labor ground truth, but the PAWS labor-op view (`sxpaa104`) is empty; real op codes live in
OWS (`sowsc09w_wrty_clm_lbr_op_vw`, 226M rows) keyed by OWS internals, needing an OWS→VIN bridge —
a follow-on build. The directly-joinable `apply.repair_cost_labor_view` overlaps only 43%
of our spine and gives no better consensus. **Conclusion:** the lift comes from the text-archetype
split (G3) + cost-tiering, not from swapping labor source.

---

## 6. Emitted Canonical Repair Library (current state)

- **Full grouping persisted** (`data/claim_signatures.parquet`): **71,957 repair signatures over
  544,121 groupable claims** (binding constraint: GSAR `veh_line_desc` at 68%).
- **Canonical Repair Library** (`out/canonical_repairs.csv`): **6,654 canonical repairs** (n≥10)
  covering 416,195 claims — **3,623 GOLD**, 3,026 SILVER, 5 HOTSPOT.
- **LLM golden solutions** (`out/golden_solutions.csv`, via `10_golden_solutions.py` + Vertex
  Gemini): **5,070 golden solutions** over the 3,623 GOLD signatures (2,176 single-repair, 2,894
  split into two; 5,069/5,070 cost-backed ≈100%).
- Example GOLD record: *P552N F-150 [15-20], causal part 6148, "long block" archetype* — 961
  claims, **cost $8,923 (median, `gsar_tot_cost_gross`), labor 16.8 hr**, canonical correction text.

### Honest limitations (v1)
- Only 58% of claims are groupable today (need GSAR vehicle line at 68%); a PAWS-side vehicle
  fallback would raise coverage.
- `gt_score` counts text cohesion as 1.0 by construction (we grouped on it); the stricter quality
  column is `solution_consensus` (cost+labor only). A few GOLD rows are language-only archetypes
  (e.g. Spanish comments) or null-cost — flagged for v1 cleanup.
- Cross-system join is VIN+date only → 32% of claims have no GSAR cost match (mostly genuinely
  absent, extended-service repairs).

---

## 7. Where things live & next moves

- **Unified dataset build:** the upstream BigQuery join that produces `repair_profile.parquet` from
  Ford's warranty warehouse (PAWS + GSAR + OWS) is proprietary and omitted from this academic release.
- **Grouping + CRL:** this `grouping/` dir — numbered scripts `01`–`12`, artifacts in `out/`.
- **Agentic distillation + serving:** `src/agents/` (the hand-off chain, region + savings agents) + `src/tools/`.

**Best next moves:** (1) extract the labor operation code once an OWS→VIN bridge exists, for a true
regional price index; (2) join FSA campaign codes for recall-type analysis; (3) k-means on log-cost
(auto-detect modes) instead of a median tier split; (4) temporal cost indexing so 2022 vs 2024
isn't read as discrepancy; (5) validation on held-out claims (label-noise reduction after
discrepancy exclusion).
