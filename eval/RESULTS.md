# ACCORD - re-derived results

Every number below is computed by the scripts in `eval/` from the committed Canonical Repair Library artifacts. No external calls, no claim-level data. Re-running a script replaces its section in place.

## a1 - Post-hoc critic verdicts

_Data root: `grouping/out_v2_public`_

Running the shipped `SolutionCriticAgent` over all 4,696 emitted Canonical Repairs gives
**4,453 PASS (94.83%) / 238 FLAG (5.07%) /
5 FAIL (0.11%)**. Reasons recorded (a solution may carry several):

- `consensus below 0.3` - 231
- `support below 5 claims` - 13
- `no cost consensus` - 5

Applied to the released library the critic demotes **0** further repairs
(5 were already marked unusable): the emitted library is
already consistent with the gates it was never actually run through.

**What those 0 demotions are, exactly.** 0 emitted repairs carry
`usable = True` while having **no cost median at all** (signatures
). Their cost tier received zero claims, so `StatsTool.band`
returned NaN rather than None, and `usable` is `bool(cost_band["median"])` - where `bool(nan)` is
truthy. So the assembly flag admits them.

The critic catches all 0, but not through the gate one would expect: `if not med` does
not fire either, because `not nan` is also False. They are caught by the *next* gate, cost-band
ordering, since `nan <= nan <= nan` evaluates False. The QA layer therefore recovers a defect the
assembly flag misses - which is a point in the critic's favour - but it does so incidentally, via NaN
comparison semantics rather than by an explicit missing-cost test. A tier that produced a genuinely
disordered band and a tier that produced no band at all are reported under the same reason string.

The practical consequence for the paper: the released coverage figure counts 0 repairs
with no cost band to stand on, and running the critic removes them.

**Scope, stated precisely.** This is the critic applied *post hoc to the finished library*, not the
critic *in loop*: it measures the structural quality of what was emitted. It cannot reproduce the one
bounded revision round, because that round would change the LLM text and the batch's
`llm_cache.jsonl` no longer exists. Gates imported from `src/config.py`
(min_support=5, min_consensus=0.3); verdict logic is the
shipped `SolutionCriticAgent`, not a reimplementation.

Artifacts: `eval/out/a1_critic_verdicts.csv`, `eval/out/a1_critic_summary.json`.

## a3 - Confidence decomposition

_Data root: `grouping/out_v2_public`_

**Verification.** Recomputing kappa = consensus x min(1, log10(n+1)/log10(31)) from the emitted
`consensus` and `support` columns reproduces the emitted `confidence` column on
**4,694 of the 4,694 rows where both quantities are non-null**
(tolerance 5e-4, half a unit in stats_tool's last rounded place) - the released library exactly matches
`stats_tool.py:54-62`. The remaining 2 row(s) carry a null consensus *and* a null
confidence (signatures 18730, 18737), so they are excluded
rather than counted as agreements: coercing both sides to zero would make them match vacuously.

**Correction to the saturation claim.** The support weight reaches 1.0 only at n >= 30, and
just **1,194 of 4,696 emitted repairs (25.4%)** clear that bar. Median support is
**15** claims, the lower quartile is 10, and the
mean support weight is **0.805** (minimum 0.202).
So the support term is not inert - it actively discounts roughly three quarters of the library, and
kappa is *not* interchangeable with the consensus fraction outside the high-support tail.

- single repair: median support 18, 31.0% saturated, mean support weight 0.863
- split tier: median support 10, 18.8% saturated, mean support weight 0.736

The mechanism is the split: GOLD requires n >= 10 at the *signature* level, but an honored two-way
split partitions those claims across two tiers, so per-tier support is roughly halved.

Artifacts: `eval/out/a3_confidence_rows.csv`, `eval/out/a3_confidence_by_split_class.csv`,
`eval/out/a3_confidence_summary.json`.

## a4 - Library audit

_Data root: `grouping/out_v2_public`_

Of 4,696 emitted Canonical Repairs, **0** has empty diagnosis or correction text,
**5** is unusable for lack of a cost median, and **5** received zero
claims in its cost tier. The empty-text row and the unusable row are *different* rows, so the
degenerate set is smaller than a single statistic implies but touches two distinct failure modes.

**Split structure.** 1,073 signatures were emitted as two repairs, with tier pairs:

- `high+low`: 1,073 signatures

The **0** `low+mid` signatures (0.0% of splits) are the
fingerprint of a truncated three-repair decision: the batch prompt behind the released run permitted
up to three repairs, `SolutionAssemblyAgent.assemble()` keeps `repairs[:2]`, so the analyst's
highest-cost scope was dropped and the remaining pair recorded as `n_repairs_detected = 2`. The
numbers on those rows are still a clean median-cut band - `cost_tier_partition` routes anything not
`"low"` to the upper half - but the prose attached to the upper band is the analyst's *middle*-scope
text. The agent-side prompt (`src/prompts/split_and_diagnose.txt`) caps at two repairs, so a re-run
through the agent chain does not reproduce this.

**Reconciliation.** 1,073 split signatures but only 1,072 have a computable tier
ratio; the difference is the signature whose low tier received no claims, leaving one tier median
undefined.

**Stale tier labels.** 319 single-repair rows still carry a `low`/`high` tier label.
Their statistics are unaffected - `cost_tier_partition` returns the whole group whenever
`n_repairs <= 1` - so these are full-group numbers under a stale label, not mis-sliced data.

**The two artifacts do not share a cost basis.** Across 2,457 single-repair signatures
present in both files, `canonical_repairs.cost_med` equals `golden_solutions.cost_med` in
**1 case(s)** (0.04%); the CRL value is
**2.44x** the golden-solution value at the median (p10
1.31x, p90 5.52x). The agent chain uses
`gsar_tot_cost_gross` (`config.COST_COL`), which `11_reassemble_from_cache.py` documents as the
corrected basis; the CRL predates that correction. **Consequence: any cost statistic quoted for the
emitted library must come from `golden_solutions.csv`.** Mixing the two silently inflates costs -
which is exactly the error that would corrupt a dispersion analysis built on CRL quartiles.

Artifacts: `eval/out/a4_library_audit.csv`, `eval/out/a4_degenerate_rows.csv`,
`eval/out/a4_tier_structure.json`.

## a5 - Savings concentration

_Data root: `grouping/out_v2_public`_

Across 2,546 single-repair canonical groups and 159,541 claims totalling
$482.0M, measured spend above consensus is **$96.8M**
(20.1% of spend, $607 per claim).

**Concentration.**

- top 1% of groups (26) carry **32.8%** of the overspend and 34.1% of claims
- top 5% of groups (128) carry **55.3%** of the overspend and 54.2% of claims
- top 10% of groups (255) carry **66.6%** of the overspend and 63.3% of claims
- top 25% of groups (637) carry **82.0%** of the overspend and 77.2% of claims
- top 50% of groups (1,273) carry **92.9%** of the overspend and 87.6% of claims

The Gini coefficient over group-level overspend is **0.746** and the largest single
group contributes $5.55M, so the total is far from evenly spread.

**But the concentration is group size, not severity.** Overspend is in fact *less* concentrated than
spend itself (Gini 0.746 vs **0.775** for total spend), and only marginally
more concentrated than claim volume (0.708). A group's overspend is therefore close to a
fixed fraction of what flows through it. The top 1%
of groups carry 32.8% of the overspend while also carrying
34.1% of the claims. Group size predicts total overspend strongly
(rank correlation 0.736) but per-claim severity barely at all (0.119),
and each group's overspend *as a share of its own spend* is essentially size-independent
(-0.005, median 23.4%). The largest groups are only
modestly worse per claim (median $658 vs $519 library-wide).

The honest reading: the ranking is a **volume-weighted work queue** - chase the biggest groups first
because that is where the dollars are - not evidence that a minority of repair types is pathologically
overspent. Per-claim severity is broadly distributed (Gini 0.400).

**Breadth sanity check.** The median group has 50.0% of its
claims above consensus, exactly what a *median* baseline mechanically implies. The total therefore
measures right-tail mass, and its magnitude depends on tail shape rather than on the count of claims
above the line - the framing the draft already adopts.

**Not computed.** Re-basing to an upper-quartile consensus requires per-claim costs and is deferred;
it cannot be derived from the group-level artifact.

Dollar amounts follow the data root. Artifacts: `eval/out/a5_savings_concentration.csv`,
`eval/out/a5_savings_top25_groups.csv`, `eval/out/a5_savings_summary.json`.

## a6 - Regional decomposition

_Data root: `grouping/out_v2_public`_

Across 50 states the pooled all-repairs median cost spans **1.67x**
(AK to SD), and the claims-weighted national anchor gives a cost index
spanning 0.76-1.26.

Decomposing that spread by component:

| Component | Spread (max/min) | Extremes | CV |
|---|---|---|---|
| Total claim cost | 1.67x | AK / SD | 0.105 |
| Labor cost | 3.05x | AK / MI | 0.148 |
| Material cost | 2.75x | MI / WA | 0.195 |
| Labor hours | 2.40x | AK / MI | 0.125 |

Note both components spread **more** than the total (3.05x and
2.75x vs 1.67x). That is only possible if they partially
offset, and they do: labor and material medians correlate at **r = -0.280** across states.
Total cost tracks labor (r = 0.567) more closely than material (r = 0.437), but
neither component alone explains the spread.

**This weakens the repair-independent multiplier rather than supporting it.** The offsetting shows
states differ not just in price *level* but in cost *composition*: labor's share of labor-plus-material
runs from 0.285 (MI) to 0.716 (WA), a
2.51x spread (1.31x after trimming the two
extreme states, so this is not one outlier). A single pooled index applies one multiplier to every
repair in a state. If states vary in labor-vs-material mix, then a labor-heavy repair and a
material-heavy repair in the same state should scale by *different* factors, and the pooled index will
misprice both - overstating one and understating the other.

We flag this as evidence **against** treating the index as repair-independent, and note the assumption
is only settled by measuring within-signature state variation, which requires the claim-level table
(`region_cost_by_signature.csv` covers just 33 signatures as extremes) and remains future work.

Artifacts: `eval/out/a6_region_components.csv`, `eval/out/a6_region_index.csv`,
`eval/out/a6_region_summary.json`.

## a2 - Separation guard

_Data root: `grouping/out_v2_public`_

The analyst proposed a two-way split on **1,361** of the 3,623 GOLD signatures;
the guard honored **1,073** and collapsed **288**, an override rate of
**21.2%**. Of those 1,361 proposals, 1,360 have a
computable ratio (one honored split has a tier that received no claims), and every distributional
statement below is over that 1,360.

**Recovering the counterfactual.** Because the split is realized as a median cut, the ratio the guard
tests is approximately the group's Q3/Q1 - a quantity emitted for every signature. Checking that
construction on the 288 collapsed signatures, where the realized ratio is
independently recorded in the rationale text: median realized 1.386
vs median Q3/Q1 1.359, with the per-signature ratio of the two centred at
0.9858 (IQR [0.970, 0.995]).
The construction is a tight, slightly conservative approximation - not an exact identity, since tier
quantiles are interpolated within the slice.

Two honest caveats about that check. It is only *available* where the guard collapsed, i.e. on
realized ratios in [1.0, 1.5), yet it is *applied* to a stratum with median
~1.67. Binning within the observable range, the factor is flat over
[1.2, 1.5) - [1.2, 1.3): 0.981, [1.3, 1.4): 0.987, [1.4, 1.5): 0.985 -
which makes extrapolation past 1.5 reasonable, though still untested. The lowest bin
([1.0, 1.2), n=15) is unstable, as expected where the
two tier medians nearly coincide.

The bias also has a known direction: Q3/Q1 **understates** the realized ratio by
~1.4%. Both consequences run *against* the claim made here -
68.3% is a **floor** on how many un-proposed signatures
clear tau, and the AUC below is an **over-estimate** of the ratio's discriminating power.

**The guard's threshold is not calibrated against within-repair dispersion.** Applying that
counterfactual to the 2,258 signatures the analyst never proposed splitting,
**68.3% of them already clear tau=1.5**
(median ratio 1.668), versus
78.8% of the 1,360 proposed ones
(median 1.781). As a classifier for what the analyst proposed, the
ratio has **AUC 0.572** - distinguishable from chance but negligible in magnitude, and
stable across support strata (n∈[10,20): 0.571 · n∈[20,50): 0.584 · n∈[50,inf): 0.580), so it is not an artifact of small-n noise.

Restricting the proposed arm to the 1,360 signatures that are *not* truncated
three-repair decisions (a4) pushes it further down, to **0.572** - indistinguishable from
chance. The residual signal in the pooled figure comes from the 388 truncated proposals, which span
three repair scopes and so genuinely separate more. On clean two-repair proposals the cost ratio
carries essentially no information about what the analyst decided.

Note that scoring honored splits alone would give 0.677, but that number is
meaningless: honored splits are *defined* by ratio >= tau, so it measures the guard's own threshold
rather than the analyst's judgement. It is reported only to pre-empt the mistake.

Clearing tau therefore **bounds** the cost separation between tiers; it does not **establish** that
two distinct repairs exist. The discriminating work is done by the analyst's reading of the claim
text, with the guard supplying a largely independent cost-side veto.

**Threshold sensitivity.** Share of proposed splits honored: 1.3 → 95.9% · 1.5 → 78.8% · 1.75 → 53.4% · 2 → 35.0% · 2.5 → 15.1%. The ratio distribution is
continuous through 1.5 with no gap, so tau is a tunable conservatism knob rather than a discovered
boundary. Sampling noise matters at this threshold too: **36.2%** of honored splits lie
within one standard error of tau (an upper bound - it treats the two tier medians as independent),
and 55.6% sit below 2.0x.

**Mechanism.** Within-repair dealer dispersion supplies the base rate: [5,10) dealers: median 1.76x, 67.2% ≥1.5x · [10,20) dealers: median 2.12x, 83.9% ≥1.5x · [20,inf) dealers: median 2.71x, 94.1% ≥1.5x. Note this is a
max/min statistic across dealers, so it grows with dealer count by construction and there are no cells
below 5 dealers - it explains *why* ordinary dispersion clears 1.5x, but it is not itself a calibrated
floor.

**Scope.** All of the above covers GOLD signatures; the 3,026 SILVER signatures never reach the
agent chain and are unexamined.

Artifacts: `eval/out/a2_tau_sweep.csv`, `eval/out/a2_signature_classes.csv`,
`eval/out/a2_discrimination_by_support.csv`, `eval/out/a2_dealer_spread_by_dealer_count.csv`,
`eval/out/a2_guard_summary.json`.

## a7 - v1 to v2 rerun churn

_Data root: `grouping/out_v2`_

Re-running the full agent chain over the 3,623 GOLD signatures (manifest:
`grouping/out_v2/rerun_manifest.json`) reproduces the v1 split decision on
**69.0%** of signatures. The 388 v1 signatures that were truncated
three-repair decisions resolved in v2 as {np.int64(2): 223, np.int64(1): 165}. Where both versions agree a signature is
a single repair, the consensus cost median moves by a relative
0.00% at the median (p90 0.00%) - the
deterministic numeric path is stable; churn is confined to the LLM's structural decision.

v2 profile: 4,696 solutions (v1: 5,070); critic verdicts {'PASS': 4453, 'FLAG': 238, 'FAIL': 5}; the bounded
revision fired on **0** signatures; 0 empty-text shell(s);
0 degraded LLM call(s); `mid` tiers: 0 (2-repair prompt
cap holds). Gate: split agreement **FAILS** the 85% bar.

Artifacts: `eval/out/a7_churn.json`, `eval/out/a7_churn_per_signature.csv`.

## b1 - same-prompt stability (panel)

_Data root: `grouping/out_v2`_

Two fresh replicates of the unmodified chain (same prompt by SHA-256, temperature 0) over the frozen
400-signature panel (`eval/panel_400.csv`; strata reweighted to population shares). Pairwise emitted
split-decision agreement (a7's metric), population-weighted:
v2 x rep1 **98.1%** (SE 0.7pp); v2 x rep2 **98.3%** (SE 0.7pp); rep1 x rep2 **99.2%** (SE 0.5pp);
mean **98.5%** - versus **69.0%** across the prompt revision (a7). Per-stratum agreement
(mean over pairs): single 99.2%, honored 96.7%, collapsed 100.0%.

Where the structure matches, tier statistics are identical to relative delta
0.00e+00 max across all pairs (the deterministic numeric path, verified end to end);
critic verdicts agree on v2 x rep1 100.0%; v2 x rep2 100.0%; rep1 x rep2 100.0%.
Diagnosis prose exact-match where structure matches: v2 x rep1 97.5%; v2 x rep2 97.5%; rep1 x rep2 96.7%.

Jittered parse-failure retries: rep1 17/400, rep2
17/400. rep1 x rep2 flip rate on jittered signatures
17.6% vs
0.0% on
first-try signatures.

Artifacts: `eval/out/b1_stability.json`, `eval/out/b1_stability_per_signature.csv`.

## b2 - LLM-numbers ablation (panel)

_Data root: `grouping/out_v2`_

The production analyst prompt, stripped of its injected StatsTool numbers and asked to estimate them
instead (grading: shipped StatsTool over all claims of the structure the model itself chose; panel of
400, temperature 0). With the production evidence (8 cost-prefixed claim comments), the group-level
cost median comes back with median absolute relative error
**2.4%** (p90
10.4%; within 10% on
89.0% of signatures); per-tier cost
medians err by 11.9% at the median, IQR width by
63.7%, and labor-hours medians - for which the
sample carries NO per-claim evidence - by 33.6%
(p90 93.9%). 0.0%
of estimated bands violate q25<=median<=q75 (the critic's hard gate). Five times the evidence
(sample_size=40) moves the group-median error to
0.0% (within 10%:
99.2%), IQR width to
32.9%, labor to
26.1%.

Schema-invalid replies: s8 0/400, s40 0/400.
Invariant (run-time truth == released v2 golden full-group median on single/collapsed signatures):
max abs delta s8 0.0, s40
0.0 over
271/271
checked. Split-proposal rate under this variant prompt: s8 55.5%, s40
40.5% (v2 analyst on the same panel:
37.5%; context only - the prompt differs, so this is not a stability
measurement).

Artifacts: `eval/out/b2_llm_numbers.json`, `eval/out/b2_llm_numbers_per_tier.csv`.

## b3 - single-call baseline (panel)

_Data root: `grouping/out_v2`_

One call per signature replaces the whole chain (40 cost-prefixed comments as evidence, no
statistics injected; panel of 400, temperature 0). Validity: 400/400 parsed,
0 schema-invalid, 0 permanent failure(s). The baseline
proposes a split on 45.8% of signatures and agrees with the
chain's analyst at the proposal level on **69.0%** (population-weighted; per stratum
{'collapsed': 0.74, 'honored': 0.68, 'single': 0.688}). Of its own 183 proposed splits, the shipped separation
guard would collapse **30.6%** (the chain's own operating point:
21.2%) - splits the single call would have shipped as two repairs with no cost-side veto.

Its own emitted records: **0.0%** of repair rows fail the critic's hard
cost gates outright (0.0% band-ordering violations,
0.0% missing cost; chain in-loop FAIL rate: 0.1%);
0.0% missing diagnosis/correction text. Numbers: group cost median
off by 0.0% at the median (p90
2.1%), labor by
27.2%. Self-reported confidence: median
0.9 (IQR 0.8-0.9) - a score with no
deterministic decomposition behind it, unlike kappa's consensus x support construction.

Artifacts: `eval/out/b3_single_call.json`, `eval/out/b3_single_call_per_signature.csv`.

## H3 GOLD gate cost-basis robustness

_Data root: `claim-level: data/claim_signatures.parquet + paws_discovery/repair_profile.parquet; shipped GOLD set: grouping/out_v2/canonical_repairs.csv`_

The shipped GOLD gate (grouping/08_build_ground_truth.py, cost basis `det_approved_amt`) reproduces
exactly from claim-level data: 3,623 GOLD signatures, an identical ID set to
`grouping/out_v2/canonical_repairs.csv` (0 missing, 0 extra). Rerunning the identical gate with the
cost basis switched to `gsar_tot_cost_gross` (the shipped pipeline's COST_COL; labor, validity
rules, n>=10 floor and 0.6 threshold unchanged) yields 3,578 GOLD signatures. Of the shipped 3,623,
3,002 survive (82.9%) and 621 drop out (17.1%); 576 signatures newly enter. Jaccard overlap between
the two GOLD sets is 0.715. One-component passes (NaN cost or NaN labor consensus, so the skipna
mean leans on the remaining components): 107 of 3,623 on the det basis (95 cost-NaN, 12 labor-NaN, 0
both) and 16 of 3,578 on the gsar basis (4 cost-NaN, 12 labor-NaN, 0 both). Full ID sets:
`eval/out/h3_gold_gate.json`.

## H addenda: artifact backing for prose claims

_Data root: `eval/out/a7_churn_per_signature.csv + grouping/out_v2 + claim-level parquet (data/, paws_discovery/)`_

(a) Truncation-excluded rerun agreement. Over all 3,623 rerun signatures, split agreement
(n_repairs_v1 == n_repairs_v2) is 2,499/3,623 (69.0%). Excluding the 388 signatures with
tier_pair_v1 == "low+mid" (v1 analyst proposed 3 repairs, library emitted 2, so the v1 decision was
truncated), of which 223 agreed, agreement is (2,499-223)/(3,623-388) = 2,276/3,235 = 70.4%
(0.7036).

(b) The 2,550 to 2,546 derivation. golden_solutions.csv holds 2,550 single-repair GOLD signatures;
group_savings.csv holds 2,546 rows. The four dropped signature_ids are 5799, 10355, 14626, 32910
(20, 14, 18, 11 member claims respectively). None of their claims carry a valid gsar_tot_cost_gross,
so grouping/12_group_savings.py has no canonical cost for them (cost_med is NaN) and no claim
survives its cost filter; their GOLD status came from the det-basis gate via the skipna
one-component path (text + labor consensus only). Details: `eval/out/h_addenda.json`.

## H stats bundle

_Data root: `grouping/out_v2`_

Uncertainty and significance for the headline eval numbers, plus a labor analogue of the a2
guard AUC and a TSB/SSM identifier check. Bootstrap and permutation seeds are 42 throughout.

**1. Labor-gap AUC.** Mirroring a2's construction with labor in place of cost (honored splits:
realized high/low tier labor-median ratio from `golden_solutions.csv`; never-proposed singles:
labor Q3/Q1 from `canonical_repairs.csv`; collapsed proposals excluded because no labor tier
record survives a collapse), the labor gap separates proposed from never-proposed signatures
with **AUC 0.469** (95% CI
[0.448, 0.490],
2000 resamples; n = 1,067 vs
2,253). Median ratios are
1.58 (honored) vs
1.58 (never proposed). Unlike cost, labor is
never thresholded by the guard, so this comparison carries no truncation artifact. The interval
sits marginally below 0.5: the labor gap on proposed splits is no larger than ordinary
within-repair labor spread, so the labor axis provides no positive discrimination of what the
analyst proposed. Caveat: the two arms sit on different labor bases (tier `labor_med_hrs` vs
`gsar_labor_hrs` quartiles).

**2. Cost AUC CI.** The a2 headline cost AUC of 0.572 gets a
bootstrap 95% CI of **[0.552,
0.590]** (2000 resamples,
n = 1,360 proposed vs 2,258 never
proposed). Above chance, but the entire interval stays in the negligible-discrimination band.

**3. SEs and tests.** (a) v1-to-v2 churn agreement is 2,499/3,623
= 69.0% with binomial SE 0.77pp. (b) B1 same-prompt
weighted agreement is 98.52% with SE
0.72pp (largest of the three per-pair SEs in
`b1_stability.json`; the pairs share runs). The stability-minus-churn difference is
**29.5pp** with combined SE
1.1pp
(z = 28). (c) Fisher exact on the flips-vs-jitter table
(3/17 jittered signatures flipped vs 0/383 first-try) gives two-sided
**p = 6.4e-05**: flips concentrate in jittered calls
far beyond chance, consistent with jitter attribution of the residual disagreement (it does not
prove every flip was caused by jitter). (d) B3 weighted rates: over-proposal
**45.5% +/- 2.3pp**,
guard-veto among proposed splits
**23.5% +/- 2.1pp**
(stratified SEs, sqrt of sum over strata of w^2 p(1-p)/n; the collapsed stratum's veto rate is
37/37 and contributes zero variance, so the second SE is slightly understated).

**4. Kappa correlates (H5).** Per-signature kappa is the `confidence` column of the v2 library;
for split signatures the **minimum** across the two emitted tiers is used throughout. (a) Kappa
vs B1 flips: 8 of 400 panel signatures flipped in at least one of the
three pairwise run comparisons (3 in rep1 x rep2 alone).
Point-biserial r = -0.052 (permutation p = 0.30); kappa of
unflipped signatures separates flipped ones with AUC 0.620.
Median kappa 0.368 (flipped) vs 0.457
(unflipped); the flipped signatures sit in panel kappa quartiles
Q1, Q1, Q1, Q1, Q2, Q2, Q3, Q4. With
8 flips the test is underpowered; the direction is suggestive, not
established. (b) Kappa vs cost-median stability: per-signature bootstrap of the claim-cost
median (500 resamples), relative CI width = (hi - lo)/median. Spearman rho =
**-0.788** (permutation p = 5e-05,
n = 400); within the single stratum alone rho =
-0.831 (p = 5e-05,
n = 250). Both p-values sit at the floor of a 20,000-permutation test.
Higher-kappa signatures do have more stable cost medians.

**5. TSB/SSM identifier distinctness (H2-lite).** Comment-trail inspection shows two usable
identifier families: TSB ids in the form yy-nnnn (also written with a space, or introduced by
the word bulletin) and SSM ids of 4 to 6 digits; boilerplate lines like "TSB or SSM: NONE"
carry no id and are not matched. 61.7% of
scanned claims mention TSB/SSM/bulletin and 35.1%
carry a parseable identifier. Tiers are the median cut of each signature's claim costs, matching
the shipped construction. Coverage is well above the 5% viability floor, so the analysis is
viable: 52.9% of honored, 58.0%
of collapsed and 49.8% of v1/v2-flipped signatures have at
least one identifier. Among signatures where BOTH tiers carry identifiers, the two tiers'
identifier sets are fully disjoint in 12.7% of honored
(n = 393), 13.4% of collapsed
(n = 127) and 12.1% of flipped
(n = 364) signatures. Counting one-sided hits as disjoint (an empty set
is disjoint by definition) the rates are 39.6%,
34.1% and 42.9%. Similar
disjointness across honored and collapsed populations means identifier separation as computed
here does not obviously validate the honored splits over the collapsed ones; treat it as a
feasibility result, not a verdict.

Artifacts: `eval/out/h_stats_bundle.json`.

## b2 - LLM-numbers ablation (panel) - claude-sonnet-5

_Data root: `grouping/out_v2`_

The same ablation under **claude-sonnet-5** through the LLMGateway gateway, s8 arm
(sample_size=8), same frozen panel, same prompt, temperature 0, graded by the same
shipped StatsTool. Group-level cost median: **4.2%** median
absolute relative error (p90 12.6%). Per-tier cost medians, the
number the library ships: **5.0%** (p90
26.5%, within 10% on
72.5%). Bands: 25.7%
(signed bias 12.2%). Labor:
30.6% (signed bias
-24.9%). Band-order violations of the critic's hard gate:
**0.0%**. Schema-invalid replies
0/400; jittered parse retries 31.
Split-proposal rate under this variant prompt 14.5% (v2 analyst on the
same panel 37.5%; context only, the prompt differs).

Artifacts: `eval/out/b2_llm_numbers_claude-sonnet-5.json`,
`eval/out/b2_llm_numbers_per_tier_claude-sonnet-5.csv`.

## b2 - LLM-numbers ablation (panel) - gpt-5.4-2026-03-05

_Data root: `grouping/out_v2`_

The same ablation under **gpt-5.4-2026-03-05** through the LLMGateway gateway, s8 arm
(sample_size=8), same frozen panel, same prompt, temperature 0, graded by the same
shipped StatsTool. Group-level cost median: **2.2%** median
absolute relative error (p90 10.8%). Per-tier cost medians, the
number the library ships: **15.0%** (p90
168.8%, within 10% on
41.7%). Bands: 57.5%
(signed bias 47.5%). Labor:
23.7% (signed bias
-7.7%). Band-order violations of the critic's hard gate:
**0.0%**. Schema-invalid replies
0/400; jittered parse retries 1.
Split-proposal rate under this variant prompt 55.2% (v2 analyst on the
same panel 37.5%; context only, the prompt differs).

Artifacts: `eval/out/b2_llm_numbers_gpt-5.4-2026-03-05.json`,
`eval/out/b2_llm_numbers_per_tier_gpt-5.4-2026-03-05.csv`.

## b3 - single-call baseline (panel) - gpt-5.4-2026-03-05

_Data root: `grouping/out_v2`_

Model: **gpt-5.4-2026-03-05**.
One call per signature replaces the whole chain (40 cost-prefixed comments as evidence, no
statistics injected; panel of 400, temperature 0). Validity: 400/400 parsed,
0 schema-invalid, 0 permanent failure(s). The baseline
proposes a split on 48.2% of signatures and agrees with the
chain's analyst at the proposal level on **60.2%** (population-weighted; per stratum
{'collapsed': 0.54, 'honored': 0.64, 'single': 0.592}). Of its own 193 proposed splits, the shipped separation
guard would collapse **26.9%** (the chain's own operating point:
21.2%) - splits the single call would have shipped as two repairs with no cost-side veto.

Its own emitted records: **0.0%** of repair rows fail the critic's hard
cost gates outright (0.0% band-ordering violations,
0.0% missing cost; chain in-loop FAIL rate: 0.1%);
0.0% missing diagnosis/correction text. Numbers: group cost median
off by 0.6% at the median (p90
6.9%), labor by
20.0%. Self-reported confidence: median
0.84 (IQR 0.78-0.89) - a score with no
deterministic decomposition behind it, unlike kappa's consensus x support construction.

Artifacts: `eval/out/b3_single_call_gpt-5.4-2026-03-05.json`,
`eval/out/b3_single_call_per_signature_gpt-5.4-2026-03-05.csv`. Population-reweighted over-proposal
and guard-veto rates with SEs (the numbers the paper quotes) come from `eval/b3_reweight.py`.

## b2 - LLM-numbers ablation (panel) - deepseekv4-flash

_Data root: `grouping/out_v2`_

The same ablation under **deepseekv4-flash** through the LLMGateway gateway, s8 arm
(sample_size=8), same frozen panel, same prompt, temperature 0, graded by the same
shipped StatsTool. Group-level cost median: **4.1%** median
absolute relative error (p90 14.1%). Per-tier cost medians, the
number the library ships: **7.6%** (p90
93.0%, within 10% on
56.9%). Bands: 78.8%
(signed bias 58.5%). Labor:
34.3% (signed bias
-28.6%). Band-order violations of the critic's hard gate:
**0.2%**. Schema-invalid replies
0/400; jittered parse retries 10.
Split-proposal rate under this variant prompt 46.2% (v2 analyst on the
same panel 37.5%; context only, the prompt differs).

Artifacts: `eval/out/b2_llm_numbers_deepseekv4-flash.json`,
`eval/out/b2_llm_numbers_per_tier_deepseekv4-flash.csv`.

## b3 - single-call baseline (panel) - deepseekv4-flash

_Data root: `grouping/out_v2`_

Model: **deepseekv4-flash**.
One call per signature replaces the whole chain (40 cost-prefixed comments as evidence, no
statistics injected; panel of 400, temperature 0). Validity: 400/400 parsed,
1 schema-invalid, 0 permanent failure(s). The baseline
proposes a split on 8.8% of signatures and agrees with the
chain's analyst at the proposal level on **64.8%** (population-weighted; per stratum
{'collapsed': 0.06, 'honored': 0.1818, 'single': 0.944}). Of its own 35 proposed splits, the shipped separation
guard would collapse **20.0%** (the chain's own operating point:
21.2%) - splits the single call would have shipped as two repairs with no cost-side veto.

Its own emitted records: **0.0%** of repair rows fail the critic's hard
cost gates outright (0.0% band-ordering violations,
0.0% missing cost; chain in-loop FAIL rate: 0.1%);
0.9% missing diagnosis/correction text. Numbers: group cost median
off by 3.4% at the median (p90
12.5%), labor by
25.0%. Self-reported confidence: median
0.92 (IQR 0.85-0.95) - a score with no
deterministic decomposition behind it, unlike kappa's consensus x support construction.

Artifacts: `eval/out/b3_single_call_deepseekv4-flash.json`,
`eval/out/b3_single_call_per_signature_deepseekv4-flash.csv`. Population-reweighted over-proposal
and guard-veto rates with SEs (the numbers the paper quotes) come from `eval/b3_reweight.py`.

## b3 - single-call baseline (panel) - claude-sonnet-5

_Data root: `grouping/out_v2`_

Model: **claude-sonnet-5**.
One call per signature replaces the whole chain (40 cost-prefixed comments as evidence, no
statistics injected; panel of 400, temperature 0). Validity: 398/400 parsed,
4 schema-invalid, 2 permanent failure(s). The baseline
proposes a split on 6.3% of signatures and agrees with the
chain's analyst at the proposal level on **64.8%** (population-weighted; per stratum
{'collapsed': 0.1, 'honored': 0.1224, 'single': 0.9675}). Of its own 25 proposed splits, the shipped separation
guard would collapse **24.0%** (the chain's own operating point:
21.2%) - splits the single call would have shipped as two repairs with no cost-side veto.

Its own emitted records: **0.0%** of repair rows fail the critic's hard
cost gates outright (0.0% band-ordering violations,
0.0% missing cost; chain in-loop FAIL rate: 0.1%);
0.0% missing diagnosis/correction text. Numbers: group cost median
off by 2.5% at the median (p90
11.7%), labor by
26.3%. Self-reported confidence: median
0.72 (IQR 0.62-0.72) - a score with no
deterministic decomposition behind it, unlike kappa's consensus x support construction.

Artifacts: `eval/out/b3_single_call_claude-sonnet-5.json`,
`eval/out/b3_single_call_per_signature_claude-sonnet-5.csv`. Population-reweighted over-proposal
and guard-veto rates with SEs (the numbers the paper quotes) come from `eval/b3_reweight.py`.
