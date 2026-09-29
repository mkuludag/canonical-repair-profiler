# Public Release — Anonymization & Rescaling Notes

This repository is a sanitized snapshot of an internal warranty-analytics research project,
prepared for use alongside an academic paper. The scientific content — methodology, agent
architecture, and all relative statistics — is unmodified.

## What was changed for this release
- **Row-level PII removed.** Verbatim technician/customer narrative columns (which carried
  customer names, VINs, dealer emails, phone numbers) were dropped from all data artifacts.
  The bundled offline sample uses synthetic, theme-flavored narrative text instead, so the
  demo and tests still exercise symptom clustering end-to-end.
- **Dealer identity pseudonymized.** Real dealer codes were replaced with `DLR-####`
  pseudonyms; state-level and spread statistics are unchanged.
- **Infrastructure identifiers redacted.** Cloud project IDs, service accounts, internal
  dataset/view names, and internal URLs were replaced with placeholders; runtime values
  inject via `CRP_*` environment variables.
- **Proprietary warehouse documentation removed.** Internal data-warehouse discovery
  material (schema catalogs, build SQL) is not part of this release.
- **Business/pitch material removed.** Internal business-case documents, product names,
  and program-level financial figures are not part of this release.
- **All dollar amounts rescaled.** Every dollar value in the data files, documentation, and
  figures has been multiplied by a single **undisclosed constant**. Relative statistics —
  ratios, percentages, consensus fractions, IQR shapes, and distributions — are exact and
  unchanged. Absolute dollar figures should not be interpreted as real costs.
- Labor **hours**, claim **counts**, and all consensus/confidence scores are real and
  unrescaled.

## Reproducibility
The upstream pipeline that builds `repair_profile.parquet` from the source warranty systems
is proprietary and omitted. Everything downstream — grouping, consensus, the agentic
distillation pipeline, the offline demo, and the test suite — runs from this repository
(`CRP_USE_SAMPLE=1`, see README §9).

## 2026-08-27 addition: paper evaluation harness and v2 library

The `eval/` harness (guard/critic/revision audits, the 400-signature panel, the B2/B3
ablations, and the churn/stability analyses reported in the accompanying paper) and the v2
released library (`grouping/out_v2_public/`) were added under the same contract:

- **`eval/out/` is not tracked.** The cached model responses and derived tables were
  generated against internal-basis data; only the scripts, prompts, `panel_400.csv`, and
  `RESULTS.md` (which stamps the data root each section was computed from) ship.
- **Gateway identifiers redacted.** The multi-provider ablation client reads its base URL,
  tenant, and audience from `CRP_LLM_*` environment variables; no internal endpoints are
  hardcoded.
- **Dollar amounts inside model-written text rescaled.** The `split_rationale` fields of
  `golden_solutions.csv` (v1 and v2) quoted dollar figures copied from raw claim evidence.
  These are now rescaled by the same undisclosed release constant as the numeric columns,
  closing a loophole that would otherwise have allowed recovery of the constant by division.
- **Verbatim narrative column dropped.** `grouping/out_v2_public/canonical_repairs.csv`
  ships without `correction_example` (raw technician text), matching the v1 export.
- The script that applies the rescale constant is itself internal and not part of this
  repository.
