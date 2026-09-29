# Canonical Repair Profiler (CRP)

### An agentic system that cleans 50 years of automotive warranty data and creates consensus ground-truth labels for downstream warranty AI products.

> **One-line value:** We turn 940,000 messy warranty claims into a library of **6,654 Canonical Repairs** — each a single agreed-upon *diagnosis + correction + labor + cost + parts* — providing cleaner ground truth for existing production AI products, surfacing **avoidable overspend equal to 18.6% of historical spend in scope** (claims above each repair's consensus cost), and keeping **future data clean** via a dealer-facing diagnosis dropdown.
>
> **Confidentiality note:** all dollar amounts in this repository (data files, docs, and figures) have been **rescaled by an undisclosed constant** for the public release. Relative statistics — ratios, percentages, consensus fractions, and distributions — are unchanged and exact.

---

## TABLE OF CONTENTS
1. [Executive Summary](#1-executive-summary)
2. [Business Impact & Innovation (Rubric: 30 pts)](#2-business-impact--innovation-rubric-30-pts)
3. [Agentic Architecture (Rubric: 25 pts)](#3-agentic-architecture-rubric-25-pts)
4. [Technical Execution (Rubric: 20 pts)](#4-technical-execution-rubric-20-pts)
5. [Documentation & Structure (Rubric: 15 pts)](#5-documentation--structure-rubric-15-pts)
6. [Testability & Video (Rubric: 10 pts)](#6-testability--video-rubric-10-pts)
7. [Challenges Overcome](#7-challenges-overcome)
8. [Engineering Quality & Reliability](#8-engineering-quality--reliability)
9. [How to Run](#9-how-to-run)
10. [Repository Map](#10-repository-map)
11. [Rubric Self-Scorecard](#11-rubric-self-scorecard)
12. [Paper Evaluation Harness](#12-paper-evaluation-harness)

---

## 1. Executive Summary

Automotive warranty claims — spanning multiple internal systems and tens of millions of records — are noisy, inconsistent, and free-text-heavy. The same physical repair is described, priced, and labored a dozen different ways by different dealers, states, and agents. That noise caps the accuracy of downstream AI products and costs analytics teams enormous manual effort.

**Canonical Repair Profiler (CRP)** is an **agentic AI pipeline** that:
1. **Unifies** the three warranty systems into one repair-line dataset (`repair_profile`, 938,307 rows × 112 columns).
2. **Groups** alike claims into **Repair Signatures** (vehicle line × causal part × symptom archetype).
3. **Distills** each group, using a **Vertex AI Gemini agent**, into ONE **Canonical Repair**: a Unified Diagnosis, a Suggested Correction, consensus labor hours, consensus cost, and a parts list — with an agentic **split decision** that separates "reseal vs replace" style sub-repairs.
4. **Flags discrepancies** — claims/dealers/states that deviate from the canonical repair — for cleansing and cost recovery.

**Result:** a **Canonical Repair Library** of **6,654 canonical repairs (3,623 GOLD)** plus **5,070 golden solutions** — clean, confidence-scored training labels and a dealer-facing diagnosis catalog.

---

## 2. Business Impact & Innovation (Rubric: 30 pts)

### 2.1 Cost Savings (Rubric: 20 pts) — better ground truth for production warranty AI
- **Service-plan parts & labor prediction:** its ceiling is **data quality**. CRP removes the dominant noise source (inconsistent diagnoses/labor/cost for the same repair), so the model trains on **consensus ground truth** instead of contradictory raw claims → **higher accuracy, fewer mispredictions, lower leakage**.
- **Automated claim adjudication:** CRP supplies a **canonical expected diagnosis + cost band per repair**, turning fuzzy approvals into **outlier-based, defensible decisions** and surfacing overpayments (we measured **same-state dealer cost spreads of 2.0×–6.9×** for the *identical* repair — direct recovery opportunity).
- **Upstream multiplier:** cleaner ground truth makes existing models measurably better and enables new clean-data-only models — better data → better models → more realized savings.
- **Bottom-up savings opportunity (measured, not modeled).** Independent of the model-uplift case, the `CostSavingsAgent` scans every single-repair canonical group and sums the spend **above the consensus cost**: across **2,172 canonical repairs / 127,465 claims**, **18.6% of historical spend in scope sits above consensus** — the addressable ceiling if every dealer executed the canonical repair at consensus. Reproduce: `python grouping/12_group_savings.py` → [`grouping/out/group_savings.csv`](grouping/out/group_savings.csv).
- **Region-aware costing.** Warranty cost varies **1.67x across US states**; the `RegionDelegatorAgent` localizes each canonical cost to the dealer's state (deterministic `RegionIndexTool`), so savings and adjudication bands are regionally fair.
- **Data cleansing at fleet scale:** the same method scales to **50 years of warranty claims**, a foundational asset reusable across enterprise analytics — not a point solution.

### 2.2 Originality (Rubric: 10 pts) — novel on four axes
1. **Consensus-as-ground-truth, with confidence.** Instead of trusting any single claim, we derive a *consensus* canonical repair per group and ship a **support × consensus confidence** weight with every label. Deviations become a **discrepancy signal** — the same pipeline produces *both* training labels *and* business savings.
2. **Agentic, LLM-decided splitting.** A Gemini agent decides — per group — whether a single causal-part code actually hides **two distinct repairs** (minor reseal vs full replacement) and writes a separate diagnosis/solution for each. This is *self-correcting granularity*, not a fixed rule.
3. **Self-correcting agentic QA.** A `SolutionCriticAgent` reviews every Canonical Repair *before it ships* and routes text defects **back to the LLM analyst** for one bounded revision round — while the **LLM never touches the numbers** (deterministic tools own labor/cost/confidence), so every dollar figure stays auditable and finance-defensible.
4. **Closes the loop on future data.** The Unified Diagnosis is designed as a **dealer dropdown at claim filing** — so CRP doesn't only clean the past 50 years, it **prevents** future noise at the source, helping dealers, Ford adjudication agents, and analytics teams remove ambiguity forever.

---

## 3. Agentic Architecture (Rubric: 25 pts)

> The judge parses `src/agents/` (task hand-offs + context sharing) and `src/tools/` (tool selection + API integration). Both are first-class, real, and runnable.

### 3.1 Design (Rubric: 15 pts) — multi-agent hand-off over a shared context
A single **`RepairContext`** blackboard (`src/agents/context.py`) is **handed off** down a chain of single-responsibility agents — each reads upstream fields and writes its own, never re-fetching:

```
GroupingAgent ─► ConsensusAgent ─► RepairAnalystAgent(LLM) ─► SolutionAssemblyAgent ─► SolutionCriticAgent ─► CostSavingsAgent ─► RegionDelegatorAgent
(signature +     (cost/labor        (split decision +          (fuse text+numbers ►       (QA gate: PASS /        ($ avoidable          (localize cost
 member claims)   consensus, IQR)    diagnosis+correction)      Canonical Repair)          FLAG / FAIL verdict)     overspend)            to dealer state)
                                            ▲                                                   │
                                            └────────────── bounded revision request ◄─────────┘
                                                (self-correcting feedback loop, max 1 round)
```

- **`GroupingAgent`** — defines the Repair Signature and gathers member claims.
- **`ConsensusAgent`** — deterministic, auditable cost/labor consensus (the LLM never invents numbers).
- **`RepairAnalystAgent`** — Vertex Gemini: the agentic **split decision** + Unified Diagnosis + Suggested Correction.
- **`SolutionAssemblyAgent`** — partitions claims into the LLM-decided cost tiers, applies the **separation guard**, assembles the final golden solution(s).
- **`SolutionCriticAgent`** — the **self-correcting QA loop**: reviews every assembled Canonical Repair with deterministic gates (text completeness, cost-band sanity `q25 ≤ med ≤ q75`, support/consensus floors) and stamps a **PASS / FLAG / FAIL** verdict. Text defects are routed **back** to `RepairAnalystAgent` as a revision request (bounded to one round — never a livelock); hard numeric failures clear the `usable` flag so a bad label cannot ship silently. The pipeline is **not one-directional**.
- **`CostSavingsAgent`** — deterministically quantifies avoidable overspend vs the consensus cost (grounded $ value).
- **`RegionDelegatorAgent`** — localizes the national solution to a dealer's state via the `RegionIndexTool` (inference-time).
- **`Orchestrator`** (`src/agents/orchestrator.py`) — constructs the tools once and threads the shared context through the chain; `run_one(signature, state=...)` (live demo) and `run_batch()` (concurrent + resumable). Full sequence diagram in [`docs/architecture.md`](docs/architecture.md).

**Explicit context hand-offs (`RepairContext`):** `GroupingAgent` writes `ctx.claims`; `ConsensusAgent` writes `ctx.consensus`; `RepairAnalystAgent` writes `ctx.llm_decision`; `SolutionAssemblyAgent` writes `ctx.solutions`; `SolutionCriticAgent` writes `ctx.critique` + a `critic_verdict` on every solution; `CostSavingsAgent` writes grounded savings fields; `RegionDelegatorAgent` writes `ctx.region` and localized cost fields. The same object is handed from agent to agent — including one **reverse hand-off** (critic → analyst) that makes the chain self-correcting.

### 3.2 Tool Selection (Rubric: 10 pts) — clean external-capability layer (`src/tools/`)
Agents never call external APIs directly; they select tools:
- **`VertexGeminiTool`** — Vertex AI Gemini (`gemini-2.5-flash`): token auto-refresh, 60s timeout, retry/back-off, strict-JSON parsing. (SDK imported lazily → testable offline.)
- **`BigQueryTool`** — BigQuery reads with auth refresh, retries, Decimal→float coercion.
- **`DataStore`** — Parquet/CSV persistence + a **resumable JSONL cache** (so a 3,600-call batch resumes, not restarts).
- **`StatsTool`** — pure consensus / IQR / cost-tier partition / confidence math.
- **`RegionIndexTool`** — deterministic per-state cost/labor index (1.67x spread) from the region study; degrades gracefully to a neutral 1.0x.
- **`auth.py`** — single source of fresh gcloud credentials (Ford CAA blocks ADC; we use refreshing user tokens).

All tool selection / project ids / model / thresholds are centralized in **[`src/config.py`](src/config.py)** (env-overridable).

---

## 4. Technical Execution (Rubric: 20 pts)

### 4.1 Execution (Rubric: 10 pts)
- The pipeline **actually ran at scale**: unified dataset built (938,307 rows), full grouping persisted (71,957 signatures), and **3,623 GOLD signatures processed by the LLM agent (all 3,623 succeeded; the 4 transient failures hit mid-batch were retried to success — 0 permanent)** → **5,070 golden solutions emitted (2,176 single-repair, 2,894 split into two)**. See **[`execution_logs.txt`](execution_logs.txt)** (a captured run: tests + sample build + offline inference + the bottom-up savings run + a live cloud run) and **[`grouping/out/golden_solutions.csv`](grouping/out/golden_solutions.csv)**.
- **99-test automated suite (96% source coverage) + CI** (`tests/`, [`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs **fully offline** (all cloud calls mocked) — `make test` / `pytest -q`. The tests already caught and fixed a real `StopIteration` bug in the separation guard.
- **Runs with no cloud:** `CRP_USE_SAMPLE=1` routes every data read to the bundled sample (`data/sample/`). `make demo` / `python cli.py infer ...` produce a real matched Canonical Repair offline.
- Single entrypoint: **`src/main.py`** (`crp infer|run|dash`, the code-track entry point) with a root-level **`cli.py`** alias for shorter commands + **`Makefile`**; structure declared in **`pyproject.toml`**.

### 4.2 Prompts (Rubric: 5 pts)
- Prompt templates are stored in the repo (externalized, not inlined):
  - **[`src/prompts/split_and_diagnose.txt`](src/prompts/split_and_diagnose.txt)** — split decision + diagnosis/correction in one strict-JSON call.
  - **[`src/prompts/match_incoming_claim.txt`](src/prompts/match_incoming_claim.txt)** — routes an incoming claim to the best canonical repair.
  - **[`src/prompts/README.md`](src/prompts/README.md)** — documents each template, its variables, and the expected JSON.

### 4.3 Error Handling (Rubric: 5 pts) — defensive throughout, and tested
- **`try/except` with classification** of transient (401/403/429/5xx/timeout) vs fatal errors → **retry with back-off** (`VertexGeminiTool`, `BigQueryTool`); covered by `tests/test_vertex_gemini_tool.py` + `tests/test_bigquery_tool.py`.
- **API timeouts** (60s) so a stuck call cannot hang the pipeline.
- **Token auto-refresh** (re-mint at 25 min) — solved the real mid-batch auth-expiry cascade; missing-gcloud and reauth errors raise **actionable** messages (`tests/test_auth.py`).
- **Graceful degradation**: a failed group falls back to a single-repair shell; bad JSON is reported (not retried forever); the `RegionIndexTool` degrades to a neutral index if its file is missing; the JSONL cache makes the batch **resumable** and skips malformed lines.

---

## 5. Documentation & Structure (Rubric: 15 pts)

- **Sample I/O (5):** **[`inputs_outputs.txt`](inputs_outputs.txt)** — exact, runnable input → output examples (offline + live). The bundled **`data/sample/*.csv`** are themselves human-readable I/O fixtures.
- **Diagram (5):** **[`docs/architecture.md`](docs/architecture.md)** — a flow diagram **and an agent sequence diagram** (Mermaid) + **[`figures/architecture.png`](figures/architecture.png)** + root **[`architecture_diagram.png`](architecture_diagram.png)**.
- **README (5):** this document + **[`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md)**, **[`docs/TESTING.md`](docs/TESTING.md)**, the per-package guides [`src/agents/README.md`](src/agents/README.md) and [`src/tools/README.md`](src/tools/README.md), and the consolidated [`grouping/METHODOLOGY.md`](grouping/METHODOLOGY.md).

**Clean structure** — product vs research is clearly separated (see the Repository Map): the runnable **product** is `app.py` + `cli.py` + `src/` + `tests/`; the **research/build** lineage lives in `grouping/` and `scripts/`.

**Results & figures:** see [`figures/`](figures/) — prioritization quadrants, group-size & cost distributions, split breakdown, and regional/dealer variance.

---

## 6. Testability & Video (Rubric: 10 pts)

- **Automated tests:** **99 pytest tests, 96% source coverage** (`tests/`) run **fully offline** (`make test`), with **CI** on push ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) across Python 3.10–3.12. They cover the stats math, every agent (incl. the separation guard, the critic's QA gates + bounded revision loop, and graceful LLM-failure fallback), the tool resilience (retries, timeouts, auth errors), and the full orchestrator hand-off chain end-to-end against the bundled sample.
- **Interactive dashboard (`app.py`, Streamlit):** a lightweight local UI over the REAL agents — Tab 1 is the **technician Claim Assistant** (vehicle → part → repair dropdowns, dealer-state cost localization, savings); Tab 2 runs the live agent hand-off chain step-by-step. Not a mockup — it calls the same `src/agents` + `src/tools`. Runs offline with `CRP_USE_SAMPLE=1`.
- **Reproducible:** numbers are deterministic (`StatsTool`), LLM JSON is cached, and the offline sample lets anyone re-run the demo with no credentials.
- **Demo video (inside the ZIP):** walks through the dashboard + a live `python cli.py run --signature <id>` / `infer` matching `inputs_outputs.txt`.

---

## 7. Challenges Overcome

We hit and **solved** real, hard problems — evidence of robustness:
1. **Cross-system identity with masked VINs.** PAWS exposes only masked VINs; we proved GSAR/OWS use the *same* masking scheme and joined on **masked VIN + repair date** (RO# and dealer codes are not comparable across systems).
2. **Empty "stub" views.** Several high-scoring views (OWS `sowsa*`, PAWS `sxpaa104`) were near-empty decoys; we caught them via row-count profiling and pivoted to the real populated sources (e.g. OWS `sowsc01`, 123M rows).
3. **Mid-batch auth expiry.** A 3,600-call LLM batch died halfway when gcloud tokens expired; we built **token auto-refresh + retry + resumable cache** and recovered **fully — all 3,623 signatures completed, the 4 transient failures retried to success (0 permanent)**.
4. **Hidden multi-repairs (reseal vs replace).** One causal-part code often hides two repairs; the LLM **agentically splits** them and we recompute per-tier numbers.
5. **Ford context-aware access blocking ADC.** We authenticate every tool with refreshing **user OAuth tokens** instead.

---

## 8. Engineering Quality & Reliability

- **Modern agentic design:** combines large-scale data engineering, unsupervised symptom clustering, and an **LLM reasoning agent (Vertex Gemini)** that makes per-group structural decisions — not a static script.
- **Robust:** every external call is wrapped with timeouts, retries, auth refresh, and graceful fallback; the batch is **resumable**; numbers are deterministic and auditable.
- **Safe for downstream use:** every Canonical Repair carries a **confidence** and a **`usable`** flag, so consumers can trust-weight or filter — no silent bad labels.
- **Scales to all of Ford:** the same pipeline generalizes from 940k PAWS claims to 88M warranty claims.

---

## 9. How to Run

> **Data note:** the raw Ford warranty extract (the 0.8 GB `repair_profile.parquet`) is **not bundled**
> (size + data privacy). But a **tiny masked sample** (`data/sample/`, 8 signatures) **is** bundled, so
> the dashboard, CLI and tests **run with no cloud** via `CRP_USE_SAMPLE=1`. The **emitted Canonical
> Repair Library** (`grouping/out/golden_solutions.csv`, `canonical_repairs.csv`) is also included so the
> outputs are inspectable. The upstream BigQuery build that produces `repair_profile.parquet` from
> Ford's warranty warehouse is proprietary and omitted from this academic release; the bundled sample
> and emitted library above are sufficient to run and inspect the system offline.

```bash
pip install -r requirements.txt        # or: make install   (adds pytest)

# ---- NO CLOUD NEEDED (bundled sample) ---------------------------------------
make test                              # 99 offline tests (all cloud calls mocked)
make demo                              # offline inference against data/sample/
CRP_USE_SAMPLE=1 streamlit run app.py  # the full dashboard, offline

# ---- LIVE (Ford GCP) --------------------------------------------------------
gcloud auth login --account <your-ford-account>     # Ford CAA: interactive user creds

streamlit run app.py                   # (D) dashboard: Claim Assistant + live pipeline (recommended demo)
python cli.py run --signature 32954    # (A) live agent build, one group (engine long block)
python cli.py run --signature 40109 --state CA   # split candidate, localized to California
python cli.py infer --vehicle "F-150" --part 6148 \
    --concern "engine knock on cold start, heavy oil consumption, misfire on cyl 1 and 4, metal in oil"  # (B) inference
python grouping/10_golden_solutions.py # (C) full resumable batch over all GOLD signatures
python grouping/12_group_savings.py    # (E) recompute the bottom-up avoidable-overspend savings
```

## 10. Repository Map

```
PRODUCT (runnable agentic system)
  app.py             Streamlit dashboard over the real agents (Claim Assistant + live pipeline)
  src/main.py        THE entry point:  crp infer | run | dash
  cli.py             root-level alias for src/main.py (shorter: python cli.py ...)
  Makefile           make test | demo | run | dash | sample | figures
  pyproject.toml     packaging + pytest config
  src/config.py      ALL project ids / model / paths / thresholds (env-overridable)
  src/agents/        7-agent hand-off chain (incl. self-correcting critic) over a shared RepairContext + orchestrator + inference
  src/tools/         VertexGemini, BigQuery, DataStore, StatsTool, RegionIndexTool, auth
  src/prompts/       externalized LLM prompt templates (+ README)
  tests/             99 offline pytest tests (96% coverage) + fixtures (conftest.py)
  data/sample/       bundled masked sample (8 signatures) so the demo + tests run with no cloud
  .github/workflows/ CI (pytest on push, Python 3.10-3.12)

DOCS / EVIDENCE
  README.md                 this file
  inputs_outputs.txt        12 exact I/O examples     execution_logs.txt captured run (tests + demo + live)
  architecture_diagram.png  system flow (root)        figures/           results + value + savings figures
  docs/                     architecture.md (flow + sequence diagram), DATA_DICTIONARY.md, TESTING.md

RESEARCH / DATA BUILD (how the library was produced; not needed to run the demo)
  grouping/          analysis + Canonical Repair Library build (01-12) + out/ artifacts + METHODOLOGY.md
  scripts/           data prep, sample builder, pipeline verification tooling
```

---

## 11. Rubric Self-Scorecard

| Category (weight) | Where to look | Evidence |
|---|---|---|
| **Business Impact (30)** — Cost Savings 20, Originality 10 | `README §2` | attributed model uplift + bottom-up avoidable overspend, 18.6% of spend in scope (`CostSavingsAgent`); region-aware costing; consensus-as-ground-truth with confidence |
| **Agentic Architecture (25)** — Design 15, Tools 10 | `src/agents/`, `src/tools/`, `docs/architecture.md` | 7-agent hand-off chain over a shared `RepairContext` with a self-correcting critic feedback loop; 5-tool capability layer (incl. `RegionIndexTool`); flow + sequence diagrams |
| **Technical Execution (20)** — Execution 10, Prompts 5, Errors 5 | `tests/`, `execution_logs.txt`, `src/prompts/`, `src/tools/` | 99-test offline suite (96% coverage) + CI; captured execution log (tests, coverage, offline + live runs, critic QA gate); 2 documented prompt templates; classified retries/timeouts/auth with tests |
| **Documentation (15)** — Sample I/O 5, Diagram 5, README 5 | `inputs_outputs.txt`, `docs/`, this README | 12 runnable I/O examples; root + Mermaid diagrams; data dictionary, testing & per-package READMEs; clean product/research split |
| **Testability & Video (10)** — Video 10 | demo video + `app.py` | live Streamlit Claim Assistant demo running an exact `inputs_outputs.txt` example |



---

## 12. Paper Evaluation Harness

`eval/` contains the audit and ablation harness behind the accompanying paper: post-hoc
critic and guard audits (`a1`-`a7`), the frozen 400-signature panel (`panel_400.csv`,
drawn by `b0_panel.py`), the rerun-stability and v1/v2 churn analyses (`b1`,
`rerun_full_chain.py`), the LLM-writes-the-numbers and single-call ablations (`b2`, `b3`,
across four providers via `llm_gateway_tool.py`), and the figure scripts
(`paper_figs.py`, `fig_verification.py`). `eval/RESULTS.md` is the re-derived results
ledger; each section stamps the data root it was computed from. Generated outputs land in
`eval/out/` (untracked - see `ANONYMIZATION_NOTES.md`). The v2 library of record, with its
run manifest, is `grouping/out_v2_public/`.
