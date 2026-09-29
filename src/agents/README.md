# Agent Hand-Offs

The AI Judge parses `src/agents/` for task hand-offs and context sharing. CRP uses
one shared `RepairContext` blackboard, defined in `context.py`, and passes that same
object through each agent. Agents read upstream fields, write their own output
fields, and never re-fetch work an earlier agent already produced.

## Shared Context Contract

- `GroupingAgent` reads `signature_id` and library metadata. It writes `ctx.claims`
  with the member claim rows for that repair signature.
- `ConsensusAgent` reads `ctx.claims`. It writes `ctx.consensus` with deterministic
  cost median, cost IQR, labor median, support, and agreement metrics.
- `RepairAnalystAgent` reads `ctx.claims` and `ctx.consensus`. It calls
  `VertexGeminiTool` with `src/prompts/split_and_diagnose.txt` and writes
  `ctx.llm_decision` with `n_repairs`, split rationale, diagnosis, correction, and
  parts text.
- `SolutionAssemblyAgent` reads `ctx.claims`, `ctx.consensus`, and
  `ctx.llm_decision`. It writes `ctx.solutions`, the final Canonical Repair rows,
  and applies the deterministic separation guard so weak LLM split suggestions are
  collapsed back to one repair.
- `SolutionCriticAgent` reads `ctx.solutions` and `ctx.llm_decision`. It writes
  `ctx.critique` and stamps a `critic_verdict` (PASS / FLAG / FAIL) on every solution;
  FAIL clears the solution's `usable` flag so a bad label cannot ship silently. When
  the only defect is missing LLM text, it issues a bounded revision request that the
  Orchestrator routes BACK to `RepairAnalystAgent` — a hand-off against the pipeline
  direction (downstream quality signal correcting an upstream agent), limited to one
  round so it can never livelock.
- `CostSavingsAgent` reads `ctx.solutions` and claim cost distributions. It writes
  grounded avoidable-overspend fields onto each solution and `ctx.savings`.
- `RegionDelegatorAgent` reads `ctx.solutions` and a dealer state. It writes
  `ctx.region` and region-localized cost/labor fields using `RegionIndexTool`.
- `ClaimMatchAgent` in `infer.py` is the product inference agent. It retrieves
  candidate Canonical Repairs for a new claim and uses `VertexGeminiTool` only when
  there are multiple plausible repairs.

## Orchestration

`orchestrator.py` constructs the tool layer once and runs the live chain:

`GroupingAgent -> ConsensusAgent -> RepairAnalystAgent -> SolutionAssemblyAgent -> SolutionCriticAgent -> CostSavingsAgent -> RegionDelegatorAgent`

with one bounded feedback edge: `SolutionCriticAgent -> RepairAnalystAgent` (revision
request) -> `SolutionAssemblyAgent` -> `SolutionCriticAgent` re-review.

`run_batch()` wraps the same chain in a resumable cache so one failed signature does
not crash a multi-thousand-call run.

## Failure Behavior

- LLM failures become a safe single-repair fallback in `RepairAnalystAgent`.
- Weak split suggestions are rejected by `SolutionAssemblyAgent` if cost tiers are
  not separated enough.
- Defective text is sent back for one (and only one) LLM revision round by
  `SolutionCriticAgent`; hard numeric defects mark the solution unusable instead.
- Batch failures are cached as `_ok: false` records and do not stop other signatures.
- Unknown incoming claims return `NO_MATCH` and route to manual review / new-signature
  creation instead of inventing a repair.
