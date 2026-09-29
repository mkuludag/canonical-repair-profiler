# Prompt templates

LLM prompts are stored here as plain-text templates (not inlined in code) so they are versioned,
reviewable, and editable without touching Python. All calls go through `VertexGeminiTool` in
strict-JSON mode; the agent fills `%(...)s` placeholders and parses the JSON response.

## `split_and_diagnose.txt`
- **Used by:** `RepairAnalystAgent` (`src/agents/repair_analyst_agent.py`).
- **Purpose:** in ONE call, decide whether a repair group is one repair or hides 2 distinct repairs
  (minor vs full replacement), and write a Unified Diagnosis + Suggested Correction per repair.
- **Variables:** `%(context)s` (vehicle line, causal part, symptom theme, claim count, cost/labor
  bands) and `%(comments)s` (a cost-spread sample of real claim narratives, each tagged with its cost).
- **Returns JSON:** `{ "n_repairs": int, "split_rationale": str, "repairs": [ { "name", "cost_tier"
  (low|high|all), "unified_diagnosis", "suggested_correction", "typical_parts" } ] }`.
- **Guardrails:** cap at 2 repairs; a proposed split is only honored if the high cost tier is >=1.5x
  the low tier (deterministic separation guard in `SolutionAssemblyAgent`), so the prompt cannot
  fabricate spurious granularity.
- **Revision round:** when the `SolutionCriticAgent` rejects the assembled text, the analyst re-sends
  this same template with a `REVISION REQUEST (from the QA critic agent): ...` suffix naming the
  defective repairs — one bounded round, same JSON contract.

## `match_incoming_claim.txt`
- **Used by:** `ClaimMatchAgent` (`src/agents/infer.py`).
- **Purpose:** route an INCOMING claim to the single best-matching Canonical Repair (the dealer
  dropdown selection) among the candidates for that vehicle + causal part.
- **Variables:** `%(vehicle)s`, `%(part)s`, `%(concern)s` (incoming text), `%(candidates)s` (the
  enumerated candidate repairs with their diagnosis/correction/labor/cost).
- **Returns JSON:** `{ "match_index": int (-1 if none fit), "confidence_0_1": float, "reason": str }`.
- **Note:** when only one candidate exists the agent short-circuits and does NOT call the LLM.

## Conventions
- Keep prompts deterministic-friendly: agents call them at `temperature=0.0` (analyst) for stable
  demos. The model only writes prose/labels; all dollar/labor numbers come from `StatsTool`.
- If you add a template, document it here (variables + expected JSON) and add a test that exercises
  the consuming agent with a `FakeLLM`.
