"""
RepairAnalystAgent — the LLM reasoning agent (Vertex Gemini).

Two responsibilities in one structured call:
  1. SPLIT DECISION (agentic): decide whether the group is really ONE repair or hides 2-3 distinct
     repairs (e.g. minor reseal vs full replacement), justified by the claims' text + cost spread.
  2. TEXT GENERATION: write a Unified Diagnosis (dealer-selectable) + Suggested Correction per repair.

It builds the prompt from the shared context (group stats + a cost-spread SAMPLE of real claim
comments) so the model sees any bimodality. All transport hardening (auth refresh, timeout, retry,
strict-JSON parsing) lives in VertexGeminiTool. The LLM's reply is schema-validated here; on any
LLM failure OR malformed payload the agent degrades gracefully to a single-repair shell, so the
pipeline always emits a solution.

HAND-OFF: reads ctx.claims + ctx.consensus, writes ctx.llm_decision, passes ctx to assembly.
Selects tools: VertexGeminiTool.
"""
import os
import numpy as np

from .context import RepairContext

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "split_and_diagnose.txt")


class RepairAnalystAgent:
    def __init__(self, gemini_tool, prompt_path: str = PROMPT_PATH, sample_size: int = 8):
        self.llm = gemini_tool
        self.sample_size = sample_size
        with open(os.path.abspath(prompt_path)) as fh:
            self.template = fh.read()

    def _build_prompt(self, ctx: RepairContext) -> str:
        g = ctx.claims.dropna(subset=["gsar_tot_cost_gross"]).sort_values("gsar_tot_cost_gross")
        if len(g) == 0:
            g = ctx.claims
        idx = np.unique(np.linspace(0, len(g) - 1, num=min(self.sample_size, len(g))).astype(int))
        samp = g.iloc[idx]
        comments = "\n".join(
            f"- [${0 if (c != c) else round(c)}] {str(t)[:280]}"
            for c, t in zip(samp["gsar_tot_cost_gross"], samp["paws_comment_trail"]))
        c = ctx.consensus
        context = (f"vehicle_line={ctx.vehicle_line}; causal_part={ctx.causal_part}; "
                   f"symptom_theme=[{ctx.archetype_theme}]; claims={c.get('n_claims')}; "
                   f"cost_median=${c.get('cost_med')}; cost_IQR=${c.get('cost_q25')}-${c.get('cost_q75')}; "
                   f"labor_median_hrs={c.get('labor_med')}")
        return self.template % {"context": context, "comments": comments}

    @staticmethod
    def _schema_error(decision: dict) -> str:
        """Validate the LLM payload against the contract in src/prompts/split_and_diagnose.txt."""
        repairs = decision.get("repairs")
        if not isinstance(repairs, list) or not repairs or not all(isinstance(r, dict) for r in repairs):
            return "schema: 'repairs' must be a non-empty list of objects"
        n = decision.get("n_repairs")
        if not isinstance(n, int) or n < 1:
            return "schema: 'n_repairs' must be a positive integer"
        return ""

    @staticmethod
    def _fallback_shell(error) -> dict:
        """Single-repair shell: downstream agents still produce a (low-confidence) solution."""
        return {"n_repairs": 1, "split_rationale": "", "repairs": [{}], "error": error}

    def analyze(self, ctx: RepairContext, feedback: str = "") -> RepairContext:
        """`feedback` carries the SolutionCriticAgent's critique on a revision round —
        it is appended to the prompt so the model sees exactly what to fix."""
        prompt = self._build_prompt(ctx)
        if feedback:
            prompt += f"\n\nREVISION REQUEST (from the QA critic agent):\n{feedback}"
        # temperature 0 for a stable, reproducible split decision (demo-safe)
        result = self.llm.generate_json(prompt, temperature=0.0)
        if result.get("_ok"):
            decision = {k: v for k, v in result.items() if k != "_ok"}
            problem = self._schema_error(decision)
            ctx.llm_decision = self._fallback_shell(problem) if problem else decision
        else:
            # graceful degradation: transport-level failure -> single-repair shell
            ctx.llm_decision = self._fallback_shell(result.get("error"))
        return ctx
