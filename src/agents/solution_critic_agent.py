"""
SolutionCriticAgent — the QA gate that turns the pipeline into a SELF-CORRECTING loop.

Every assembled Canonical Repair is reviewed BEFORE it is emitted, against deterministic
gates (no LLM judges the numbers — they stay auditable):

  hard gates  (verdict FAIL, solution marked unusable):
    - no cost consensus (cost_med missing/zero)
    - cost band out of order (q25 <= median <= q75 violated)
  soft gates  (verdict FLAG, solution emitted but marked for analyst review):
    - missing diagnosis/correction text (after the revision round below)
    - support below CRP_CRITIC_MIN_SUPPORT claims
    - consensus below CRP_CRITIC_MIN_CONSENSUS

FEEDBACK LOOP (the agentic part): when the only defect is missing/empty LLM text and the
upstream LLM call did not already fail, the critic issues a revision request. The Orchestrator
routes that critique BACK to the RepairAnalystAgent (whose prompt gains the critique), the
SolutionAssemblyAgent re-assembles, and the critic re-reviews — bounded to ONE round so a
disagreement can never loop forever. This is a hand-off AGAINST the pipeline direction:
downstream quality signal correcting an upstream agent.

HAND-OFF: reads ctx.solutions + ctx.llm_decision, writes ctx.critique and a `critic_verdict`
onto every solution; FAIL additionally clears the solution's `usable` flag.
Deterministic — selects no external tools, so the QA gate also runs fully offline.
"""
from collections import Counter

from .context import RepairContext
from ..config import CRITIC_MIN_SUPPORT, CRITIC_MIN_CONSENSUS


def verdict_summary(critique: dict) -> str:
    """One-line 'N PASS / N FLAG / N FAIL' summary, shared by the CLI and the dashboard."""
    return (f"{critique.get('n_pass', 0)} PASS / {critique.get('n_flag', 0)} FLAG / "
            f"{critique.get('n_fail', 0)} FAIL"
            + (" (after 1 revision round)" if critique.get("revised") else ""))


class SolutionCriticAgent:
    def __init__(self, min_support: int = CRITIC_MIN_SUPPORT,
                 min_consensus: float = CRITIC_MIN_CONSENSUS):
        self.min_support = min_support
        self.min_consensus = min_consensus

    def _verdict(self, s: dict) -> tuple:
        """Return (verdict, hard_problems, soft_flags) for one assembled solution."""
        problems, flags = [], []
        med, q25, q75 = s.get("cost_med"), s.get("cost_q25"), s.get("cost_q75")
        if not med:
            problems.append("no cost consensus")
        elif q25 is not None and q75 is not None and not (q25 <= med <= q75):
            problems.append(f"cost band out of order (q25={q25}, med={med}, q75={q75})")
        diagnosis = str(s.get("unified_diagnosis") or "").strip()
        correction = str(s.get("suggested_correction") or "").strip()
        if not diagnosis or not correction:
            flags.append("missing diagnosis/correction text")
        if (s.get("support") or 0) < self.min_support:
            flags.append(f"support below {self.min_support} claims")
        if s.get("consensus") is not None and s["consensus"] < self.min_consensus:
            flags.append(f"consensus below {self.min_consensus}")
        verdict = "FAIL" if problems else ("FLAG" if flags else "PASS")
        return verdict, problems, flags

    def review(self, ctx: RepairContext, revised: bool = False) -> RepairContext:
        """Stamp a verdict on every solution and record the critique on the context."""
        verdicts = []
        for s in ctx.solutions:
            verdict, problems, flags = self._verdict(s)
            s["critic_verdict"] = verdict
            if verdict == "FAIL":
                s["usable"] = False
            verdicts.append({"repair_name": s.get("repair_name"), "verdict": verdict,
                             "reasons": problems + flags})
        counts = Counter(v["verdict"] for v in verdicts)
        ctx.critique = {
            "verdicts": verdicts,
            "n_pass": counts["PASS"],
            "n_flag": counts["FLAG"],
            "n_fail": counts["FAIL"],
            "revised": revised,
        }
        return ctx

    def revision_request(self, ctx: RepairContext) -> str:
        """Critique text for the RepairAnalystAgent, or "" when a revision would be pointless.

        Only text defects are revisable (the LLM owns the prose; the numbers are deterministic),
        and only when the upstream LLM call succeeded (a fallback shell would just fail again)
        and we have not already revised once (bounded loop).
        """
        if ctx.critique.get("revised") or ctx.llm_decision.get("error"):
            return ""
        text_defects = [
            f"- repair '{v['repair_name']}': {'; '.join(r for r in v['reasons'] if 'text' in r)}"
            for v in ctx.critique.get("verdicts", [])
            if any("text" in r for r in v["reasons"])
        ]
        if not text_defects:
            return ""
        return ("Your previous answer was reviewed by a QA critic and rejected for these "
                "repairs:\n" + "\n".join(text_defects) +
                "\nReturn the SAME JSON structure with a complete, dealer-selectable "
                "unified_diagnosis and suggested_correction for EVERY repair.")
