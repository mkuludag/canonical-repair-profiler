"""
Orchestrator — wires the tools + agents and runs the Canonical Repair Profiling pipeline.

This file makes the AGENTIC ARCHITECTURE explicit: it constructs the tool layer once, then passes a
single shared RepairContext through the agent hand-off chain:

    GroupingAgent -> ConsensusAgent -> RepairAnalystAgent(LLM) -> SolutionAssemblyAgent
        -> SolutionCriticAgent (QA gate; may send the context BACK to the analyst for ONE
           bounded revision round) -> CostSavingsAgent -> RegionDelegatorAgent (optional)

`run_one(signature_id)` processes a single repair group end-to-end and returns its golden
solution(s) — this is the path demonstrated live in the submission video.
`run_batch(signature_ids)` runs many signatures concurrently with a resumable cache; a failed
signature is logged + recorded in the cache and never crashes the batch.

Run live demo:
    python -m src.agents.orchestrator --signature 411
"""
import argparse
import concurrent.futures as cf
import json
import logging

from ..tools import BigQueryTool, VertexGeminiTool, DataStore, StatsTool, RegionIndexTool
from .grouping_agent import GroupingAgent
from .consensus_agent import ConsensusAgent
from .repair_analyst_agent import RepairAnalystAgent
from .solution_assembly_agent import SolutionAssemblyAgent
from .solution_critic_agent import SolutionCriticAgent, verdict_summary
from .cost_savings_agent import CostSavingsAgent
from .region_delegator_agent import RegionDelegatorAgent

log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, root: str = "."):
        # --- tool layer (constructed once, shared by all agents) ---
        self.datastore = DataStore(root=root)
        self.gemini = VertexGeminiTool()
        self.stats = StatsTool()
        self.bq = BigQueryTool()  # available for region/dealer discrepancy tooling
        self.region_index = RegionIndexTool()
        # --- agent layer ---
        self.grouping = GroupingAgent(self.datastore)
        self.consensus = ConsensusAgent(self.stats)
        self.analyst = RepairAnalystAgent(self.gemini)
        self.assembly = SolutionAssemblyAgent(self.stats)
        self.critic = SolutionCriticAgent()
        self.savings = CostSavingsAgent(self.stats)
        self.region = RegionDelegatorAgent(self.region_index)

    def run_one_ctx(self, signature_id: int, verbose: bool = False, state: str = "") -> "RepairContext":
        """Full agent hand-off chain for one signature, returning the whole RepairContext.

        The context carries everything the batch consumer needs beyond the solutions themselves:
        the critic's verdicts and revised flag, the analyst's decision (and any transport error),
        and the savings fields. `run_one` wraps this and keeps the original solutions-only API.
        """
        ctx = self.grouping.load_signature(signature_id)        # GroupingAgent
        ctx = self.consensus.enrich(ctx)                        # ConsensusAgent
        ctx = self.analyst.analyze(ctx)                         # RepairAnalystAgent (LLM)
        ctx = self.assembly.assemble(ctx)                       # SolutionAssemblyAgent
        ctx = self.critic.review(ctx)                           # SolutionCriticAgent (QA gate)
        feedback = self.critic.revision_request(ctx)
        if feedback:                                            # bounded feedback loop: ONE round
            log.info("critic requested a revision for signature %s", signature_id)
            ctx = self.analyst.analyze(ctx, feedback=feedback)  # critique -> analyst
            ctx = self.assembly.assemble(ctx)                   # re-assemble
            ctx = self.critic.review(ctx, revised=True)         # re-review (final verdicts)
        ctx = self.savings.analyze(ctx)                         # CostSavingsAgent ($ grounding)
        if state:
            ctx = self.region.apply(ctx, state)                 # RegionDelegatorAgent (localize)
        if verbose:
            print(ctx.summary())
            print(f"critic: {verdict_summary(ctx.critique)}")
            print(json.dumps(ctx.solutions, indent=2, default=str))
        return ctx

    def run_one(self, signature_id: int, verbose: bool = False, state: str = ""):
        """Back-compat wrapper: the golden solution(s) for one signature (see run_one_ctx)."""
        return self.run_one_ctx(signature_id, verbose=verbose, state=state).solutions

    def run_batch(self, signature_ids, workers: int = 8, cache_rel: str = "grouping/out/llm_cache.jsonl"):
        done = self.datastore.load_cache(cache_rel, ok_only=True)
        todo = [s for s in signature_ids if s not in done]
        results = []

        def _work(sid):
            try:
                ctx = self.run_one_ctx(int(sid))
                # Cache the critique + decision metadata alongside the solutions: the revision-fire
                # count and verdict distribution are results in their own right, and the cache is
                # the run's durable record.
                self.datastore.append_cache(cache_rel, {
                    "signature_id": int(sid), "_ok": True, "solutions": ctx.solutions,
                    "critique": {k: ctx.critique.get(k) for k in ("n_pass", "n_flag", "n_fail", "revised")},
                    "n_repairs": ctx.llm_decision.get("n_repairs"),
                    "llm_error": ctx.llm_decision.get("error") or None,
                    # >1 means the reply came from a jittered parse-failure retry, not temp 0
                    "llm_attempts": ctx.llm_decision.get("_attempts"),
                })
                return ctx.solutions
            except Exception as e:  # never let one group crash the batch
                log.warning("signature %s failed: %s", sid, str(e)[:200])
                self.datastore.append_cache(cache_rel, {"signature_id": int(sid), "_ok": False, "error": str(e)[:200]})
                return []

        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            for sols in ex.map(_work, todo):
                results.extend(sols)
        return results


def main():
    ap = argparse.ArgumentParser(description="Canonical Repair Profiling - agent orchestrator")
    ap.add_argument("--signature", type=int, required=True, help="signature_id to process live")
    ap.add_argument("--state", default="", help="optional US state to localize cost, e.g. CA")
    args = ap.parse_args()
    Orchestrator().run_one(args.signature, verbose=True, state=args.state)


if __name__ == "__main__":
    main()
