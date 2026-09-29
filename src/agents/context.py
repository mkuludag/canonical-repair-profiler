"""
RepairContext — the shared "blackboard" object that is HANDED OFF between agents.

This is the explicit context-sharing mechanism of our agentic architecture: each agent reads the
fields it needs and writes the fields it produces, then passes the SAME context to the next agent.
No agent re-fetches what an upstream agent already computed.

Hand-off chain (see orchestrator.py) — field -> writer -> downstream readers:
  GroupingAgent      -> writes signature keys + ctx.claims (read by Consensus/Analyst/Assembly/Savings)
  ConsensusAgent     -> writes ctx.consensus (read by Analyst prompt + Assembly)
  RepairAnalystAgent -> writes ctx.llm_decision (n_repairs, per-repair diagnosis + correction; read by Assembly)
  SolutionAssemblyAgent -> writes ctx.solutions, the final Canonical Repair records
  SolutionCriticAgent   -> writes ctx.critique + per-solution verdicts; may hand the context BACK
                           to the RepairAnalystAgent for one bounded revision round
  CostSavingsAgent      -> augments ctx.solutions with grounded $ savings + writes ctx.savings
  RegionDelegatorAgent  -> (optional) writes region-localized cost onto the solutions + ctx.region
"""
from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd


@dataclass
class RepairContext:
    signature_id: int
    vehicle_line: str
    causal_part: str
    archetype: int
    archetype_theme: str = ""

    # written by GroupingAgent
    claims: Optional[pd.DataFrame] = None          # member claims (cost, labor, comment text)

    # written by ConsensusAgent
    consensus: dict = field(default_factory=dict)  # {cost_*, labor_*, cost_consensus, labor_consensus}

    # written by RepairAnalystAgent (LLM)
    llm_decision: dict = field(default_factory=dict)  # {n_repairs, split_rationale, repairs:[...]}

    # written by SolutionAssemblyAgent
    solutions: list = field(default_factory=list)  # final Canonical Repair rows

    # written by SolutionCriticAgent (QA gate; may send the context BACK to the analyst once)
    critique: dict = field(default_factory=dict)  # {verdicts:[...], n_pass, n_flag, n_fail, revised}

    # written by CostSavingsAgent (optional, grounded $ value)
    savings: dict = field(default_factory=dict)  # {avoidable_overspend, savings_per_claim, ...}

    # written by RegionDelegatorAgent (optional, inference-time localization)
    region: dict = field(default_factory=dict)  # {state, cost_index, labor_index}

    def summary(self) -> str:
        return (f"signature={self.signature_id} {self.vehicle_line}/{self.causal_part} "
                f"archetype={self.archetype} claims={0 if self.claims is None else len(self.claims)} "
                f"n_repairs={self.llm_decision.get('n_repairs','?')} solutions={len(self.solutions)}")
