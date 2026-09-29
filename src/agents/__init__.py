"""
src/agents — the multi-agent pipeline for Canonical Repair Profiling.

Agents collaborate over a shared RepairContext (see context.py). Each agent has a single
responsibility and HANDS OFF the enriched context to the next:

    GroupingAgent          -> defines the repair signature + gathers member claims
    ConsensusAgent         -> deterministic cost/labor consensus (auditable numbers)
    RepairAnalystAgent     -> LLM: split decision + unified diagnosis + suggested correction
    SolutionAssemblyAgent  -> fuses text + numbers into the final Canonical Repair record
    SolutionCriticAgent    -> QA gate: verdict per solution; may send the context BACK to the
                              analyst for one bounded revision round (self-correcting loop)
    CostSavingsAgent       -> quantifies $ saved by converging a group to its canonical cost
    RegionDelegatorAgent   -> localizes the national solution to a dealer's state (inference time)

The Orchestrator constructs the tools (src/tools) once and threads the context through the chain.
"""
from .context import RepairContext
from .grouping_agent import GroupingAgent
from .consensus_agent import ConsensusAgent
from .repair_analyst_agent import RepairAnalystAgent
from .solution_assembly_agent import SolutionAssemblyAgent
from .solution_critic_agent import SolutionCriticAgent
from .cost_savings_agent import CostSavingsAgent
from .region_delegator_agent import RegionDelegatorAgent
from .orchestrator import Orchestrator

__all__ = ["RepairContext", "GroupingAgent", "ConsensusAgent", "RepairAnalystAgent",
           "SolutionAssemblyAgent", "SolutionCriticAgent", "CostSavingsAgent",
           "RegionDelegatorAgent", "Orchestrator"]
