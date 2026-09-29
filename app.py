"""
Canonical Repair Profiler - dashboard (Streamlit).

A lightweight, presentable UI over the REAL agent pipeline (no mockups):
  Tab 1  "Claim Assistant"  -> a technician/agent narrows a claim with cascading dropdowns
                               (vehicle -> causal part -> repair), localizes cost to their state,
                               and sees the consensus solution + the savings vs costlier claims.
  Tab 2  "Agent Pipeline"   -> run the multi-agent chain on a repair group, step by step, live.
  Tab 3  "About"            -> architecture + business value.

Run:
    streamlit run app.py            (or: python cli.py dash)
Offline:
    CRP_USE_SAMPLE=1 streamlit run app.py   (uses the bundled sample; no BigQuery/Vertex data reads)
Prereq for live cloud calls: `gcloud auth login` (Vertex Gemini + BigQuery use the refreshing token).
"""
import numpy as np
import pandas as pd
import streamlit as st

from src import config as C
from src.tools import VertexGeminiTool, DataStore, StatsTool, RegionIndexTool
from src.agents.context import RepairContext
from src.agents.consensus_agent import ConsensusAgent
from src.agents.repair_analyst_agent import RepairAnalystAgent
from src.agents.solution_assembly_agent import SolutionAssemblyAgent
from src.agents.solution_critic_agent import SolutionCriticAgent, verdict_summary
from src.agents.cost_savings_agent import CostSavingsAgent
from src.agents.region_delegator_agent import RegionDelegatorAgent
from src.agents.infer import ClaimMatchAgent

CRL = C.SAMPLE_CANONICAL_REPAIRS if C.use_sample() else C.CANONICAL_REPAIRS

# Curated demo repair groups (signature_id -> label) for the live pipeline tab.
DEMO_SIGNATURES = {
    32954: "F-150 [15-20] - Engine Long Block - 961 claims - ~$8.9k - single repair",
    40109: "Super Duty [17-22] - Oil Pan - LLM SPLIT: reseal (~$1.6k) vs replace (~$2.6k)",
    71778: "Transit Connect [13-24] - Transmission - 172 claims - single repair",
}

st.set_page_config(page_title="Canonical Repair Profiler", page_icon="wrench", layout="wide")


# ---------- cached resources / data ----------
@st.cache_resource(show_spinner=False)
def _agents():
    gemini = VertexGeminiTool()
    region_tool = RegionIndexTool()
    return {
        "consensus": ConsensusAgent(StatsTool()),
        "analyst": RepairAnalystAgent(gemini),
        "assembly": SolutionAssemblyAgent(StatsTool()),
        "critic": SolutionCriticAgent(),
        "savings": CostSavingsAgent(),
        "region_tool": region_tool,
        "region": RegionDelegatorAgent(region_tool),
        "matcher": ClaimMatchAgent(DataStore(), gemini),
        "crl": pd.read_csv(CRL),
    }


@st.cache_data(show_spinner=False)
def _golden() -> pd.DataFrame:
    path = C.SAMPLE_GOLDEN_SOLUTIONS if C.use_sample() else C.GOLDEN_SOLUTIONS
    df = pd.read_csv(path)
    if "usable" in df.columns:
        df = df[df["usable"] == True]  # noqa: E712
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _group_savings() -> pd.DataFrame:
    try:
        return pd.read_csv(C.GROUP_SAVINGS)
    except (FileNotFoundError, OSError):
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_group_claims(signature_id: int) -> pd.DataFrame:
    if C.use_sample():
        claims = pd.read_csv(C.SAMPLE_CLAIMS)
        claims = claims[claims["signature_id"] == signature_id].copy()
    else:
        sig = pd.read_parquet(C.CLAIM_SIGNATURES, columns=["request_r", "signature_id"])
        sig["request_r"] = pd.to_numeric(sig["request_r"], errors="coerce").astype("Int64")
        members = sig[sig["signature_id"] == signature_id]["request_r"]
        rp = pd.read_parquet(C.REPAIR_PROFILE,
                             columns=["request_r", "paws_comment_trail", C.COST_COL, C.LABOR_COL])
        rp["request_r"] = pd.to_numeric(rp["request_r"], errors="coerce").astype("Int64")
        claims = rp[rp["request_r"].isin(members)].copy()
    for c in (C.COST_COL, C.LABOR_COL):
        claims[c] = pd.to_numeric(claims[c], errors="coerce")
    claims.loc[(claims[C.COST_COL] <= 0) | (claims[C.COST_COL] >= C.COST_CAP), C.COST_COL] = np.nan
    return claims


def _fmt(x, prefix="$"):
    return f"{prefix}{x:,.0f}" if isinstance(x, (int, float)) and x == x else "-"


def _solution_card(s: dict, region_state: str = ""):
    st.markdown(f"#### {s.get('repair_name','Canonical Repair')}")
    consensus = s.get("consensus")
    support = s.get("support", s.get("tier_n_claims"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Labor (median)", f"{s.get('labor_med_hrs','-')} hrs")

    # region-localized cost when a state is selected, else national
    if region_state and s.get("cost_med_regional") is not None:
        c2.metric(f"Cost in {region_state}", _fmt(s.get("cost_med_regional")),
                  delta=_fmt(s["cost_med_regional"] - s["cost_med"], "$") + " vs national",
                  help=f"IQR \\${s.get('cost_q25_regional')}-\\${s.get('cost_q75_regional')} "
                       f"(index {s.get('region_cost_index')}x)")
    else:
        c2.metric("Cost (median)", _fmt(s.get("cost_med")),
                  help=f"IQR \\${s.get('cost_q25')}-\\${s.get('cost_q75')}")

    if consensus is not None:
        c3.metric("Consensus", f"{consensus:.0%}", help="how strongly the claims agree on cost & labor")
    else:
        c3.metric("Confidence", f"{s.get('confidence','-')}")
    c4.metric("Support", f"{support} claims", help="historical claims backing this repair")

    st.markdown(f"**Unified Diagnosis** _(dealer dropdown selection)_  \n{s.get('unified_diagnosis','')}")
    st.markdown(f"**Suggested Correction**  \n{s.get('suggested_correction','')}")
    if s.get("typical_parts"):
        st.markdown(f"**Typical Parts:** {s.get('typical_parts')}")

    # per-repair savings vs costlier claims in the same group (well-scoped; not a top-line claim)
    sav = s.get("savings_per_claim")
    if sav:
        st.success(
            f"Standardizing this repair to the consensus saves ~{_fmt(sav)}/claim "
            f"({s.get('savings_pct_claims_above','?')}% of claims currently run above the consensus cost).")
    if region_state and s.get("region_note"):
        st.caption(s["region_note"])
    if s.get("cost_tier") and s.get("cost_tier") != "all":
        rationale = str(s.get("split_rationale", "")).replace("$", "\\$")
        st.caption(f"cost tier: {s.get('cost_tier')}  -  split rationale: {rationale}")


def _enrich(sol: dict, region_state: str, A) -> dict:
    """Attach group savings + region localization to a library solution row (inference-time agents)."""
    out = dict(sol)
    gsav = _group_savings()
    if not gsav.empty and "signature_id" in gsav.columns:
        hit = gsav[gsav["signature_id"] == sol.get("signature_id")]
        if len(hit):
            r = hit.iloc[0]
            out["savings_per_claim"] = float(r["savings_per_claim"])
            out["savings_avoidable_overspend"] = float(r["avoidable_overspend"])
            out["savings_pct_claims_above"] = float(r["pct_claims_above"])
            out["savings_n_claims"] = int(r["n_claims"])
    if region_state:
        out = A["region"].adjust_solution(out, region_state)
    return out


# ---------- header ----------
st.title("Canonical Repair Profiler")
st.caption("Turning 940k noisy Ford warranty claims into clean, confidence-scored Canonical Repairs - "
           "powered by Vertex AI Gemini + BigQuery.")
if C.use_sample():
    st.info("Running in OFFLINE SAMPLE mode (CRP_USE_SAMPLE=1): data is the bundled sample slice.")
h1, h2, h3, h4 = st.columns(4)
h1.metric("Unified claims", "938,307")
h2.metric("Canonical Repairs", "6,654")
h3.metric("Golden solutions", "5,070")
h4.metric("Avoidable overspend", "18.6%",
          help="Share of historical spend in scope above each repair's consensus cost - "
               "the measured bottom-up savings ceiling (see README).")

A = _agents()
gold = _golden()
tab_infer, tab_pipeline, tab_about = st.tabs(
    ["1. Claim Assistant", "2. Agent Pipeline (live)", "About"])

# ===================== TAB 1: CLAIM ASSISTANT (technician dropdowns) =====================
with tab_infer:
    st.subheader("Claim Assistant - find the right repair, cost & labor in seconds")
    st.caption("The dealer / assessing-agent workflow: narrow the claim with dropdowns, localize the "
               "cost to your state, and get the consensus diagnosis, correction, parts and labor.")

    cc1, cc2, cc3 = st.columns(3)
    vehicles = sorted(gold["vehicle_line"].dropna().unique())
    vehicle = cc1.selectbox("Vehicle line", vehicles,
                            index=min(2, len(vehicles) - 1) if vehicles else 0)
    parts = sorted(gold[gold["vehicle_line"] == vehicle]["causal_part"].astype(str).unique())
    part = cc2.selectbox("Causal part", parts)
    states = ["National (no state)"] + A["region_tool"].states()
    region_pick = cc3.selectbox("Dealer state (localize cost)", states)
    region_state = "" if region_pick.startswith("National") else region_pick

    cand = gold[(gold["vehicle_line"] == vehicle) & (gold["causal_part"].astype(str) == part)]
    cand = cand.reset_index(drop=True)

    concern = st.text_area("Optional: technician / customer description (helps pick among repairs)",
                           "", height=70, placeholder="e.g. oil leak from front cover, gasket seepage")

    if len(cand) == 0:
        st.warning("No canonical repair exists yet for this vehicle + part (would route to manual review).")
    elif len(cand) == 1:
        st.success("One canonical repair for this vehicle + part.")
        _solution_card(_enrich(cand.iloc[0].to_dict(), region_state, A), region_state)
    else:
        st.info(f"This vehicle + part maps to **{len(cand)} distinct canonical repairs** "
                "(e.g. a minor repair vs a full replacement). Pick one, or let the agent choose from your text.")
        if concern.strip() and st.button("Let the agent pick the best match", type="primary"):
            with st.spinner("Agent matching against the candidates (Vertex Gemini)..."):
                res = A["matcher"].match(vehicle, part, concern)
            if res.get("status") == "MATCHED":
                st.success(f"Agent picked match #{res['match_index']} "
                           f"(confidence {res.get('confidence_0_1')}) - {res.get('reason','')}")
                _solution_card(_enrich(res["solution"], region_state, A), region_state)
            else:
                st.warning(f"{res.get('status')}: {res.get('message', res.get('reason',''))}")
        for i, row in cand.iterrows():
            with st.expander(f"Repair option {i+1}: {row['repair_name']}  -  ~{_fmt(row['cost_med'])}"):
                _solution_card(_enrich(row.to_dict(), region_state, A), region_state)

# ===================== TAB 2: AGENT PIPELINE =====================
with tab_pipeline:
    st.subheader("Run the multi-agent pipeline on a repair group")
    st.caption("Watch the agent hand-off chain process a real group of claims into a Canonical Repair "
               "- including the agent's own decision on whether the group is one repair or several.")
    sid = st.selectbox("Repair group (signature):", list(DEMO_SIGNATURES),
                       format_func=lambda k: DEMO_SIGNATURES[k])
    state_pick = st.selectbox("Localize to state (optional):",
                              ["National (no state)"] + A["region_tool"].states(), key="pipe_state")
    pipe_state = "" if state_pick.startswith("National") else state_pick

    if st.button("Run agent pipeline", type="primary"):
        row = A["crl"][A["crl"]["signature_id"] == sid].iloc[0]
        ctx = RepairContext(signature_id=int(sid), vehicle_line=row["vehicle_line"],
                            causal_part=str(row["causal_part"]), archetype=int(row["arch"]),
                            archetype_theme=str(row.get("arch_theme", "")))
        with st.status("Running agent hand-off chain...", expanded=True) as status:
            st.write("**GroupingAgent** -> gathering the group's member claims...")
            ctx.claims = _load_group_claims(sid)
            st.write(f"  -> {len(ctx.claims):,} claims for {ctx.vehicle_line} / part {ctx.causal_part}")
            st.write("**ConsensusAgent** -> computing deterministic cost/labor consensus...")
            ctx = A["consensus"].enrich(ctx)
            st.write(f"  -> cost median ${ctx.consensus.get('cost_med')}, "
                     f"labor median {ctx.consensus.get('labor_med')} hrs")
            st.write("**RepairAnalystAgent** (Vertex Gemini) -> split decision + diagnosis + correction...")
            ctx = A["analyst"].analyze(ctx)
            st.write(f"  -> proposed **{ctx.llm_decision.get('n_repairs')} repair(s)** (validated next)")
            st.write("**SolutionAssemblyAgent** -> fusing LLM text with per-tier numbers + separation guard...")
            ctx = A["assembly"].assemble(ctx)
            st.write("**SolutionCriticAgent** -> QA gate: reviewing every solution (PASS/FLAG/FAIL)...")
            ctx = A["critic"].review(ctx)
            feedback = A["critic"].revision_request(ctx)
            if feedback:
                st.write("  -> critique handed BACK to the RepairAnalystAgent (one bounded revision round)...")
                ctx = A["analyst"].analyze(ctx, feedback=feedback)
                ctx = A["assembly"].assemble(ctx)
                ctx = A["critic"].review(ctx, revised=True)
            st.write(f"  -> verdicts: {verdict_summary(ctx.critique)}")
            st.write("**CostSavingsAgent** -> quantifying avoidable overspend vs consensus...")
            ctx = A["savings"].analyze(ctx)
            if pipe_state:
                st.write(f"**RegionDelegatorAgent** -> localizing cost to {pipe_state}...")
                ctx = A["region"].apply(ctx, pipe_state)
            st.write(f"  -> final: **{len(ctx.solutions)} Canonical Repair(s)**")
            status.update(label="Pipeline complete", state="complete", expanded=False)

        final_n = len(ctx.solutions)
        proposed = int(ctx.llm_decision.get("n_repairs", 1) or 1)
        rationale = (ctx.solutions[0].get("split_rationale", "") if ctx.solutions else "").replace("$", "\\$")
        if final_n > 1:
            st.info(f"The agent identified **{final_n} distinct repairs** in this group "
                    f"(minor repair vs full replacement). Rationale: {rationale}")
        elif proposed > 1:
            st.info("The agent considered splitting this group but **validated it as a single repair** - "
                    f"the cost tiers were not separated enough to be distinct. {rationale}")
        else:
            st.success("The agent determined this group is **one clean repair**.")
        for sol in ctx.solutions:
            _solution_card(sol, pipe_state)
            st.divider()

# ===================== TAB 3: ABOUT =====================
with tab_about:
    st.markdown(
        "**Canonical Repair Profiler (CRP)** distills noisy warranty claims into one consensus "
        "*Canonical Repair* per repair type - a clean training label and a dealer-selectable diagnosis.\n\n"
        "**Agent chain (shared `RepairContext`):** GroupingAgent -> ConsensusAgent -> RepairAnalystAgent "
        "(Vertex Gemini) -> SolutionAssemblyAgent -> SolutionCriticAgent (QA gate with a bounded "
        "revision loop back to the analyst) -> CostSavingsAgent -> RegionDelegatorAgent.\n\n"
        "**Tools (`src/tools`):** Vertex AI Gemini, BigQuery, a local DataStore, a deterministic "
        "StatsTool, and a RegionIndexTool (numbers come from data; only diagnosis/correction text is LLM).\n\n"
        "**Business value:** cleaner consensus ground truth makes existing production warranty AI "
        "products measurably better and enables new clean-data-only models. (A 50-year historical "
        "look-back at avoidable overspend is detailed in the README.)")
