"""
GroupingAgent - assigns warranty claims to a Repair Signature and gathers a group's member claims.

A Repair Signature = vehicle_line x causal_part x symptom_archetype (the unit a Canonical Repair is
built for). For the live/demo path this agent loads a single signature's claims from the DataStore
and seeds a RepairContext; the batch path reuses the persisted full grouping
(data/claim_signatures.parquet) produced by the offline grouping build.

Offline mode: when CRP_USE_SAMPLE=1 the agent reads the tiny bundled sample (data/sample/*.csv)
instead of the full parquet, so the whole pipeline runs with no BigQuery/Vertex data access.

HAND-OFF: writes ctx.claims, then passes ctx downstream to the ConsensusAgent.
"""
import numpy as np
import pandas as pd

from .context import RepairContext
from .. import config as C


class GroupingAgent:
    def __init__(self, datastore, repair_profile_path=C.REPAIR_PROFILE,
                 signatures_path=C.CLAIM_SIGNATURES, crl_path=C.CANONICAL_REPAIRS):
        self.ds = datastore
        self.sample = C.use_sample()
        self.repair_profile_path = repair_profile_path
        self.signatures_path = signatures_path
        self.crl_path = C.SAMPLE_CANONICAL_REPAIRS if self.sample else crl_path
        # Optional batch acceleration, populated by preload(): {signature_id: claims DataFrame} plus
        # the CRL indexed by signature_id. The live one-signature path never needs them.
        self._claims_index = None
        self._crl_index = None

    def preload(self, signature_ids) -> int:
        """Load the CRL and every requested signature's member claims into memory in ONE pass.

        run_one() re-reads the CRL and the full repair-profile parquet per signature, which is fine
        live but prohibitive for a multi-thousand-signature batch. Rows receive the same numeric
        coercion and cost-cap cleaning as _load_claims, so downstream agents see identical data
        either way. Returns the number of signatures indexed.
        """
        wanted = {int(s) for s in signature_ids}
        crl = self.ds.read_csv(self.crl_path)
        self._crl_index = crl[crl["signature_id"].isin(wanted)].set_index("signature_id")
        if self.sample:
            claims = self.ds.read_csv(C.SAMPLE_CLAIMS)
            claims = claims[claims["signature_id"].isin(wanted)].copy()
        else:
            sig = self.ds.read_parquet(self.signatures_path, columns=["request_r", "signature_id"])
            sig["request_r"] = pd.to_numeric(sig["request_r"], errors="coerce").astype("Int64")
            sig = sig[sig["signature_id"].isin(wanted)]
            rp = self.ds.read_parquet(
                self.repair_profile_path,
                columns=["request_r", "paws_comment_trail", C.COST_COL, C.LABOR_COL])
            rp["request_r"] = pd.to_numeric(rp["request_r"], errors="coerce").astype("Int64")
            claims = sig.merge(rp, on="request_r", how="inner")
        for c in (C.COST_COL, C.LABOR_COL):
            claims[c] = pd.to_numeric(claims[c], errors="coerce")
        claims.loc[(claims[C.COST_COL] <= 0) | (claims[C.COST_COL] >= C.COST_CAP), C.COST_COL] = np.nan
        self._claims_index = {int(s): g.copy() for s, g in claims.groupby("signature_id")}
        return len(self._claims_index)

    def load_signature(self, signature_id: int) -> RepairContext:
        """Build a RepairContext for one signature: its keys, theme, and member claims."""
        if self._crl_index is not None and signature_id in self._crl_index.index:
            row = self._crl_index.loc[signature_id]
            ctx = RepairContext(
                signature_id=int(signature_id),
                vehicle_line=row["vehicle_line"], causal_part=str(row["causal_part"]),
                archetype=int(row["arch"]), archetype_theme=str(row.get("arch_theme", "")),
            )
            ctx.claims = self._load_claims(signature_id)
            return ctx
        crl = self.ds.read_csv(self.crl_path)
        match = crl[crl["signature_id"] == signature_id]
        if len(match) == 0:
            raise KeyError(f"signature_id {signature_id} not found in {self.crl_path}")
        row = match.iloc[0]
        ctx = RepairContext(
            signature_id=int(signature_id),
            vehicle_line=row["vehicle_line"], causal_part=str(row["causal_part"]),
            archetype=int(row["arch"]), archetype_theme=str(row.get("arch_theme", "")),
        )
        ctx.claims = self._load_claims(signature_id)
        return ctx

    def _load_claims(self, signature_id: int) -> pd.DataFrame:
        if self._claims_index is not None:
            # Preloaded batch path: already coerced and cost-capped by preload(). A signature whose
            # claims all fell out of the join gets an empty frame, same as the cold path would build.
            claims = self._claims_index.get(int(signature_id))
            if claims is None:
                claims = pd.DataFrame(columns=["request_r", "paws_comment_trail", C.COST_COL, C.LABOR_COL])
            return claims.copy()
        if self.sample:
            claims = self.ds.read_csv(C.SAMPLE_CLAIMS)
            claims = claims[claims["signature_id"] == signature_id].copy()
        else:
            sig = self.ds.read_parquet(self.signatures_path, columns=["request_r", "signature_id"])
            sig["request_r"] = pd.to_numeric(sig["request_r"], errors="coerce").astype("Int64")
            members = sig[sig["signature_id"] == signature_id]["request_r"]
            rp = self.ds.read_parquet(
                self.repair_profile_path,
                columns=["request_r", "paws_comment_trail", C.COST_COL, C.LABOR_COL])
            rp["request_r"] = pd.to_numeric(rp["request_r"], errors="coerce").astype("Int64")
            claims = rp[rp["request_r"].isin(members)].copy()
        for c in (C.COST_COL, C.LABOR_COL):
            claims[c] = pd.to_numeric(claims[c], errors="coerce")
        claims.loc[(claims[C.COST_COL] <= 0) | (claims[C.COST_COL] >= C.COST_CAP), C.COST_COL] = np.nan
        return claims
