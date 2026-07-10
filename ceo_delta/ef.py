"""v3.0 — EF-driven planning adaptation. The shared coordinate system.

E and F live in ONE basis: [partition, flow, role, scale].

  F (RequiredStructure) — what structure the fingerprint REQUIRES. Derived
    deterministically from the TaskFingerprint (category -> target). No LLM.

  E (EFTensor) — how far the REALIZED plan+execution deviates from F, per
    axis. Computed from graph topology + output embeddings + behavioral
    bands. No LLM in the measurement path.

The verdict is a pure function of E (all axes below threshold -> good).
There is NO second error model. This is the whole point of v3.0: the error
tensor is the *dual of the fingerprint* — error is measured in fingerprint
space, so it is invariant to surface task phrasing.

Three of four axes (flow, partition, scale) are pure structure/embedding
math. Role reuses the existing behavioral bands. Drift is GONE from the
planning tensor — it was execution-quality geometry, not planning error.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .embeddings import cosine
from .error_tensor import compute_role

EF_AXES: Tuple[str, ...] = ("partition", "flow", "role", "scale")


# ---------------------------------------------------------------------------
# F — required structure (deterministic map from fingerprint)
# ---------------------------------------------------------------------------

@dataclass
class RequiredStructure:
    flow: str
    decomposability: str
    target_depth: int
    required_roles: Set[str]

    def as_dict(self) -> Dict:
        return {
            "flow": self.flow,
            "decomposability": self.decomposability,
            "target_depth": self.target_depth,
            "required_roles": sorted(self.required_roles),
        }


def compute_required(fp) -> RequiredStructure:
    """The fingerprint IS the specification. This reads it, deterministically."""
    stance = fp.epistemic_stance
    roles: Set[str] = set()
    if stance == "retrieval":
        roles = {"retriever", "synthesizer"}
    elif stance in ("synthesis", "generation"):
        roles = {"synthesizer"}
    elif stance == "verification":
        roles = {"verifier"}
    if fp.output_contract == "verification":
        roles.add("verifier")
    if fp.requires_verifier():
        roles.add("verifier")
    target_depth = fp.depth_cap()
    if fp.information_flow == "recursive":
        # deepen_recursive requires depth>=3 to approximate recursive flow; a
        # low complexity cap can otherwise set target_depth=2, making the
        # flow and scale axes mutually unsatisfiable (an F-level contradiction).
        target_depth = max(target_depth, 3)
    return RequiredStructure(
        flow=fp.information_flow,
        decomposability=fp.decomposability,
        target_depth=target_depth,
        required_roles=roles,
    )


# ---------------------------------------------------------------------------
# Graph signature — pure topology, deterministic
# ---------------------------------------------------------------------------

def _critical_path(dag) -> int:
    memo: Dict[str, int] = {}

    def depth(nid: str) -> int:
        if nid in memo:
            return memo[nid]
        node = dag.node(nid)
        if not node or not node.dependencies:
            memo[nid] = 1
        else:
            memo[nid] = 1 + max(depth(d) for d in node.dependencies)
        return memo[nid]

    return max((depth(n.node_id) for n in dag.nodes), default=0)


def graph_signature(dag) -> Dict:
    deps = {n.node_id: list(n.dependencies) for n in dag.nodes}
    ids = set(deps)
    indeg = {nid: 0 for nid in ids}
    outdeg = {nid: 0 for nid in ids}
    for nid, ds in deps.items():
        for d in ds:
            if d in ids:
                indeg[nid] += 1
                outdeg[d] += 1
    roots = [nid for nid in ids if indeg[nid] == 0]
    sinks = [nid for nid in ids if outdeg[nid] == 0]
    n = len(ids) or 1
    chain_nodes = sum(1 for nid in ids if len(deps[nid]) == 1)
    return {
        "root_count": len(roots),
        "sink_count": len(sinks),
        "max_in": max(indeg.values(), default=0),
        "max_out": max(outdeg.values(), default=0),
        "chain_ratio": round(chain_nodes / n, 3),
        "has_join": any(v >= 2 for v in indeg.values()),
        "depth": _critical_path(dag),
        "n_nodes": n,
    }


# ---------------------------------------------------------------------------
# E — per-axis deviation from F
# ---------------------------------------------------------------------------

def _flow_error(sig: Dict, req: RequiredStructure) -> float:
    """Deviation between realized DAG shape and the flow F requires. Pure graph."""
    f = req.flow
    if f == "divergent":
        pen = 0.0
        # Parallel gathering exists in EITHER form: multiple independent roots,
        # or a coordinator node fanning out to >=2 branches. Both are the same
        # information flow; penalizing the coordinator form was a false positive
        # (the metric counted roots but never checked branching).
        has_parallel_gather = sig["root_count"] >= 2 or sig["max_out"] >= 2
        if not has_parallel_gather:
            pen += 0.5   # no parallel gathering anywhere
        if not sig["has_join"]:
            pen += 0.5   # branches never converge
        return min(1.0, pen)
    if f == "convergent":
        pen = 0.0
        if sig["max_in"] < 2:
            pen += 0.6   # nothing actually merges
        if sig["sink_count"] > 1:
            pen += 0.4   # multiple unmerged endpoints
        return min(1.0, pen)
    if f == "sequential":
        pen = 0.0
        if sig["max_out"] > 1:
            pen += 0.5   # branching where a chain was required
        if sig["root_count"] > 1:
            pen += 0.5
        return min(1.0, pen)
    if f == "recursive":
        # nested/self-similar approximated as sufficient depth
        return 0.0 if sig["depth"] >= 3 else 1.0
    return 0.0


def _partition_error(dag, trace) -> Tuple[float, Dict]:
    """Redundancy pole of partition error: mean semantic overlap among
    sibling nodes (identical dependency set). Over-partition = siblings doing
    the same work. Pure embedding math."""
    by_id = {r.node_id: r for r in trace.results}
    nodes = dag.nodes
    sims: List[float] = []
    pairs: Dict[Tuple[str, str], float] = {}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            # Siblings = same dependency set. Root nodes (empty deps) ARE
            # siblings here: in a divergent plan the parallel fan-out roots are
            # the primary over-partition risk, so they must be checked. (The
            # old echo metric excluded roots to avoid false positives at
            # threshold 0.85; partition measures mean sibling cosine at a lower
            # threshold, where genuinely-differentiated roots sit safely below.)
            if set(nodes[i].dependencies) == set(nodes[j].dependencies):
                a, b = by_id.get(nodes[i].node_id), by_id.get(nodes[j].node_id)
                if a and b and a.output_embedding and b.output_embedding:
                    s = cosine(a.output_embedding, b.output_embedding)
                    sims.append(s)
                    pairs[(nodes[i].node_id, nodes[j].node_id)] = round(s, 4)
    if not sims:
        return 0.0, {}
    return statistics.fmean(sims), pairs


def _role_error(dag, trace, weights, bands, required_roles: Set[str]) -> Tuple[float, Dict]:
    """Behavioral-band error (existing compute_role) + penalty for a required
    functional role being entirely absent from the plan."""
    role, node_excess = compute_role(dag, trace, weights, bands)
    present = {n.roles.functional for n in dag.nodes}
    missing = [r for r in required_roles if r not in present]
    return min(1.0, role + 0.3 * len(missing)), node_excess


def _scale_error(sig: Dict, req: RequiredStructure) -> float:
    """Structural magnitude vs complexity-implied depth budget. Overage
    penalized hard, underage mildly. Pure graph.

    The asymmetry is deliberate, not an oversight: a plan that is one level
    shallower than target is treated as half as bad as one level deeper,
    because under-decomposing is the cheaper failure mode (fewer nodes, less
    coordination surface) and a shallow plan can still satisfy the task even
    if it doesn't fully exploit the target depth. One consequence to be aware
    of: on a target_depth=2 task, one level of underage yields error 0.25,
    which sits under ef_scale_threshold (0.34) and so never fires -- shallow
    plans against a depth-2 target are effectively unenforced on this axis.
    That's accepted, not hidden.
    """
    target = req.target_depth or 1
    realized = sig["depth"]
    if realized > target:
        return min(1.0, (realized - target) / target)
    return min(1.0, 0.5 * (target - realized) / target)


@dataclass
class EFTensor:
    partition: float = 0.0
    flow: float = 0.0
    role: float = 0.0
    scale: float = 0.0
    partition_pairs: Dict = field(default_factory=dict)
    role_excess: Dict = field(default_factory=dict)
    signature: Dict = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        return {
            "partition": round(self.partition, 4),
            "flow": round(self.flow, 4),
            "role": round(self.role, 4),
            "scale": round(self.scale, 4),
        }

    def worst_node(self, axis: str) -> str:
        """Which node the directive should target for this axis. Flow/scale
        are topology-level (no single node)."""
        if axis == "role" and self.role_excess:
            return max(self.role_excess, key=lambda n: sum(self.role_excess[n].values()))
        if axis == "partition" and self.partition_pairs:
            worst_pair = max(self.partition_pairs, key=self.partition_pairs.get)
            return worst_pair[1]
        return ""


def compute_ef(dag, trace, required: RequiredStructure, weights, role_bands) -> EFTensor:
    sig = graph_signature(dag)
    flow = _flow_error(sig, required)
    partition, pairs = _partition_error(dag, trace)
    role, role_excess = _role_error(dag, trace, weights, role_bands, required.required_roles)
    scale = _scale_error(sig, required)
    return EFTensor(
        partition=partition, flow=flow, role=role, scale=scale,
        partition_pairs=pairs, role_excess=role_excess, signature=sig,
    )


# ---------------------------------------------------------------------------
# Verdict — pure function of E (one error model, no parallel scoring)
# ---------------------------------------------------------------------------

def verdict_from_ef(ef: EFTensor, thresholds: Dict[str, float],
                    mixed_margin: float = 0.15) -> Tuple[str, bool]:
    d = ef.as_dict()
    excess = {ax: d[ax] - thresholds[ax] for ax in EF_AXES}
    worst = max(excess.values())
    n_over = sum(1 for e in excess.values() if e > 0)
    if worst <= 0:
        return "good", True
    if n_over == 1 and worst <= mixed_margin:
        return "mixed", False
    return "poor", False


def final_verdict(ef_dict, q_dict, ef_thresholds, q_thresholds, mixed_margin=0.15):
    """Combined verdict over the FULL basis (structural E axes + content Q axes),
    same uniform gate as verdict_from_ef -- ONE error model, extended. Structure
    and content are gated by the same rule; the delivered verdict now reflects
    answer quality, not just plan shape."""
    from .quality import Q_AXES
    excess = {ax: ef_dict[ax] - ef_thresholds[ax] for ax in EF_AXES}
    excess.update({ax: q_dict[ax] - q_thresholds[ax] for ax in Q_AXES})
    worst = max(excess.values())
    n_over = sum(1 for e in excess.values() if e > 0)
    if worst <= 0:
        return "good", True
    if n_over == 1 and worst <= mixed_margin:
        return "mixed", False
    return "poor", False
