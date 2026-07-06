"""Error State Tensor E = [drift, echo, cascade, role, resource].

Replaces the scalar Δe = w1*(1-fp) + w2*mismatch + w3*echo with an
unreduced 5-dimensional vector. No weighted sum ever collapses these
into one number — comparison/decisioning happens via gates, banding,
and rule-table matching on the vector itself (see ESCALATION.md).

E_diverge (NLI contradiction) was scoped and dropped — cost/complexity
not justified for the hackathon window.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .embeddings import cosine
from .role_features import behavioral_profile, role_error


@dataclass
class ErrorTensor:
    drift: float = 0.0
    echo: float = 0.0
    cascade: float = 0.0
    role: float = 0.0
    resource: float = 0.0
    # per-node breakdown, for directive targeting / escalation case records
    node_role_excess: Dict[str, Dict[str, float]] = field(default_factory=dict)
    node_drift: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        return {
            "drift": round(self.drift, 4),
            "echo": round(self.echo, 4),
            "cascade": round(self.cascade, 4),
            "role": round(self.role, 4),
            "resource": round(self.resource, 4),
        }

    def vector(self) -> List[float]:
        """Ordered vector for similarity math. Order is fixed everywhere."""
        return [self.drift, self.echo, self.cascade, self.role, self.resource]

    DIMS = ("drift", "echo", "cascade", "role", "resource")


# ---------------------------------------------------------------------------
# Dimension computation
# ---------------------------------------------------------------------------

def compute_drift(
    dag, trace, weights: Dict[str, float], alpha: float = 0.5
) -> Tuple[float, Dict[str, float]]:
    """E_drift: centrality-weighted blend of local (output vs intent) and
    global (output vs task target) cosine deviation."""
    by_id = {r.node_id: r for r in trace.results}
    total_w = sum(weights.get(n.node_id, 1.0) for n in dag.nodes) or 1.0
    total = 0.0
    per_node: Dict[str, float] = {}
    target_emb = dag.task_embedding

    for n in dag.nodes:
        r = by_id.get(n.node_id)
        if not r or not r.output_embedding:
            continue
        intent_emb = n.expected_output_fingerprint or r.output_embedding
        local = 1.0 - cosine(r.output_embedding, intent_emb)
        glob = 1.0 - cosine(r.output_embedding, target_emb) if target_emb else local
        node_drift = alpha * local + (1 - alpha) * glob
        w = weights.get(n.node_id, 1.0)
        total += w * node_drift
        per_node[n.node_id] = round(node_drift, 4)

    return total / total_w, per_node


def compute_echo(dag, trace, weights: Dict[str, float], cosine_threshold: float) -> float:
    """E_echo: mean off-diagonal similarity among sibling-node outputs
    (same dependency set). Reuses the existing sibling-pairing logic."""
    by_id = {r.node_id: r for r in trace.results}
    nodes = dag.nodes
    sims: List[float] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if set(nodes[i].dependencies) == set(nodes[j].dependencies) and nodes[i].dependencies:
                a, b = by_id.get(nodes[i].node_id), by_id.get(nodes[j].node_id)
                if a and b and a.output_embedding and b.output_embedding:
                    sims.append(cosine(a.output_embedding, b.output_embedding))
    if not sims:
        return 0.0
    return statistics.fmean(sims)


def compute_cascade(
    dag, trace, weights: Dict[str, float], node_drift: Dict[str, float], gamma: float = 1.5
) -> float:
    """E_cascade: max over root nodes of distance-decayed drift propagation
    to descendants, weighted by descendant-output similarity to root.

    Gated on actual node errors -- if nothing failed, cascade is 0.0.
    Without this gate, high semantic similarity between ancestor/descendant
    nodes (expected on any coherent run) registers as cascade even when
    every node succeeded, which is a false positive by construction.
    """
    by_id = {r.node_id: r for r in trace.results}

    # gate: no errors anywhere → no cascade to measure
    if not any(r.error for r in trace.results):
        return 0.0

    children: Dict[str, List[str]] = {n.node_id: [] for n in dag.nodes}
    for n in dag.nodes:
        for dep in n.dependencies:
            if dep in children:
                children[dep].append(n.node_id)

    def descendants_with_dist(root: str) -> List[Tuple[str, int]]:
        out, frontier, dist = [], [root], 0
        seen = {root}
        while frontier:
            dist += 1
            nxt = []
            for nid in frontier:
                for c in children.get(nid, []):
                    if c not in seen:
                        seen.add(c)
                        out.append((c, dist))
                        nxt.append(c)
            frontier = nxt
        return out

    best = 0.0
    for n in dag.nodes:
        root_r = by_id.get(n.node_id)
        if not root_r or not root_r.output_embedding:
            continue
        score = 0.0
        for desc_id, dist in descendants_with_dist(n.node_id):
            desc_r = by_id.get(desc_id)
            if not desc_r or not desc_r.output_embedding:
                continue
            sim = cosine(root_r.output_embedding, desc_r.output_embedding)
            d_local = node_drift.get(desc_id, 0.0)
            score += (d_local * sim) / (gamma ** max(0, dist - 1))
        best = max(best, score)
    return best


def compute_role(
    dag, trace, weights: Dict[str, float], bands: Dict
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """E_role: centrality-weighted sum of band-excess across nodes."""
    by_id = {r.node_id: r for r in trace.results}
    total_w = sum(weights.get(n.node_id, 1.0) for n in dag.nodes) or 1.0
    total = 0.0
    per_node: Dict[str, Dict[str, float]] = {}

    for n in dag.nodes:
        r = by_id.get(n.node_id)
        if not r:
            continue
        parent_outputs = [
            by_id[d].output for d in n.dependencies if d in by_id and by_id[d].output
        ]
        sigma = behavioral_profile(r.output, parent_outputs)
        err, excess = role_error(sigma, bands, n.roles.functional)
        w = weights.get(n.node_id, 1.0)
        total += w * err
        if any(v > 0 for v in excess.values()):
            per_node[n.node_id] = excess

    return total / total_w, per_node


def compute_resource(
    total_tokens: int, wallclock_s: float, cost_max: float, latency_max: float,
    tokens_per_usd: float = 1_000_000, lambda_cost: float = 0.5, lambda_lat: float = 0.5,
) -> float:
    """E_resource: zero under budget, linear+quadratic penalty past ceiling.
    Bug fix vs original spec: original max(x, x^2) penalized x<1 (under
    budget) at value x, which punishes efficiency. This version is zero
    for x<=1.
    """
    cost_usd = total_tokens / tokens_per_usd
    x_cost = cost_usd / max(1e-9, cost_max)
    x_lat = wallclock_s / max(1e-9, latency_max)

    def penalty(x: float) -> float:
        over = max(0.0, x - 1.0)
        return over + over ** 2

    return lambda_cost * penalty(x_cost) + lambda_lat * penalty(x_lat)


# ---------------------------------------------------------------------------
# Gates / dedup
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    resource_gate_fired: bool = False
    cascade_drift_dedup_applied: bool = False


def apply_gates(tensor: ErrorTensor, cost_max: float, latency_max: float,
                 cascade_threshold: float, drift_node_threshold: float) -> GateResult:
    """Mutates tensor in place (suppresses double-counted drift on cascade
    roots) and reports which gates fired. Resource gate is checked by the
    caller (it short-circuits the whole directive path, doesn't just mutate)."""
    gr = GateResult()

    if tensor.cascade > cascade_threshold:
        # find drift-elevated nodes that are plausibly the cascade root:
        # any node whose own drift is elevated AND has descendants in the trace.
        elevated = [
            nid for nid, d in tensor.node_drift.items() if d > drift_node_threshold
        ]
        if elevated:
            # suppress the elevated drift's contribution to the graph-level
            # drift aggregate -- attribute fault to cascade instead.
            suppressed_total = sum(tensor.node_drift[nid] for nid in elevated)
            n = max(1, len(tensor.node_drift))
            tensor.drift = max(0.0, tensor.drift - suppressed_total / n)
            gr.cascade_drift_dedup_applied = True

    return gr


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------

class HysteresisTracker:
    """Per-(node, dimension) one-iteration grace margin after a refine.

    Only ever REMOVES a dimension from "still violated" -- never discounts
    a dimension that hasn't just been refined. State lives for the duration
    of one run (reset per orchestrator.run() call).
    """

    def __init__(self, margin: float = 0.15):
        self.margin = margin
        self._just_refined: Dict[Tuple[str, str], bool] = {}

    def mark_refined(self, node_id: str, dim: str) -> None:
        self._just_refined[(node_id, dim)] = True

    def clear(self) -> None:
        self._just_refined.clear()

    def effective_threshold(self, node_id: str, dim: str, base_threshold: float) -> float:
        if self._just_refined.get((node_id, dim)):
            return base_threshold * (1 + self.margin)
        return base_threshold

    def advance_iteration(self) -> None:
        """Call once per iteration AFTER threshold checks: clears grace so
        next iteration's checks (for dims not freshly refined) use base."""
        self._just_refined.clear()


# ---------------------------------------------------------------------------
# Ranking for directive construction
# ---------------------------------------------------------------------------

@dataclass
class ViolatedDim:
    dim: str
    excess: float  # value - threshold, >0 only


def rank_violations(
    tensor: ErrorTensor, thresholds: Dict[str, float],
    hysteresis: Optional[HysteresisTracker] = None,
    primary_node: Optional[str] = None,
) -> List[ViolatedDim]:
    """Returns real (post-hysteresis) violations sorted by magnitude desc.
    Hysteresis only filters dims that were *just* refined; everything else
    checks against base threshold, full stop.
    """
    vals = tensor.as_dict()
    out: List[ViolatedDim] = []
    for dim in ErrorTensor.DIMS:
        base = thresholds.get(dim, 0.4)
        eff = base
        if hysteresis and primary_node:
            eff = hysteresis.effective_threshold(primary_node, dim, base)
        if vals[dim] > eff:
            out.append(ViolatedDim(dim=dim, excess=round(vals[dim] - base, 4)))
    out.sort(key=lambda v: v.excess, reverse=True)
    return out


def cluster_primary_secondary(
    violations: List[ViolatedDim], cluster_ratio: float = 0.6
) -> Tuple[Optional[ViolatedDim], List[ViolatedDim]]:
    """Splits ranked violations into (primary, secondary_list).
    Secondary dims are those within cluster_ratio of the max excess --
    i.e. close enough to the worst offender to deserve co-mention.
    If nothing clears 0.6x the max, only primary is returned.
    """
    if not violations:
        return None, []
    primary = violations[0]
    threshold = primary.excess * cluster_ratio
    secondary = [v for v in violations[1:] if v.excess >= threshold]
    return primary, secondary