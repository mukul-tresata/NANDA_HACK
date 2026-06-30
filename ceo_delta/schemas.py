"""Core structural data schemas and AgentRegistry logic.

Seeding specialized agents directly at min_confidence baseline to break
the monopolization trap from the generic agent entry.

v0.4 changes:
    - DeltaDirective dataclass: typed control signal from Delta to CEO
    - Why dataclass enriched with directive_received and directive_response
    - ExecutionTrace gains iteration field for multi-pass loop tracking
"""
from __future__ import annotations

import math
import statistics
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Delta -> CEO control signal
# ---------------------------------------------------------------------------

@dataclass
class DeltaDirective:
    action: Literal["surface", "refine", "replan"]
    reason: str                          # human-readable explanation
    replan_hint: str = ""                # injected into CEO prompt on replan
    refinement_targets: List[str] = field(default_factory=list)  # node ids to fix
    confidence: float = 1.0             # how confident Delta is in this directive
    iteration: int = 0                  # which loop iteration produced this
    escalated: bool = False             # True if LLM fallback was used


@dataclass
class TaskFingerprint:
    """Structural signature of a task, produced by Research.clarify().

    Shape axes (embedded -> drive handbook retrieval / CEO topology+depth priors):
        information_flow : divergent | convergent | sequential | recursive
        epistemic_stance : retrieval | synthesis | generation | verification
        output_contract  : artifact | comparison | verification | ranking
        decomposability   : independent | coupled

    Modifier axes (NOT embedded -> direct scalar/gate, never used for similarity):
        complexity        : low | medium | high      -> scales suggested depth
        domain_volatility : stable | evolving | contested -> gates mandatory verifier
    """
    information_flow: str = "sequential"
    epistemic_stance: str = "synthesis"
    output_contract: str = "artifact"
    decomposability: str = "coupled"
    complexity: str = "medium"
    domain_volatility: str = "stable"
    embedding: List[float] = field(default_factory=list)  # shape-axes embedding only

    def shape_string(self) -> str:
        """The string that gets embedded -- shape axes ONLY, never modifiers."""
        return (
            f"information_flow:{self.information_flow} "
            f"epistemic_stance:{self.epistemic_stance} "
            f"output_contract:{self.output_contract} "
            f"decomposability:{self.decomposability}"
        )

    def depth_cap(self) -> int:
        """Complexity scales suggested depth. Deterministic, not LLM-inferred."""
        return {"low": 2, "medium": 3, "high": 4}.get(self.complexity, 3)

    def requires_verifier(self) -> bool:
        """Volatility gates mandatory verification. Deterministic, not a suggestion."""
        return self.domain_volatility in ("evolving", "contested")

# ---------------------------------------------------------------------------
# Planning schemas
# ---------------------------------------------------------------------------

@dataclass
class Roles:
    structural: str = "linear"
    functional: str = "generic"
    epistemic: str = "generalist"


@dataclass
class Why:
    task_type_recognized: str = ""
    topology_chosen: str = ""
    depth_chosen: int = 0
    alternatives_rejected: str = ""
    priors_used: str = ""
    exploratory: bool = False
    directive_received: str = ""   # what Delta told CEO this iteration
    directive_response: str = ""   # how CEO interpreted and acted on it


@dataclass
class Node:
    node_id: str
    intent: str
    roles: Roles = field(default_factory=Roles)
    dependencies: List[str] = field(default_factory=list)
    why: Why = field(default_factory=Why)
    expected_output_fingerprint: List[float] = field(default_factory=list)
    assigned_agent_id: Optional[str] = None


@dataclass
class DAG:
    task: str
    task_embedding: List[float]
    topology: str
    depth: int
    nodes: List[Node]
    why_topology: str = ""
    why_depth: str = ""
    exploratory: bool = False
    dag_id: str = field(default_factory=lambda: _id("dag"))

    def node(self, nid: str) -> Optional[Node]:
        return next((n for n in self.nodes if n.node_id == nid), None)

    def in_degree(self) -> Dict[str, int]:
        counts: Dict[str, int] = {n.node_id: 0 for n in self.nodes}
        for n in self.nodes:
            for dep in n.dependencies:
                if dep in counts:
                    counts[dep] += 1
        return counts

    def centrality_weights(self) -> Dict[str, float]:
        deg = self.in_degree()
        if not deg:
            return {}
        max_deg = max(deg.values()) or 1
        return {nid: 0.5 + (d / max_deg) for nid, d in deg.items()}


# ---------------------------------------------------------------------------
# Execution schemas
# ---------------------------------------------------------------------------

@dataclass
class NodeResult:
    node_id: str
    intent: str
    output: str
    output_embedding: List[float]
    cost_tokens: int
    latency_s: float
    role_function_match: bool
    fingerprint_match: float
    gated: bool
    error: Optional[str] = None


@dataclass
class ExecutionTrace:
    dag_id: str
    task: str
    results: List[NodeResult]
    total_tokens: int
    wallclock_s: float
    started_at: float = field(default_factory=time.time)
    iteration: int = 0              # which loop iteration this trace belongs to


# ---------------------------------------------------------------------------
# Handbook schemas
# ---------------------------------------------------------------------------

@dataclass
class HandbookEntry:
    task_embedding: List[float]
    task_summary: str
    topology_votes: Dict[str, int] = field(default_factory=dict)
    depth_votes: Dict[str, int] = field(default_factory=dict)
    topology_chosen: str = ""
    depth_chosen: int = 0
    topology_outcome: str = ""
    revision: str = ""
    decision_points: List[str] = field(default_factory=list)
    confidence: int = 0
    contested: bool = False
    entry_id: str = field(default_factory=lambda: _id("hb"))
    updated_at: float = field(default_factory=time.time)
    # directive effectiveness tracking
    directive_outcomes: Dict[str, Dict] = field(default_factory=dict)
    # key: directive action+reason hash, value: {fires: int, fixed: int}


# ---------------------------------------------------------------------------
# AgentCard schemas
# ---------------------------------------------------------------------------

@dataclass
class PerformanceRecord:
    run_id: str
    node_id: str
    task_class: str
    functional_role: str
    fingerprint_match: float
    role_function_match: bool
    error: Optional[str]
    latency_s: float
    cost_tokens: int
    centrality_weight: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentCardStatic:
    agent_id: str
    name: str
    description: str
    supported_roles: List[str]
    security_clearance: float = 0.5
    protocols: List[str] = field(default_factory=lambda: ["https"])
    endpoint: str = ""
    registered_at: float = field(default_factory=time.time)


@dataclass
class AgentCardDynamic:
    trust_score: float = 0.5
    confidence: int = 0
    per_role_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_class_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    echo_incidents: int = 0
    hub_failure_count: int = 0
    last_revision: str = ""
    last_updated: float = field(default_factory=time.time)
    records: List[PerformanceRecord] = field(default_factory=list)


@dataclass
class AgentCard:
    static: AgentCardStatic
    dynamic: AgentCardDynamic = field(default_factory=AgentCardDynamic)

    @property
    def agent_id(self) -> str:
        return self.static.agent_id

    @property
    def trust_score(self) -> float:
        return self.dynamic.trust_score

    @property
    def confidence(self) -> int:
        return self.dynamic.confidence

    def supports_role(self, functional_role: str) -> bool:
        return (
            functional_role in self.static.supported_roles
            or "generic" in self.static.supported_roles
        )

    def meets_security(self, required: float) -> bool:
        return self.static.security_clearance >= required

    def update(self, record: PerformanceRecord) -> None:
        d = self.dynamic
        d.records.append(record)
        d.confidence += 1

        total_w = sum(r.centrality_weight for r in d.records) or 1.0
        fp_score = sum(r.fingerprint_match * r.centrality_weight for r in d.records) / total_w
        role_score = sum(
            (1.0 if r.role_function_match else 0.0) * r.centrality_weight
            for r in d.records
        ) / total_w
        err_score = 1.0 - sum(
            (1.0 if r.error else 0.0) * r.centrality_weight for r in d.records
        ) / total_w
        d.trust_score = round(0.4 * fp_score + 0.3 * role_score + 0.3 * err_score, 3)

        if record.error and record.centrality_weight > 1.0:
            d.hub_failure_count += 1

        role = record.functional_role
        if role not in d.per_role_stats:
            d.per_role_stats[role] = {
                "runs": 0, "fp_mean": 0.0,
                "error_rate": 0.0, "role_match_rate": 0.0,
            }
        rs = d.per_role_stats[role]
        n = rs["runs"]
        rs["fp_mean"] = round((rs["fp_mean"] * n + record.fingerprint_match) / (n + 1), 3)
        rs["error_rate"] = round(
            (rs["error_rate"] * n + (1.0 if record.error else 0.0)) / (n + 1), 3
        )
        rs["role_match_rate"] = round(
            (rs["role_match_rate"] * n + (1.0 if record.role_function_match else 0.0)) / (n + 1), 3
        )
        rs["runs"] = n + 1

        cls = record.task_class
        if cls not in d.per_class_stats:
            d.per_class_stats[cls] = {"runs": 0, "fp_mean": 0.0, "error_rate": 0.0}
        cs = d.per_class_stats[cls]
        m = cs["runs"]
        cs["fp_mean"] = round((cs["fp_mean"] * m + record.fingerprint_match) / (m + 1), 3)
        cs["error_rate"] = round(
            (cs["error_rate"] * m + (1.0 if record.error else 0.0)) / (m + 1), 3
        )
        cs["runs"] = m + 1

        d.last_updated = time.time()


class AgentRegistry:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self._cards: Dict[str, AgentCard] = {}
        self._seed_default_agents()

    def register(self, card: AgentCard) -> None:
        self._cards[card.agent_id] = card

    def get(self, agent_id: str) -> Optional[AgentCard]:
        return self._cards.get(agent_id)

    def resolve_for_node(
        self, functional_role: str, task_class: str = "reasoning"
    ) -> Optional[AgentCard]:
        candidates = [c for c in self._cards.values() if c.supports_role(functional_role)]
        if not candidates:
            return None

        min_conf = self.cfg.agent_card_min_confidence if self.cfg else 3
        confident = [c for c in candidates if c.confidence >= min_conf]
        pool = confident if confident else candidates

        def score(card: AgentCard) -> float:
            cls_stats = card.dynamic.per_class_stats.get(task_class)
            if cls_stats and cls_stats["runs"] >= 2:
                return cls_stats["fp_mean"] * (1.0 - cls_stats["error_rate"])
            return card.trust_score

        return max(pool, key=score)

    def record_performance(self, agent_id: str, record: PerformanceRecord) -> None:
        card = self._cards.get(agent_id)
        if card:
            card.update(record)

    def summary(self) -> List[Dict[str, Any]]:
        out = []
        for c in self._cards.values():
            out.append({
                "agent_id": c.agent_id,
                "name": c.static.name,
                "roles": c.static.supported_roles,
                "security": c.static.security_clearance,
                "trust": c.trust_score,
                "confidence": c.confidence,
                "hub_failures": c.dynamic.hub_failure_count,
                "per_role": c.dynamic.per_role_stats,
            })
        return out

    def _seed_default_agents(self) -> None:
        defaults = [
            ("agent_retriever",   "Default Retriever",   ["retriever"],                          0.5),
            ("agent_synthesizer", "Default Synthesizer", ["synthesizer"],                        0.5),
            ("agent_verifier",    "Default Verifier",    ["verifier"],                           0.7),
            ("agent_generic",     "Default Generic",
             ["generic", "retriever", "synthesizer", "verifier"],                                0.5),
        ]
        min_conf = self.cfg.agent_card_min_confidence if self.cfg else 3
        for aid, name, roles, sec in defaults:
            card = AgentCard(
                static=AgentCardStatic(
                    agent_id=aid,
                    name=name,
                    description=f"Default {name} agent (seeded)",
                    supported_roles=roles,
                    security_clearance=sec,
                )
            )
            card.dynamic.confidence = min_conf
            self.register(card)


def dag_to_dict(d: DAG) -> Dict[str, Any]:
    return asdict(d)