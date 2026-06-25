"""Core data structures: nodes, DAGs, traces, handbook entries, agent cards.

v0.3 — AgentCard schema.

AgentCard is a single document per agent carrying both static identity and
dynamic performance history. It is the unit NANDA registers and propagates.

    Static section  — set at registration, rarely changes
    Dynamic section — updated by Delta after every run

CEO queries AgentRegistry (local store, NANDA-backed in production) to resolve
each DAG node to a specific agent. The query filters by:
    1. functional role match
    2. security_level >= task security requirement
    3. confidence floor (min runs before card is trusted)
    4. trust_score above threshold

Delta writes PerformanceRecord entries to the card after each audit.
The card's trust_score and per-role stats are recomputed from these records.

Node gains one new field: assigned_agent_id — the card CEO resolved for it.
Kernel reads this to inject card context into the execution prompt.
"""
from __future__ import annotations

import math
import statistics
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Existing core schemas (unchanged except Node gains assigned_agent_id)
# ---------------------------------------------------------------------------

@dataclass
class Roles:
    structural: str = "linear"      # fan-out / join / linear / hub
    functional: str = "generic"     # retriever / synthesizer / verifier / ...
    epistemic: str = "generalist"   # specialist / generalist


@dataclass
class Why:
    task_type_recognized: str = ""
    topology_chosen: str = ""
    depth_chosen: int = 0
    alternatives_rejected: str = ""
    priors_used: str = ""
    exploratory: bool = False


@dataclass
class Node:
    node_id: str
    intent: str
    roles: Roles = field(default_factory=Roles)
    dependencies: List[str] = field(default_factory=list)
    why: Why = field(default_factory=Why)
    expected_output_fingerprint: List[float] = field(default_factory=list)
    assigned_agent_id: Optional[str] = None   # NEW — set by CEO via AgentRegistry


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
        """How many nodes depend on each node (blocked-by count).

        n1 -> n2, n1 -> n3, n2 -> n4:
            n1: 2  (n2 and n3 both depend on n1)
            n2: 1  (n4 depends on n2)
            n3: 0  (sink)
            n4: 0  (sink)
        """
        counts: Dict[str, int] = {n.node_id: 0 for n in self.nodes}
        for n in self.nodes:
            for dep in n.dependencies:
                if dep in counts:
                    counts[dep] += 1
        return counts

    def centrality_weights(self) -> Dict[str, float]:
        """Normalize in_degree to [0.5, 1.5].

        Hub node (max dependents) -> 1.5
        Sink node (no dependents) -> 0.5
        Linear scale between them.
        Never zero — even leaf failures matter.
        """
        deg = self.in_degree()
        if not deg:
            return {}
        max_deg = max(deg.values()) or 1
        return {nid: 0.5 + (d / max_deg) for nid, d in deg.items()}


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


# ---------------------------------------------------------------------------
# AgentCard schema (new)
# ---------------------------------------------------------------------------

@dataclass
class PerformanceRecord:
    """One run's worth of performance data for a specific agent on a specific
    node. Delta writes one of these per node per run.

    task_class    — from Research's structured brief (research-heavy, reasoning,
                    tool-use, verification, simple). Allows per-class stats.
    security_req  — security level the task required (0.0-1.0). Lets us track
                    whether this agent performs differently under high-stakes tasks.
    """
    run_id: str
    node_id: str
    task_class: str                 # research-heavy / reasoning / tool-use / verification / simple
    functional_role: str            # retriever / synthesizer / verifier / generic
    fingerprint_match: float        # 0.0-1.0
    role_function_match: bool
    error: Optional[str]
    latency_s: float
    cost_tokens: int
    centrality_weight: float        # how critical this node was in its DAG
    security_req: float = 0.0       # security level required for this task
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentCardStatic:
    """Identity section — set at registration, changes rarely.

    security_clearance: float [0.0-1.0]
        How trusted this agent is for sensitive tasks.
        0.0 = untrusted / unknown
        0.5 = standard
        1.0 = fully verified / high-trust

    supported_roles: List[str]
        Functional roles this agent claims to support.
        CEO uses this for initial candidate filtering before checking dynamic stats.

    protocols: List[str]
        Communication protocols supported (e.g. "a2a", "mcp", "https").
        Used by NANDA for routing.
    """
    agent_id: str
    name: str
    description: str
    supported_roles: List[str]          # functional roles this agent handles
    security_clearance: float = 0.5     # [0.0-1.0] — how trusted for sensitive tasks
    protocols: List[str] = field(default_factory=lambda: ["https"])
    endpoint: str = ""                  # where to reach this agent
    registered_at: float = field(default_factory=time.time)


@dataclass
class AgentCardDynamic:
    """Performance section — updated by Delta after every run.

    trust_score: float [0.0-1.0]
        Recomputed from records. Weighted average of fingerprint match,
        role function match rate, and inverse error rate.
        Centrality-weighted — performance on hub nodes counts more.

    confidence: int
        Total number of runs contributing to this card's stats.
        CEO ignores cards below cfg.agent_card_min_confidence.

    per_role_stats: Dict[str, Dict]
        Per functional-role breakdown. CEO can compare "how does this agent
        perform specifically as a retriever" vs "as a synthesizer".

    per_class_stats: Dict[str, Dict]
        Per task-class breakdown. CEO can check "how does this agent perform
        on research-heavy tasks specifically".

    echo_incidents: int
        How many times this agent produced echoing output.
        A high echo count on a hub node is a strong negative signal.

    hub_failure_count: int
        How many times this agent failed on a hub node (centrality_weight > 1.0).
        Directly maps to cascade amplification risk.

    last_revision: str
        Most recent Delta feedback note about this agent.
    """
    trust_score: float = 0.5            # recomputed from records
    confidence: int = 0                 # total runs
    per_role_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_class_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    echo_incidents: int = 0
    hub_failure_count: int = 0
    last_revision: str = ""
    last_updated: float = field(default_factory=time.time)
    records: List[PerformanceRecord] = field(default_factory=list)


@dataclass
class AgentCard:
    """Single source of truth for an agent's identity and performance.

    One card per agent. CEO reads it. Delta writes to dynamic section.
    NANDA propagates the whole card — consumers see static + dynamic together.

    Usage:
        card = AgentCard(static=AgentCardStatic(...))
        card.update(record)         # Delta calls this after each run
        score = card.trust_score    # CEO reads this for filtering
    """
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
        return functional_role in self.static.supported_roles or "generic" in self.static.supported_roles

    def meets_security(self, required: float) -> bool:
        """True if this agent's clearance meets the task's security requirement."""
        return self.static.security_clearance >= required

    def update(self, record: PerformanceRecord) -> None:
        """Ingest one PerformanceRecord and recompute dynamic stats.
        Called by Delta after each node's execution is audited.
        """
        d = self.dynamic
        d.records.append(record)
        d.confidence += 1

        # recompute trust score from all records
        # centrality-weighted: hub node performance counts more
        total_w = sum(r.centrality_weight for r in d.records) or 1.0
        fp_score = sum(r.fingerprint_match * r.centrality_weight for r in d.records) / total_w
        role_score = sum(
            (1.0 if r.role_function_match else 0.0) * r.centrality_weight
            for r in d.records
        ) / total_w
        err_score = 1.0 - sum(
            (1.0 if r.error else 0.0) * r.centrality_weight
            for r in d.records
        ) / total_w
        # weighted blend: fingerprint 40%, role match 30%, no-error 30%
        d.trust_score = round(0.4 * fp_score + 0.3 * role_score + 0.3 * err_score, 3)

        # hub failure count
        if record.error and record.centrality_weight > 1.0:
            d.hub_failure_count += 1

        # per-role stats
        role = record.functional_role
        if role not in d.per_role_stats:
            d.per_role_stats[role] = {
                "runs": 0, "fp_mean": 0.0, "error_rate": 0.0, "role_match_rate": 0.0
            }
        rs = d.per_role_stats[role]
        n = rs["runs"]
        rs["fp_mean"] = round((rs["fp_mean"] * n + record.fingerprint_match) / (n + 1), 3)
        rs["error_rate"] = round((rs["error_rate"] * n + (1.0 if record.error else 0.0)) / (n + 1), 3)
        rs["role_match_rate"] = round(
            (rs["role_match_rate"] * n + (1.0 if record.role_function_match else 0.0)) / (n + 1), 3
        )
        rs["runs"] = n + 1

        # per-class stats
        cls = record.task_class
        if cls not in d.per_class_stats:
            d.per_class_stats[cls] = {"runs": 0, "fp_mean": 0.0, "error_rate": 0.0}
        cs = d.per_class_stats[cls]
        m = cs["runs"]
        cs["fp_mean"] = round((cs["fp_mean"] * m + record.fingerprint_match) / (m + 1), 3)
        cs["error_rate"] = round((cs["error_rate"] * m + (1.0 if record.error else 0.0)) / (m + 1), 3)
        cs["runs"] = m + 1

        d.last_updated = time.time()


# ---------------------------------------------------------------------------
# AgentRegistry — local store, NANDA-backed in production
# ---------------------------------------------------------------------------

class AgentRegistry:
    """Local in-memory registry of AgentCards.

    In production this syncs with NANDA's index. For the MVP it is an
    in-memory dict that survives the process lifetime.

    CEO calls resolve_for_node() to get the best agent for a given node.
    Delta calls record_performance() to update a card after a run.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg  # caller must pass cfg; use config.DEFAULT at call site
        self._cards: Dict[str, AgentCard] = {}
        self._seed_default_agents()

    def register(self, card: AgentCard) -> None:
        self._cards[card.agent_id] = card

    def get(self, agent_id: str) -> Optional[AgentCard]:
        return self._cards.get(agent_id)

    def resolve_for_node(
        self,
        functional_role: str,
        security_req: float = 0.0,
        task_class: str = "reasoning",
    ) -> Optional[AgentCard]:
        """Find the best agent for a node given role, security, and task class.

        Filtering pipeline (in order):
            1. supports the required functional role
            2. security_clearance >= security_req
            3. confidence >= cfg.agent_card_min_confidence (or is the only option)
            4. among survivors, rank by trust_score

        Returns None if no card passes all filters — caller falls back to
        generic LLM execution.
        """
        candidates = [
            c for c in self._cards.values()
            if c.supports_role(functional_role)
            and c.meets_security(security_req)
        ]
        if not candidates:
            return None

        # separate confident cards from cold-start ones
        confident = [
            c for c in candidates
            if c.confidence >= self.cfg.agent_card_min_confidence
        ]
        pool = confident if confident else candidates  # fall back to cold-start if no confident ones

        # rank by task-class-specific fp_mean if available, else overall trust_score
        def score(card: AgentCard) -> float:
            cls_stats = card.dynamic.per_class_stats.get(task_class)
            if cls_stats and cls_stats["runs"] >= 2:
                # penalize class-specific error rate
                return cls_stats["fp_mean"] * (1.0 - cls_stats["error_rate"])
            return card.trust_score

        return max(pool, key=score)

    def record_performance(self, agent_id: str, record: PerformanceRecord) -> None:
        """Delta calls this. Updates the card and recomputes dynamic stats."""
        card = self._cards.get(agent_id)
        if card:
            card.update(record)

    def summary(self) -> List[Dict[str, Any]]:
        """Human-readable summary for CLI display."""
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
        """Seed one generic agent per functional role so the system works
        out-of-the-box without a live NANDA registry.

        These represent the fallback LLM executor for each role.
        Security clearance 0.5 (standard). Trust starts at 0.5 (neutral).
        """
        defaults = [
            ("agent_retriever",   "Default Retriever",   ["retriever"],              0.5),
            ("agent_synthesizer", "Default Synthesizer", ["synthesizer"],             0.5),
            ("agent_verifier",    "Default Verifier",    ["verifier"],               0.7),
            ("agent_generic",     "Default Generic",     ["generic", "retriever",
                                                          "synthesizer", "verifier"], 0.5),
        ]
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
            self.register(card)


def dag_to_dict(d: DAG) -> Dict[str, Any]:
    return asdict(d)