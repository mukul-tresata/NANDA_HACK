"""Core data structures: nodes, DAGs, traces, handbook entries."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


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
    exploratory: bool = False       # cold-start / low-confidence flag


@dataclass
class Node:
    node_id: str
    intent: str
    roles: Roles = field(default_factory=Roles)
    dependencies: List[str] = field(default_factory=list)
    why: Why = field(default_factory=Why)
    expected_output_fingerprint: List[float] = field(default_factory=list)


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


@dataclass
class NodeResult:
    node_id: str
    intent: str
    output: str
    output_embedding: List[float]
    cost_tokens: int
    latency_s: float
    role_function_match: bool
    fingerprint_match: float        # cosine(expected, actual)
    gated: bool                     # passed intent gate
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
    """One learned record. Topology and depth tracked separately (shape can be
    right while depth is wrong). Multi-way conflict handled via vote tallies."""
    task_embedding: List[float]
    task_summary: str
    # vote tallies: option -> support count (multi-way, not binary)
    topology_votes: Dict[str, int] = field(default_factory=dict)
    depth_votes: Dict[str, int] = field(default_factory=dict)
    topology_chosen: str = ""
    depth_chosen: int = 0
    topology_outcome: str = ""      # good / poor / mixed
    revision: str = ""              # Delta's recommendation
    decision_points: List[str] = field(default_factory=list)
    confidence: int = 0             # number of runs supporting current revision
    contested: bool = False
    entry_id: str = field(default_factory=lambda: _id("hb"))
    updated_at: float = field(default_factory=time.time)


def dag_to_dict(d: DAG) -> Dict[str, Any]:
    out = asdict(d)
    return out
