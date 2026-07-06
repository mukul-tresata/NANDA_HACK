"""v3.6 — warm-start planning from the cached best_plan (learning consume-side).

ef_store.upsert_case has always WRITTEN a per-fingerprint best_plan (the
lowest threshold-relative worst-excess plan ever seen for this species), but
nothing ever read it back — the descent's own memory of "what already worked"
was write-only. This module is the read side: it reconstructs a runnable DAG
from the exhaustive best_plan dict descent.py now persists, WITHOUT calling
the planning LLM at all.

Only structure is persisted (topology/depth/nodes/intents/roles/deps).
Runtime-derivable data -- expected_output_fingerprint embeddings and assigned
agents -- is deliberately NOT persisted and is recomputed here / by the
caller, exactly mirroring ceo.CEO._build_dag / _resolve_agents, so a
reconstructed DAG is indistinguishable in shape from one the CEO would have
planned fresh.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .embeddings import embed
from .schemas import DAG, Node, Roles, Why


def dag_from_best_plan(best_plan: Dict, task: str, task_embedding: List[float]) -> Optional[DAG]:
    """Rebuild a runnable DAG from a persisted exhaustive best_plan, or None
    if best_plan is empty/malformed (defensive -- callers fall back to a
    normal CEO.plan() in that case)."""
    if not best_plan:
        return None
    nodes_raw = best_plan.get("nodes") or []
    if not nodes_raw:
        return None

    topology = str(best_plan.get("topology", "linear"))
    depth = int(best_plan.get("depth", 1) or 1)
    why_topology = str(best_plan.get("why_topology", ""))
    why_depth = str(best_plan.get("why_depth", ""))

    nodes: List[Node] = []
    for nr in nodes_raw:
        intent = str(nr.get("intent", "")).strip() or "unspecified"
        functional_role = str(nr.get("functional", "generic"))
        # SAME rule as ceo._build_dag: synthesizer nodes fingerprint against
        # the parent task (task fidelity), everyone else against their own
        # intent string.
        fingerprint_target = task_embedding if functional_role == "synthesizer" else embed(intent)
        nodes.append(Node(
            node_id=str(nr.get("node_id") or f"n{len(nodes)+1}"),
            intent=intent,
            roles=Roles(
                structural=str(nr.get("structural", topology)),
                functional=functional_role,
                epistemic=str(nr.get("epistemic", "generalist")),
            ),
            dependencies=[str(d) for d in (nr.get("dependencies") or [])],
            why=Why(
                priors_used="warmstart",
                topology_chosen=topology,
                depth_chosen=depth,
            ),
            expected_output_fingerprint=fingerprint_target,
        ))

    return DAG(
        task=task,
        task_embedding=task_embedding,
        topology=topology,
        depth=depth,
        nodes=nodes,
        why_topology=why_topology,
        why_depth=why_depth,
        exploratory=False,
    )
