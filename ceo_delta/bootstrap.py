"""Cold-start bootstrap (limitation #1).

On run 1 the handbook is empty: no priors, no similarity matches, demo looks
broken. Two-pronged fix:

  1. seed_handbook(): write synthetic, low-confidence entries distilled from the
     four papers that inspired the architecture (KAIJU, Latency-Aware DAG,
     POLARIS, Plan-then-Execute). These give CEO *something* to retrieve.
  2. CEO additionally runs in explicit exploratory mode for the first
     cfg.cold_start_runs runs (handled in ceo.py / orchestrator.py), flagging
     every WHY annotation low-confidence so Delta knows to weight them lightly.

Seed entries carry confidence=1 (clearly weak) so a single real run can
override them.

v2.1: same cold-start problem exists one level down -- escalations.jsonl
(the case-based precedent store that replaced delta_rules.py) is also
empty on run 1, with retrieval gated behind MIN_CASES per role anyway, so
seeding matters less for "does retrieval work" and more for "is there
*something* to look at on day one." seed_escalations() handles that
target. Kept in this file alongside seed_handbook() because both are the
same concern (cold start) on two different stores, not two unrelated
responsibilities -- splitting them would separate steps of one workflow
rather than separate genuinely independent things.
"""
from __future__ import annotations

from typing import List

from .embeddings import embed
from .handbook import Handbook
from .schemas import HandbookEntry
from .escalation import EscalationCase, EscalationStore, _case_id

# ---------------------------------------------------------------------------
# Handbook seeding (unchanged from v2.0)
# ---------------------------------------------------------------------------

# task-class prototypes -> recommended (topology, depth) from the papers
_HANDBOOK_SEEDS = [
    ("information_flow:divergent epistemic_stance:retrieval output_contract:artifact decomposability:independent",
     "fan-out", 2, "Plan-then-Execute: parallel retrieval branches before synthesis"),
    ("information_flow:recursive epistemic_stance:synthesis output_contract:artifact decomposability:coupled",
     "hierarchical", 3, "Plan-then-Execute: hierarchical plan for decomposable reasoning"),
    ("information_flow:sequential epistemic_stance:retrieval output_contract:artifact decomposability:independent",
     "fan-out", 1, "Latency-Aware DAG: short critical path, parallel where possible"),
    ("information_flow:sequential epistemic_stance:generation output_contract:artifact decomposability:coupled",
     "linear", 2, "KAIJU: intent-gated execution, decouple planning from tool firing"),
    ("information_flow:convergent epistemic_stance:verification output_contract:verification decomposability:coupled",
     "join", 2, "adversarial verification: multiple checkers join into a verdict"),
    ("information_flow:sequential epistemic_stance:retrieval output_contract:artifact decomposability:independent",
     "linear", 1, "trivial task: shallow linear plan, avoid over-planning"),
    ("information_flow:recursive epistemic_stance:generation output_contract:ranking decomposability:coupled",
     "hierarchical", 3, "POLARIS: meta-learner pattern, deeper plan to expose decision points"),
]


def seed_handbook(hb: Handbook) -> int:
    n = 0
    for summary, topo, depth, revision in _HANDBOOK_SEEDS:
        emb = embed(summary)
        entry = HandbookEntry(
            task_embedding=emb,
            task_summary=summary,
            topology_votes={topo: 1},
            depth_votes={str(depth): 1},
            topology_chosen=topo,
            depth_chosen=depth,
            topology_outcome="seed",
            revision=f"[SEED|low-confidence] {revision}",
            decision_points=["seeded from inspiring papers"],
            confidence=1,
            contested=False,
        )
        hb.entries.append(entry)
        n += 1
    return n


def ensure_seeded(hb: Handbook) -> None:
    if not hb.entries:
        seed_handbook(hb)


# ---------------------------------------------------------------------------
# Escalation case seeding (v2.1 -- new)
# ---------------------------------------------------------------------------

_ESCALATION_SEEDS = [
    # retriever: collapsed into narrative instead of expanding facts
    dict(
        node_context={"role": "retriever", "deps": "[]"},
        e_tensor_excess={"drift": 0.30, "echo": 0.0, "cascade": 0.0, "role": 0.55, "resource": 0.0},
        directive_issued={
            "action": "refine", "target": "<node>",
            "constraint": "retriever must expand entity count beyond parent set (comp>1.0); currently compressing into prose",
            "why": "retriever role collapsed into narrative summary instead of fact expansion",
        },
        case_summary="retriever produced narrative summary instead of expanded fact manifest",
    ),
    # synthesizer: failed to compress, just concatenated parents
    dict(
        node_context={"role": "synthesizer", "deps": "[n1,n2,n3]"},
        e_tensor_excess={"drift": 0.15, "echo": 0.0, "cascade": 0.0, "role": 0.40, "resource": 0.0},
        directive_issued={
            "action": "refine", "target": "<node>",
            "constraint": "synthesizer must compress upstream entities below 0.7 ratio and increase structural formatting",
            "why": "synthesizer concatenated parent outputs rather than cross-referencing and compressing",
        },
        case_summary="synthesizer failed to compress; output ~= concatenation of parent nodes",
    ),
    # verifier: generated new claims instead of auditing
    dict(
        node_context={"role": "verifier", "deps": "[n1]"},
        e_tensor_excess={"drift": 0.20, "echo": 0.0, "cascade": 0.0, "role": 0.50, "resource": 0.0},
        directive_issued={
            "action": "refine", "target": "<node>",
            "constraint": "verifier must cite parent claims explicitly and avoid generating new unverified statistics",
            "why": "verifier drifted into synthesis mode, producing confident ungrounded numbers",
        },
        case_summary="verifier generated new statistics instead of auditing parent claims (verifier drift failure)",
    ),
    # cascade: hub node error propagating downstream
    dict(
        node_context={"role": "synthesizer", "deps": "[n1,n2]"},
        e_tensor_excess={"drift": 0.10, "echo": 0.0, "cascade": 0.45, "role": 0.0, "resource": 0.0},
        directive_issued={
            "action": "replan", "target": "topology",
            "constraint": "isolate hub node with a guard/verifier node before synthesis; reduce fan-out blast radius",
            "why": "upstream hub failure propagating downstream to dependent synthesis node",
        },
        case_summary="hub-origin error cascading through dependency chain to synthesizer",
    ),
    # echo: siblings converging on identical output
    dict(
        node_context={"role": "retriever", "deps": "[]"},
        e_tensor_excess={"drift": 0.0, "echo": 0.50, "cascade": 0.0, "role": 0.0, "resource": 0.0},
        directive_issued={
            "action": "replan", "target": "topology",
            "constraint": "differentiate sibling node intents explicitly; reduce fan-out width if differentiation fails",
            "why": "parallel retriever nodes converging on near-identical content",
        },
        case_summary="sibling fan-out nodes producing near-identical outputs (echo collapse)",
    ),
]


def seed_escalations(store: EscalationStore) -> int:
    import time
    cases = []
    for spec in _ESCALATION_SEEDS:
        cases.append(EscalationCase(
            case_id=_case_id(),
            timestamp=time.time(),
            run_id="seed",
            task_species={
                "flow": "divergent", "epistemic_stance": "synthesis",
                "output_contract": "artifact", "complexity": "medium",
            },
            node_context=spec["node_context"],
            e_tensor_raw=spec["e_tensor_excess"],
            e_tensor_excess=spec["e_tensor_excess"],
            gates_applied={"resource_gate_fired": False, "cascade_drift_dedup_applied": False},
            directive_issued=spec["directive_issued"],
            case_summary=spec["case_summary"],
        ))
    store.seed_synthetic(cases)
    return len(cases)


def ensure_escalations_seeded(store: EscalationStore) -> None:
    if any(c.synthetic for c in store._cases):
        return  # already seeded
    seed_escalations(store)