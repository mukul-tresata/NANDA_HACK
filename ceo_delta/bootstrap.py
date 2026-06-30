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
"""
from __future__ import annotations

from typing import List

from .embeddings import embed
from .handbook import Handbook
from .schemas import HandbookEntry

# task-class prototypes -> recommended (topology, depth) from the papers
_SEEDS = [
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
    for summary, topo, depth, revision in _SEEDS:
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
