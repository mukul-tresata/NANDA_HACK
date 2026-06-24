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
    ("research heavy task retrieve recent papers survey literature gather sources",
     "fan-out", 2, "Plan-then-Execute: parallel retrieval branches before synthesis"),
    ("multi step reasoning analysis decompose problem into subproblems",
     "hierarchical", 3, "Plan-then-Execute: hierarchical plan for decomposable reasoning"),
    ("latency sensitive fast turnaround quick answer time critical",
     "fan-out", 1, "Latency-Aware DAG: short critical path, parallel where possible"),
    ("tool use authorize external actions execute api calls side effects",
     "linear", 2, "KAIJU: intent-gated execution, decouple planning from tool firing"),
    ("verify validate check correctness audit cross examine claims",
     "join", 2, "adversarial verification: multiple checkers join into a verdict"),
    ("simple direct factual single answer lookup definition",
     "linear", 1, "trivial task: shallow linear plan, avoid over-planning"),
    ("strategy optimization improve policy meta learning adapt over runs",
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
