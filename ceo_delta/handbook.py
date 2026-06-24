"""Planning Handbook — an in-memory vector DB of learned planning entries.

Two instances are created (one for CEO, one for Research). Delta writes to both.
Persists to JSON so learning survives across process runs.

Limitation #4 (multi-way conflict) is resolved here, not with a binary boolean:
each entry keeps *vote tallies* over topology and depth options. `resolve()`
declares a winner only when one option beats the runner-up by a margin AND the
distribution's normalized entropy is low enough; otherwise the entry is marked
`contested` and the leading option is exposed for transparency.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from .config import Config, DEFAULT
from .embeddings import cosine
from .schemas import HandbookEntry


def _entropy(counts: List[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    ps = [c / total for c in counts if c > 0]
    h = -sum(p * math.log(p) for p in ps)
    hmax = math.log(len(ps)) if len(ps) > 1 else 1.0
    return h / hmax if hmax > 0 else 0.0


class Handbook:
    def __init__(self, name: str, cfg: Config | None = None, path: Optional[str] = None):
        self.name = name
        self.cfg = cfg or DEFAULT
        self.path = path
        self.entries: List[HandbookEntry] = []
        if path and os.path.exists(path):
            self.load()

    # -- retrieval ------------------------------------------------------------
    def query(self, task_embedding: List[float], top_k: int | None = None
              ) -> List[Tuple[HandbookEntry, float]]:
        top_k = top_k or self.cfg.handbook_top_k
        scored = [(e, cosine(task_embedding, e.task_embedding)) for e in self.entries]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def best_match(self, task_embedding: List[float]) -> Tuple[Optional[HandbookEntry], float]:
        q = self.query(task_embedding, top_k=1)
        return (q[0][0], q[0][1]) if q else (None, 0.0)

    def density(self, task_embedding: List[float], radius: float = 0.6) -> int:
        """Neighborhood density == implicit confidence signal."""
        return sum(1 for e in self.entries if cosine(task_embedding, e.task_embedding) >= radius)

    # -- writing --------------------------------------------------------------
    def upsert_votes(self, task_embedding: List[float], task_summary: str,
                     topology: str, depth: int, outcome_good: bool,
                     revision: str = "", merge_radius: float = 0.8) -> HandbookEntry:
        """Add a vote for (topology, depth). Merges into a nearby entry if one
        exists, else creates a new one. Then re-resolves the conflict state."""
        entry, sim = self.best_match(task_embedding)
        if entry is None or sim < merge_radius:
            entry = HandbookEntry(task_embedding=task_embedding, task_summary=task_summary)
            self.entries.append(entry)

        if outcome_good:
            entry.topology_votes[topology] = entry.topology_votes.get(topology, 0) + 1
            dk = str(depth)
            entry.depth_votes[dk] = entry.depth_votes.get(dk, 0) + 1
        else:
            # a poor outcome is a negative signal: it does NOT add a vote, but it
            # decays the option that was used so the tally tracks reality
            entry.topology_votes[topology] = max(0, entry.topology_votes.get(topology, 0) - 1)

        if revision:
            entry.revision = revision
        self.resolve(entry)
        entry.updated_at = _now()
        return entry

    # -- limitation #4: multi-way conflict resolution -------------------------
    def resolve(self, entry: HandbookEntry) -> None:
        topo_winner, topo_settled = self._resolve_dist(entry.topology_votes)
        depth_winner, depth_settled = self._resolve_dist(entry.depth_votes)

        entry.topology_chosen = topo_winner or entry.topology_chosen
        if depth_winner is not None:
            entry.depth_chosen = int(depth_winner)

        total = sum(entry.topology_votes.values())
        entry.confidence = entry.topology_votes.get(entry.topology_chosen, 0)
        # contested if topology unsettled (covers 5-3-2 style splits, not just 2-way)
        entry.contested = (total >= self.cfg.conflict_min_votes) and not topo_settled

    def _resolve_dist(self, votes: Dict[str, int]) -> Tuple[Optional[str], bool]:
        items = [(k, v) for k, v in votes.items() if v > 0]
        if not items:
            return None, False
        items.sort(key=lambda x: x[1], reverse=True)
        total = sum(v for _, v in items)
        leader, lead_v = items[0]
        runner_v = items[1][1] if len(items) > 1 else 0
        margin = (lead_v - runner_v) / total
        ent = _entropy([v for _, v in items])
        settled = (total >= self.cfg.conflict_min_votes
                   and margin >= self.cfg.conflict_dominance_margin
                   and ent <= self.cfg.conflict_entropy_threshold)
        return leader, settled

    def contested_entries(self) -> List[HandbookEntry]:
        return [e for e in self.entries if e.contested]

    # -- persistence ----------------------------------------------------------
    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump([asdict(e) for e in self.entries], f, indent=2)

    def load(self) -> None:
        with open(self.path) as f:
            raw = json.load(f)
        self.entries = [HandbookEntry(**e) for e in raw]


def _now() -> float:
    import time
    return time.time()
