"""v3.0 — F-keyed trajectory store. The persistent E-F coupling.

Case-based memory, keyed on the fingerprint's shape_string (the structural
species — a small controlled vocabulary, so exact-match == same species).
This is deliberately case-based, not vote-based: LLMs learn from concrete
exemplars, not abstract rules, so we store what actually happened.

Per fingerprint class we persist:
  - the convergence trajectory (initial E -> final E, iterations)
  - learned_moves: per-axis move_ids that DROVE an axis below threshold
    (the deterministic wins + the escalation wins) — these get prioritized
    next time the same species appears, so plans start closer to F
  - escalation_moves: LLM-authored moves from earned escalation. Folding
    these back into the repertoire is how the deterministic vocabulary GROWS
    (b feeds a).
  - irreducible_counts: per-axis count of times an axis exhausted its moves.
    Escalation (b) only unlocks once this crosses a threshold — escalation
    must be EARNED by repeated proven irreducibility, not triggered on one
    stumble.
  - side_effect_counts: per-move, per-OTHER-axis count of times applying
    that move caused a previously-satisfied axis to newly cross its own
    threshold (collateral damage — e.g. adding a node to fix role pushing
    scale over budget). This captures the full ripple of a move across
    every dimension, not just the one it targeted, giving a genuinely
    holistic (fingerprint, plan, move) -> (full error ripple) record.
    Same "earn it, don't react to one incident" discipline as escalation:
    a move is only DEPRIORITIZED (never removed) once a specific collateral
    axis crosses the gate a repeated number of times for this fingerprint.

Everything here is deterministic bookkeeping. The only non-determinism in
the whole system is the CONTENT the LLM authors inside an escalation move,
and even that only fires when the counter says it's earned.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from .embeddings import cosine


def _blank_case(shape_key: str, embedding: List[float]) -> Dict:
    return {
        "shape_key": shape_key,
        "shape_embedding": embedding,
        "runs": 0,
        "best_plan": {},
        # THRESHOLD-RELATIVE worst excess (max over axes of E[ax]-threshold[ax])
        # for the stored best_plan -- NEVER raw max(E.values()). Raw axis
        # magnitudes aren't comparable across axes with different thresholds
        # (partition's threshold is 0.70, role's is 0.35 -- comparing their
        # raw values head-to-head is apples to oranges). This field used to
        # be named best_worst_axis and store the raw max; renamed when that
        # bug was found and fixed (see descent.py Descent._worst_excess,
        # the same threshold-relative metric, now the only "which plan is
        # better" comparison anywhere in the codebase).
        "best_worst_excess": float("inf"),
        "initial_ef": {},
        "final_ef": {},
        "iterations": 0,
        "learned_moves": {},          # axis -> [move_id]
        "escalation_moves": {},       # axis -> [{move_id, axis, constraint, action}]
        "irreducible_counts": {},     # axis -> int
        "side_effect_counts": {},     # move_id -> {other_axis: int}
        "final_q": {},                # v3.5 -- last run's Q-tensor (groundedness/relevance), display only
    }


class EFStore:
    def __init__(self, path: str = ".ceo_delta/ef_cases.json"):
        self.path = path
        self.cases: Dict[str, Dict] = {}
        if path and os.path.exists(path):
            self.load()

    # -- lifecycle ----------------------------------------------------------
    def ensure(self, shape_key: str, embedding: List[float]) -> Dict:
        if shape_key not in self.cases:
            self.cases[shape_key] = _blank_case(shape_key, embedding)
        return self.cases[shape_key]

    def get(self, shape_key: str) -> Optional[Dict]:
        return self.cases.get(shape_key)

    def best_match(self, embedding: List[float]) -> Tuple[Optional[Dict], float]:
        """Nearest fingerprint class by shape embedding — used for cross-F
        generalization (a novel-but-similar species inherits priors)."""
        best, best_sim = None, 0.0
        for c in self.cases.values():
            emb = c.get("shape_embedding") or []
            if not emb:
                continue
            s = cosine(embedding, emb)
            if s > best_sim:
                best, best_sim = c, s
        return best, best_sim

    # -- move memory (b feeds a) --------------------------------------------
    def learned_moves_for(self, shape_key: str, axis: str) -> List[str]:
        c = self.cases.get(shape_key)
        if not c:
            return []
        return list(c["learned_moves"].get(axis, []))

    def escalation_moves_for(self, shape_key: str, axis: str) -> List[Dict]:
        c = self.cases.get(shape_key)
        if not c:
            return []
        return list(c["escalation_moves"].get(axis, []))

    def add_learned_move(self, shape_key: str, axis: str, move_id: str) -> None:
        c = self.cases.get(shape_key)
        if not c:
            return
        lst = c["learned_moves"].setdefault(axis, [])
        if move_id not in lst:
            lst.append(move_id)

    def add_escalation_move(self, shape_key: str, axis: str, desc: Dict) -> None:
        c = self.cases.get(shape_key)
        if not c:
            return
        lst = c["escalation_moves"].setdefault(axis, [])
        if all(d.get("move_id") != desc.get("move_id") for d in lst):
            lst.append(desc)

    # -- irreducibility counter (gates earned escalation) -------------------
    def bump_irreducible(self, shape_key: str, axis: str) -> int:
        c = self.cases.get(shape_key)
        if not c:
            return 0
        c["irreducible_counts"][axis] = c["irreducible_counts"].get(axis, 0) + 1
        return c["irreducible_counts"][axis]

    def reset_irreducible(self, shape_key: str, axis: str) -> None:
        """An axis is only irreducible while it is CURRENTLY stuck. The moment
        a move (deterministic or escalated) drives it below threshold, the axis
        is solved for this species, so its counter resets to 0.

        This is the fix for the escalation-permanently-dead bug: the old gate
        was `count >= threshold AND no move ever learned`, so a single early
        success wrote a permanent 'learned' move that disqualified escalation
        forever -- even as the count climbed to 6 on later failures. With the
        counter reset-on-success, the count now tracks CONSECUTIVE-recent
        failures since the last win, and eligibility is a clean `count >=
        threshold`: if the learned move were still working, the count would
        keep resetting and never re-arm escalation; when it stops working, the
        count climbs and escalation correctly re-opens."""
        c = self.cases.get(shape_key)
        if not c:
            return
        if axis in c.get("irreducible_counts", {}):
            c["irreducible_counts"][axis] = 0

    # -- side-effect profile (captures ripples across ALL axes, not just the
    # one a move targeted) -- gates move deprioritization -------------------
    def bump_side_effect(self, shape_key: str, move_id: str, other_axis: str) -> int:
        c = self.cases.get(shape_key)
        if not c:
            return 0
        per_move = c.setdefault("side_effect_counts", {}).setdefault(move_id, {})
        per_move[other_axis] = per_move.get(other_axis, 0) + 1
        return per_move[other_axis]

    def deprioritized_moves(self, shape_key: str, threshold: int) -> List[str]:
        """move_ids that have EARNED deprioritization: some collateral axis
        crossed threshold at least `threshold` times when this move was
        applied for this fingerprint class. Never removes a move — callers
        should still keep it available as a last resort."""
        c = self.cases.get(shape_key)
        if not c:
            return []
        out = []
        for move_id, axis_counts in c.get("side_effect_counts", {}).items():
            if any(cnt >= threshold for cnt in axis_counts.values()):
                out.append(move_id)
        return out

    # -- trajectory ---------------------------------------------------------
    def upsert_case(self, shape_key: str, shape_embedding: List[float],
                    best_plan: Dict, initial_ef: Dict, final_ef: Dict,
                    iterations: int, worst_excess: float,
                    final_q: Optional[Dict] = None) -> None:
        """worst_excess MUST be threshold-relative (max over axes of
        E[ax]-threshold[ax]), computed by the caller (Descent, which holds
        the thresholds) -- never re-derived here from raw final_ef values.
        Comparing raw axis magnitudes to decide "best" is invalid across axes
        with different thresholds; that was a real, shipped bug (this store
        and orchestrator.py's answer-selection both had independent copies of
        it, and disagreed with each other as a result).

        final_q (v3.5) is display-only bookkeeping -- the Q-tensor from the
        run that produced final_ef. Does not participate in best-plan
        selection (worst_excess is still structural-only, unchanged)."""
        c = self.ensure(shape_key, shape_embedding)
        c["runs"] += 1
        c["initial_ef"] = initial_ef
        c["final_ef"] = final_ef
        c["iterations"] = iterations
        c["final_q"] = final_q or {}
        if worst_excess <= c.get("best_worst_excess", float("inf")):
            c["best_worst_excess"] = worst_excess
            c["best_plan"] = best_plan

    # -- persistence --------------------------------------------------------
    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"cases": self.cases}, f, indent=2)

    def load(self) -> None:
        with open(self.path) as f:
            raw = json.load(f)
        self.cases = raw.get("cases", {})
        # backfill fields added in later versions so old on-disk cases don't KeyError
        for c in self.cases.values():
            c.setdefault("side_effect_counts", {})
            c.setdefault("final_q", {})  # v3.5 -- absent on cases persisted before Q-factor existed
            # best_worst_axis (raw max, buggy) -> best_worst_excess (threshold-
            # relative, correct). Old data can't be converted (we no longer
            # have the per-run thresholds it was computed against), so treat
            # it as unknown -- the next upsert_case call will set it correctly
            # rather than trusting a value computed by the old broken formula.
            if "best_worst_axis" in c and "best_worst_excess" not in c:
                c["best_worst_excess"] = float("inf")
            c.setdefault("best_worst_excess", float("inf"))
