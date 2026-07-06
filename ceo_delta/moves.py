"""v3.0 — Deterministic repair repertoire.

The structural repair is chosen by TABLE LOOKUP from the worst EF axis, not
by asking the LLM to "reconsider from first principles." This is what makes
the descent deterministic: which structural edit to attempt is a function of
(axis, required F, current graph signature, what's already been tried), never
of the model's mood.

The LLM only fills in node INTENT wording once the structure is fixed. The
move's `constraint` is the exact instruction injected into CEO's prompt.

APPLICABILITY (v3.2): the earlier version listed every move for an axis and
tried them in list order regardless of what F required. That let the descent
attempt structurally-contradictory repairs in one run -- e.g. flow.split_to_
parallel (make it divergent) THEN flow.linearize (make it sequential), or
scale.collapse_layer THEN scale.add_layer (opposite directions). Trying a
move that pushes the axis the WRONG way is not descent, it's a random walk,
and it was the direct cause of the observed thrashing. Now `applicable_moves`
returns only the moves whose structural direction can actually reduce the
locked axis given the current (F, signature) -- so the descent only ever
takes steps that CAN help, and an axis with no applicable move is honestly
"exhausted" (-> irreducible -> eventually escalates) rather than faking work
by trying doomed moves.

Escalation-authored moves (the LLM's earned contributions, see descent.py)
are folded into the candidate list at runtime -- a successful escalation
becomes a permanent deterministic move for that fingerprint class (b feeds a).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class Move:
    move_id: str
    axis: str
    constraint: str
    action: str = "replan"   # "refine" (local edit) or "replan" (structural)


MOVE_SET: Dict[str, List[Move]] = {
    "flow": [
        Move(
            "flow.split_to_parallel", "flow",
            "STRUCTURAL FIX (flow): this task needs DIVERGENT information flow but your "
            "plan has no parallel gathering. Create it in EITHER form: >=2 INDEPENDENT "
            "root nodes (no dependencies), OR one lightweight coordinator that fans out "
            "to >=2 parallel branches. Each parallel branch must gather a DISTINCT "
            "sub-aspect, and exactly ONE join node must depend on all of them.",
            "replan",
        ),
        Move(
            "flow.add_join", "flow",
            "STRUCTURAL FIX (flow): your parallel branches never converge. Add exactly one "
            "join/synthesizer node that depends on ALL parallel branches and merges them "
            "into a single downstream output.",
            "replan",
        ),
        Move(
            "flow.linearize", "flow",
            "STRUCTURAL FIX (flow): this task needs SEQUENTIAL flow but your plan branches. "
            "Rebuild as a single chain where each node depends on exactly one predecessor.",
            "replan",
        ),
        Move(
            "flow.add_merge", "flow",
            "STRUCTURAL FIX (flow): this task is CONVERGENT — multiple distinct inputs must "
            "merge into one judgment. Ensure a single terminal node depends on >=2 upstream "
            "nodes and explicitly reconciles them.",
            "replan",
        ),
        Move(
            "flow.deepen_recursive", "flow",
            "STRUCTURAL FIX (flow): this task is RECURSIVE — it decomposes into smaller "
            "versions of itself — but your plan is too shallow to express that nesting. "
            "Build a chain of at least 3 dependency levels where each level handles one "
            "layer of the recursion (base case -> one unwinding step -> the general case), "
            "each node depending on the level below it.",
            "replan",
        ),
    ],
    "partition": [
        Move(
            "partition.differentiate", "partition",
            "STRUCTURAL FIX (partition): two sibling nodes with identical dependencies are "
            "producing near-duplicate outputs (redundant fan-out). Rewrite each sibling's "
            "intent so it covers a STRICTLY DISJOINT sub-topic along an orthogonal axis — "
            "no scope overlap between siblings.",
            "refine",
        ),
        Move(
            "partition.merge_redundant", "partition",
            "STRUCTURAL FIX (partition): redundant siblings persist after differentiation. "
            "Merge the overlapping sibling nodes into a single node that covers their union.",
            "replan",
        ),
    ],
    "role": [
        Move(
            "role.add_missing", "role",
            "STRUCTURAL FIX (role): a functional role REQUIRED by this task type is missing "
            "from the plan entirely. Add a node with that role in the correct position.",
            "replan",
        ),
        Move(
            "role.realign", "role",
            "STRUCTURAL FIX (role): a node's actual behavior does not match its assigned "
            "functional role. Either reassign its role to match what it does, or rewrite its "
            "intent to behave as its role demands — retriever=EXPAND+cite sources, "
            "synthesizer=COMPRESS+cross-reference, verifier=AUDIT claims+cite.",
            "refine",
        ),
    ],
    "scale": [
        Move(
            "scale.collapse_layer", "scale",
            "STRUCTURAL FIX (scale): the plan is DEEPER than the task complexity warrants. "
            "Collapse one dependency layer — remove an intermediate level and connect its "
            "children directly to its parents.",
            "replan",
        ),
        Move(
            "scale.add_layer", "scale",
            "STRUCTURAL FIX (scale): the plan is too SHALLOW for the task complexity. Add one "
            "decomposition layer between the roots and the final node.",
            "replan",
        ),
    ],
}

_BY_ID: Dict[str, Move] = {m.move_id: m for lst in MOVE_SET.values() for m in lst}


# ---------------------------------------------------------------------------
# Applicability: which moves can actually reduce this axis in THIS state?
# Each returns move_ids in preferred order (most surgical first).
# ---------------------------------------------------------------------------

def _flow_applicable(sig: Dict, req) -> List[str]:
    f = getattr(req, "flow", None)
    if f == "sequential":
        return ["flow.linearize"]
    if f == "divergent":
        has_gather = sig.get("root_count", 0) >= 2 or sig.get("max_out", 0) >= 2
        if not has_gather:
            return ["flow.split_to_parallel", "flow.add_join"]
        if not sig.get("has_join"):
            return ["flow.add_join", "flow.split_to_parallel"]
        # gather + join present but flow-E still over: rebuild the gather
        return ["flow.split_to_parallel", "flow.add_join"]
    if f == "convergent":
        return ["flow.add_merge"]
    if f == "recursive":
        return ["flow.deepen_recursive"]
    return []


def _scale_applicable(sig: Dict, req) -> List[str]:
    depth = sig.get("depth", 0)
    target = getattr(req, "target_depth", 0) or 1
    if depth > target:
        return ["scale.collapse_layer"]
    if depth < target:
        return ["scale.add_layer"]
    return []   # at target -> scale-E is 0 by construction; nothing to do


def _role_applicable(sig: Dict, req, present_roles: Set[str]) -> List[str]:
    required = getattr(req, "required_roles", set()) or set()
    missing = set(required) - set(present_roles or set())
    if missing:
        # a required role is absent -> add it first, realign as fallback
        return ["role.add_missing", "role.realign"]
    # all required roles present but behavior out of band -> realign only
    return ["role.realign"]


def _partition_applicable(sig: Dict, req, present_roles: Set[str]) -> List[str]:
    # differentiate (surgical) then merge (structural fallback) -- these are
    # escalating, not contradictory, so both always apply when partition fires.
    return ["partition.differentiate", "partition.merge_redundant"]


def applicable_moves(axis: str, sig: Optional[Dict], req, present_roles: Optional[Set[str]] = None,
                     extra: Optional[List[Move]] = None,
                     priority_ids: Optional[List[str]] = None,
                     deprioritized_ids: Optional[List[str]] = None) -> List[Move]:
    """Moves whose structural direction can reduce `axis` given the current
    (F, signature) state, ordered: learned/priority first, base-applicable
    next, earned-but-collateral (deprioritized) last. Persisted escalation
    moves (`extra`) are folded in as candidates (b feeds a)."""
    sig = sig or {}
    present_roles = present_roles or set()
    if axis in ("role", "partition"):
        base_ids = (_role_applicable if axis == "role" else _partition_applicable)(sig, req, present_roles)
    elif axis == "flow":
        base_ids = _flow_applicable(sig, req)
    elif axis == "scale":
        base_ids = _scale_applicable(sig, req)
    else:
        base_ids = []

    moves: List[Move] = [_BY_ID[mid] for mid in base_ids if mid in _BY_ID]
    # fold in persisted escalation moves for this axis (earned deterministic vocab)
    for m in (extra or []):
        if all(m.move_id != x.move_id for x in moves):
            moves.append(m)

    rank = {mid: i for i, mid in enumerate(priority_ids)} if priority_ids else {}
    depr = set(deprioritized_ids or [])

    def sort_key(m: Move):
        return (1 if m.move_id in depr else 0, rank.get(m.move_id, len(rank)))

    moves.sort(key=sort_key)
    return moves


def select_move(axis: str, tried: Set[str], moves: List[Move]) -> Optional[Move]:
    """First not-yet-tried move from the (already applicability-filtered,
    already-ordered) candidate list. No direction logic here anymore --
    applicable_moves guarantees every candidate points the right way."""
    for m in moves:
        if m.move_id not in tried:
            return m
    return None
