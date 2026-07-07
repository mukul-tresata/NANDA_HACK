"""v3.3 — Deterministic structural repair operators (the actual descent step).

THE FIX this module exists for
------------------------------
v3.0-3.2 described a "deterministic coordinate descent" but never implemented
the descent *operator*. Every repair move -- flow.*, scale.*, role.*,
partition.* -- was realized identically: the orchestrator looped back to
`ceo.plan(directive=...)`, a full stochastic LLM regeneration. The move table
only rewrote the PROMPT; it never transformed the graph. Consequences, all
visible in the evidence battery:
  - a "scale.collapse_layer" directive did not remove a layer, it *asked the
    model to please build shallower* and the model rebuilt the whole DAG,
    lighting up other axes;
  - the incumbent-rejection guard then froze progress at iteration 0's cold
    plan (worst-excess flat across the run while 4 moves were "applied");
  - `refine` and `replan` were indistinguishable (the loop only branched on
    `surface`), so no move was ever surgical.

For the two PURE-TOPOLOGY axes (flow, scale) the repair is a function of the
graph alone, so it does NOT need the LLM at all. This module implements those
moves as pure `DAG -> DAG` transforms. Applying a direction-correct move now
provably moves its axis the right way by construction -- the descent becomes a
real contraction with a bounded, reproducible trajectory.

The LLM is still used to author CONTENT (a new node's intent wording) and for
the genuinely content-dependent half of the partition axis
(partition.differentiate -- inventing distinct intents for siblings that are
redundant), which stays on the ceo.plan refine path. Structure is
deterministic; only text is sampled. (v3.4: role.add_missing / role.realign
joined the deterministic set below -- see their docstrings for why the role
axis's missing-role term is pure set membership, not content judgment. v3.5:
partition.merge_redundant joined too -- collapsing redundant siblings is a
graph transform once redundancy is detected via intent-embedding similarity,
see its docstring.)

Every operator:
  - takes an incumbent DAG and returns a NEW DAG (deep-copied; incumbent is
    never mutated -- the descent needs the incumbent intact for rejection);
  - preserves existing nodes' intents / roles / fingerprints wherever the
    topology allows (surgical, not a rebuild);
  - gives any newly-created node a deterministic template intent derived from
    the task, so the kernel can execute it without a planning call;
  - is idempotent-safe: if the graph is already in the target shape it returns
    (dag, changed=False) so the descent can honestly mark the axis exhausted
    instead of thrashing.
"""
from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Tuple

from .embeddings import cosine, embed
from .schemas import DAG, Node, Roles, Why


# ---------------------------------------------------------------------------
# depth / topology helpers (longest-path levels -- same metric ef.py uses)
# ---------------------------------------------------------------------------

def _levels(dag: DAG) -> Dict[str, int]:
    """Longest-path level of each node (roots = 1). Matches ef._critical_path."""
    memo: Dict[str, int] = {}
    ids = {n.node_id for n in dag.nodes}

    def lvl(nid: str) -> int:
        if nid in memo:
            return memo[nid]
        node = dag.node(nid)
        deps = [d for d in (node.dependencies if node else []) if d in ids]
        memo[nid] = 1 if not deps else 1 + max(lvl(d) for d in deps)
        return memo[nid]

    for n in dag.nodes:
        lvl(n.node_id)
    return memo


def _depth(dag: DAG) -> int:
    lv = _levels(dag)
    return max(lv.values(), default=0)


def _roots(dag: DAG) -> List[str]:
    ids = {n.node_id for n in dag.nodes}
    return [n.node_id for n in dag.nodes
            if not any(d in ids for d in n.dependencies)]


def _sinks(dag: DAG) -> List[str]:
    """Nodes nothing depends on."""
    depended: set = set()
    for n in dag.nodes:
        depended.update(n.dependencies)
    return [n.node_id for n in dag.nodes if n.node_id not in depended]


def _topo_order(dag: DAG) -> List[Node]:
    """Deterministic topological order (ties broken by existing node order)."""
    ids = {n.node_id for n in dag.nodes}
    order: List[Node] = []
    placed: set = set()
    pending = list(dag.nodes)
    # Kahn-ish, but preserving list order for determinism; guard against cycles.
    changed = True
    while pending and changed:
        changed = False
        still: List[Node] = []
        for n in pending:
            if all((d not in ids) or (d in placed) for d in n.dependencies):
                order.append(n)
                placed.add(n.node_id)
                changed = True
            else:
                still.append(n)
        pending = still
    order.extend(pending)  # any cycle remnant: append in original order
    return order


def _clone(dag: DAG) -> DAG:
    return copy.deepcopy(dag)


def _new_node(node_id: str, task: str, task_emb, role: str, intent: str,
              deps: List[str]) -> Node:
    return Node(
        node_id=node_id,
        intent=intent,
        roles=Roles(structural="synthetic", functional=role, epistemic="generalist"),
        dependencies=list(deps),
        why=Why(task_type_recognized="deterministic-structural-op",
                directive_response="inserted by graph_ops (no LLM)"),
        expected_output_fingerprint=task_emb,
    )


def _fresh_id(dag: DAG, stem: str) -> str:
    existing = {n.node_id for n in dag.nodes}
    i = len(dag.nodes) + 1
    while f"{stem}{i}" in existing:
        i += 1
    return f"{stem}{i}"


def _resync_depth_topology(dag: DAG, topology: str) -> None:
    dag.depth = _depth(dag)
    dag.topology = topology


# ---------------------------------------------------------------------------
# SCALE operators (change critical-path depth by exactly 1, deterministically)
# ---------------------------------------------------------------------------

def collapse_layer(dag: DAG, req=None) -> Tuple[DAG, bool]:
    """Remove one dependency layer -> depth strictly decreases by (at least) 1.

    Deletes the deepest interior node that actually gates the critical path and
    rewires its dependents onto its dependencies. If no removal can shorten the
    graph (already minimal), returns changed=False so the axis is honestly
    exhausted rather than thrashed.

    `req` (optional RequiredStructure): if the chosen candidate's removal
    would drop a `req.required_roles` member out of the graph's present-role
    set, that candidate is skipped in favor of the next deepest one (role
    preservation is a pure set-membership check on the trial graph, no LLM
    judgment). If every non-sink candidate would break a required role, the
    move honestly refuses: (dag, False).
    """
    d0 = _depth(dag)
    if d0 <= 1 or len(dag.nodes) <= 1:
        return dag, False

    required_roles = getattr(req, "required_roles", None) if req else None

    sinks = set(_sinks(dag))
    levels = _levels(dag)
    # candidates: non-sink nodes (removing a node with dependents can shorten a
    # path). Deepest first -- collapsing near the bottom of the critical path is
    # the most surgical.
    candidates = sorted(
        [n for n in dag.nodes if n.node_id not in sinks],
        key=lambda n: levels[n.node_id], reverse=True,
    )
    for cand in candidates:
        if required_roles:
            # Refuse ONLY if this candidate is the SOLE carrier of a required
            # role that is currently PRESENT -- removing it would drop that role
            # from the graph. A required role that is already absent is a
            # role-axis problem, not collapse's concern, and must not block
            # collapse (else a graph missing any required role can never be
            # depth-repaired at all).
            role = cand.roles.functional
            if role in required_roles and not any(
                n.roles.functional == role for n in dag.nodes if n.node_id != cand.node_id
            ):
                continue  # sole carrier of a present required role -- skip it
        trial = _clone(dag)
        cid = cand.node_id
        cdeps = [d for d in cand.dependencies]
        trial.nodes = [n for n in trial.nodes if n.node_id != cid]
        for n in trial.nodes:
            if cid in n.dependencies:
                rewired = [d for d in n.dependencies if d != cid]
                for cd in cdeps:
                    if cd != n.node_id and cd not in rewired:
                        rewired.append(cd)
                n.dependencies = rewired
        if _depth(trial) < d0:
            _resync_depth_topology(trial, "linear" if _depth(trial) == 1 else dag.topology)
            return trial, True
    return dag, False


def add_layer(dag: DAG) -> Tuple[DAG, bool]:
    """Insert one decomposition layer -> depth increases by exactly 1.

    A single new node becomes the sole root; every former root depends on it.
    Every path lengthens by one, so depth += 1 with no other structural change
    (former parallel roots stay parallel via the new node's fan-out, so a
    satisfied divergent/flow axis is preserved).
    """
    roots = _roots(dag)
    if not roots:
        return dag, False
    trial = _clone(dag)
    nid = _fresh_id(trial, "scaffold")
    stem = (dag.task or "the task")[:80]
    scaffold = _new_node(
        nid, dag.task, dag.task_embedding, role="retriever",
        intent=f"Establish the foundational context and scope needed before addressing: {stem}",
        deps=[],
    )
    root_set = set(roots)
    for n in trial.nodes:
        if n.node_id in root_set:
            n.dependencies = [nid]
    trial.nodes.insert(0, scaffold)
    _resync_depth_topology(trial, "hierarchical")
    return trial, True


# ---------------------------------------------------------------------------
# FLOW operators (reshape edges toward the required flow, deterministically)
# ---------------------------------------------------------------------------

def linearize(dag: DAG) -> Tuple[DAG, bool]:
    """Rebuild as a single chain: one root, no branching (sequential flow).

    Preserves every node's intent/role -- only the dependency edges are
    rewritten, in topological order, so downstream context is still coherent.
    """
    order = _topo_order(dag)
    if len(order) <= 1:
        return dag, False
    # already a single chain? (root_count==1 and every node out-degree<=1)
    if len(_roots(dag)) == 1:
        outdeg: Dict[str, int] = {n.node_id: 0 for n in dag.nodes}
        for n in dag.nodes:
            for d in n.dependencies:
                if d in outdeg:
                    outdeg[d] += 1
        if all(v <= 1 for v in outdeg.values()):
            return dag, False
    trial = _clone(dag)
    by_id = {n.node_id: n for n in trial.nodes}
    prev: Optional[str] = None
    for n in order:
        node = by_id[n.node_id]
        node.dependencies = [prev] if prev else []
        prev = node.node_id
    _resync_depth_topology(trial, "linear")
    return trial, True


def add_join(dag: DAG) -> Tuple[DAG, bool]:
    """Add exactly one join node depending on all current sinks (divergent
    branches must converge)."""
    sinks = _sinks(dag)
    if len(sinks) < 2:
        return dag, False
    trial = _clone(dag)
    nid = _fresh_id(trial, "join")
    stem = (dag.task or "the task")[:80]
    join = _new_node(
        nid, dag.task, dag.task_embedding, role="synthesizer",
        intent=f"Integrate and cross-reference all parallel findings into one coherent answer to: {stem}",
        deps=list(sinks),
    )
    trial.nodes.append(join)
    _resync_depth_topology(trial, "join")
    return trial, True


def split_to_parallel(dag: DAG) -> Tuple[DAG, bool]:
    """Create parallel gathering: >=2 independent roots that converge on one
    join. Existing non-terminal nodes become independent gatherers; the current
    terminal node becomes (or gains) the join."""
    if len(dag.nodes) < 2:
        return dag, False
    trial = _clone(dag)
    order = _topo_order(trial)
    join_node = order[-1]
    gatherers = [n for n in trial.nodes if n.node_id != join_node.node_id]
    if len(gatherers) < 2:
        # only one non-terminal node -- clone it into a second gatherer so the
        # fan-out is real, not nominal.
        nid = _fresh_id(trial, "branch")
        stem = (dag.task or "the task")[:80]
        extra = _new_node(
            nid, dag.task, dag.task_embedding, role="retriever",
            intent=f"Gather a distinct, non-overlapping sub-aspect of: {stem}",
            deps=[],
        )
        trial.nodes.insert(len(trial.nodes) - 1, extra)
        gatherers = gatherers + [extra]
    for g in gatherers:
        g.dependencies = []
    join_node.dependencies = [g.node_id for g in gatherers]
    _resync_depth_topology(trial, "fan-out")
    return trial, True


def add_merge(dag: DAG) -> Tuple[DAG, bool]:
    """Convergent flow: ensure a single terminal node with in-degree >=2 that
    reconciles multiple upstream inputs."""
    sinks = _sinks(dag)
    trial = _clone(dag)
    # already convergent: exactly one sink and it merges >=2 inputs
    if len(sinks) == 1:
        only = trial.node(sinks[0])
        if only and len([d for d in only.dependencies]) >= 2:
            return dag, False
    nid = _fresh_id(trial, "merge")
    stem = (dag.task or "the task")[:80]
    deps = sinks if len(sinks) >= 2 else [n.node_id for n in trial.nodes if n.node_id not in sinks] + sinks
    deps = list(dict.fromkeys(deps))  # de-dup, keep order
    if len(deps) < 2:
        return dag, False
    merge = _new_node(
        nid, dag.task, dag.task_embedding, role="synthesizer",
        intent=f"Reconcile the distinct upstream inputs into a single judgment on: {stem}",
        deps=deps,
    )
    trial.nodes.append(merge)
    _resync_depth_topology(trial, "join")
    return trial, True


def deepen_recursive(dag: DAG, target_depth: int = 3) -> Tuple[DAG, bool]:
    """Recursive flow is approximated by sufficient nesting depth (>=3). Add
    decomposition layers until the critical path reaches target_depth."""
    changed = False
    cur = dag
    guard = 0
    while _depth(cur) < target_depth and guard < target_depth + 2:
        cur, did = add_layer(cur)
        changed = changed or did
        guard += 1
        if not did:
            break
    if changed:
        _resync_depth_topology(cur, "hierarchical")
    return cur, changed


# ---------------------------------------------------------------------------
# ROLE operators (v3.4 -- close the missing-role / mis-assigned-role gap)
# ---------------------------------------------------------------------------
#
# ef.py's _role_error is: compute_role(...) + 0.3 * len(missing), where
# missing = required functional roles absent from the plan's present set.
# That second term is a pure set-membership fact about dag.nodes' roles --
# it needs no LLM judgment to fix. These two operators close it
# deterministically:
#   - role.add_missing : insert a node carrying the absent role.
#   - role.realign      : relabel an over-represented node onto the absent
#                          role instead (no new node, pure field change).
# Both generalize ceo.force_verifier's pattern (which remains as the
# volatility-gated verifier guarantee; these operators are the general,
# any-required-role version used by the descent).

def role_add_missing(dag: DAG, required: Optional[set] = None) -> Tuple[DAG, bool]:
    """Insert exactly one node for the first (sorted) missing required
    functional role. Position policy:
      - verifier / synthesizer -> depends on all current sinks (terminal,
        it audits/integrates what's already been produced).
      - retriever -> a new root (no dependencies), feeding the graph.
      - anything else -> safe default: depends on current sinks.

    Idempotent: if `required` is falsy or nothing is missing, returns
    (dag, False) so the descent can mark the role axis exhausted.
    """
    if not required:
        return dag, False
    present = {n.roles.functional for n in dag.nodes}
    missing = sorted(r for r in required if r not in present)
    if not missing:
        return dag, False

    target_role = missing[0]
    trial = _clone(dag)
    nid = _fresh_id(trial, f"{target_role}")
    stem = (dag.task or "the task")[:80]

    if target_role == "verifier":
        deps = _sinks(trial)
        intent = (
            f"Audit all upstream claims for unverified statistics, figures, "
            f"or causal claims lacking cited sources, in service of: {stem}"
        )
        topology = "join"
    elif target_role == "synthesizer":
        deps = _sinks(trial)
        intent = f"Integrate all upstream findings into one coherent answer to: {stem}"
        topology = "join"
    elif target_role == "retriever":
        deps = []
        intent = f"Gather the foundational information needed to address: {stem}"
        topology = "hierarchical"
    else:
        deps = _sinks(trial)
        intent = f"Provide the missing {target_role} function for: {stem}"
        topology = trial.topology

    node = _new_node(nid, dag.task, dag.task_embedding, role=target_role,
                      intent=intent, deps=deps)
    if deps:
        trial.nodes.append(node)
    else:
        trial.nodes.insert(0, node)
    _resync_depth_topology(trial, topology)
    return trial, True


def role_realign(dag: DAG, required: Optional[set] = None) -> Tuple[DAG, bool]:
    """Relabel an over-represented node's functional role onto a genuinely
    required-but-absent role. Pure field change: no node added or removed,
    no dependency edges touched.

    RATIONALE: realign only fires toward a role that is actually required
    and currently absent, and only reassigns a node whose current role is
    over-represented (so no required role goes missing as a side effect).
    That means it strictly reduces _role_error's 0.3*len(missing) term
    without inventing structure or gaming the behavioral-band term -- it
    never relabels arbitrarily just to move the metric.

    Idempotent: if `required` is falsy or nothing is missing, or no safe
    reassignment target exists, returns (dag, False).
    """
    if not required:
        return dag, False
    present_counts: Dict[str, int] = {}
    for n in dag.nodes:
        present_counts[n.roles.functional] = present_counts.get(n.roles.functional, 0) + 1
    missing = sorted(r for r in required if r not in present_counts)
    if not missing:
        return dag, False
    target_role = missing[0]

    # candidate donor roles: over-represented (count > 1) AND not themselves
    # a required role that would go missing if we take one away.
    def safe_donor(role: str) -> bool:
        if present_counts.get(role, 0) <= 1:
            return False
        if role in required and present_counts[role] - 1 < 1:
            return False
        return True

    order = _topo_order(dag)
    chosen: Optional[Node] = None
    for n in reversed(order):
        if safe_donor(n.roles.functional):
            chosen = n
            break
    if chosen is None:
        return dag, False

    trial = _clone(dag)
    victim = trial.node(chosen.node_id)
    victim.roles.functional = target_role
    _resync_depth_topology(trial, dag.topology)
    return trial, True


# ---------------------------------------------------------------------------
# PARTITION operator (v3.5 -- close the redundant-siblings half of the axis)
# ---------------------------------------------------------------------------
#
# ef.py's _partition_error measures mean cosine similarity of SIBLING nodes
# (identical dependency sets) over their OUTPUT embeddings: high similarity
# means siblings are doing redundant work. Repairing that has two distinct
# halves:
#   - partition.merge_redundant : two siblings are redundant -> collapse them
#                                  into one. This IS a graph transform (drop a
#                                  node, rewire dependents) once redundancy is
#                                  detected, so it belongs here.
#   - partition.differentiate   : siblings are NOT redundant but need to be
#                                  steered apart -> inventing distinct intents
#                                  is irreducibly semantic, so that half stays
#                                  on the stochastic LLM refine path.
# This operator cannot see runtime OUTPUT embeddings (it only sees the graph
# pre-execution), so it uses INTENT-embedding similarity as a deterministic
# proxy for redundancy. That's a safe approximation: the descent re-measures
# the real partition error on actual outputs after the move and rejects it
# via the normal incumbent-rejection guard if the merge didn't help.

PARTITION_MERGE_SIM = 0.80


def partition_merge_redundant(dag: DAG, partition_pairs: Optional[Dict] = None,
                               req=None) -> Tuple[DAG, bool]:
    """Collapse the single most-redundant sibling pair into one node.

    Siblings = nodes sharing an identical dependency set (root nodes, with
    empty deps, are siblings of each other too -- a divergent fan-out's
    parallel roots are the primary over-partition risk, mirroring
    ef._partition_error's own comment).

    Redundancy target selection (v3.6 -- Fix A): when `partition_pairs` (the
    EFTensor's real, OUTPUT-embedding-based sibling similarity dict, keyed
    (id_i,id_j)->cosine) is provided and non-empty, the merge target is the
    highest-similarity pair FROM THAT MEASURED DATA -- the axis's actual
    worst pair. This replaces the old intent-embedding guess, which false-
    fired on distinct-but-similarly-worded siblings (e.g. per-offer eval
    nodes, flight/hotel retrievers) that read similar in intent text but do
    genuinely distinct work, as measured by their real outputs.

    Only when `partition_pairs` is None/empty (cold, pre-execution -- no
    output embeddings exist yet) does this fall back to the previous INTENT
    embedding cosine similarity proxy across all sibling groups.

    Either way: if the winning pair's similarity is below PARTITION_MERGE_SIM,
    the siblings are genuinely distinct work -- merging them would destroy
    needed parallelism, so this returns (dag, False) untouched. Same for: no
    sibling pairs at all, or fewer than 2 nodes total.

    Flow-preservation guard (v3.6 -- Fix B): before committing, the trial
    DAG's graph-signature is computed AFTER the proposed merge. If `req` is
    given and its flow is divergent, and the merge would drop BOTH
    root_count<2 AND max_out<2 (breaking the parallel-gather floor), or its
    flow is convergent and the merge would drop the sink's max_in<2, the
    merge is refused: (dag, False). This is pure arithmetic on the trial
    signature -- it turns the partition->flow coupling into an honest no-op
    instead of a flow-spiking move that the descent's rejection logic would
    have to catch downstream.

    Otherwise merges the winning pair: keeps the lexicographically smaller
    node_id, drops the other, and rewires every remaining node's
    dependencies from the dropped id onto the kept id (de-duplicated, no
    self-dependency). This is the DETERMINISTIC half of partition repair;
    partition.differentiate (making genuinely distinct siblings read as
    distinct) stays on the LLM path -- see the module comment above.
    """
    if len(dag.nodes) < 2:
        return dag, False

    existing = {n.node_id for n in dag.nodes}
    best_pair: Optional[Tuple[str, str]] = None
    best_sim = -2.0

    if partition_pairs:
        for (i, j), sim in partition_pairs.items():
            if i not in existing or j not in existing:
                continue  # stale entry from a prior graph shape
            if sim > best_sim:
                best_sim = sim
                best_pair = (i, j)

    if best_pair is None:
        # cold path (no measured output data yet) -- fall back to the
        # intent-embedding proxy across sibling groups.
        groups: Dict[frozenset, List[Node]] = {}
        for n in dag.nodes:
            key = frozenset(n.dependencies)
            groups.setdefault(key, []).append(n)

        for members in groups.values():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    sim = cosine(embed(a.intent), embed(b.intent))
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (a.node_id, b.node_id)

    if best_pair is None or best_sim < PARTITION_MERGE_SIM:
        return dag, False

    keep_id, drop_id = sorted(best_pair)  # lexicographically smaller kept

    trial = _clone(dag)
    existing_ids = {n.node_id for n in trial.nodes} - {drop_id}
    trial.nodes = [n for n in trial.nodes if n.node_id != drop_id]
    for n in trial.nodes:
        rewired: List[str] = []
        for d in n.dependencies:
            if d not in existing_ids:
                continue  # strip dangling / dropped refs
            nd = keep_id if d == drop_id else d
            if nd == n.node_id:
                continue  # never depend on self
            if nd not in rewired:
                rewired.append(nd)
        n.dependencies = rewired

    if req is not None:
        from .ef import graph_signature  # local import: avoid module-load cycle
        trial_sig = graph_signature(trial)
        flow = getattr(req, "flow", None)
        if flow == "divergent" and trial_sig["root_count"] < 2 and trial_sig["max_out"] < 2:
            return dag, False  # would break the parallel-gather floor
        if flow == "convergent" and trial_sig["max_in"] < 2:
            return dag, False  # would break the merge floor

    _resync_depth_topology(trial, trial.topology)
    return trial, True


# ---------------------------------------------------------------------------
# Registry — which move_ids are deterministic graph transforms
# ---------------------------------------------------------------------------

# move_id -> operator. Pure-topology (flow, scale) moves, the two
# set-membership-driven role moves (role.add_missing, role.realign), and the
# redundant-sibling-collapse half of partition (partition.merge_redundant)
# live here -- all are provable graph/field transforms needing no LLM
# judgment. partition.differentiate stays on the LLM refine path (genuinely
# content-dependent: inventing distinct intents for non-redundant siblings
# requires semantic judgment, not just counting).
DET_OPS: Dict[str, Callable[..., Tuple[DAG, bool]]] = {
    "scale.collapse_layer": collapse_layer,
    "scale.add_layer": add_layer,
    "flow.linearize": linearize,
    "flow.add_join": add_join,
    "flow.split_to_parallel": split_to_parallel,
    "flow.add_merge": add_merge,
    "flow.deepen_recursive": deepen_recursive,
    "role.add_missing": role_add_missing,
    "role.realign": role_realign,
    "partition.merge_redundant": partition_merge_redundant,
}


def is_deterministic_move(move_id: str) -> bool:
    return move_id in DET_OPS


def apply_move(dag: DAG, move_id: str, required: Optional[set] = None,
               partition_pairs: Optional[Dict] = None, req=None) -> Tuple[DAG, bool]:
    """Apply a deterministic structural move to a copy of `dag`. Returns
    (new_dag, changed). `changed=False` means the graph is already in the
    target shape (axis exhausted) -- the caller must NOT loop on it.

    `required` (a set/iterable of required functional role names, or None)
    is consumed by the role.* operators (which role is missing) and by
    scale.collapse_layer (v3.6 -- Fix C: which role must not be dropped).

    `partition_pairs` (the EFTensor's measured output-similarity dict, or
    None) and `req` (the RequiredStructure, or None) are consumed only by
    partition.merge_redundant (v3.6 -- Fix A/B: real worst-pair selection
    and the flow-preservation guard).

    All other moves are pure functions of the graph and ignore these extra
    kwargs -- existing callers passing only (dag, move_id) keep working
    unchanged.
    """
    op = DET_OPS.get(move_id)
    if op is None:
        return dag, False
    if move_id.startswith("role."):
        return op(dag, required)
    if move_id == "scale.collapse_layer":
        return op(dag, req)
    if move_id == "partition.merge_redundant":
        return op(dag, partition_pairs, req)
    return op(dag)


def move_id_from_directive(directive) -> Optional[str]:
    """Extract the base move_id from a Descent directive whose reason is
    'ef_move:<move_id>'. Escalated / non-descent directives return None (they
    stay on the LLM path -- their constraint is free-text, not a graph op)."""
    reason = getattr(directive, "reason", "") or ""
    if reason.startswith("ef_move:"):
        return reason.split(":", 1)[1]
    return None
