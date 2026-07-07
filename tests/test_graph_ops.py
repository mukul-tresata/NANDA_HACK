"""Offline proof that the deterministic structural operators (graph_ops.py)
move their axis the correct direction, by construction, with NO LLM.

This is the direct evidence for the demo's claims 1/2/5/8 on the two
pure-topology axes: given a fixed incumbent DAG, each move is a pure function
whose effect on flow/scale error is provable and reproducible.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ceo_delta import graph_ops as G
from ceo_delta.ef import (
    RequiredStructure, graph_signature, _flow_error, _scale_error, _critical_path,
)
from ceo_delta.schemas import DAG, Node, Roles, Why


def _node(nid, deps, role="generic"):
    return Node(node_id=nid, intent=f"intent {nid}",
                roles=Roles(functional=role), dependencies=list(deps),
                expected_output_fingerprint=[0.0])


def _dag(nodes, topology="linear"):
    d = DAG(task="demo task", task_embedding=[0.0] * 4, topology=topology,
            depth=1, nodes=nodes)
    d.depth = _critical_path(d)
    return d


def _req(flow="sequential", target_depth=2):
    return RequiredStructure(flow=flow, decomposability="coupled",
                             target_depth=target_depth, required_roles=set())


# ---------------------------------------------------------------------------
# SCALE
# ---------------------------------------------------------------------------

def test_collapse_layer_strictly_reduces_depth():
    # chain of depth 4
    dag = _dag([_node("a", []), _node("b", ["a"]), _node("c", ["b"]), _node("d", ["c"])])
    assert _critical_path(dag) == 4
    out, changed = G.collapse_layer(dag)
    assert changed
    assert _critical_path(out) == 3            # exactly one layer gone
    assert _critical_path(dag) == 4            # incumbent untouched (deep copy)


def test_collapse_reduces_scale_error_toward_target():
    dag = _dag([_node("a", []), _node("b", ["a"]), _node("c", ["b"]), _node("d", ["c"])])
    req = _req(target_depth=2)
    before = _scale_error(graph_signature(dag), req)
    out, _ = G.collapse_layer(dag)
    after = _scale_error(graph_signature(out), req)
    assert after < before                       # direction-correct, monotone


def test_collapse_exhausts_at_minimum():
    dag = _dag([_node("a", []), _node("b", ["a"])])  # depth 2
    out, changed = G.collapse_layer(dag)
    assert changed and _critical_path(out) == 1
    out2, changed2 = G.collapse_layer(out)          # depth 1 -> nothing to do
    assert changed2 is False


def test_add_layer_increases_depth_by_one():
    dag = _dag([_node("a", []), _node("b", ["a"])])  # depth 2
    out, changed = G.add_layer(dag)
    assert changed
    assert _critical_path(out) == 3
    assert len(out.nodes) == len(dag.nodes) + 1
    # new node authored with a real intent (executable without a planning call)
    assert all(n.intent.strip() for n in out.nodes)


def test_add_layer_preserves_parallel_gather():
    # two independent roots + join  (divergent shape)
    dag = _dag([_node("r1", []), _node("r2", []), _node("j", ["r1", "r2"])], "fan-out")
    req = _req(flow="divergent", target_depth=4)
    assert _flow_error(graph_signature(dag), req) == 0.0
    out, _ = G.add_layer(dag)
    # divergent still satisfied after deepening (fan-out preserved via new node)
    assert _flow_error(graph_signature(out), req) == 0.0


# ---------------------------------------------------------------------------
# FLOW
# ---------------------------------------------------------------------------

def test_linearize_makes_sequential_flow_zero():
    # branching plan: one root fanning to two, then a join
    dag = _dag([_node("r", []), _node("b1", ["r"]), _node("b2", ["r"]), _node("j", ["b1", "b2"])], "fan-out")
    req = _req(flow="sequential")
    assert _flow_error(graph_signature(dag), req) > 0.0
    out, changed = G.linearize(dag)
    assert changed
    sig = graph_signature(out)
    assert sig["root_count"] == 1 and sig["max_out"] <= 1
    assert _flow_error(sig, req) == 0.0
    assert len(out.nodes) == len(dag.nodes)     # surgical: no nodes added/dropped


def test_linearize_idempotent_on_a_chain():
    dag = _dag([_node("a", []), _node("b", ["a"]), _node("c", ["b"])])
    out, changed = G.linearize(dag)
    assert changed is False                     # already a chain


def test_split_to_parallel_makes_divergent_flow_zero():
    dag = _dag([_node("a", []), _node("b", ["a"]), _node("c", ["b"])])  # a chain
    req = _req(flow="divergent")
    assert _flow_error(graph_signature(dag), req) > 0.0
    out, changed = G.split_to_parallel(dag)
    assert changed
    sig = graph_signature(out)
    assert (sig["root_count"] >= 2 or sig["max_out"] >= 2) and sig["has_join"]
    assert _flow_error(sig, req) == 0.0


def test_add_join_makes_branches_converge():
    dag = _dag([_node("r1", []), _node("r2", [])], "fan-out")   # two sinks, no join
    req = _req(flow="divergent")
    before = _flow_error(graph_signature(dag), req)
    out, changed = G.add_join(dag)
    assert changed
    sig = graph_signature(out)
    assert sig["has_join"]
    assert _flow_error(sig, req) <= before


def test_add_merge_makes_convergent_flow_zero():
    dag = _dag([_node("r1", []), _node("r2", []), _node("r3", [])], "fan-out")  # 3 sinks
    req = _req(flow="convergent")
    assert _flow_error(graph_signature(dag), req) > 0.0
    out, changed = G.add_merge(dag)
    assert changed
    assert _flow_error(graph_signature(out), req) == 0.0


def test_deepen_recursive_reaches_depth_three():
    dag = _dag([_node("a", []), _node("b", ["a"])])  # depth 2
    req = _req(flow="recursive")
    assert _flow_error(graph_signature(dag), req) == 1.0
    out, changed = G.deepen_recursive(dag)
    assert changed
    assert _critical_path(out) >= 3
    assert _flow_error(graph_signature(out), req) == 0.0


# ---------------------------------------------------------------------------
# determinism / purity
# ---------------------------------------------------------------------------

def test_operators_are_pure_and_reproducible():
    dag = _dag([_node("r", []), _node("b1", ["r"]), _node("b2", ["r"]), _node("j", ["b1", "b2"])], "fan-out")
    o1, _ = G.linearize(dag)
    o2, _ = G.linearize(dag)
    # same input -> identical topology out, twice (claim 8, at the operator level)
    assert [(n.node_id, n.dependencies) for n in o1.nodes] == \
           [(n.node_id, n.dependencies) for n in o2.nodes]
    # incumbent never mutated
    assert [(n.node_id, n.dependencies) for n in dag.nodes] == \
           [("r", []), ("b1", ["r"]), ("b2", ["r"]), ("j", ["b1", "b2"])]


def test_registry_only_covers_topology_axes():
    assert G.is_deterministic_move("scale.collapse_layer")
    assert G.is_deterministic_move("flow.linearize")
    # v3.4: role.* is a pure set-membership / field transform -- it joined
    # the deterministic set (see role_add_missing / role_realign docstrings).
    assert G.is_deterministic_move("role.realign")
    assert G.is_deterministic_move("role.add_missing")
    # partition.* stays content-dependent (needs semantic judgment) -> LLM path
    assert not G.is_deterministic_move("partition.differentiate")


# ---------------------------------------------------------------------------
# ROLE
# ---------------------------------------------------------------------------

def test_role_add_missing_inserts_verifier_depending_on_sinks():
    dag = _dag([_node("a", [], role="retriever"), _node("b", ["a"], role="synthesizer")])
    required = {"synthesizer", "verifier"}
    sinks_before = set(G._sinks(dag))
    out, changed = G.role_add_missing(dag, required=required)
    assert changed is True
    assert len(out.nodes) == len(dag.nodes) + 1
    new_nodes = [n for n in out.nodes if n.roles.functional == "verifier"]
    assert len(new_nodes) == 1
    assert set(new_nodes[0].dependencies) == sinks_before
    # incumbent untouched
    assert len(dag.nodes) == 2


def test_role_add_missing_noop_when_nothing_missing():
    dag = _dag([_node("a", [], role="retriever"), _node("b", ["a"], role="verifier")])
    out, changed = G.role_add_missing(dag, required={"retriever", "verifier"})
    assert changed is False
    assert out is dag


def test_role_add_missing_noop_when_required_none():
    dag = _dag([_node("a", [], role="retriever")])
    out, changed = G.role_add_missing(dag, required=None)
    assert changed is False
    assert out is dag


def test_role_add_missing_idempotent_once_role_present():
    dag = _dag([_node("a", [], role="retriever"), _node("b", ["a"], role="synthesizer")])
    required = {"synthesizer", "verifier"}
    out, changed = G.role_add_missing(dag, required=required)
    assert changed is True
    out2, changed2 = G.role_add_missing(out, required=required)
    assert changed2 is False
    assert out2 is out


def test_role_realign_reassigns_overrepresented_node_to_absent_role():
    # two retrievers, no verifier -- retriever is over-represented (count 2),
    # verifier is required and absent. Realign should relabel the LAST
    # retriever in topological order onto "verifier".
    dag = _dag([
        _node("a", [], role="retriever"),
        _node("b", ["a"], role="retriever"),
        _node("c", ["b"], role="synthesizer"),
    ])
    required = {"retriever", "verifier"}
    out, changed = G.role_realign(dag, required=required)
    assert changed is True
    present_after = {n.roles.functional for n in out.nodes}
    assert "verifier" in present_after
    # the topologically-last retriever ("b") was relabeled, not the hub/root
    assert out.node("b").roles.functional == "verifier"
    assert out.node("a").roles.functional == "retriever"
    # incumbent untouched (deep copy)
    assert dag.node("b").roles.functional == "retriever"


def test_role_realign_noop_when_required_none():
    dag = _dag([_node("a", [], role="retriever"), _node("b", ["a"], role="retriever")])
    out, changed = G.role_realign(dag, required=None)
    assert changed is False
    assert out is dag


def test_role_realign_noop_when_nothing_missing():
    dag = _dag([_node("a", [], role="retriever"), _node("b", ["a"], role="verifier")])
    out, changed = G.role_realign(dag, required={"retriever", "verifier"})
    assert changed is False
    assert out is dag


def test_role_realign_noop_when_no_safe_donor():
    # no role is over-represented -> nothing safe to reassign
    dag = _dag([_node("a", [], role="retriever"), _node("b", ["a"], role="synthesizer")])
    out, changed = G.role_realign(dag, required={"retriever", "synthesizer", "verifier"})
    assert changed is False
    assert out is dag


# ---------------------------------------------------------------------------
# apply_move integration contract (dag, move_id, required=None)
# ---------------------------------------------------------------------------

def test_apply_move_dispatches_role_add_missing_with_required():
    dag = _dag([_node("a", [], role="retriever"), _node("b", ["a"], role="synthesizer")])
    out, changed = G.apply_move(dag, "role.add_missing", required={"synthesizer", "verifier"})
    assert changed is True
    assert any(n.roles.functional == "verifier" for n in out.nodes)


def test_apply_move_backward_compatible_without_required():
    # flow/scale moves still work called WITHOUT the required kwarg at all.
    dag = _dag([_node("a", []), _node("b", ["a"]), _node("c", ["b"]), _node("d", ["c"])])
    out, changed = G.apply_move(dag, "scale.collapse_layer")
    assert changed is True
    assert _critical_path(out) == 3


# ---------------------------------------------------------------------------
# PARTITION (v3.5 -- partition.merge_redundant)
# ---------------------------------------------------------------------------

def _node_intent(nid, deps, intent, role="generic"):
    return Node(node_id=nid, intent=intent,
                roles=Roles(functional=role), dependencies=list(deps),
                expected_output_fingerprint=[0.0])


def test_partition_merge_redundant_collapses_near_identical_sibling_roots():
    # two sibling roots (empty deps, identical intent-ish text) feeding a join
    dag = _dag([
        _node_intent("r2", [], "Retrieve current flight prices from Bangalore to Bangkok"),
        _node_intent("r1", [], "Retrieve current flight prices from Bangalore to Bangkok"),
        _node_intent("j", ["r1", "r2"], "Integrate the retrieved flight price data"),
    ], "fan-out")
    out, changed = G.partition_merge_redundant(dag)
    assert changed is True
    assert len(out.nodes) == len(dag.nodes) - 1
    join = out.node("j")
    assert join is not None
    # rewired onto the kept (lexicographically smaller) node, no dangling ref
    assert "r2" not in join.dependencies
    assert "r1" in join.dependencies
    existing_ids = {n.node_id for n in out.nodes}
    for n in out.nodes:
        assert n.node_id not in n.dependencies          # no self-dependency
        assert len(n.dependencies) == len(set(n.dependencies))  # no dup deps
        assert all(d in existing_ids for d in n.dependencies)   # no dangling
    # incumbent untouched
    assert len(dag.nodes) == 3


def test_partition_merge_redundant_keeps_lexicographically_smaller_id():
    dag = _dag([
        _node_intent("zeta", [], "Retrieve current flight prices from Bangalore to Bangkok"),
        _node_intent("alpha", [], "Retrieve current flight prices from Bangalore to Bangkok"),
    ], "fan-out")
    out, changed = G.partition_merge_redundant(dag)
    assert changed is True
    assert len(out.nodes) == 1
    assert out.nodes[0].node_id == "alpha"


def test_partition_merge_redundant_noop_when_siblings_are_distinct():
    dag = _dag([
        _node_intent("r1", [], "Retrieve current flight prices from Bangalore to Bangkok"),
        _node_intent("r2", [], "Summarize Thai visa rules for Indian passport holders"),
        _node_intent("j", ["r1", "r2"], "Integrate findings"),
    ], "fan-out")
    out, changed = G.partition_merge_redundant(dag)
    assert changed is False
    assert out is dag


def test_partition_merge_redundant_noop_when_no_sibling_pairs():
    # every node has a distinct dependency set -> no siblings at all
    dag = _dag([
        _node_intent("a", [], "Retrieve flight prices"),
        _node_intent("b", ["a"], "Summarize Thai visa rules"),
        _node_intent("c", ["a", "b"], "Integrate findings"),
    ])
    out, changed = G.partition_merge_redundant(dag)
    assert changed is False
    assert out is dag


def test_partition_merge_redundant_noop_below_two_nodes():
    dag = _dag([_node_intent("a", [], "Retrieve flight prices")])
    out, changed = G.partition_merge_redundant(dag)
    assert changed is False
    assert out is dag


def test_apply_move_dispatches_partition_merge_redundant_without_kwargs():
    dag = _dag([
        _node_intent("r1", [], "Retrieve current flight prices from Bangalore to Bangkok"),
        _node_intent("r2", [], "Retrieve current flight prices from Bangalore to Bangkok"),
        _node_intent("j", ["r1", "r2"], "Integrate the retrieved flight price data"),
    ], "fan-out")
    out, changed = G.apply_move(dag, "partition.merge_redundant")
    assert changed is True
    assert len(out.nodes) == len(dag.nodes) - 1


# ---------------------------------------------------------------------------
# v3.6 -- Fix A/B/C/D
# ---------------------------------------------------------------------------

def test_partition_merge_redundant_uses_real_partition_pairs_not_intent_guess():
    # intent text makes r1/r2 look near-identical (would win on the OLD intent
    # proxy), but the REAL measured partition_pairs says r1/r3 is the true
    # worst (highest-similarity) pair -- Fix A must merge THAT pair instead.
    dag = _dag([
        _node_intent("r1", [], "Retrieve current flight prices from Bangalore to Bangkok"),
        _node_intent("r2", [], "Retrieve current flight prices from Bangalore to Bangkok"),
        _node_intent("r3", [], "Summarize Thai visa rules for Indian passport holders"),
        _node_intent("j", ["r1", "r2", "r3"], "Integrate findings"),
    ], "fan-out")
    partition_pairs = {("r1", "r3"): 0.95, ("r1", "r2"): 0.10, ("r2", "r3"): 0.05}
    out, changed = G.partition_merge_redundant(dag, partition_pairs=partition_pairs)
    assert changed is True
    ids = {n.node_id for n in out.nodes}
    # r1 (kept, lexicographically smaller of r1/r3) survives, r3 (dropped) gone,
    # r2 (not part of the real worst pair) is untouched.
    assert "r3" not in ids
    assert "r1" in ids
    assert "r2" in ids


def test_partition_merge_redundant_refuses_when_it_would_break_divergent_gather_floor():
    # exactly 2 roots, no branching coordinator (max_out < 2) -- merging the
    # only two roots would drop root_count to 1 and max_out stays < 2,
    # breaking the divergent flow's parallel-gather floor. Fix B must refuse.
    dag = _dag([
        _node_intent("r1", [], "Retrieve current flight prices from Bangalore to Bangkok"),
        _node_intent("r2", [], "Retrieve current flight prices from Bangalore to Bangkok"),
        _node_intent("j", ["r1", "r2"], "Integrate the retrieved flight price data"),
    ], "fan-out")
    req = _req(flow="divergent")
    partition_pairs = {("r1", "r2"): 0.95}
    out, changed = G.partition_merge_redundant(dag, partition_pairs=partition_pairs, req=req)
    assert changed is False
    assert out is dag


def test_collapse_layer_never_deletes_the_sole_verifier():
    # chain a -> b -> c(verifier) -> d ; c is the DEEPEST non-sink candidate
    # (would be picked first) and is the sole required verifier. Fix C must
    # skip it and fall through to the next-deepest safe candidate (b)
    # instead, never deleting the verifier.
    dag = _dag([
        _node("a", [], role="retriever"),
        _node("b", ["a"], role="generic"),
        _node("c", ["b"], role="verifier"),
        _node("d", ["c"], role="synthesizer"),
    ])
    req = _req(target_depth=2)
    req.required_roles = {"verifier"}
    out, changed = G.collapse_layer(dag, req=req)
    assert changed is True
    assert any(n.roles.functional == "verifier" for n in out.nodes)
    assert out.node("c") is not None            # the verifier node itself survives
    # incumbent untouched regardless
    assert any(n.roles.functional == "verifier" for n in dag.nodes)


def test_compute_required_floors_recursive_target_depth_at_three():
    class _FP:
        epistemic_stance = "synthesis"
        output_contract = "artifact"
        information_flow = "recursive"
        decomposability = "coupled"

        def requires_verifier(self):
            return False

        def depth_cap(self):
            return 2  # low complexity cap, below the recursive-flow floor

    from ceo_delta.ef import compute_required
    req = compute_required(_FP())
    assert req.target_depth >= 3
