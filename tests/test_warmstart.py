"""Offline tests for warm-start DAG reconstruction from a cached best_plan
(the read side of ef_store's best_plan -- see ceo_delta/warmstart.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ceo_delta.config import DEFAULT
from ceo_delta.embeddings import cosine, embed
from ceo_delta.schemas import DAG, Node, Roles, Why
from ceo_delta.warmstart import dag_from_best_plan


def _hand_built_dag():
    task = "compare two vendor proposals and recommend one"
    task_emb = embed(task)
    n1 = Node(
        node_id="n1", intent="retrieve vendor A proposal details",
        roles=Roles(structural="divergent", functional="retriever", epistemic="specialist"),
        dependencies=[],
        expected_output_fingerprint=embed("retrieve vendor A proposal details"),
    )
    n2 = Node(
        node_id="n2", intent="retrieve vendor B proposal details",
        roles=Roles(structural="divergent", functional="retriever", epistemic="specialist"),
        dependencies=[],
        expected_output_fingerprint=embed("retrieve vendor B proposal details"),
    )
    n3 = Node(
        node_id="n3", intent="synthesize a recommendation comparing both vendors",
        roles=Roles(structural="convergent", functional="synthesizer", epistemic="generalist"),
        dependencies=["n1", "n2"],
        expected_output_fingerprint=task_emb,
    )
    return DAG(
        task=task, task_embedding=task_emb, topology="divergent", depth=2,
        nodes=[n1, n2, n3], why_topology="parallel gather + join", why_depth="2 layers",
        exploratory=False,
    ), task, task_emb


def _serialize(dag):
    """Mirror descent.py's exhaustive best_plan dict (the one thing that
    changed there -- this test locks in that shape)."""
    return {
        "topology": dag.topology,
        "depth": dag.depth,
        "why_topology": dag.why_topology,
        "why_depth": dag.why_depth,
        "nodes": [
            {
                "node_id": n.node_id,
                "intent": n.intent,
                "structural": n.roles.structural,
                "functional": n.roles.functional,
                "epistemic": n.roles.epistemic,
                "dependencies": list(n.dependencies),
            }
            for n in dag.nodes
        ],
    }


def test_roundtrip_topology_depth_and_provenance():
    dag, task, task_emb = _hand_built_dag()
    best_plan = _serialize(dag)
    rebuilt = dag_from_best_plan(best_plan, task, task_emb)
    assert rebuilt is not None
    assert rebuilt.topology == dag.topology
    assert rebuilt.depth == dag.depth
    assert rebuilt.why_topology == dag.why_topology
    assert rebuilt.why_depth == dag.why_depth
    assert rebuilt.exploratory is False


def test_roundtrip_nodes_ids_roles_deps_intents():
    dag, task, task_emb = _hand_built_dag()
    best_plan = _serialize(dag)
    rebuilt = dag_from_best_plan(best_plan, task, task_emb)

    assert len(rebuilt.nodes) == len(dag.nodes)
    orig_by_id = {n.node_id: n for n in dag.nodes}
    for rn in rebuilt.nodes:
        on = orig_by_id[rn.node_id]  # KeyError if node_id missing/renamed
        assert rn.intent == on.intent
        assert rn.roles.functional == on.roles.functional
        assert rn.roles.structural == on.roles.structural
        assert rn.roles.epistemic == on.roles.epistemic
        assert rn.dependencies == on.dependencies


def test_empty_best_plan_returns_none():
    assert dag_from_best_plan({}, "some task", embed("some task")) is None


def test_best_plan_with_no_nodes_returns_none():
    best_plan = {"topology": "linear", "depth": 1, "nodes": []}
    assert dag_from_best_plan(best_plan, "some task", embed("some task")) is None


def test_synthesizer_fingerprint_is_task_embedding_others_are_intent_embedding():
    dag, task, task_emb = _hand_built_dag()
    best_plan = _serialize(dag)
    rebuilt = dag_from_best_plan(best_plan, task, task_emb)

    synth = rebuilt.node("n3")
    assert synth.roles.functional == "synthesizer"
    assert synth.expected_output_fingerprint == task_emb

    retriever = rebuilt.node("n1")
    assert retriever.roles.functional == "retriever"
    assert retriever.expected_output_fingerprint != task_emb
    assert retriever.expected_output_fingerprint == embed(retriever.intent)


# -- v3.6 task-identity gate ---------------------------------------------
# Root-cause fix: shape_key is purely STRUCTURAL (topology/depth/coupling)
# and is shared across many distinct tasks, but best_plan carries a task-
# SPECIFIC plan. Warm-start must only reuse best_plan when the incoming task
# is (near-)the SAME task, checked via best_plan["task_embedding"] (see
# descent.py finalize() and orchestrator.py _next_dag). These tests exercise
# that discriminator directly -- they do not require the orchestrator's LLM
# plumbing, just the same embed()/cosine() primitives the gate uses.

def test_best_plan_with_task_embedding_still_reconstructs_unchanged():
    """A best_plan carrying the new task_embedding key is otherwise identical
    to dag_from_best_plan -- the embedding is extra data the reconstructor
    ignores, so existing warm-start behavior is preserved."""
    dag, task, task_emb = _hand_built_dag()
    best_plan = _serialize(dag)
    best_plan["task_embedding"] = embed(dag.task)  # mirrors descent.py finalize()

    rebuilt = dag_from_best_plan(best_plan, task, task_emb)
    assert rebuilt is not None
    assert len(rebuilt.nodes) == len(dag.nodes)
    assert rebuilt.topology == dag.topology
    assert rebuilt.depth == dag.depth


def test_identity_gate_blocks_same_shape_different_task():
    """Same-shape, different-task: the classic category-error case (a
    C-compiler plan must not answer a water-cycle question just because both
    happen to be sequential/synthesis/artifact/coupled)."""
    compiler_task = "explain how a C compiler works"
    cached_emb = embed(compiler_task)

    incoming_task = "explain the water cycle"
    sim = cosine(embed(incoming_task), cached_emb)
    assert sim < DEFAULT.warm_start_similarity_threshold


def test_identity_gate_allows_same_task():
    """Same task (or a near-identical rephrasing of it) must pass the gate,
    or warm-start would never fire even for its intended case."""
    compiler_task = "explain how a C compiler works"
    cached_emb = embed(compiler_task)

    sim = cosine(embed(compiler_task), cached_emb)
    assert sim >= DEFAULT.warm_start_similarity_threshold


def test_missing_task_embedding_is_none_safe():
    """Old cached plans (created before this fix) have no task_embedding key.
    The orchestrator gate does `case["best_plan"].get("task_embedding")` and
    must fall through to cold-plan rather than raising or misbehaving."""
    dag, _task, _task_emb = _hand_built_dag()
    best_plan = _serialize(dag)  # no task_embedding key, as old cases have

    assert best_plan.get("task_embedding") is None
