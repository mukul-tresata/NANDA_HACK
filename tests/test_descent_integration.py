"""Integration proof: the orchestrator's directive loop realizes a
deterministic flow/scale move as a GRAPH TRANSFORM on the incumbent (no
planning LLM call), not as a stochastic replan.

This is the seam that turns the descent from "propose + regenerate + reject"
(which stalled at the cold plan) into an actual coordinate-descent step.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ceo_delta.config import Config
from ceo_delta.ef import graph_signature, _flow_error, _scale_error, _critical_path
from ceo_delta.orchestrator import Orchestrator
from ceo_delta.schemas import DAG, DeltaDirective, Node, Roles, TaskFingerprint


def _orch(tmpdir):
    cfg = Config()
    cfg.llm_allow_stub = True            # never touches the network in this test
    return Orchestrator(cfg, workdir=str(tmpdir))


def _fp(flow, complexity="low"):
    fp = TaskFingerprint(information_flow=flow, epistemic_stance="synthesis",
                         output_contract="artifact", decomposability="coupled",
                         complexity=complexity)
    fp.embedding = [0.1, 0.2, 0.3, 0.4]   # fixed, avoids re-embedding noise
    return fp


def _branching_dag():
    nodes = [
        Node(node_id="r", intent="root", roles=Roles(functional="retriever"), dependencies=[]),
        Node(node_id="b1", intent="branch1", roles=Roles(functional="retriever"), dependencies=["r"]),
        Node(node_id="b2", intent="branch2", roles=Roles(functional="retriever"), dependencies=["r"]),
        Node(node_id="j", intent="join", roles=Roles(functional="synthesizer"), dependencies=["b1", "b2"]),
    ]
    d = DAG(task="demo", task_embedding=[0.0] * 4, topology="fan-out", depth=2, nodes=nodes)
    d.depth = _critical_path(d)
    return d


def test_loop_applies_linearize_as_graph_transform(tmp_path):
    orch = _orch(tmp_path)
    orch.delta.begin_run(_fp("sequential"))
    descent = orch.delta._descent
    descent._incumbent_dag = _branching_dag()

    req = descent.req
    before = _flow_error(graph_signature(descent._incumbent_dag), req)
    assert before > 0.0                              # branching violates sequential

    directive = DeltaDirective(action="replan", reason="ef_move:flow.linearize")
    new_dag, used_det = orch._next_dag(
        current_task="demo", prev_directive=directive,
        task_class="synthesis", fingerprint=_fp("sequential"),
    )

    assert used_det is True                          # graph transform path, not ceo.plan
    assert _flow_error(graph_signature(new_dag), req) == 0.0
    # incumbent untouched (descent needs it intact for rejection)
    assert _flow_error(graph_signature(descent._incumbent_dag), req) == before


def test_loop_applies_collapse_as_graph_transform(tmp_path):
    orch = _orch(tmp_path)
    orch.delta.begin_run(_fp("sequential", complexity="low"))  # target_depth = 2
    descent = orch.delta._descent
    # deep chain: depth 4, target 2 -> scale over threshold
    chain = [Node(node_id=c, intent=c, roles=Roles(functional="generic"),
                  dependencies=([p] if p else []))
             for c, p in [("a", None), ("b", "a"), ("c", "b"), ("d", "c")]]
    dag = DAG(task="demo", task_embedding=[0.0] * 4, topology="linear", depth=4, nodes=chain)
    dag.depth = _critical_path(dag)
    descent._incumbent_dag = dag

    req = descent.req
    before = _scale_error(graph_signature(dag), req)
    directive = DeltaDirective(action="replan", reason="ef_move:scale.collapse_layer")
    new_dag, used_det = orch._next_dag("demo", directive, "synthesis", _fp("sequential"))

    assert used_det is True
    assert _critical_path(new_dag) == 3              # exactly one layer collapsed
    assert _scale_error(graph_signature(new_dag), req) < before


def test_content_move_falls_back_to_ceo_plan(tmp_path):
    orch = _orch(tmp_path)
    orch.delta.begin_run(_fp("sequential"))
    orch.delta._descent._incumbent_dag = _branching_dag()

    called = {"plan": False}
    real_plan = orch.ceo.plan

    def spy(*a, **k):
        called["plan"] = True
        return real_plan(*a, **k)

    orch.ceo.plan = spy
    directive = DeltaDirective(action="refine", reason="ef_move:role.realign")
    _, used_det = orch._next_dag("demo", directive, "synthesis", _fp("sequential"))
    assert used_det is False
    assert called["plan"] is True                    # content axes still use the LLM
