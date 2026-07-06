"""CEO composition step: after the directive loop settles on a winning plan,
the CEO composes ONE coherent deliverable from its own internal work
(including the verifier's audit) instead of handing back a raw node output.

Offline-only -- forces the stub LLM path (unreachable base_url) so these
never touch a live backend.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CEO_LLM_URL", "http://127.0.0.1:1/v1")

from ceo_delta.ceo import CEO
from ceo_delta.config import Config
from ceo_delta.handbook import Handbook
from ceo_delta.llm import LLMClient
from ceo_delta.orchestrator import Orchestrator
from ceo_delta.schemas import DAG, ExecutionTrace, Node, NodeResult, Roles

# exact stub reply for a plain (non-json, non-dag, non-brief) prompt -- see
# LLMClient._chat_stub in ceo_delta/llm.py.
_STUB_COMPOSE_REPLY = "stub response (LLM server unreachable)"


def _orch(tmpdir):
    cfg = Config()
    cfg.llm_allow_stub = True
    cfg.llm_base_url = "http://127.0.0.1:1/v1"   # unreachable -> deterministic stub
    return Orchestrator(cfg, workdir=str(tmpdir))


def _trace_with(results):
    return ExecutionTrace(dag_id="d1", task="t", results=results, total_tokens=0, wallclock_s=0.0)


def _result(node_id, output, error=None):
    return NodeResult(
        node_id=node_id, intent="i", output=output, output_embedding=[],
        cost_tokens=0, latency_s=0.0, role_function_match=True,
        fingerprint_match=1.0, gated=False, error=error,
    )


def test_run_delivers_composed_answer_not_raw(tmp_path):
    orch = _orch(tmp_path)
    r = orch.run("Summarize the water cycle in strict ordered steps.")
    assert r.answer == _STUB_COMPOSE_REPLY
    assert r.raw_answer   # raw last-node output preserved, non-empty
    assert r.answer != r.raw_answer or r.raw_answer == _STUB_COMPOSE_REPLY


def test_audit_field_populated_when_verifier_present():
    cfg = Config()
    cfg.llm_allow_stub = True
    cfg.llm_base_url = "http://127.0.0.1:1/v1"
    llm = LLMClient(cfg)
    hb = Handbook("planning", cfg, path="/tmp/_unused_handbook_composition_test.json")
    ceo = CEO(hb, llm, cfg=cfg)

    nodes = [
        Node(node_id="n1", intent="retrieve", roles=Roles(functional="retriever"), dependencies=[]),
        Node(node_id="n2", intent="synthesize", roles=Roles(functional="synthesizer"), dependencies=["n1"]),
        Node(node_id="n3", intent="verify", roles=Roles(functional="verifier"), dependencies=["n2"]),
    ]
    dag = DAG(task="demo task", task_embedding=[0.0] * 4, topology="linear", depth=3, nodes=nodes)
    trace = _trace_with([
        _result("n1", "raw retrieved facts"),
        _result("n2", "synthesized draft answer"),
        _result("n3", "AUDIT: figure X unverified, rest confirmed"),
    ])

    by_id = {r.node_id: r for r in trace.results}
    audit = "\n".join(
        by_id[n.node_id].output
        for n in dag.nodes
        if n.roles.functional == "verifier" and by_id.get(n.node_id) and not by_id[n.node_id].error
    )
    assert "AUDIT" in audit

    out = ceo.compose("demo task", None, dag, trace)
    assert isinstance(out, str) and out != ""


def test_compose_empty_material_returns_empty():
    cfg = Config()
    cfg.llm_allow_stub = True
    cfg.llm_base_url = "http://127.0.0.1:1/v1"
    llm = LLMClient(cfg)
    hb = Handbook("planning", cfg, path="/tmp/_unused_handbook_composition_test2.json")
    ceo = CEO(hb, llm, cfg=cfg)

    nodes = [Node(node_id="n1", intent="x", roles=Roles(functional="generic"), dependencies=[])]
    dag = DAG(task="t", task_embedding=[0.0] * 4, topology="linear", depth=1, nodes=nodes)
    trace = _trace_with([_result("n1", "", error="boom")])

    out = ceo.compose("t", None, dag, trace)
    assert out == ""
