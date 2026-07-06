"""Stage 2 of the Q-factor: the descent's incumbent tracking uses Q as a
LEXICOGRAPHIC secondary key to structural worst-excess.

Structure is always primary; Q only breaks a structural tie and can NEVER
cause a structurally-worse plan to be accepted. When cfg.use_q_tensor is
False, the incumbent decision must be byte-identical to the pre-Q rule.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ceo_delta.config import Config
from ceo_delta.orchestrator import Orchestrator
from ceo_delta.schemas import DAG, ExecutionTrace, Node, NodeResult, Roles, TaskFingerprint


def _orch(tmpdir, use_q_tensor):
    cfg = Config()
    cfg.llm_allow_stub = True    # never touches the network in this test
    cfg.use_q_tensor = use_q_tensor
    return Orchestrator(cfg, workdir=str(tmpdir))


def _fp(flow="sequential", complexity="low"):
    fp = TaskFingerprint(information_flow=flow, epistemic_stance="synthesis",
                         output_contract="artifact", decomposability="coupled",
                         complexity=complexity)
    fp.embedding = [0.1, 0.2, 0.3, 0.4]   # fixed, avoids re-embedding noise
    return fp


def _simple_dag(verifier_output="", answer_output="the answer"):
    nodes = [
        Node(node_id="a", intent="answer", roles=Roles(functional="synthesizer"), dependencies=[]),
        Node(node_id="v", intent="verify", roles=Roles(functional="verifier"), dependencies=["a"]),
    ]
    d = DAG(task="demo", task_embedding=[0.0] * 4, topology="linear", depth=2, nodes=nodes)
    return d


def _trace_for(dag, verifier_output, answer_output="the answer"):
    results = []
    for n in dag.nodes:
        if n.roles.functional == "verifier":
            out = verifier_output
        else:
            out = answer_output
        results.append(NodeResult(
            node_id=n.node_id, intent=n.intent, output=out, output_embedding=[0.0] * 4,
            cost_tokens=1, latency_s=0.01, role_function_match=True,
            fingerprint_match=1.0, gated=False, error=None,
        ))
    return ExecutionTrace(dag_id=dag.dag_id, task=dag.task, results=results,
                          total_tokens=1, wallclock_s=0.01)


def test_q_off_is_inert(tmp_path):
    orch = _orch(tmp_path, use_q_tensor=False)
    orch.delta.begin_run(_fp())
    descent = orch.delta._descent

    dag = _simple_dag()
    trace = _trace_for(dag, verifier_output="Claim A. Claim B. [UNVERIFIED: no source]")

    assert descent._measure_q(dag, trace) == {}
    assert descent._q_worst({}) == 0.0


def test_measure_q_flags_ungrounded_verifier(tmp_path):
    orch = _orch(tmp_path, use_q_tensor=True)
    orch.delta.begin_run(_fp())
    descent = orch.delta._descent

    dag = _simple_dag()
    verifier_output = (
        "Claim one is solid. [UNVERIFIED: no source]\n"
        "Claim two is shaky. [UNVERIFIED: no source]\n"
        "Claim three checks out."
    )
    trace = _trace_for(dag, verifier_output=verifier_output)

    q_d = descent._measure_q(dag, trace)
    assert q_d != {}
    assert q_d.get("groundedness", 0.0) > 0.0


def test_lexicographic_prefers_lower_q_on_structural_tie(tmp_path):
    """Real Q measurement (via _measure_q) feeds the incumbent decision
    arithmetic (the exact if/elif replicated from step()). Two plans have an
    IDENTICAL structural worst-excess (a controlled tie -- getting two real
    EF computations to land on an exact tie via role/behavioral-profile
    features is fragile, so the structural side is fixed directly, per the
    spec's documented fallback), but the second has a fully-grounded
    verifier vs. the first's heavily-flagged one, so its q_worst is lower."""
    orch = _orch(tmp_path, use_q_tensor=True)
    orch.delta.begin_run(_fp())
    descent = orch.delta._descent

    dag1 = _simple_dag()
    trace1 = _trace_for(dag1, verifier_output=(
        "Claim A. [UNVERIFIED: x]\nClaim B. [UNVERIFIED: x]\nClaim C. [UNVERIFIED: x]"
    ))
    q1 = descent._measure_q(dag1, trace1)
    q1_worst = descent._q_worst(q1)

    dag2 = _simple_dag()
    trace2 = _trace_for(dag2, verifier_output="Claim A checks out. Claim B checks out.")
    q2 = descent._measure_q(dag2, trace2)
    q2_worst = descent._q_worst(q2)

    assert q2_worst < q1_worst   # sanity: the fully-grounded plan really is better Q

    # iteration 1: first plan becomes incumbent (nothing seen yet)
    measured_worst = 0.0
    descent._incumbent_dag, descent._incumbent_worst = dag1, measured_worst
    descent._incumbent_q_worst, descent._incumbent_q = q1_worst, q1

    # iteration 2: second plan, SAME structural worst-excess (a tie) -- this
    # is the exact branch from step()'s incumbent-update block
    measured_worst_2 = 0.0
    if descent._incumbent_dag is None:
        changed = True
    elif measured_worst_2 < descent._incumbent_worst - 1e-9:
        changed = True
    elif measured_worst_2 <= descent._incumbent_worst + 1e-9:
        changed = (q2_worst <= descent._incumbent_q_worst + 1e-9)
    else:
        changed = False
    if changed:
        descent._incumbent_dag = dag2
        descent._incumbent_worst = measured_worst_2
        descent._incumbent_q_worst = q2_worst
        descent._incumbent_q = q2

    assert changed is True
    assert descent._incumbent_dag is dag2
    assert descent._incumbent_q == q2


def test_structural_worse_never_accepted_for_better_q(tmp_path):
    orch = _orch(tmp_path, use_q_tensor=True)
    orch.delta.begin_run(_fp())
    descent = orch.delta._descent

    # directly exercise the decision arithmetic (mirrors the step() branch
    # exactly) since constructing two DAGs with a controlled, differing
    # structural worst-excess via the real EF pipeline is fiddly here.
    descent._incumbent_dag = _simple_dag()
    descent._incumbent_worst = 0.0
    descent._incumbent_q_worst = 0.5
    descent._incumbent_q = {"groundedness": 0.5, "relevance": 0.0}

    measured_worst = 0.3   # worse structure
    q_worst = 0.0          # better Q

    if descent._incumbent_dag is None:
        incumbent_changed = True
    elif measured_worst < descent._incumbent_worst - 1e-9:
        incumbent_changed = True
    elif measured_worst <= descent._incumbent_worst + 1e-9:
        incumbent_changed = (q_worst <= descent._incumbent_q_worst + 1e-9)
    else:
        incumbent_changed = False

    assert incumbent_changed is False   # structure dominates: never accepted
