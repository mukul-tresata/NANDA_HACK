"""Property test for the PRESERVE guarantee: across an arbitrary sequence of
measured plans, the incumbent's worst-axis excess is monotone non-increasing.

This is the claim the demo's DESCEND panel and PROJECT_OVERVIEW both lean on
("a good repair is kept; a worse re-roll is measured and rejected"). It was
previously defended only by specific worked examples (test_descent_q.py,
test_descent_integration.py); this pins it as a property over randomized
inputs, driven through the real Descent.step() uphill-rejection logic
(descent.py:249-283), not a reimplementation of that logic.

No network/LLM calls: llm_allow_stub=True and use_q_tensor defaults False, so
_measure_q short-circuits before touching dag/trace.
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ceo_delta.config import Config
from ceo_delta.ef import EFTensor
from ceo_delta.orchestrator import Orchestrator
from ceo_delta.schemas import DAG, Node, Roles, TaskFingerprint


def _orch(tmpdir):
    cfg = Config()
    cfg.llm_allow_stub = True  # never touches the network
    return Orchestrator(cfg, workdir=str(tmpdir))


def _fp():
    fp = TaskFingerprint(information_flow="sequential", epistemic_stance="synthesis",
                          output_contract="artifact", decomposability="coupled",
                          complexity="low")
    fp.embedding = [0.1, 0.2, 0.3, 0.4]
    return fp


def _fixed_dag():
    nodes = [
        Node(node_id="n1", intent="a", roles=Roles(functional="retriever"), dependencies=[]),
        Node(node_id="n2", intent="b", roles=Roles(functional="synthesizer"), dependencies=["n1"]),
    ]
    return DAG(task="property-test", task_embedding=[0.0] * 4, topology="linear",
               depth=2, nodes=nodes)


def _worst_excess(ef_d, th) -> float:
    return max(ef_d[ax] - th[ax] for ax in ("partition", "flow", "role", "scale"))


def test_incumbent_worst_excess_is_monotone_over_random_trajectories(tmp_path):
    rng = random.Random(20260710)  # fixed seed: deterministic, reproducible failures
    n_trials, n_iterations = 200, 8
    dag = _fixed_dag()
    # One Orchestrator reused across trials -- begin_run() builds a fresh Descent
    # per trial (cheap; delta.py:251), so trials stay independent without paying
    # to reconstruct the Orchestrator (embedding model, handbook load) 200 times.
    orch = _orch(tmp_path)

    for trial in range(n_trials):
        orch.delta.begin_run(_fp())
        descent = orch.delta._descent
        th = descent.thresholds()

        prev_worst = None
        for iteration in range(n_iterations):
            ef = EFTensor(
                partition=rng.random(), flow=rng.random(),
                role=rng.random(), scale=rng.random(),
            )
            descent.step(dag, trace=None, ef=ef, iteration=iteration)

            worst_now = _worst_excess(descent._incumbent_ef.as_dict(), th)
            if prev_worst is not None:
                assert worst_now <= prev_worst + 1e-9, (
                    f"trial {trial}, iter {iteration}: incumbent worst-axis excess "
                    f"increased ({prev_worst} -> {worst_now}) -- PRESERVE violated"
                )
            prev_worst = worst_now

            # incumbent_trace must record the same monotone sequence we just checked
            recorded = _worst_excess(descent.incumbent_trace[-1], th)
            assert abs(recorded - worst_now) < 1e-9


def test_incumbent_never_worse_than_the_first_measurement(tmp_path):
    """A direct corollary: whatever the incumbent ends on, it's never worse
    than iteration 0's measurement -- the loop can only hold or improve."""
    rng = random.Random(2026)
    dag = _fixed_dag()
    orch = _orch(tmp_path)

    for trial in range(50):
        orch.delta.begin_run(_fp())
        descent = orch.delta._descent
        th = descent.thresholds()

        first_ef = EFTensor(partition=rng.random(), flow=rng.random(),
                             role=rng.random(), scale=rng.random())
        descent.step(dag, trace=None, ef=first_ef, iteration=0)
        first_worst = _worst_excess(descent._incumbent_ef.as_dict(), th)

        for iteration in range(1, 6):
            ef = EFTensor(partition=rng.random(), flow=rng.random(),
                          role=rng.random(), scale=rng.random())
            descent.step(dag, trace=None, ef=ef, iteration=iteration)

        final_worst = _worst_excess(descent._incumbent_ef.as_dict(), th)
        assert final_worst <= first_worst + 1e-9
