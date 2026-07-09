#!/usr/bin/env python3
"""Attractor / fixed-point test — the honest replacement for "deterministic
seed hardcoding."

CEO stays fully free and creative at iteration 0 (no changes to ceo.py --
this is deliberate: narrowing CEO's initial choice to a templated skeleton
would make it "a script wearing an LLM costume" for structure, which is a
real thing to want to avoid). Instead of proving the SEED is identical run to
run, we prove the CONVERGED STRUCTURE is identical run to run, regardless of
what CEO guessed first. That is a strictly stronger systems claim: a control
loop whose fixed point is independent of initial condition.

Design:
  - Run the SAME task N times, each against a FRESH .ceo_delta store (deleted
    between reps) so no cross-run move-priority learning can shortcut the
    result -- each rep's CEO is planning cold, with no memory of prior reps.
  - Capture CEO's iteration-0 structural guess (topology, depth, node count)
    each rep -- this is EXPECTED to vary, since CEO is a real creative
    planner sampling from an LLM.
  - Capture the FINAL delivered structure's shape signature (depth,
    root_count, has_join, max_out, functional-role multiset) each rep -- this
    is the claim under test: it should NOT vary, because the deterministic
    flow/scale repair operators (graph_ops.py) drive any starting guess to
    the same F-satisfying shape.

Usage:
  python3 scripts/attractor_test.py --reps 5
  python3 scripts/attractor_test.py --reps 5 --task "..." --workdir-root .attractor_runs
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ceo_delta.config import Config
from ceo_delta.ef import graph_signature
from ceo_delta.orchestrator import Orchestrator

DEFAULT_TASK = (
    "Trace how a compiler transforms source code into machine code through "
    "its ordered passes: lexical analysis, then parsing, then semantic "
    "analysis, then optimization, then code generation, where each pass "
    "operates strictly on the output of the previous pass and cannot begin "
    "until the previous one completes."
)


def _require_live_backend(cfg) -> None:
    try:
        urllib.request.urlopen(cfg.llm_base_url, timeout=5)
    except urllib.error.HTTPError:
        pass
    except Exception as e:
        print(f"ERROR: vLLM backend at {cfg.llm_base_url} is unreachable ({e}). "
              f"Set CEO_LLM_URL or start the server before running this script.")
        sys.exit(1)


def _shape(dag) -> dict:
    """Structural signature, independent of node wording -- topology only."""
    sig = graph_signature(dag)
    roles = sorted(n.roles.functional for n in dag.nodes)
    return {
        "depth": sig["depth"],
        "root_count": sig["root_count"],
        "sink_count": sig["sink_count"],
        "max_out": sig["max_out"],
        "has_join": sig["has_join"],
        "n_nodes": sig["n_nodes"],
        "role_multiset": roles,
    }


def run_rep(task: str, workdir: str, rep: int, temperature: float | None = None) -> dict:
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)  # fresh store -- no cross-run learning shortcut
    cfg = Config()
    if temperature is not None:
        # Scoped to THIS test only -- the production default (0.0) stays
        # untouched everywhere else. temp=0 makes CEO's cold-start guess
        # byte-identical every rep by construction (same prompt -> same
        # greedy output), which means the test can only ever show trivial
        # reproducibility, never the attractor claim (convergence FROM
        # varying starts). A nonzero temperature here lets CEO's iteration-0
        # guess actually vary, so "distinct seeds, 1 final shape" becomes
        # a meaningful result instead of a foregone conclusion.
        cfg.llm_temperature = temperature
    orch = Orchestrator(cfg, workdir=workdir)
    r = orch.run(task, task_label=f"attractor.rep{rep}")

    seed_shape = None
    if r.ef_trace:
        # iteration-0's raw plan shape isn't stored directly on RunResult, but
        # the first ef_trace entry's signature came from CEO's cold plan
        # before any repair -- reconstruct it from descent's own recorded
        # detail trace (signature is captured pre-move each iteration).
        pass
    return {
        "rep": rep,
        "seed_signature": r.ef_detail[0]["signature"] if r.ef_detail else None,
        "final_signature": _shape(r.dag) if r.dag else None,
        "iterations": r.iterations,
        "verdict": r.report.verdict if r.report else None,
        "final_ef": r.report.ef_tensor if r.report else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--workdir-root", default=".attractor_runs")
    ap.add_argument("--out", default="attractor_results.jsonl")
    ap.add_argument("--temperature", type=float, default=None,
                     help="Override llm_temperature for THIS test only (production "
                          "default is 0.0/deterministic). Use e.g. 0.7 to let CEO's "
                          "cold-start guess actually vary across reps -- otherwise "
                          "every rep's seed is byte-identical by construction and "
                          "the run can only show trivial reproducibility, not the "
                          "attractor/fixed-point property.")
    args = ap.parse_args()

    cfg = Config()
    _require_live_backend(cfg)

    records = []
    with open(args.out, "w") as fout:
        for i in range(1, args.reps + 1):
            print(f"[rep {i}/{args.reps}] running (fresh store)...")
            rec = run_rep(args.task, f"{args.workdir_root}/rep{i}", i, temperature=args.temperature)
            records.append(rec)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            print(f"  seed={rec['seed_signature']}")
            print(f"  final={rec['final_signature']}  iterations={rec['iterations']} "
                  f"verdict={rec['verdict']}")

    analyze(records)


def analyze(records):
    print("\n" + "=" * 60)
    print("ATTRACTOR / FIXED-POINT ANALYSIS")
    seeds = [json.dumps(r["seed_signature"], sort_keys=True) for r in records]
    finals = [json.dumps(r["final_signature"], sort_keys=True) for r in records]

    n_distinct_seeds = len(set(seeds))
    n_distinct_finals = len(set(finals))
    print(f"\n  {len(records)} reps of the SAME task, fresh store each time.")
    print(f"  distinct seed (iteration-0) shapes : {n_distinct_seeds}")
    print(f"  distinct FINAL (converged) shapes  : {n_distinct_finals}")

    if n_distinct_seeds > 1 and n_distinct_finals == 1:
        print("\n  RESULT: CEO's cold-start guess varied across reps (real, "
              "creative, non-templated planning) but the deterministic repair "
              "loop drove every rep to the IDENTICAL final structure.")
        print("  This is a fixed point independent of initial condition -- "
              "the strong form of reproducibility/generalization claims 7/8.")
    elif n_distinct_finals == 1:
        print("\n  RESULT: final shape identical across reps (seed also "
              "happened to be identical every time -- CEO's guess didn't "
              "vary for this task, so this run doesn't exercise the "
              "attractor property, only trivial reproducibility).")
    else:
        print("\n  RESULT: final shape VARIED across reps -- the descent did "
              "NOT converge to a unique fixed point for this task. Inspect "
              "per-rep ef_trace/move_log to see which axis diverged.")
        for r in records:
            print(f"    rep {r['rep']}: final={r['final_signature']}")


if __name__ == "__main__":
    main()
