#!/usr/bin/env python3
"""Cross-run learning + task-family generalization, in one harness.

Why one test proves both claims at once:

  ef_store.py keys learned moves and irreducibility counters on
  fingerprint.shape_string() -- the STRUCTURAL axes only (information_flow,
  epistemic_stance, output_contract, decomposability). It never sees the raw
  task text. So if we run three surface-DIFFERENT tasks that share one
  fingerprint species back to back, and iterations-to-converge / initial
  error trend down across them, that improvement is structurally incapable
  of being literal memorization -- the system has never seen any of these
  exact words before. It can only be transferring a learned MOVE for that
  species. Cross-run learning and structural generalization (M2) are the
  same observation here, not two experiments.

Design:
  - 3 families (fingerprint species), 3 surface-different variants each.
  - 1 control task in a 4th, never-before-seen species, run LAST.
    Its first-run numbers should look like a typical "variant 1" (cold),
    not like a "variant 3" (warmed) -- that rules out the confound of
    "the whole session just gets easier over time" instead of
    "this specific species got easier because we've seen it before."

Usage:
  # needs the vLLM server reachable (CEO_LLM_URL, default http://10.8.0.23:8001/v1)
  python3 scripts/cross_run_learning_test.py
  python3 scripts/cross_run_learning_test.py --out results.jsonl --workdir .ceo_delta_demo

Then:
  python3 scripts/plot_learning_curve.py results.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ceo_delta.config import Config
from ceo_delta.orchestrator import Orchestrator

FAMILIES = {
    "A_divergent_enumerate": [
        "What are all the different aspects of what makes a person trustworthy?",
        "What are the key dimensions of a resilient distributed system?",
        "Break down the core components of a healthy organizational culture.",
    ],
    "B_sequential_pipeline": [
        "Trace how a compiler transforms source code into machine code through "
        "its ordered passes: lexical analysis, then parsing, then semantic "
        "analysis, then optimization, then code generation, where each pass "
        "operates strictly on the output of the previous pass and cannot begin "
        "until the previous one completes.",
        "Walk through how an HTTP request is processed by a web server from "
        "socket accept to response flush, where each stage strictly depends on "
        "the prior stage's completed output and cannot start early.",
        "Explain the TCP three-way handshake and the data transfer that "
        "follows it as a strict sequence of dependent steps, where no step can "
        "begin before the previous one completes.",
    ],
    "C_recursive_exhaustive": [
        "Explain what a stack data structure is, but do it exhaustively -- a "
        "deeply layered, multi-level structured breakdown covering every "
        "mechanism, operation, and nuance in comprehensive depth.",
        "Explain binary search trees exhaustively, as a deeply nested "
        "breakdown covering every layer of structure and operation in "
        "comprehensive depth.",
        "Provide an exhaustive, recursively layered explanation of how "
        "quicksort partitions and recurses on subarrays, covering every level "
        "of the recursion in comprehensive depth.",
    ],
}

CONTROL_TASK = (
    "Compare the CAP-theorem tradeoffs of three real distributed databases and "
    "render a verdict on which is the best fit for a write-heavy workload."
)


def _require_live_backend(cfg) -> None:
    """Fail loud if the vLLM backend isn't reachable, rather than silently
    falling through to the offline stub -- this harness exists to gather
    evidence from the REAL model."""
    import urllib.request
    import urllib.error
    try:
        urllib.request.urlopen(cfg.llm_base_url, timeout=5)
    except urllib.error.HTTPError:
        pass  # server responded (even with an error status) -- it's alive
    except Exception as e:
        print(f"ERROR: vLLM backend at {cfg.llm_base_url} is unreachable ({e}). "
              f"Set CEO_LLM_URL or start the server before running this script.")
        sys.exit(1)


def run_one(orch: Orchestrator, task: str, family: str, variant: int) -> dict:
    t0 = time.time()
    tokens_before = orch.llm.total_tokens
    r = orch.run(task, task_label=f"{family}.v{variant}")
    wallclock = round(time.time() - t0, 2)
    tokens = orch.llm.total_tokens - tokens_before

    initial_ef = r.ef_trace[0] if r.ef_trace else {}
    # BUG FIX: this used to be r.ef_trace[-1] -- the descent loop's last RAW
    # iteration, which can be a worse iteration than the one actually
    # returned to the user (orchestrator picks the best iteration by Pareto
    # dominance, see delta.DeltaReport.comparison_key / dominates()). Use the
    # tensor tied to the report that was actually selected instead, so
    # final_ef reflects the delivered answer, not a discarded attempt.
    final_ef = (r.report.ef_tensor if r.report and r.report.ef_tensor else None) \
        or (r.ef_trace[-1] if r.ef_trace else {})
    thresholds = {
        "partition": orch.cfg.ef_partition_threshold, "flow": orch.cfg.ef_flow_threshold,
        "role": orch.cfg.ef_role_threshold, "scale": orch.cfg.ef_scale_threshold,
    }
    initial_worst_excess = max(
        (initial_ef.get(ax, 0.0) - thresholds[ax] for ax in thresholds), default=0.0
    )
    final_worst_excess = max(
        (final_ef.get(ax, 0.0) - thresholds[ax] for ax in thresholds), default=0.0
    )
    # incumbent worst-axis excess per iteration -- the honest monotone descent
    # curve (best plan seen so far, never regresses). This is the direct
    # evidence for the "descent is monotone" claim.
    incumbent_worst_curve = [
        round(max(e.get(ax, 0.0) - thresholds[ax] for ax in thresholds), 4)
        for e in (r.ef_incumbent_trace or [])
    ]
    moves = [
        {"axis": m.get("axis"), "move": m.get("move"), "converged": m.get("converged"),
         "before": m.get("before"), "after": m.get("after"),
         "side_effects": m.get("side_effects")}
        for m in (r.ef_move_log or [])
    ]

    return {
        "family": family,
        "variant": variant,
        "task": task,
        "fingerprint": {
            "flow": r.fingerprint.information_flow,
            "stance": r.fingerprint.epistemic_stance,
            "contract": r.fingerprint.output_contract,
            "decomposability": r.fingerprint.decomposability,
        },
        "iterations": r.iterations,
        "initial_ef": initial_ef,
        "final_ef": final_ef,
        "initial_worst_excess": round(initial_worst_excess, 4),
        "final_worst_excess": round(final_worst_excess, 4),
        "incumbent_worst_curve": incumbent_worst_curve,
        "verdict": r.report.verdict if r.report else None,
        "moves": moves,
        "wallclock_s": wallclock,
        "tokens": tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cross_run_learning_results.jsonl")
    ap.add_argument("--workdir", default=".ceo_delta_demo")
    ap.add_argument("--fresh", action="store_true",
                     help="wipe --workdir first for a clean, presentable baseline")
    args = ap.parse_args()

    if args.fresh and os.path.isdir(args.workdir):
        shutil.rmtree(args.workdir)

    cfg = Config()
    _require_live_backend(cfg)  # the offline stub returns an identical trivial
                                 # DAG for every task -- a silent stub run would
                                 # make every 'family' look the same
    orch = Orchestrator(cfg, workdir=args.workdir)

    records = []
    with open(args.out, "w") as fout:
        for family, variants in FAMILIES.items():
            for i, task in enumerate(variants, 1):
                print(f"[{family} v{i}/{len(variants)}] running...")
                rec = run_one(orch, task, family, i)
                records.append(rec)
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                print(f"  -> iterations={rec['iterations']} "
                      f"initial_worst_excess={rec['initial_worst_excess']} "
                      f"final_worst_excess={rec['final_worst_excess']} "
                      f"verdict={rec['verdict']} moves={[m['move'] for m in rec['moves']]}")

        print(f"[CONTROL (never seen) v1/1] running...")
        rec = run_one(orch, CONTROL_TASK, "D_control_unseen", 1)
        records.append(rec)
        fout.write(json.dumps(rec) + "\n")
        print(f"  -> iterations={rec['iterations']} "
              f"initial_worst_excess={rec['initial_worst_excess']} "
              f"final_worst_excess={rec['final_worst_excess']} "
              f"verdict={rec['verdict']}")

    print(f"\nwrote {len(records)} records to {args.out}")
    print("next: python3 scripts/plot_learning_curve.py", args.out)


if __name__ == "__main__":
    main()
