"""Benchmark: 3 tasks x 5 runs — tracks whether the system improves with experience.

Tasks are chosen to stress different skills:
  Task A (research-heavy)  — "What makes distributed databases resilient to network partitions?"
  Task B (reasoning/plan)  — "Design a migration plan from monolith to microservices"
  Task C (verification)    — "Compare and verify the trade-offs between batch vs stream processing"

Each task is run 5 times on the SAME Orchestrator so the handbook accumulates.
After all runs, prints a per-task progression table and a summary.
"""
from __future__ import annotations

import json
import sys
import time

from ceo_delta.orchestrator import Orchestrator

TASKS = {
    "A_research": "What are the key factors that make a distributed database system resilient to network partitions?",
    "B_planning": "Design a step-by-step migration plan to move a monolithic e-commerce application to microservices",
    "C_verification": "Compare and verify the trade-offs between batch processing and stream processing for real-time analytics",
}

RUNS = 5


def run_task(orch: Orchestrator, label: str, task: str, run_idx: int) -> dict:
    t0 = time.time()
    result = orch.run(task, auto_clarify=True, task_label=label)
    elapsed = round(time.time() - t0, 2)

    rep = result.report
    dag = result.dag
    brief = result.structured_brief

    return {
        "task": label,
        "run": run_idx + 1,
        "topology": dag.topology,
        "depth": dag.depth,
        "nodes": len(dag.nodes),
        "task_class": brief.task_class,
        "replanned": result.replanned,
        "verdict": rep.verdict,
        "fp_mean": rep.semantic.get("fingerprint_match_mean", 0.0),
        "w_fp": rep.semantic.get("weighted_fingerprint_match", 0.0),
        "brief_drift": rep.semantic.get("brief_drift", 0.0),
        "error_rate": rep.failure.get("error_rate", 0.0),
        "w_error_rate": rep.failure.get("weighted_error_rate", 0.0),
        "echo_rate": rep.failure.get("echoing_rate", 0.0),
        "plan_exec_align": rep.structural.get("plan_exec_alignment", 0.0),
        "elapsed_s": elapsed,
        "answer_snippet": result.answer[:120].replace("\n", " "),
    }


def print_table(rows: list[dict]) -> None:
    headers = [
        "task", "run", "topology", "depth", "nodes", "task_class",
        "replanned", "verdict", "fp_mean", "w_fp", "brief_drift",
        "error_rate", "plan_exec_align", "elapsed_s",
    ]
    widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}

    def fmt_row(r):
        return "  ".join(str(r.get(h, "")).ljust(widths[h]) for h in headers)

    sep = "  ".join("-" * widths[h] for h in headers)
    print(fmt_row({h: h for h in headers}))
    print(sep)
    prev_task = None
    for r in rows:
        if r["task"] != prev_task and prev_task is not None:
            print(sep)
        prev_task = r["task"]
        print(fmt_row(r))


def print_trend(rows: list[dict]) -> None:
    print("\n=== IMPROVEMENT TREND (first run -> last run per task) ===")
    for label in TASKS:
        task_rows = [r for r in rows if r["task"] == label]
        if len(task_rows) < 2:
            continue
        first, last = task_rows[0], task_rows[-1]
        delta_fp   = round(last["fp_mean"] - first["fp_mean"], 3)
        delta_w_fp = round(last["w_fp"] - first["w_fp"], 3)
        delta_err  = round(last["error_rate"] - first["error_rate"], 3)
        delta_drift = round(last["brief_drift"] - first["brief_drift"], 3)
        print(
            f"  {label:15s}  fp_mean {first['fp_mean']:.3f}->{last['fp_mean']:.3f} "
            f"({'+' if delta_fp>=0 else ''}{delta_fp})  "
            f"w_fp {first['w_fp']:.3f}->{last['w_fp']:.3f} ({'+' if delta_w_fp>=0 else ''}{delta_w_fp})  "
            f"error {first['error_rate']:.3f}->{last['error_rate']:.3f} ({'+' if delta_err>=0 else ''}{delta_err})  "
            f"drift {first['brief_drift']:.3f}->{last['brief_drift']:.3f} ({'+' if delta_drift>=0 else ''}{delta_drift})  "
            f"verdicts: {[r['verdict'] for r in task_rows]}"
        )


def main():
    print(f"=== CEO-DELTA BENCHMARK  ({len(TASKS)} tasks x {RUNS} runs) ===\n")

    # Fresh orchestrator per run-set so handbook starts clean each time.
    # We want to observe learning within the session, not carry over old state.
    import shutil, os
    workdir = "/tmp/ceo_bench"
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir, exist_ok=True)

    orch = Orchestrator(workdir=workdir, run_log_path=f"{workdir}/runs.jsonl")

    all_rows = []
    total = len(TASKS) * RUNS
    done = 0

    for label, task in TASKS.items():
        print(f"\n--- Task {label} ---")
        print(f"    {task}\n")
        for i in range(RUNS):
            print(f"  run {i+1}/{RUNS} ...", end=" ", flush=True)
            try:
                row = run_task(orch, label, task, i)
                all_rows.append(row)
                print(
                    f"verdict={row['verdict']}  fp={row['fp_mean']:.3f}  "
                    f"topo={row['topology']}  d={row['depth']}  nodes={row['nodes']}  "
                    f"elapsed={row['elapsed_s']}s"
                )
            except Exception as e:
                print(f"ERROR: {e}")
                all_rows.append({"task": label, "run": i + 1, "verdict": "error", "fp_mean": 0.0,
                                  "w_fp": 0.0, "brief_drift": 0.0, "error_rate": 1.0,
                                  "w_error_rate": 1.0, "echo_rate": 0.0, "plan_exec_align": 0.0,
                                  "topology": "?", "depth": 0, "nodes": 0, "task_class": "?",
                                  "replanned": False, "elapsed_s": 0.0, "answer_snippet": str(e)})
            done += 1

    print("\n\n=== FULL RESULTS TABLE ===\n")
    print_table(all_rows)
    print_trend(all_rows)

    # Registry state after all runs
    print("\n=== AGENT REGISTRY STATE ===")
    for entry in orch.registry.summary():
        print(
            f"  {entry['agent_id']:20s}  trust={entry['trust']:.3f}  "
            f"conf={entry['confidence']}  hub_fails={entry['hub_failures']}  "
            f"roles={entry['roles']}"
        )

    # Handbook summary
    print("\n=== HANDBOOK STATE (ceo) ===")
    entries = orch.ceo_hb.entries
    print(f"  total entries: {len(entries)}")
    for e in entries[:10]:
        print(
            f"  [{e.entry_id}] conf={e.confidence}  topo={e.topology_chosen}  "
            f"d={e.depth_chosen}  contested={e.contested}  "
            f"summary={e.task_summary[:60]}"
        )

    out_path = f"{workdir}/benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
