#!/usr/bin/env python3
"""CLI for the CEO-Delta architecture.

  python cli.py run "your task here"
  python cli.py run "task" --satisfaction 0.9     # standard run + meta signal
  python cli.py reflect                            # force a reflection session
  python cli.py demo                               # multi-run learning demo
  python cli.py handbook                           # dump handbook state
"""
from __future__ import annotations

import argparse
import json
import sys

from ceo_delta import Orchestrator


def _print_run(r):
    print("=" * 70)
    print(f"TASK: {r.task}")
    if r.clarification:
        print(f"CLARIFY: {r.clarification}")
    print(f"\nDAG: topology={r.dag.topology} depth={r.dag.depth} "
          f"nodes={len(r.dag.nodes)} exploratory={r.dag.exploratory}")
    print(f"  why_topology: {r.dag.why_topology[:120]}")
    print(f"  why_depth   : {r.dag.why_depth[:120]}")
    for n in r.dag.nodes:
        print(f"   - {n.node_id} [{n.roles.structural}/{n.roles.functional}] "
              f"deps={n.dependencies} :: {n.intent[:70]}")
    print(f"\nRESEARCH brief drift={r.brief.drift:.2f} material={r.brief.materiality} "
          f"replanned={r.replanned}")
    print(f"\nMETRICS:")
    print(f"  structural : {r.report.structural}")
    print(f"  runtime    : {r.report.runtime}")
    print(f"  failure    : {r.report.failure}")
    print(f"  semantic   : {r.report.semantic}")
    print(f"  verdict    : {r.report.verdict}")
    print(f"  CEO  fb    : {r.report.ceo_feedback}")
    print(f"  RSCH fb    : {r.report.research_feedback}")
    print(f"  granular   : {r.report.granular_entries}")
    if r.reflection and r.reflection.triggered:
        print(f"\nREFLECTION: {r.reflection.reason} | explorations="
              f"{r.reflection.explorations} tokens={r.reflection.tokens_spent} "
              f"resolved={r.reflection.resolved} notes={r.reflection.notes}")
    print(f"\nANSWER:\n{r.answer[:1200]}")
    print("=" * 70)


def cmd_run(args):
    orch = Orchestrator()
    r = orch.run(args.task, user_satisfaction=args.satisfaction)
    _print_run(r)


def cmd_reflect(args):
    orch = Orchestrator()
    log = orch.reflect_now()
    print(json.dumps(log.__dict__, indent=2, default=str))


def cmd_handbook(args):
    orch = Orchestrator()
    for hb in (orch.ceo_hb, orch.research_hb):
        print(f"\n### {hb.name} handbook ({len(hb.entries)} entries)")
        for e in hb.entries:
            print(f"  [{e.entry_id}] topo={e.topology_chosen}({dict(e.topology_votes)}) "
                  f"depth={e.depth_chosen} conf={e.confidence} "
                  f"contested={e.contested} :: {e.task_summary[:50]}")


def cmd_demo(args):
    orch = Orchestrator()
    tasks = [
        "Summarize recent research on DAG-based multi-agent orchestration",
        "Quickly tell me what intent-gated execution means",
        "Compare KAIJU and POLARIS approaches to agent planning",
        "Survey papers on reducing critical path latency in agent DAGs",
        "Verify whether echoing is a real failure mode in multi-agent systems",
        "Survey recent papers on DAG-based multi-agent orchestration",  # near-dup -> density
    ]
    for i, t in enumerate(tasks):
        print(f"\n\n########## RUN {i+1} ##########")
        r = orch.run(t)
        _print_run(r)
    print("\n\n========== FINAL HANDBOOK ==========")
    cmd_handbook(args)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ceo-delta")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run"); pr.add_argument("task")
    pr.add_argument("--satisfaction", type=float, default=None)
    pr.set_defaults(fn=cmd_run)

    sub.add_parser("reflect").set_defaults(fn=cmd_reflect)
    sub.add_parser("handbook").set_defaults(fn=cmd_handbook)
    sub.add_parser("demo").set_defaults(fn=cmd_demo)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
