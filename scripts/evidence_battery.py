#!/usr/bin/env python3
"""The full presentation-evidence battery, one script, one pass.

Answers, in order, with one phase each:

  0. noise_floor          -- how much do iterations/error vary on an EASY
                             task with nothing to fix, purely from LLM
                             stochasticity? Calibrates every other phase --
                             without this you can't tell signal from noise
                             anywhere else in this battery.
  1. repeat_convergence   -- same HARD task, verbatim, 5x in a row against a
                             persistent store. Same wording removes the
                             "different variant hit a different axis"
                             confound entirely: any change in which move
                             fires, its order, or iteration count is
                             unambiguously the store, not task variance.
                             This task is also known (from a prior manual
                             run) to trigger scale->partition collateral
                             damage, so it doubles as the side-effect /
                             deprioritization test (ef_side_effect_
                             deprioritize_threshold=2).
  2. control_pre/mid/post -- ONE never-before-seen species, run 3x at 3
                             different points in the session (before,
                             middle, after the family sweep). If its
                             numbers stay flat across all 3 while the
                             families below visibly improve, that rules out
                             "the whole session just gets easier" as the
                             explanation for any within-family trend.
  3. family_A/B/C         -- 3 fingerprint species x 6 surface-different
                             variants each. Enough repetitions per species
                             for the same axis to plausibly fail twice
                             naturally (not forced), which is what's needed
                             to see a learned-move promotion on real,
                             varied language -- the actual M2 generalization
                             claim, not the repeat_convergence phase's easier
                             verbatim-repeat version of it.
  4. interference_check   -- family A's FIRST variant, run again verbatim
                             after B and C have both run in between. Confirms
                             A's learned moves survive unrelated species
                             traffic in the shared store (keys are per-
                             fingerprint, not global) rather than getting
                             overwritten or diluted.
  5. escalation_trigger   -- a task deliberately chosen to expose a known
                             gap in moves.py: flow=recursive with a plan
                             CEO tends to leave shallow. None of the 4
                             flow moves (split_to_parallel/add_join/
                             linearize/add_merge) address "increase depth
                             for recursive," so this axis is a good bet to
                             be genuinely irreducible via the deterministic
                             table twice in a row. By
                             ef_irreducible_escalate_threshold=2, the 3rd
                             identical rep should unlock a real LLM-authored
                             escalation move (its move_id will contain
                             ".esc."); a 4th rep checks whether a successful
                             escalation folds back into the deterministic
                             repertoire (b feeds a). This is a best-effort
                             trigger, not a guarantee -- CEO might just
                             happen to plan deep enough on its own, in which
                             case this phase will show that instead, which
                             is also worth knowing.

All records share one JSONL schema (phase-tagged) so one downstream plot
script can render every panel from one file.

Usage:
  # needs the vLLM server reachable (CEO_LLM_URL, default http://10.8.0.23:8001/v1)
  python3 scripts/evidence_battery.py --fresh
  python3 scripts/plot_evidence_battery.py evidence_battery_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ceo_delta.config import Config
from ceo_delta.orchestrator import Orchestrator
from scripts.cross_run_learning_test import run_one as _run_one

EASY_TASK = "Summarize the main phases of the water cycle."

HARD_TASK = (
    "Provide an exhaustive, recursively layered explanation of how quicksort "
    "partitions and recurses on subarrays, covering every level of the "
    "recursion in comprehensive depth."
)

CONTROL_TASK = (
    "Compare the CAP-theorem tradeoffs of three real distributed databases and "
    "render a verdict on which is the best fit for a write-heavy workload."
)

FAMILIES = {
    "A_divergent_enumerate": [
        "What are all the different aspects of what makes a person trustworthy?",
        "What are the key dimensions of a resilient distributed system?",
        "Break down the core components of a healthy organizational culture.",
        "Map the key drivers of employee burnout.",
        "Identify the major factors influencing climate migration.",
        "Outline the primary considerations in choosing a programming language "
        "for a new project.",
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
        "Explain the strictly ordered stages of CPU instruction execution: "
        "fetch, decode, execute, memory access, writeback, where each stage "
        "cannot begin until the previous stage's result is available.",
        "Walk through the sequential stages of forming a new git commit "
        "object step by step, where each step depends on the previous step's "
        "completed output.",
        "Trace the strictly ordered stages of DNS resolution from browser "
        "cache check to authoritative server response, where each stage "
        "depends on the previous stage failing to resolve the query.",
    ],
    "C_recursive_exhaustive": [
        "Explain what a stack data structure is, but do it exhaustively -- a "
        "deeply layered, multi-level structured breakdown covering every "
        "mechanism, operation, and nuance in comprehensive depth.",
        "Explain binary search trees exhaustively, as a deeply nested "
        "breakdown covering every layer of structure and operation in "
        "comprehensive depth.",
        "Provide an exhaustive, recursively layered explanation of how merge "
        "sort divides and recurses on subarrays, covering every level of the "
        "recursion in comprehensive depth.",
        "Give an exhaustive, recursively layered explanation of the "
        "Fibonacci recursive call tree, covering every level of the "
        "recursion in comprehensive depth.",
        "Explain the recursive structure of a filesystem directory tree "
        "exhaustively, layer by layer, in comprehensive depth.",
        "Provide an exhaustive, recursively layered explanation of how a "
        "recursive descent parser handles nested expressions, covering every "
        "level of the recursion in comprehensive depth.",
    ],
}

# Deliberately short/ambiguous -- likely to get planned shallow (depth<3)
# despite carrying a recursive-flow fingerprint, which is exactly the gap
# moves.py's flow repertoire doesn't cover (see module docstring, phase 5).
ESCALATION_TASK = (
    "Recursively explain how a recursive function calls itself until it "
    "hits a base case."
)


def _require_live_backend(cfg) -> None:
    """Fail loud if the vLLM backend isn't reachable, rather than silently
    falling through to the offline stub -- this battery exists to gather
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


def _print_rec(rec: dict) -> None:
    print(f"  -> iterations={rec['iterations']} "
          f"initial_worst_excess={rec['initial_worst_excess']} "
          f"final_worst_excess={rec['final_worst_excess']} "
          f"verdict={rec['verdict']} "
          f"moves={[m['move'] for m in rec['moves']]}")


def run_phase(orch, fout, records, phase, family, task, variant):
    print(f"[{phase} | {family} v{variant}] running...")
    rec = _run_one(orch, task, family, variant)
    rec["phase"] = phase
    records.append(rec)
    fout.write(json.dumps(rec) + "\n")
    fout.flush()
    _print_rec(rec)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evidence_battery_results.jsonl")
    ap.add_argument("--workdir", default=".ceo_delta_battery")
    ap.add_argument("--fresh", action="store_true", default=True,
                     help="wipe --workdir first (default on -- this battery "
                          "is meant to be a clean, reproducible dataset)")
    ap.add_argument("--no-fresh", dest="fresh", action="store_false")
    args = ap.parse_args()

    if args.fresh and os.path.isdir(args.workdir):
        shutil.rmtree(args.workdir)

    cfg = Config()
    _require_live_backend(cfg)  # the offline stub returns an identical trivial
                                 # DAG for every task -- a silent stub run would
                                 # make every phase look the same
    orch = Orchestrator(cfg, workdir=args.workdir)

    records = []
    with open(args.out, "w") as fout:

        # -- phase 0: noise floor -----------------------------------------
        print("\n=== PHASE 0: noise floor (easy task x3, nothing should need fixing) ===")
        for i in range(1, 4):
            run_phase(orch, fout, records, "noise_floor", "easy", EASY_TASK, i)

        # -- phase 1: repeat-to-convergence + collateral -------------------
        print("\n=== PHASE 1: repeat-to-convergence (hard task x5, persistent store) ===")
        for i in range(1, 6):
            run_phase(orch, fout, records, "repeat_convergence", "hard", HARD_TASK, i)

        # -- phase 2a: control, pre -----------------------------------------
        print("\n=== PHASE 2: control species (never seen), rep 1/3 (pre-family) ===")
        run_phase(orch, fout, records, "control", "control", CONTROL_TASK, 1)

        # -- phase 3a: family A ---------------------------------------------
        print("\n=== PHASE 3: family A (divergent/enumerate), 6 variants ===")
        for i, task in enumerate(FAMILIES["A_divergent_enumerate"], 1):
            run_phase(orch, fout, records, "family_A", "A_divergent_enumerate", task, i)

        # -- phase 2b: control, mid -----------------------------------------
        print("\n=== PHASE 2: control species, rep 2/3 (mid-family) ===")
        run_phase(orch, fout, records, "control", "control", CONTROL_TASK, 2)

        # -- phase 3b: family B ---------------------------------------------
        print("\n=== PHASE 3: family B (sequential/pipeline), 6 variants ===")
        for i, task in enumerate(FAMILIES["B_sequential_pipeline"], 1):
            run_phase(orch, fout, records, "family_B", "B_sequential_pipeline", task, i)

        # -- phase 3c: family C ---------------------------------------------
        print("\n=== PHASE 3: family C (recursive/exhaustive), 6 variants ===")
        for i, task in enumerate(FAMILIES["C_recursive_exhaustive"], 1):
            run_phase(orch, fout, records, "family_C", "C_recursive_exhaustive", task, i)

        # -- phase 2c: control, post -----------------------------------------
        print("\n=== PHASE 2: control species, rep 3/3 (post-family) ===")
        run_phase(orch, fout, records, "control", "control", CONTROL_TASK, 3)

        # -- phase 4: interference check --------------------------------------
        print("\n=== PHASE 4: interference check (family A's v1 task, repeated verbatim "
              "after B and C ran in between) ===")
        run_phase(orch, fout, records, "interference_check", "A_divergent_enumerate",
                  FAMILIES["A_divergent_enumerate"][0], 99)

        # -- phase 5: escalation trigger --------------------------------------
        print("\n=== PHASE 5: escalation trigger (recursive-flow / shallow-depth gap, x4) ===")
        for i in range(1, 5):
            run_phase(orch, fout, records, "escalation_trigger", "escalation",
                      ESCALATION_TASK, i)

    print(f"\nwrote {len(records)} records to {args.out}")
    esc_fired = [r for r in records if any(".esc." in (m["move"] or "") for m in r["moves"])]
    print(f"escalation moves observed in: {[ (r['phase'], r['variant']) for r in esc_fired ] or 'none'}")
    print("next: python3 scripts/plot_evidence_battery.py", args.out)


if __name__ == "__main__":
    main()
