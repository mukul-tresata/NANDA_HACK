"""Demo battery — the fresh-store rebuild + full evidence sweep.

Runs a fingerprint-spanning task battery against a FRESH store (so learned_moves,
the exhaustive best_plan, and correct final_q all accumulate on the post-fix
code), then a SECOND PASS on a groundable subset to DEMONSTRATE warm-start
firing (learning consume-side) rather than just storing it.

Run at the production default temperature (0.0 / deterministic) -- this is the
demo condition, not the temp>0 attractor stress test.

Usage:
    python3 -u scripts/demo_battery.py
    (budget ~45-90 min; run as a background job)
"""
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ceo_delta import Orchestrator

WORKDIR = ".demo_battery"
shutil.rmtree(WORKDIR, ignore_errors=True)  # FRESH rebuild -- never mix pre-fix data

# (category, task). Ordering groups same-fingerprint tasks so learned_moves
# accumulate within a species across the pass.
TASKS = [
    # --- Sequential ---
    ("Sequential", "Explain how a modern C compiler transforms C source into an executable."),
    ("Sequential", "Explain the entire water cycle from evaporation to groundwater recharge."),
    ("Sequential", "Describe how an HTTP request travels from a browser to a server and back."),

    # --- Divergent ---
    ("Divergent", "Plan a two-week Japan itinerary with budget, transportation, food, and weather considerations."),
    ("Divergent", "Compare Python, Rust, Go, Java, and C++ across performance, ecosystem, safety, concurrency, learning curve, and deployment."),
    ("Divergent", "Design a home lab for AI experimentation under three different budgets ($500, $1500, $5000)."),

    # --- Convergent ---
    ("Convergent", "Given three conflicting job offers (Offer A: $180k, strong culture, slow growth, NYC; "
                   "Offer B: $210k, average culture, fast growth, remote; Offer C: $160k, great culture, "
                   "fast growth, Austin), recommend the best one and justify the tradeoffs."),
    ("Convergent", "Review a proposed software architecture with latency, scalability, maintainability, and "
                   "developer productivity constraints, then recommend a final design."),
    ("Convergent", "Synthesize the current consensus from these three research abstracts on retrieval-augmented "
                   "generation. Abstract 1: 'RAG reduces hallucination by grounding generations in retrieved "
                   "passages, but retrieval quality bounds answer quality.' Abstract 2: 'Fine-tuning and RAG are "
                   "complementary; RAG handles fresh facts, fine-tuning handles style and format.' Abstract 3: "
                   "'Poor retrieval can be worse than no retrieval, injecting distractor context that degrades "
                   "faithfulness.' What is the consensus?"),

    # --- Recursive ---
    ("Recursive", "Explain recursion to someone who doesn't understand recursion."),
    ("Recursive", "Explain how interpreters execute recursive functions by tracing a concrete recursive function."),
    ("Recursive", "Describe how hierarchical planning works using hierarchical planning itself as the organizing principle."),

    # --- Retrieval / Synthesis / Generation ---
    ("Generation", "Write a complete design document for a distributed rate limiter."),
    ("Generation", "Generate a semester-long learning roadmap for GPUs and modern AI accelerators."),
    ("Generation", "Write a short science-fiction story with these constraints -- characters: a lighthouse keeper "
                   "and a stranded AI probe; theme: memory and forgetting; ending: they part without ever truly "
                   "understanding each other."),

    # --- Deliberately ungroundable (Q discriminators) ---
    ("Ungroundable", "Should I buy NVIDIA stock tomorrow?"),
    ("Ungroundable", "Summarize today's biggest AI news."),
    ("Ungroundable", "What restaurants near me are currently open and have less than a 15-minute wait?"),
]

# Second-pass replay subset: groundable tasks that should converge good on pass 1
# (best_worst_excess < 0) and therefore WARM-START on pass 2. Ungroundable tasks
# are excluded -- they end poor, never cache a satisfying best_plan, never warm-start.
REPLAY = [
    "Explain how a modern C compiler transforms C source into an executable.",
    "Compare Python, Rust, Go, Java, and C++ across performance, ecosystem, safety, concurrency, learning curve, and deployment.",
    "Given three conflicting job offers (Offer A: $180k, strong culture, slow growth, NYC; "
    "Offer B: $210k, average culture, fast growth, remote; Offer C: $160k, great culture, "
    "fast growth, Austin), recommend the best one and justify the tradeoffs.",
    "Write a complete design document for a distributed rate limiter.",
    "Explain the entire water cycle from evaporation to groundwater recharge.",
    "Explain how interpreters execute recursive functions by tracing a concrete recursive function.",
]


def _warmstarted(r) -> bool:
    return any(getattr(n.why, "priors_used", "") == "warmstart" for n in r.dag.nodes)


def _run(o, task, category, phase):
    t0 = time.time()
    r = o.run(task)
    fp = r.fingerprint
    shape = f"{fp.information_flow}/{fp.epistemic_stance}/{fp.output_contract}/{fp.decomposability}"
    warm = _warmstarted(r)
    print(f"[{phase}] {category:12} {task[:52]:52} -> verdict={r.report.verdict:6} "
          f"Q={r.report.q_tensor} iters={r.iterations} "
          f"{'WARM' if warm else 'cold'} ({time.time()-t0:.0f}s)", flush=True)
    print(f"             fingerprint={shape}", flush=True)
    return {"task": task, "category": category, "verdict": r.report.verdict,
            "q": r.report.q_tensor, "shape": shape, "warm": warm, "iters": r.iterations}


def main():
    o = Orchestrator(workdir=WORKDIR)

    print("=" * 70)
    print("PASS 1 -- fresh-store rebuild (fingerprint-spanning battery)")
    print("=" * 70)
    pass1 = [_run(o, task, cat, "P1") for cat, task in TASKS]

    print("\n" + "=" * 70)
    print("PASS 2 -- replay groundable subset (should WARM-START from pass 1)")
    print("=" * 70)
    pass2 = [_run(o, task, "replay", "P2") for task in REPLAY]

    # -- summary ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    shapes = sorted({r["shape"] for r in pass1})
    print(f"\nDistinct fingerprint species populated: {len(shapes)}")
    for s in shapes:
        print(f"  - {s}")

    verdicts = {}
    for r in pass1:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    print(f"\nPass-1 verdict distribution: {verdicts}")

    warm_hits = sum(1 for r in pass2 if r["warm"])
    print(f"\nPass-2 warm-start: {warm_hits}/{len(pass2)} replayed tasks reused a cached plan")
    for r in pass2:
        print(f"  - {'WARM' if r['warm'] else 'COLD'} (iters={r['iters']}): {r['task'][:60]}")

    # -- honest rubric check ---------------------------------------------------
    print("\nEXPECT: sequential/comparison/generation/convergent self-contained tasks")
    print("  -> good, groundedness 0.0 (no false-fire on entity-dense explanations);")
    print("  itinerary/home-lab -> mixed/poor (need live prices);")
    print("  stock/news/restaurants -> poor (Q discriminators);")
    print("  pass-2 replay -> WARM on all groundable species that converged good in pass 1.")


if __name__ == "__main__":
    main()
