"""Test 3 of the demo-readiness check: does the verdict/Q-tensor discriminate
sensibly across a variety of task types, without crashing? One task
(stock picks) is deliberately ungroundable and should read poor/mixed;
the rest are self-contained or well-scoped and should read good.

Usage: python3 -u scripts/generalization_sweep.py
"""
import os
import shutil
import sys
import time

# scripts/ is a sibling of ceo_delta/, not a parent -- put the repo root on
# the path so `import ceo_delta` resolves regardless of cwd (same shim
# attractor_test.py uses).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ceo_delta import Orchestrator

WORKDIR = ".sweep_test"
shutil.rmtree(WORKDIR, ignore_errors=True)

TASKS = [
    "Explain how the water cycle works",
    "Explain how a compiler's lexer, parser, and codegen stages fit together",
    "What are the best stocks to buy right now",
    "Explain the CAP theorem and how it applies to distributed databases",
    "Explain how a microservices architecture handles service discovery",
]

o = Orchestrator(workdir=WORKDIR)
for t in TASKS:
    print(f">> running: {t}", flush=True)
    t0 = time.time()
    r = o.run(t)
    print(f"   done in {time.time()-t0:.1f}s -> verdict={r.report.verdict} "
          f"Q={r.report.q_tensor}", flush=True)

print("\nExpect: stocks task -> poor/mixed (ungroundable, forces verifier). "
      "The other 4 -> good, without any false-fire on entity-dense-but-"
      "self-contained explanations (the bug just fixed).")
