#!/usr/bin/env python3
"""Role-axis focused re-test: is role-E under-sensitive because of CENTRALITY
DILUTION, or because the injection was too weak?

We inject the same kind of role defect (a node whose behavior mismatches its
label) at three positions of increasing centrality/count, and watch role-E:
  peripheral : one retriever ROOT (centrality ~0.5) given a synthesis task
  central    : the synthesizer JOIN (centrality ~1.5) given a retriever task
  multi      : two retrievers + the synthesizer, all mislabeled

If role-E fires on `central`/`multi` but not `peripheral`, dilution is the cause
(and role detection is centrality-gated — arguably correct). If it fires
nowhere, the metric is genuinely insensitive to this defect.

Usage: python scripts/role_retest.py --reps 6
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ceo_delta.config import Config, ROLE_BANDS
from ceo_delta.embeddings import embed
from ceo_delta.ef import compute_required, compute_ef
from ceo_delta.kernel import Kernel
from ceo_delta.llm import LLMClient
from scripts.research.perturbation_harness import base_dag, realize, BASE_FP, _trace_from

SYNTH_TASK = "compress and cross-reference everything into one tight synthesis; do not list raw facts"
RETR_TASK = "retrieve and list raw individual facts and sources; do NOT summarize, compress, or combine"

# (label kept as-is on the node; only the TASK is flipped to force off-role behavior)
MODES = {
    "peripheral (root retriever)": {"r1": SYNTH_TASK},
    "central (synthesizer join)":  {"s1": RETR_TASK},
    "multi (2 retr + synth)":      {"r1": SYNTH_TASK, "r2": SYNTH_TASK, "s1": RETR_TASK},
}


def inject(mapping):
    d = base_dag()
    changed = set()
    for nid, intent in mapping.items():
        n = d.node(nid)
        n.intent = intent
        n.expected_output_fingerprint = embed(intent)
        changed.add(nid)
    return d, changed



def _require_live_backend(cfg) -> None:
    """Fail loud if the configured LLM backend isn't reachable, rather than
    silently falling through to the offline stub -- these scripts exist to
    gather evidence from the REAL model, so a quiet stub run would produce
    misleading (uniformly trivial) results without any error."""
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6)
    args = ap.parse_args()

    cfg = Config()
    _require_live_backend(cfg)
    kernel = Kernel(LLMClient(cfg), None, cfg)
    required = compute_required(BASE_FP)
    thr = cfg.ef_role_threshold

    base = base_dag()
    base_results = {r.node_id: r for r in kernel.execute(base).results}
    base_ef = compute_ef(base, _trace_from(base, base_results), required,
                         base.centrality_weights(), ROLE_BANDS)
    w = base.centrality_weights()
    print(f"base role-E = {base_ef.as_dict()['role']:.3f}  (threshold {thr})")
    print(f"node centralities: {dict((k, round(v,2)) for k,v in w.items())}\n")

    for name, mapping in MODES.items():
        vals, fires, excerpts = [], 0, None
        for _ in range(args.reps):
            d, changed = inject(mapping)
            trace = realize(d, changed, base_results, kernel)
            ef = compute_ef(d, trace, required, d.centrality_weights(), ROLE_BANDS)
            rv = ef.as_dict()["role"]
            vals.append(rv)
            fires += int(rv > thr)
            excerpts = ef.role_excess   # per-node band excess, last rep
        mean = statistics.fmean(vals)
        print(f"[{name}]")
        print(f"  role-E mean={mean:.3f}  range=({min(vals):.3f},{max(vals):.3f})  "
              f"fired {fires}/{args.reps}")
        # show which nodes carried band-excess (was the violation even detected
        # pre-dilution?)
        if excerpts:
            per = {nid: round(sum(v.values()), 3) for nid, v in excerpts.items()}
            print(f"  per-node band-excess (last rep): {per}")
        else:
            print(f"  per-node band-excess: none detected")
        print()


if __name__ == "__main__":
    main()
