#!/usr/bin/env python3
"""Perturbation / defect-injection harness — non-circular threshold + axis validation.

We inject KNOWN structural defects at the ASSIGNMENT level (never at the
metric's direct input), let the plan execute, and measure the E vector. Because
the ground-truth label ("I injected a partition defect of severity k") is set by
us and is independent of the metric, this is non-circular for the axes where the
injection and the metric are operationally distinct:

  partition (VALIDITY): give sibling nodes OVERLAPPING INTENTS, let them generate
     INDEPENDENTLY, and measure whether the emergent outputs turn out redundant.
     The metric must EARN the detection — the LLM could still differentiate them.
  role (VALIDITY): give a node a role label that MISMATCHES its task, generate,
     and measure whether behavioral profiling catches it.
  scale / flow (CONSISTENCY): injecting these manipulates exactly the graph
     property the metric reads (depth / join-presence), so a fire is near-
     guaranteed. This is a consistency check on the implementation, not an
     empirical validity test — these axes are DEFINITIONAL. Labeled as such.

Two distribution-level, cherry-pick-proof outputs:
  - THRESHOLD VALIDITY: does measured-E on an axis climb with injected severity
    and cross its threshold where severity is genuinely bad?  (severity->E)
  - DISCRIMINANT VALIDITY: is the dominant INJECTED axis the dominant MEASURED
    axis?  (attribution accuracy + confusion matrix over all trials)

Usage:
  python scripts/perturbation_harness.py --trials 60
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ceo_delta.config import Config, ROLE_BANDS
from ceo_delta.embeddings import embed
from ceo_delta.ef import compute_required, compute_ef, EF_AXES
from ceo_delta.kernel import Kernel
from ceo_delta.llm import LLMClient
from ceo_delta.schemas import DAG, Node, Roles, TaskFingerprint, NodeResult

BASE_TASK = "What are the key factors that make a distributed database resilient to network partitions"

# Fixed healthy species: divergent gather -> synthesize -> verify.
BASE_FP = TaskFingerprint(
    information_flow="divergent", epistemic_stance="synthesis",
    output_contract="artifact", decomposability="independent", complexity="medium",
)
BASE_FP.embedding = embed(BASE_FP.shape_string())

RETRIEVER_INTENTS = [
    "retrieve consensus mechanisms relevant to partition tolerance",
    "retrieve replication strategies used under network partitions",
    "retrieve failure detection and recovery mechanisms for partitions",
]
SYNTH_INTENT = "synthesize the retrieved factors into one coherent explanation"
VERIFY_INTENT = "verify the synthesis for internal consistency and coverage"


def base_dag() -> DAG:
    nodes = []
    for i, intent in enumerate(RETRIEVER_INTENTS, 1):
        nodes.append(Node(f"r{i}", intent,
                          Roles("fan-out", "retriever", "specialist"), [],
                          expected_output_fingerprint=embed(intent)))
    nodes.append(Node("s1", SYNTH_INTENT, Roles("join", "synthesizer", "generalist"),
                      ["r1", "r2", "r3"], expected_output_fingerprint=embed(SYNTH_INTENT)))
    nodes.append(Node("v1", VERIFY_INTENT, Roles("join", "verifier", "specialist"),
                      ["s1"], expected_output_fingerprint=embed(VERIFY_INTENT)))
    return DAG(task=BASE_TASK, task_embedding=embed(BASE_TASK),
               topology="fan-out", depth=3, nodes=nodes)


# -- injections: return (perturbed_dag, set_of_nodes_needing_regeneration) -----

def inject_partition(k: int):
    """Give k sibling retrievers r1's intent (overlapping assignment). They still
    generate independently -> metric must earn detecting the redundancy."""
    d = base_dag()
    changed = set()
    for i in range(2, 2 + k):        # r2 (and r3) copy r1's intent
        node = d.node(f"r{i}")
        node.intent = RETRIEVER_INTENTS[0]
        node.expected_output_fingerprint = embed(node.intent)
        changed.add(node.node_id)
    return d, changed


def inject_role(k: int):
    """Give k retriever-LABELED nodes a synthesis TASK -> behavioral mismatch."""
    d = base_dag()
    changed = set()
    for i in range(1, 1 + k):
        node = d.node(f"r{i}")
        node.intent = "compress and cross-reference the topic into a tight synthesis"
        node.expected_output_fingerprint = embed(node.intent)
        # label STAYS 'retriever' -> role-E should catch the behavior mismatch
        changed.add(node.node_id)
    return d, changed


def inject_scale(k: int):
    """Insert k pass-through layers to push depth past target (structural only)."""
    d = base_dag()
    prev = "s1"
    insert = []
    for j in range(k):
        nid = f"pad{j}"
        insert.append(Node(nid, "pass through intermediate result",
                           Roles("linear", "generic", "generalist"), [prev],
                           expected_output_fingerprint=embed("passthrough")))
        prev = nid
    d.node("v1").dependencies = [prev]
    # splice pad nodes in before v1
    d.nodes = d.nodes[:-1] + insert + [d.node("v1")]
    return d, set()   # no intent changes -> no regeneration


def inject_flow(_k: int):
    """Linearize: chain the retrievers instead of parallel (structural only)."""
    d = base_dag()
    d.node("r2").dependencies = ["r1"]
    d.node("r3").dependencies = ["r2"]
    d.node("s1").dependencies = ["r3"]
    for n in d.nodes:
        n.roles.structural = "linear"
    d.topology = "linear"
    return d, set()   # no intent changes -> reuse base outputs


INJECTORS = {
    "partition": (inject_partition, [1, 2]),
    "role":      (inject_role,      [1, 2]),
    "scale":     (inject_scale,     [1, 2, 3]),
    "flow":      (inject_flow,      [1]),
}
VALIDITY = {"partition", "role"}   # the rest are consistency checks


def realize(dag, changed, base_results, kernel):
    """Build a trace: reuse base outputs for unchanged nodes, regenerate changed
    ones (in dep order), synthesize placeholders for brand-new nodes."""
    results = {}
    for n in dag.nodes:
        if n.node_id in base_results and n.node_id not in changed:
            results[n.node_id] = base_results[n.node_id]
    # regenerate changed nodes in dependency order
    pending = [n for n in dag.nodes if n.node_id in changed]
    for n in pending:
        results[n.node_id] = kernel._run_node(n, dag, results, "")
    # placeholders for new structural nodes (scale pads) — generic role, no
    # siblings, so they don't perturb partition/role E.
    for n in dag.nodes:
        if n.node_id not in results:
            results[n.node_id] = NodeResult(
                n.node_id, n.intent, "[structural pad]", embed("pad"),
                1, 0.0, True, 0.0, True, None)
    from ceo_delta.schemas import ExecutionTrace
    ordered = [results[n.node_id] for n in dag.nodes]
    return ExecutionTrace(dag.dag_id, dag.task, ordered,
                          sum(r.cost_tokens for r in ordered), 0.0)



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
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--out", default="perturbation_results.jsonl")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--analyze-only", metavar="PATH",
                     help="skip execution entirely; re-analyze an existing "
                          "results JSONL (e.g. perturbation_results.jsonl) "
                          "including the non-circular threshold derivation")
    args = ap.parse_args()

    TH = {"partition": Config().ef_partition_threshold, "flow": Config().ef_flow_threshold,
          "role": Config().ef_role_threshold, "scale": Config().ef_scale_threshold}
    if args.analyze_only:
        records = [json.loads(l) for l in open(args.analyze_only) if l.strip()]
        analyze(records, TH)
        derive_thresholds(records, TH)
        return

    cfg = Config()
    _require_live_backend(cfg)
    random.seed(args.seed)
    kernel = Kernel(LLMClient(cfg), None, cfg)
    required = compute_required(BASE_FP)
    TH = {"partition": cfg.ef_partition_threshold, "flow": cfg.ef_flow_threshold,
          "role": cfg.ef_role_threshold, "scale": cfg.ef_scale_threshold}

    print("executing base plan once (for output reuse)...")
    base = base_dag()
    base_results = {r.node_id: r for r in kernel.execute(base).results}
    base_ef = compute_ef(base, _trace_from(base, base_results), required,
                         base.centrality_weights(), ROLE_BANDS)
    print(f"base plan E: {base_ef.as_dict()}  (should be ~healthy)\n")

    records = []
    with open(args.out, "w") as fout:
        for t in range(args.trials):
            axis = random.choice(list(INJECTORS))
            fn, sevs = INJECTORS[axis]
            sev = random.choice(sevs)
            dag, changed = fn(sev)
            trace = realize(dag, changed, base_results, kernel)
            ef = compute_ef(dag, trace, required, dag.centrality_weights(), ROLE_BANDS)
            measured = ef.as_dict()
            rec = {"trial": t, "injected_axis": axis, "severity": sev,
                   "validity": axis in VALIDITY, "measured_E": measured}
            records.append(rec)
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            dom = max(EF_AXES, key=lambda a: measured[a] - TH[a])
            hit = "OK" if dom == axis else f"->{dom}"
            print(f"[{t+1}/{args.trials}] inject {axis:<9} sev={sev} "
                  f"-> E={measured} dominant={dom} {hit}")

    analyze(records, TH)
    derive_thresholds(records, TH)


def _trace_from(dag, results):
    from ceo_delta.schemas import ExecutionTrace
    ordered = [results[n.node_id] for n in dag.nodes if n.node_id in results]
    return ExecutionTrace(dag.dag_id, dag.task, ordered, 0, 0.0)


# Known SIDE EFFECTS: an injector aimed at one axis can incidentally alter a
# DIFFERENT axis's ground-truth structure, which would silently contaminate
# that other axis's "known-clean" pool if not excluded. Verified by reading
# each injector:
#   inject_flow linearizes r1->r2->r3->s1->v1 (depth 3 -> 5) -- ALWAYS also
#     perturbs scale's ground truth, regardless of flow's own severity.
#   inject_role at severity>=2 sets >=2 SIBLING root nodes (identical
#     dependency set: both roots) to byte-identical intent text -- that IS a
#     partition-style redundancy by the partition metric's own definition
#     (same-deps siblings), so it contaminates partition's ground truth.
#   inject_partition / inject_scale: verified no structural/textual overlap
#     with any other axis's measured quantity -- no side effect.
_SIDE_EFFECTS = {
    "flow": lambda sev: {"scale"},
    "role": lambda sev: {"partition"} if sev >= 2 else set(),
    "partition": lambda sev: set(),
    "scale": lambda sev: set(),
}


def derive_thresholds(records, TH):
    """Non-circular threshold derivation.

    calibrate_ef.py buckets healthy/violated using the CURRENT threshold
    itself, so "clean separation" is near-guaranteed at the boundary by
    construction -- it's a self-consistency check, not independent evidence.

    Here, ground truth comes from the EXPERIMENT DESIGN, never from any
    threshold: in each trial, exactly one axis was deliberately injected
    (KNOWN-VIOLATED, at a known severity) and the other three axes were left
    untouched (KNOWN-CLEAN, in that same trial) -- UNLESS the injector has a
    documented side effect on that axis too (_SIDE_EFFECTS above), in which
    case that trial is excluded from the clean pool rather than silently
    mislabeled. Pooling across all 60 trials gives large, independent
    clean/violated populations per axis without referencing TH anywhere in
    the split. The gap between those two ground-truth populations is what a
    threshold should sit in.
    """
    print("\n" + "=" * 60)
    print("NON-CIRCULAR THRESHOLD DERIVATION (ground truth = injection design, "
          "not current TH; side-effect-contaminated trials excluded from clean)")
    for axis in EF_AXES:
        clean, excluded = [], 0
        for r in records:
            if r["injected_axis"] == axis:
                continue
            side = _SIDE_EFFECTS.get(r["injected_axis"], lambda sev: set())(r["severity"])
            if axis in side:
                excluded += 1
                continue
            clean.append(r["measured_E"][axis])
        violated = [r["measured_E"][axis] for r in records if r["injected_axis"] == axis]
        tag = "VALIDITY" if axis in VALIDITY else "consistency"
        excl_note = f"  ({excluded} side-effect-contaminated trials excluded)" if excluded else ""
        print(f"\n  [{axis:<9} {tag:<11}] current TH={TH[axis]}{excl_note}")
        print(f"    known-clean    (n={len(clean)}): {_summary(clean)}")
        print(f"    known-violated (n={len(violated)}): {_summary(violated)}")
        if not clean or not violated:
            print("    insufficient data to derive")
            continue
        c_hi = max(clean)
        v_lo = min(violated)
        if v_lo > c_hi:
            mid = round((c_hi + v_lo) / 2, 3)
            verdict = f"CLEAN SEPARATION -> data-supported threshold ~= {mid}"
        else:
            overlap = c_hi - v_lo
            # percentile-based fallback: threshold at the point that best
            # trades off false-fire vs missed-fire, reported not prescribed
            c_sorted = sorted(clean)
            p95 = c_sorted[min(len(c_sorted) - 1, int(0.95 * (len(c_sorted) - 1)))]
            verdict = (f"OVERLAP of {overlap:.3f} (populations blend) -- "
                       f"clean p95={p95:.3f} is a defensible floor")
        agree = "MATCHES current" if abs((c_hi + v_lo) / 2 - TH[axis]) < 0.05 else "DIFFERS from current"
        print(f"    {verdict}  [{agree}]")


def _summary(vals):
    if not vals:
        return "n=0"
    vals = sorted(vals)
    q = lambda p: vals[min(len(vals) - 1, int(p * (len(vals) - 1)))]
    return (f"min={vals[0]:.3f} p25={q(.25):.3f} med={q(.5):.3f} "
            f"p75={q(.75):.3f} max={vals[-1]:.3f} mean={statistics.fmean(vals):.3f}")


def analyze(records, TH):
    print("\n" + "=" * 60)
    print("THRESHOLD VALIDITY  (measured E on injected axis, by severity)")
    for axis in INJECTORS:
        rows = [r for r in records if r["injected_axis"] == axis]
        if not rows:
            continue
        tag = "VALIDITY" if axis in VALIDITY else "consistency"
        by_sev = {}
        for r in rows:
            by_sev.setdefault(r["severity"], []).append(r["measured_E"][axis])
        cells = "  ".join(f"sev{s}:{statistics.fmean(v):.3f}(n={len(v)})"
                          for s, v in sorted(by_sev.items()))
        fired = sum(1 for r in rows if r["measured_E"][axis] > TH[axis])
        print(f"  [{axis:<9} {tag:<11} thr={TH[axis]}]  {cells}  | fired {fired}/{len(rows)}")

    print("\nDISCRIMINANT VALIDITY  (does injected axis dominate measured E?)")
    correct = 0
    conf = {a: {b: 0 for b in EF_AXES} for a in INJECTORS}
    for r in records:
        m = r["measured_E"]
        dom = max(EF_AXES, key=lambda a: m[a] - TH[a])
        conf[r["injected_axis"]][dom] += 1
        if dom == r["injected_axis"]:
            correct += 1
    print(f"  attribution accuracy: {correct}/{len(records)} = {correct/len(records)*100:.0f}%")
    print("  confusion (row=injected, col=dominant measured):")
    print("             " + "".join(f"{a[:6]:>8}" for a in EF_AXES))
    for a in INJECTORS:
        print(f"    {a:<9}" + "".join(f"{conf[a][b]:>8}" for b in EF_AXES))


if __name__ == "__main__":
    main()
