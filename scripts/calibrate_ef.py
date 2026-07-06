#!/usr/bin/env python3
"""EF threshold calibration DIAGNOSTIC.

Reports the actual per-axis distribution of E values across the run corpus so
we can set thresholds from data instead of hand-guessing. Deliberately a
REPORTER, not an auto-setter — it tells you which axes have enough contrast to
calibrate and which are still too thin (n too small) to touch without
overfitting.

For each axis it separates two populations from the per-iteration ef_trace:
  - HEALTHY: values at iterations where the run's verdict was good / the axis
    was not the one being corrected.
  - VIOLATED: values at iterations where the axis exceeded its current
    threshold (the pre-correction moments — the ones that matter).

A good threshold sits in the GAP between these two clouds. If they overlap
heavily, or VIOLATED has too few samples, the axis is NOT ready to calibrate.

Usage:
    python scripts/calibrate_ef.py [path/to/runs.jsonl]
"""
from __future__ import annotations

import json
import statistics
import sys

AXES = ("partition", "flow", "role", "scale")
CURRENT = {"partition": 0.70, "flow": 0.25, "role": 0.35, "scale": 0.34}
MIN_VIOLATED_SAMPLES = 5   # below this, refuse to recommend a threshold


def load(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collect(rows):
    """Return per-axis {'healthy': [...], 'violated': [...]} from ef_trace."""
    buckets = {ax: {"healthy": [], "violated": []} for ax in AXES}
    n_traces = 0
    for r in rows:
        trace = r.get("ef_trace") or []
        if not trace:
            continue
        n_traces += 1
        for step in trace:
            for ax in AXES:
                v = step.get(ax)
                if v is None:
                    continue
                if v > CURRENT[ax]:
                    buckets[ax]["violated"].append(v)
                else:
                    buckets[ax]["healthy"].append(v)
    return buckets, n_traces


def summarize(vals):
    if not vals:
        return "n=0"
    vals = sorted(vals)
    q = lambda p: vals[min(len(vals) - 1, int(p * (len(vals) - 1)))]
    return (f"n={len(vals)} min={vals[0]:.3f} p25={q(.25):.3f} "
            f"med={q(.5):.3f} p75={q(.75):.3f} max={vals[-1]:.3f} "
            f"mean={statistics.fmean(vals):.3f}")


def recommend(ax, healthy, violated):
    if len(violated) < MIN_VIOLATED_SAMPLES:
        return (f"  ⚠ INSUFFICIENT DATA — only {len(violated)} violated sample(s) "
                f"(need >={MIN_VIOLATED_SAMPLES}). Leave threshold at {CURRENT[ax]} "
                f"(provisional). Generate targeted correction events first.")
    if not healthy:
        return "  ⚠ no healthy samples — cannot locate the gap."
    h_hi = max(healthy)
    v_lo = min(violated)
    if v_lo > h_hi:
        mid = round((h_hi + v_lo) / 2, 3)
        return (f"  ✓ CLEAN SEPARATION — healthy tops out at {h_hi:.3f}, violated "
                f"starts at {v_lo:.3f}. Data-supported threshold ≈ {mid} "
                f"(current {CURRENT[ax]}).")
    overlap = h_hi - v_lo
    return (f"  ~ OVERLAP of {overlap:.3f} (healthy up to {h_hi:.3f}, violated "
            f"down to {v_lo:.3f}) — populations blend; threshold is a judgment "
            f"call, not a clean cut. Current {CURRENT[ax]}.")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else ".ceo_delta/runs.jsonl"
    try:
        rows = load(path)
    except FileNotFoundError:
        print(f"No corpus at {path}. Run some tasks first (with the v3.0 logger).")
        return
    buckets, n_traces = collect(rows)
    print(f"=== EF CALIBRATION DIAGNOSTIC ===")
    print(f"corpus: {len(rows)} runs, {n_traces} with EF traces\n")
    if n_traces == 0:
        print("No ef_trace found in any run. These runs predate the v3.0 logger "
              "fix — re-run tasks to accumulate a calibratable corpus.")
        return
    for ax in AXES:
        h, v = buckets[ax]["healthy"], buckets[ax]["violated"]
        print(f"[{ax}]  (current threshold = {CURRENT[ax]})")
        print(f"  healthy : {summarize(h)}")
        print(f"  violated: {summarize(v)}")
        print(recommend(ax, h, v))
        print()


if __name__ == "__main__":
    main()
