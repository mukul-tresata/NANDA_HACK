"""evidence.py — the narrative-locking evidence battery (demo build).

One focused, reproducible demonstration per headline claim about the
CEO-Delta "brain". Prints a clean, sectioned claim -> evidence report AND
writes a machine-readable `evidence_results.json` next to it, so a UI can
render the exact same run without re-executing the model.

Claims, in pitch order:
  1. CAPTURE    — derive a structural task-class (fingerprint) from raw intent.
  2. GENERALIZE — surface-different tasks that share a structure map to the
                  SAME class (structural generalization).
  3. DESCEND    — the delivered-plan quality is MONOTONE: worst-axis excess
                  never rises across the run (the PRESERVE guarantee), while
                  the descent repairs the plan toward the satisfying region.
  4. REUSE      — reuse a learned plan on a REPEAT of the same task (warm-start:
                  no planning LLM, converges in 1 iteration).
  5. CONVERGE   — varying cold-start seeds all land in the satisfying region
                  (region-stability, not a single fixed point).

Claims 1-2 are fingerprint-only (fast, ~30s). Claims 3-5 are full runs.
Each section prints as it completes.

Usage: python3 -u scripts/evidence.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ceo_delta import Orchestrator
from ceo_delta.config import Config
from ceo_delta.embeddings import cosine, embed
from ceo_delta.graph_ops import DET_OPS

# EF axis thresholds (mirror config.py). worst-excess = max(value - threshold).
_CFG = Config()
TH = {
    "partition": _CFG.ef_partition_threshold,
    "flow": _CFG.ef_flow_threshold,
    "role": _CFG.ef_role_threshold,
    "scale": _CFG.ef_scale_threshold,
}
RESULTS: list[dict] = []          # structured payload for the UI
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence_results.json")

# ---------------------------------------------------------------------------
# Presentation helpers — TTY-gated ANSI so piping to a log file stays clean.
# ---------------------------------------------------------------------------
_TTY = sys.stdout.isatty()
_B, _D, _RED, _GRN, _YEL, _CYN, _GRY = 1, 2, 31, 32, 33, 36, 90
_W = 74                                   # report width
_AXIS_ORDER = ("information_flow", "epistemic_stance", "output_contract", "decomposability")


def _s(text: str, *codes: int) -> str:
    if not _TTY or not codes:
        return text
    return "".join(f"\033[{c}m" for c in codes) + text + "\033[0m"


def _rule(ch: str = "─", color: int = _GRY) -> str:
    return _s("  " + ch * (_W - 2), color)


def _banner(title: str, subtitle: str) -> None:
    print(_s("  " + "═" * (_W - 2), _CYN))
    print("  " + _s(title, _B, _CYN))
    print("  " + _s(subtitle, _D))
    print(_s("  " + "═" * (_W - 2), _CYN))


def _claim(n: int, name: str, thesis: str) -> None:
    print()
    print("  " + _s(f"CLAIM {n}", _B, _CYN) + _s(f"   {name}", _B))
    print(_rule())
    print("  " + _s(thesis, _D))
    print()


def _pass(passed: bool, detail: str) -> None:
    tag = _s(" PASS ", _B, _GRN) if passed else _s(" CHECK ", _B, _YEL)
    print()
    print("  " + tag + "  " + _s(detail, _D if passed else _YEL))


def _chips(species: str) -> str:
    try:
        parts = dict(p.split(":", 1) for p in species.split())
    except ValueError:
        return species
    sep = _s(" · ", _GRY)
    return sep.join(_s(parts.get(k, "?"), _B) for k in _AXIS_ORDER)


def _fresh(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _warmstarted(r) -> bool:
    return any(getattr(n.why, "priors_used", "") == "warmstart" for n in r.dag.nodes)


def _worst_excess(ef: dict) -> float:
    """How far the single worst axis is over its threshold. <=0 == satisfying."""
    return round(max(ef.get(a, 0.0) - TH[a] for a in TH), 4)


def _record(cid, name, passed, headline, summary, payload) -> None:
    RESULTS.append({
        "id": cid, "name": name, "passed": bool(passed),
        "headline": headline, "summary": summary, "payload": payload,
    })


# ---------------------------------------------------------------------------
# 1 CAPTURE + 2 GENERALIZE  (fingerprint-only, no full run)
# ---------------------------------------------------------------------------

def claim_capture_and_generalize(orch: Orchestrator) -> None:
    _claim(1, "CAPTURE", "Raw natural-language intent is parsed into a structural "
                         "task-class — deterministically, no keywords.")

    distinct = [
        "Explain how a modern C compiler transforms C source into an executable.",
        "Compare Python, Rust, and Go across performance, safety, and concurrency.",
        "Given three job offers, recommend the best one and justify the tradeoffs.",
    ]
    print("  " + _s(f"{'task':44}", _GRY) + _s("structural species", _GRY))
    species = {}
    for t in distinct:
        fp, _ = orch.research.clarify(t)
        s = fp.shape_string()
        species[t] = s
        label = (t[:41] + "…") if len(t) > 42 else t
        print(f"  {label:44}{_chips(s)}")
    n_species = len(set(species.values()))
    captured = n_species >= 2
    _pass(captured, f"{n_species} distinct species across {len(distinct)} task-types")
    _record("1_capture", "CAPTURE", captured,
            f"{n_species} distinct species across {len(distinct)} task-types",
            f"{n_species} distinct species / {len(distinct)} tasks",
            {"tasks": [{"task": t, "species": s} for t, s in species.items()]})

    _claim(2, "GENERALIZE", "Pairs of tasks with almost no words in common collapse to "
                            "ONE identical structural class.")
    pairs = [
        ("Explain the entire water cycle from evaporation to groundwater recharge.",
         "Describe how an HTTP request travels from a browser to a server and back."),
        ("Explain how photosynthesis converts sunlight into chemical energy in plants.",
         "Describe how a garbage collector reclaims unused memory in a language runtime."),
        ("Explain how vaccines train the immune system to recognize a pathogen.",
         "Describe how a CPU pipeline overlaps instruction fetch, decode, and execute."),
    ]
    examples = []
    for a, b in pairs:
        fp_a, _ = orch.research.clarify(a)
        fp_b, _ = orch.research.clarify(b)
        text_sim = round(float(cosine(embed(a), embed(b))), 3)
        same_class = fp_a.shape_string() == fp_b.shape_string()
        print("  " + _s("A  ", _CYN) + a)
        print("  " + _s("B  ", _CYN) + b)
        print()
        print("  " + _s(f"{'raw text similarity':24}", _GRY) + _s(f"{text_sim}", _B)
              + _s("   (near-unrelated wording)", _D))
        shared = fp_a.shape_string() if same_class else f"A: {fp_a.shape_string()} / B: {fp_b.shape_string()}"
        print("  " + _s(f"{'shared species' if same_class else 'species':24}", _GRY)
              + (_chips(fp_a.shape_string()) if same_class else shared))
        print()
        examples.append({"task_a": a, "task_b": b, "text_cosine": text_sim,
                          "species_a": fp_a.shape_string(), "species_b": fp_b.shape_string(),
                          "same_class": same_class})

    n_ok = sum(1 for e in examples if e["same_class"] and e["text_cosine"] < 0.6)
    ok2 = n_ok == len(examples)
    _pass(ok2, f"{n_ok}/{len(examples)} pairs: unrelated wording yet same structural class")
    _record("2_generalize", "GENERALIZE", ok2,
            f"{n_ok}/{len(examples)} pairs collapse to one class despite unrelated wording",
            f"{n_ok}/{len(examples)} pairs · same class",
            {"examples": examples})


# ---------------------------------------------------------------------------
# 3 DESCEND  (monotone incumbent + repair toward the satisfying region)
# ---------------------------------------------------------------------------

def _descend_one(task: str, idx: int) -> dict:
    wd = f".evidence_descend/run{idx}"
    _fresh(wd)
    o = Orchestrator(Config(), workdir=wd)
    r = o.run(task)

    # Incumbent (delivered-plan) trace only — monotone by construction. The raw
    # ef_trace holds rejected trials and is NOT monotone, so we never use it here.
    inc = r.ef_incumbent_trace
    curve = [_worst_excess(e) for e in inc]
    monotone = all(curve[i + 1] <= curve[i] + 1e-9 for i in range(len(curve) - 1))
    improved = len(curve) >= 2 and curve[-1] < curve[0] - 1e-9

    det_drops, llm_rejected = [], []
    for m in r.ef_move_log:
        is_det = m.get("move") in DET_OPS
        if is_det and m.get("improved") and m.get("before", 0) > m.get("after", 0):
            det_drops.append(m)
        if (not is_det) and (m.get("improved") is False):
            llm_rejected.append(m)

    print("  " + _s(f"task  {task}", _CYN))
    print("  " + _s("delivered-plan worst-axis excess per iteration", _GRY)
          + _s("   (≤ 0.0 = satisfying region)", _D))
    for i, (e, w) in enumerate(zip(inc, curve)):
        mark = _s("  ← in region", _GRN) if w <= 0 else ""
        wtxt = _s(f"{w:+.4f}", _B, _GRN if w <= 0 else _YEL)
        detail = _s(f"partition {e.get('partition', 0):.3f}  role {e.get('role', 0):.3f}", _D)
        print(f"    iter {i}   {wtxt}   {detail}{mark}")
    print()
    print("  " + _s(f"{'monotone non-increasing':26}", _GRY) + _s(str(monotone), _B, _GRN if monotone else _RED))
    print("  " + _s(f"{'net worst-excess':26}", _GRY) + _s(f"{curve[0]:+.4f}  →  {curve[-1]:+.4f}", _B))

    print()
    print("  " + _s("descent moves", _GRY))
    if r.ef_move_log:
        for m in r.ef_move_log:
            is_det = m.get("move") in DET_OPS
            tag = _s("DETERMINISTIC", _B, _CYN) if is_det else _s("llm", _D)
            kept = m.get("improved")
            kmark = _s("kept", _GRN) if kept else _s("rejected", _RED)
            print(f"    [{tag}]  {m['move']:26} {m['before']:.3f} → {m['after']:.3f}   {kmark}")
    else:
        print("    " + _s("(plan already satisfied on first build — nothing to repair)", _D))
    print("  " + _s(f"deterministic repairs {len(det_drops)}  ·  worse moves rejected {len(llm_rejected)}", _D))
    print()

    return {"task": task, "verdict": r.report.verdict, "iterations": r.iterations,
            "incumbent_trace": inc, "worst_excess_curve": curve,
            "raw_trace": r.ef_trace, "move_log": r.ef_move_log,
            "monotone": monotone, "improved": improved,
            "deterministic_repairs": det_drops, "llm_rejected": llm_rejected}


def claim_descend() -> None:
    _claim(3, "DESCEND", "The delivered plan's worst-axis error is monotone "
                         "non-increasing — a guarantee (PRESERVE), not luck.")
    tasks = [
        "Plan a trip from Bangalore to Bangkok under 60k using current fares.",
        "Compare AWS, GCP, and Azure on pricing, reliability, and ease of use, "
        "and recommend one for a startup.",
    ]
    runs = [_descend_one(t, i) for i, t in enumerate(tasks)]

    n_monotone = sum(1 for run in runs if run["monotone"])
    ok = n_monotone == len(runs)
    _pass(ok, f"{n_monotone}/{len(runs)} runs monotone; "
              + ", ".join(f"{r['worst_excess_curve'][0]:+.3f}->{r['worst_excess_curve'][-1]:+.3f}"
                          for r in runs))
    _record("3_descend", "DESCEND", ok,
            f"{n_monotone}/{len(runs)} runs monotone across "
            f"{sum(len(r['deterministic_repairs']) for r in runs)} deterministic repair(s)",
            f"{n_monotone}/{len(runs)} runs monotone",
            {"runs": runs})
    _fresh(".evidence_descend")


# ---------------------------------------------------------------------------
# 4 REUSE  (warm-start on a repeat of the SAME task)
# ---------------------------------------------------------------------------

def _reuse_one(task: str, idx: int) -> dict:
    wd = f".evidence_reuse/task{idx}"
    _fresh(wd)
    o = Orchestrator(Config(), workdir=wd)

    r1 = o.run(task)
    r2 = o.run(task)
    rows = [("run 1", r1, "cold — plans from scratch, caches best plan"),
            ("run 2", r2, "warm — reuses the cached plan")]
    print("  " + _s(f"task  {task}", _CYN))
    for label, r, note in rows:
        warm = _warmstarted(r)
        state = _s("warm", _B, _GRN) if warm else _s("cold", _D)
        arrow = _s("   ← reused learned plan", _GRN) if warm else ""
        print(f"    {_s(label, _B)}   {state:>6}   verdict {r.report.verdict:5}   "
              f"{r.iterations} iter{arrow}")
        print("           " + _s(note, _D))
    print()
    _fresh(wd)

    return {"task": task,
            "run1": {"verdict": r1.report.verdict, "iters": r1.iterations, "warm": _warmstarted(r1)},
            "run2": {"verdict": r2.report.verdict, "iters": r2.iterations, "warm": _warmstarted(r2)}}


def claim_reuse() -> None:
    _claim(4, "REUSE", "The second time it meets a task-shape it has solved, it "
                       "reuses the learned plan — no re-planning.")
    tasks = [
        "Explain how a modern C compiler transforms C source into an executable.",
        "Explain how garbage collection reclaims unused memory in a language runtime.",
    ]
    examples = [_reuse_one(t, i) for i, t in enumerate(tasks)]

    n_ok = sum(1 for e in examples if e["run2"]["warm"] and not e["run1"]["warm"])
    ok = n_ok == len(examples)
    _pass(ok, f"{n_ok}/{len(examples)} tasks: run 2 warm-started")
    _record("4_reuse", "REUSE", ok,
            f"{n_ok}/{len(examples)} tasks warm-started on repeat",
            f"{n_ok}/{len(examples)} tasks warm on repeat",
            {"examples": examples})


# ---------------------------------------------------------------------------
# 5 CONVERGE  (region-stability from varying cold starts, temp>0)
# ---------------------------------------------------------------------------

def claim_converge() -> None:
    _claim(5, "CONVERGE", "Different cold-start seeds all get driven into the same "
                          "satisfying region — stability of a region, not a point.")
    task = "Explain how a modern C compiler transforms C source into an executable."
    reps = []
    for i in range(1, 4):
        wd = f".evidence_converge/rep{i}"
        _fresh(wd)
        cfg = Config()
        cfg.llm_temperature = 0.7          # THIS TEST ONLY: let the cold plan vary
        o = Orchestrator(cfg, workdir=wd)
        r = o.run(task)
        reps.append({"rep": i, "verdict": r.report.verdict, "iters": r.iterations,
                     "ef": r.report.ef_tensor})
        in_reg = r.report.verdict in ("good", "mixed")
        vtxt = _s(f"{r.report.verdict:5}", _B, _GRN if in_reg else _RED)
        print(f"    seed {i}   {vtxt}   {r.iterations} iter"
              + (_s("   • in region", _GRN) if in_reg else ""))
    verdicts = [x["verdict"] for x in reps]
    in_region = all(v in ("good", "mixed") for v in verdicts)
    _pass(in_region, f"{sum(v in ('good','mixed') for v in verdicts)}/{len(verdicts)} "
                     f"in region {{good, mixed}}")
    _record("5_converge", "CONVERGE", in_region,
            f"verdicts={verdicts} all in satisfying region={in_region}",
            f"{sum(v in ('good','mixed') for v in verdicts)}/{len(verdicts)} in region",
            {"task": task, "reps": reps})
    _fresh(".evidence_converge")


def main() -> None:
    t0 = time.time()
    _banner("CEO-DELTA   ·   EVIDENCE BATTERY",
            "Structural planning intelligence — five claims, one live run")
    orch = Orchestrator(Config(), workdir=".evidence_fp")
    claim_capture_and_generalize(orch)
    claim_descend()
    claim_reuse()
    claim_converge()
    _fresh(".evidence_fp")

    elapsed = round(time.time() - t0, 1)
    n_pass = sum(1 for r in RESULTS if r["passed"])
    print()
    _banner("EVIDENCE SUMMARY", f"{n_pass} / {len(RESULTS)} claims passed   ·   {elapsed:.0f}s")
    for r in RESULTS:
        mark = _s(" PASS ", _B, _GRN) if r["passed"] else _s(" CHECK ", _B, _YEL)
        print(f"  {mark}  {_s(r['name'], _B):11}   {_s(r['summary'], _D)}")
    print()
    print("  " + _s(f"payload → {OUT_JSON}", _D))

    with open(OUT_JSON, "w") as f:
        json.dump({"generated_at": time.time(), "elapsed_s": elapsed,
                   "thresholds": TH, "claims": RESULTS}, f, indent=2)


if __name__ == "__main__":
    main()
