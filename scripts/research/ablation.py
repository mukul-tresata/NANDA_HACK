"""ablation.py — does the ARCHITECTURE lift capability, or is it just scaffolding?

Tests the hypothesis: routing a task through CEO-Delta improves the base model's
inherent reasoning beyond what a generic agentic scaffold (CoT + decompose +
verify + best-of-k) achieves. The only comparison that supports that hypothesis
is the RESIDUAL of the full system over a generic scaffold:

    acc(4 full CEO-Delta)  −  acc(3 generic scaffold)

acc(4) > acc(1 raw) only re-proves that scaffolding helps (already known).
acc(4) > acc(3) is the claim. Condition 5 (self-consistency / best-of-k) is the
literal "best-of-3" strawman — the full system is DETERMINISTIC (temp=0), so
(4) matching or beating a non-deterministic (5) is the determinism narrative
made quantitative.

All conditions run on the SAME Qwen backend (ceo_delta.llm.LLMClient), so the
model is held constant and only the orchestration around it varies. GSM8K is
first because its answers are numbers — uniform, robust extraction, no
multiple-choice letter-mapping sink that could mask a real lift.

Usage:
    python3 -u scripts/ablation.py --n 5                       # smoke test
    python3 -u scripts/ablation.py --n 100 --conditions direct,cot,scaffold,ceo
    python3 -u scripts/ablation.py --n 200 --conditions ceo,scaffold,sc --sc-k 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ceo_delta import Orchestrator
from ceo_delta.config import Config
from ceo_delta.llm import LLMClient

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".ablation_cache")
OUT_JSON = os.path.join(HERE, "ablation_results.json")
GSM8K_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
             "master/grade_school_math/data/test.jsonl")
BBH_URL = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{task}.json"

# Curated diverse subset: 10 genuinely different reasoning SHAPES with clean,
# robustly-extractable answer formats (MC letter / yes-no / true-false /
# valid-invalid / count). This is the point of BBH for us — each task is a
# different structure, so CEO-Delta's fingerprint adaptation actually has
# something to adapt to (unlike homogeneous GSM8K).
BBH_TASKS = [
    "logical_deduction_three_objects",   # MC (A/B/C)
    "date_understanding",                # MC
    "reasoning_about_colored_objects",   # MC
    "temporal_sequences",                # MC
    "navigate",                          # Yes/No
    "web_of_lies",                       # Yes/No
    "causal_judgement",                  # Yes/No
    "boolean_expressions",               # True/False
    "formal_fallacies",                  # valid/invalid
    "object_counting",                   # integer
]

# HARD subset: tasks where Qwen-35B is NOT saturated (~40-70%), so there is real
# headroom for any method to show a lift. The "easy" BBH_TASKS above ceiling'd at
# direct=100% (n=10 pilot), which cannot discriminate. Extraction is still clean:
# these are mostly multiple-choice (A/B/C...) plus a couple structured-string /
# integer tasks, all covered by _extract_bbh + _norm_bbh.
HARD_BBH_TASKS = [
    "tracking_shuffled_objects_seven_objects",  # MC
    "logical_deduction_seven_objects",          # MC (harder than 3-object)
    "geometric_shapes",                         # MC (A-K)
    "hyperbaton",                               # MC (A/B) adjective order
    "salient_translation_error_detection",      # MC (A-F)
    "penguins_in_a_table",                      # MC
    "snarks",                                   # MC (A/B) sarcasm
    "disambiguation_qa",                        # MC
    "multistep_arithmetic_two",                 # integer (often negative)
    "word_sorting",                             # sorted word list (string)
]

# generic across datasets; the model infers the answer form from the question.
ANSWER_FMT = "End your response with a line exactly like: The answer is <answer>."
DATASET = "gsm8k"   # set in main(); routes extraction + scoring


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_gsm8k(n: int, seed: int) -> list[dict]:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "gsm8k_test.jsonl")
    if not os.path.exists(path):
        print(f"[data] downloading GSM8K test set -> {path}", flush=True)
        urllib.request.urlretrieve(GSM8K_URL, path)
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            gold = o["answer"].split("####")[-1].strip()
            rows.append({"question": o["question"], "gold": _to_num(gold)})
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def load_bbh(n: int, seed: int, tasks: list[str] | None = None) -> list[dict]:
    """Mixed sample across diverse BBH task types — each item is potentially a
    different reasoning shape, which is the whole reason to use BBH here."""
    tasks = tasks or BBH_TASKS
    os.makedirs(CACHE, exist_ok=True)
    rows = []
    for t in tasks:
        path = os.path.join(CACHE, f"bbh_{t}.json")
        if not os.path.exists(path):
            print(f"[data] downloading BBH/{t}", flush=True)
            urllib.request.urlretrieve(BBH_URL.format(task=t), path)
        data = json.load(open(path))
        for ex in data["examples"]:
            rows.append({"question": ex["input"], "gold": ex["target"], "task": t})
    rng = random.Random(seed)
    rng.shuffle(rows)          # interleave tasks so the sample is shape-diverse
    return rows[:n]


# ---------------------------------------------------------------------------
# Answer extraction — ONE function, applied identically to every condition.
# ---------------------------------------------------------------------------

_NUM = r"-?\$?\s*[\d,]*\.?\d+"


def _to_num(s):
    if s is None:
        return None
    s = str(s).replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        f = float(s)
        return int(f) if f == int(f) else round(f, 4)
    except ValueError:
        return None


def _extract_gsm8k(text: str):
    """Prefer an explicit 'answer is X', then a '#### X', else the last number."""
    if not text:
        return None
    for pat in (r"answer\s*is\s*(" + _NUM + r")",
                r"####\s*(" + _NUM + r")"):
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            return _to_num(m[-1])
    nums = re.findall(_NUM, text)
    return _to_num(nums[-1]) if nums else None


def _norm_bbh(s: str):
    """Normalize a BBH answer for exact-match: strip parens/punct/case/spaces."""
    if s is None:
        return None
    s = str(s).strip().rstrip(".").strip()
    s = s.strip("()").strip()          # "(A)" -> "A"
    s = re.sub(r"\s+", " ", s).lower()
    return s or None


def _extract_bbh(text: str):
    """Grab what follows 'answer is', else the last non-empty line, normalized."""
    if not text:
        return None
    m = re.search(r"answer\s*is[:\s]*\(?\s*([A-Za-z0-9 ,'\-/]+?)\s*\)?\s*(?:\.|\n|$)",
                  text + "\n", re.IGNORECASE)
    if m:
        return _norm_bbh(m.group(1))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return _norm_bbh(lines[-1]) if lines else None


def extract_answer(text: str):
    return _extract_bbh(text) if DATASET == "bbh" else _extract_gsm8k(text)


# Boolean-family synonyms: a yes/no question and a true/false question are the
# same question in disguise. Qwen answered "False" where gold was "No" (web_of_lies
# item 7 in the pilot) — correct reasoning, wrong token. Merging these classes
# measures REASONING, not which surface word the model happened to pick.
_BOOL_SYN = {
    "yes": "y", "true": "y", "correct": "y", "valid": "y",
    "no": "n", "false": "n", "incorrect": "n", "invalid": "n",
}


def _canon_bbh(s):
    n = _norm_bbh(s)
    return _BOOL_SYN.get(n, n)


def correct(pred, gold) -> bool:
    if pred is None or gold is None:
        return False
    if DATASET == "bbh":
        return _canon_bbh(pred) == _canon_bbh(gold)
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Conditions — each returns (pred_number, raw_text_for_audit)
# ---------------------------------------------------------------------------

def cond_direct(llm: LLMClient, q: str):
    out = llm.chat([{"role": "user",
                     "content": f"Solve this problem. {ANSWER_FMT}\n\n{q}"}],
                   temperature=0.0, tag="direct")
    return extract_answer(out), out


def cond_cot(llm: LLMClient, q: str):
    out = llm.chat([{"role": "user",
                     "content": f"Solve this problem. Think step by step, showing "
                                f"your work. {ANSWER_FMT}\n\n{q}"}],
                   temperature=0.0, tag="cot")
    return extract_answer(out), out


def cond_scaffold(llm: LLMClient, q: str):
    """Generic decompose -> solve -> verify. Deterministic (temp=0). NO
    fingerprint, NO EF tensor, NO descent, NO PRESERVE — the 'any engineer
    could write this in 20 lines' control that condition 4 must beat."""
    steps = llm.chat([{"role": "user",
                       "content": f"Break this problem into a short numbered list "
                                  f"of the sub-steps needed to solve it. Do NOT "
                                  f"solve it yet.\n\n{q}"}],
                     temperature=0.0, tag="scaffold.decompose")
    draft = llm.chat([{"role": "user",
                       "content": f"Problem:\n{q}\n\nSub-steps:\n{steps}\n\nNow "
                                  f"work through each sub-step and solve. {ANSWER_FMT}"}],
                     temperature=0.0, tag="scaffold.solve")
    checked = llm.chat([{"role": "user",
                         "content": f"Problem:\n{q}\n\nProposed solution:\n{draft}\n\n"
                                    f"Check the arithmetic and logic carefully. If it "
                                    f"is correct, restate the answer. If it is wrong, "
                                    f"redo it correctly. {ANSWER_FMT}"}],
                       temperature=0.0, tag="scaffold.verify")
    return extract_answer(checked), checked


def cond_sc(llm: LLMClient, q: str, k: int = 3):
    """Self-consistency / best-of-k: k CoT samples at temp>0, majority vote on
    the extracted answer. The literal 'best-of-3' strawman — and NON-deterministic
    by construction, unlike the full system."""
    votes, raws = [], []
    for i in range(k):
        out = llm.chat([{"role": "user",
                         "content": f"Solve this problem. Think step by step. "
                                    f"{ANSWER_FMT}\n\n{q}"}],
                       temperature=0.7, tag=f"sc.{i}")
        raws.append(out)
        a = extract_answer(out)
        if a is not None:
            votes.append(a)
    if not votes:
        return None, raws
    winner = Counter(votes).most_common(1)[0][0]
    return winner, raws


def cond_ceo(q: str, item_idx: int):
    """Full CEO-Delta. Fresh per-item store so we measure CAPABILITY, not
    cross-item learning (that's a separate, deliberate experiment)."""
    wd = os.path.join(CACHE, f"ceo_wd/item_{item_idx}")
    shutil.rmtree(wd, ignore_errors=True)
    try:
        # same format instruction the other conditions get, so compose emits
        # an extractable answer — keeps the comparison fair.
        r = Orchestrator(Config(), workdir=wd).run(f"{q}\n\n{ANSWER_FMT}")
        return extract_answer(r.answer), r.answer
    finally:
        shutil.rmtree(wd, ignore_errors=True)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def bootstrap_ci(a: list[bool], b: list[bool], iters: int = 2000, seed: int = 0):
    """95% CI on acc(a) - acc(b) over paired items (resample item indices)."""
    rng = random.Random(seed)
    n = len(a)
    if n == 0:
        return (0.0, 0.0)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(a[i] for i in idx) / n
        db = sum(b[i] for i in idx) / n
        diffs.append(da - db)
    diffs.sort()
    return (round(diffs[int(0.025 * iters)], 4), round(diffs[int(0.975 * iters)], 4))


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Paired discordant counts for a (treatment) vs b (control)."""
    a_only = sum(1 for x, y in zip(a, b) if x and not y)   # a right, b wrong
    b_only = sum(1 for x, y in zip(a, b) if y and not x)   # b right, a wrong
    both = sum(1 for x, y in zip(a, b) if x and y)
    neither = sum(1 for x, y in zip(a, b) if not x and not y)
    return {"treatment_only": a_only, "control_only": b_only,
            "both": both, "neither": neither}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CONDITIONS = ["direct", "cot", "scaffold", "sc", "ceo"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--conditions", default="direct,cot,scaffold,ceo")
    ap.add_argument("--sc-k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--task-set", default="easy", choices=["easy", "hard"],
                    help="bbh only: 'easy' (saturated pilot subset) or 'hard' "
                         "(non-ceiling'd tasks with real headroom)")
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--hard-only", action="store_true",
                    help="keep only items raw Qwen (direct) gets WRONG, then measure "
                         "RECOVERY rate on that base-failure subset — where a lift can appear")
    ap.add_argument("--pool", type=int, default=400,
                    help="max items to scan when collecting --n base-model failures")
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conds:
        if c not in CONDITIONS:
            raise SystemExit(f"unknown condition {c!r}; choose from {CONDITIONS}")

    global DATASET
    if args.dataset not in ("gsm8k", "bbh"):
        raise SystemExit("dataset must be gsm8k or bbh")
    DATASET = args.dataset
    bbh_tasks = HARD_BBH_TASKS if args.task_set == "hard" else BBH_TASKS
    load = ((lambda n, seed: load_bbh(n, seed, bbh_tasks))
            if DATASET == "bbh" else load_gsm8k)

    llm = LLMClient(Config())

    if args.hard_only:
        pool = load(args.pool, args.seed)
        print(f"[hard-only] scanning up to {len(pool)} items for base-model failures "
              f"(target {args.n})...", flush=True)
        items, scanned = [], 0
        for j, it in enumerate(pool):
            scanned = j + 1
            pred, raw = cond_direct(llm, it["question"])
            if not correct(pred, it["gold"]):
                it["_direct"] = {"pred": pred, "raw": raw, "correct": False}
                items.append(it)
                if len(items) >= args.n:
                    break
            if scanned % 20 == 0:
                print(f"  scanned {scanned}, found {len(items)} hard", flush=True)
        fail_rate = len(items) / scanned if scanned else 0.0
        print(f"[hard-only] using {len(items)} base-model failures "
              f"(scanned {scanned}, base fail-rate {fail_rate*100:.1f}%)", flush=True)
        print(f"[hard-only] metric is RECOVERY RATE — fraction of base failures each "
              f"method rescues", flush=True)
    else:
        items = load(args.n, args.seed)
    print(f"[run] dataset={DATASET} n={len(items)} conditions={conds} "
          f"hard_only={args.hard_only}", flush=True)
    results = {c: [] for c in conds}      # per-condition list[bool]
    audit = []                            # per-item detail
    t0 = time.time()

    for i, it in enumerate(items):
        q, gold = it["question"], it["gold"]
        row = {"idx": i, "gold": gold, "conds": {}}
        for c in conds:
            ts = time.time()
            if c == "direct":
                if "_direct" in it:                       # reuse hard-only pre-filter pass
                    pred, raw = it["_direct"]["pred"], it["_direct"]["raw"]
                else:
                    pred, raw = cond_direct(llm, q)
            elif c == "cot":
                pred, raw = cond_cot(llm, q)
            elif c == "scaffold":
                pred, raw = cond_scaffold(llm, q)
            elif c == "sc":
                pred, raw = cond_sc(llm, q, args.sc_k)
            elif c == "ceo":
                pred, raw = cond_ceo(q, i)
            ok = correct(pred, gold)
            results[c].append(ok)
            row["conds"][c] = {"pred": pred, "correct": ok,
                               "secs": round(time.time() - ts, 1),
                               "raw": (raw if isinstance(raw, str) else raw)}
        audit.append(row)
        flags = " ".join(f"{c}={'✓' if row['conds'][c]['correct'] else '✗'}" for c in conds)
        print(f"  item {i:>3} gold={gold!s:>8}  {flags}", flush=True)

    # ---- report ----
    metric = "RECOVERY RATE (of base-model failures)" if args.hard_only else "ACCURACY"
    print("\n" + "=" * 60, flush=True)
    print(f"{metric}  (n={len(items)}, {DATASET}"
          + (", hard-only" if args.hard_only else "") + ")", flush=True)
    print("=" * 60, flush=True)
    acc = {}
    for c in conds:
        k = sum(results[c])
        acc[c] = k / len(items) if items else 0.0
        print(f"  {c:10} {acc[c]*100:5.1f}%   ({k}/{len(items)})", flush=True)

    residuals = {}
    if "ceo" in conds:
        print("\nRESIDUALS (the hypothesis lives here):", flush=True)
        for base in ("scaffold", "sc", "cot", "direct"):
            if base in conds:
                diff = acc["ceo"] - acc[base]
                ci = bootstrap_ci(results["ceo"], results[base], seed=args.seed)
                mc = mcnemar(results["ceo"], results[base])
                residuals[f"ceo_minus_{base}"] = {"delta": round(diff, 4), "ci95": ci, "mcnemar": mc}
                star = "  <-- THE CLAIM" if base == "scaffold" else ""
                print(f"  ceo − {base:9} = {diff*100:+5.1f}%   95%CI [{ci[0]*100:+.1f}%, "
                      f"{ci[1]*100:+.1f}%]   (ceo-only {mc['treatment_only']}, "
                      f"{base}-only {mc['control_only']}){star}", flush=True)

    elapsed = round(time.time() - t0, 1)
    print(f"\n  total {elapsed}s", flush=True)

    with open(args.out, "w") as f:
        json.dump({"dataset": args.dataset, "n": len(items), "conditions": conds,
                   "seed": args.seed, "elapsed_s": elapsed,
                   "accuracy": acc, "residuals": residuals, "audit": audit}, f, indent=2)
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
