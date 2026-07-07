# CEO-Delta — Project Overview

> A self-improving multi-agent **planning + execution** engine. You give it a
> raw task in natural language; it derives the task's *structure*, plans a DAG
> of specialized agents, executes them, measures how far the realized plan
> deviates from the structure the task *required*, and repairs the plan with
> deterministic graph surgery — never delivering a worse plan than it started
> with, and reusing what it learned on the next task of the same shape.
>
> This document is exhaustive on purpose: it is meant to be handed to another
> agent to build a **demo UI**. The last section ("UI-Builder Guide") lists the
> exact data structures you can render.

---

## 0. The one-paragraph pitch

Most agent frameworks plan by prompting an LLM and hoping. CEO-Delta separates
**what structure a task requires** (the *fingerprint* F, derived
deterministically) from **what structure the plan actually realized** (measured
as an error tensor E in the *same coordinate system* as F). Because E and F
live in one basis, improving a plan becomes literal **coordinate descent**:
lock onto the worst over-threshold axis, apply a deterministic graph transform,
re-measure, confirm it dropped. A hard guarantee (PRESERVE) makes the delivered
plan's worst axis **monotone non-increasing** — bad LLM re-rolls are measured
and rejected, so the loop can only stand still or improve. What it learns is
keyed to the task's *shape*, so a surface-different task with the same structure
inherits the fix.

---

## 1. Core mental model — E, F, and the shared basis

Everything hinges on **one coordinate system with four structural axes**:

```
BASIS = [ partition , flow , role , scale ]
```

- **F = RequiredStructure** — what structure the task *requires*. Derived
  deterministically from the task fingerprint (no LLM). "This is a divergent,
  retrieval task, so it needs parallel gathering + a synthesizer + depth ~3."
- **E = EFTensor** — how far the *realized* plan+execution deviates from F, per
  axis, each in `[0, 1]`. "Your plan's flow error is 0.0 (good) but partition
  error is 0.84 (redundant siblings)."

`E` is the **dual of the fingerprint**: error is measured *in fingerprint
space*, so it is invariant to how the task was phrased. Two tasks that read
completely differently but share a fingerprint are measured — and repaired —
the same way.

A second, **content** tensor rides alongside for answer quality:

```
Q_BASIS = [ groundedness , relevance ]   # generation-side RAGAS, no LLM judge
```

The final verdict is a **single uniform gate** over the full basis
(4 structural + 2 content axes): every axis under its threshold → `good`; one
axis barely over → `mixed`; otherwise → `poor`. There is exactly **one** error
model — no parallel scoring, no weighted sums collapsing axes into a scalar.

### 1.1 The axes, precisely

| Axis | Measures | Computed from | Threshold |
|------|----------|---------------|-----------|
| `partition` | over-partition: redundant sibling nodes doing the same work | mean cosine of sibling outputs (same dependency set) | 0.70 |
| `flow` | realized DAG shape vs required information flow (divergent/convergent/sequential/recursive) | graph topology (roots, joins, fan-out, depth) | 0.25 |
| `role` | behavioral misalignment + a required functional role being absent | behavioral bands on node output + missing-role penalty | 0.35 |
| `scale` | realized depth vs complexity-implied depth budget | critical-path depth vs target | 0.34 |
| `groundedness` | claims not backed by retrieved/verified evidence | verifier `[UNVERIFIED]` markers + entity overlap | (content) |
| `relevance` | answer's semantic distance from the actual task | cosine(answer, task) | (content) |

`worst_excess(E) = max(value − threshold)` over the axes. `≤ 0` means the plan
sits in the **satisfying region**.

---

## 2. The PRESERVE guarantee (the thing that makes descent *predictable*)

This is the headline property and lives entirely in `descent.py`:

> The plan delivered is never worse (by worst-axis excess) than the best plan
> seen this run. Across the incumbents, worst-axis excess is **monotone
> non-increasing**. A change that is structurally inert delivers a
> byte-identical plan.

Mechanically (`descent.py`, `step()`): each iteration measures the new plan's
`worst_excess`. The new plan becomes the **incumbent** only if it is no worse
than the current incumbent (with a lexicographic Q tiebreak on structural
ties). Otherwise it is **rejected** and the descent keeps building from the
incumbent. One bad LLM replan can no longer poison the rest of the run.

**Why this matters for the demo:** it converts "watch it maybe get better" into
"watch a curve that provably only goes down." The `ef_incumbent_trace` is that
curve. The raw `ef_trace` is diagnostic (it shows the *trials*, including ones
that got rejected).

---

## 3. The run pipeline (end to end)

`Orchestrator.run(task)` — the single public entrypoint. One call does:

```
raw task
  │
  ▼
[1] RESEARCH.clarify(task)            research.py
     → TaskFingerprint (F-inputs: flow/stance/contract/decomposability
        + complexity/volatility modifiers)  ← this is CAPTURE
     → TaskSpecifics (structured_intent, constraints)
  │
  ▼
[2] compute_required(fingerprint)     ef.py   → RequiredStructure F
  │
  ▼
┌── DIRECTIVE / DESCENT LOOP (max_iter) ──────────────────────────────┐
│ [3] _next_dag(...)                   orchestrator.py                  │
│      • iteration 0: WARM-START if a cached best_plan exists for this  │
│        fingerprint (warmstart.py), else CEO cold-plans (ceo.py)       │
│      • later iterations: if the directive names a DETERMINISTIC move, │
│        apply it as a pure graph transform (graph_ops.py) — no LLM     │
│ [4] force_verifier if volatility requires it (structural gate)        │
│ [5] iteration 0 only: RESEARCH.investigate() secondary pass          │
│      (won't clobber a warm-started plan)                              │
│ [6] KERNEL.execute(dag)             kernel.py → ExecutionTrace        │
│      (retriever nodes call grounding.retrieve(); others run the LLM)  │
│ [7] DELTA.audit(dag, trace)         delta.py                          │
│      → compute_ef() → EFTensor E    ef.py                             │
│      → Descent.step() picks the worst axis, emits a DeltaDirective    │
│        (surface | refine | replan) naming the repair move             │
│      → PRESERVE: incumbent updated only if not worse  ← DESCEND       │
│ [8] track best_* mirrors the descent's incumbent decision            │
│ if directive.action == "surface": break                              │
└──────────────────────────────────────────────────────────────────────┘
  │
  ▼
[9]  end_run: persist this fingerprint's trajectory + learned moves
       to ef_store (F-keyed case memory)     ← this is what REUSE reads
  │
  ▼
[10] CEO.compose(...)                 ceo.py
       one coherent answer in a single voice from the winning DAG's work
  │
  ▼
[11] Q-factor: compute_quality(...)   quality.py
       measure groundedness+relevance on the DELIVERED answer,
       fold into final_verdict over the full 6-axis basis
  │
  ▼
[12] reflection (periodic), persist, return RunResult
```

### The two learning loops

- **Intra-run (DESCEND):** the directive loop above. Deterministic graph
  surgery repairs the plan *within* a single `run()`.
- **Cross-run (REUSE):** `ef_store` writes the lowest-worst-excess plan ever
  seen *per fingerprint species*. `warmstart.py` reads it back on the next
  task of the same shape and reconstructs a runnable DAG, so the second run of
  a shape skips cold planning (converges in 1 iteration).

---

## 4. Module map (`ceo_delta/`, ~6.8k LOC)

**Orchestration & control**
- `orchestrator.py` (536) — the `run()` pipeline above; `RunResult`; wires
  every subsystem; owns the directive loop and best-plan tracking.
- `delta.py` (803) — the Delta agent: audits a plan/trace, computes E, owns the
  `Descent` object, emits `DeltaDirective`s; `DeltaReport` (verdict + tensors).
- `descent.py` (593) — **coordinate descent + PRESERVE**. `step()` does
  incumbent update with uphill rejection; maintains `axis_trace`,
  `incumbent_trace` (the monotone curve), `move_log`, `detail_trace`.

**Structure: F and E**
- `schemas.py` (399) — all core dataclasses: `TaskFingerprint`, `DAG`, `Node`,
  `Roles`, `Why`, `ExecutionTrace`, `NodeResult`, `HandbookEntry`, `AgentCard`,
  `AgentRegistry`, `DeltaDirective`.
- `ef.py` (284) — `RequiredStructure` F; `compute_required()`;
  `graph_signature()`; per-axis error functions; `EFTensor`;
  `verdict_from_ef()` and `final_verdict()` (the uniform gate).
- `error_tensor.py` (308) — legacy v2 execution tensor (drift/echo/cascade/
  role/resource) + hysteresis + violation ranking. Still used for some
  directive/escalation continuity; the planning verdict is the v3 EF path.
- `role_features.py` (114) — deterministic behavioral profile of a node's
  output vs per-role BANDS → `role_error`. Backs the `role` axis.

**Planning & execution**
- `research.py` (254) — Research agent (upstream **parser**, not a learner):
  raw intent → `(TaskFingerprint, TaskSpecifics)`. `clarify()` is CAPTURE.
- `ceo.py` (433) — CEO planner: fingerprint → handbook query → agent
  resolution → full `DAG`; `replan()`, `force_verifier()`, and `compose()`
  (the single-voice deliverable).
- `kernel.py` (203) — execution kernel: runs the DAG, flows upstream outputs
  into downstream nodes, returns an `ExecutionTrace`.
- `grounding.py` (143) — retriever nodes call `retrieve()` (backed by a web
  search seam) instead of hallucinating. Backend-agnostic.
- `llm.py` (204) — vLLM OpenAI-compatible client (self-hosted Qwen-family);
  defensive JSON extraction; optional deterministic stub.
- `embeddings.py` (31) — `embed()` / `cosine()` via `all-MiniLM-L6-v2`
  (384-dim unit vectors). The geometry behind every cosine in the system.

**Repair repertoire (the deterministic operators)**
- `graph_ops.py` (688) — pure graph transforms + `DET_OPS` registry +
  `apply_move()`. See §5.
- `moves.py` (228) — table-lookup that picks *which* structural edit to attempt
  from `(worst axis, F, current signature, what's been tried)` — never "ask the
  LLM to reconsider." This is what makes descent deterministic.

**Learning & memory**
- `ef_store.py` (229) — F-keyed case memory (keyed on `shape_string`, a small
  controlled vocabulary so exact-match == same species). Writes per-species
  `best_plan` (lowest worst-excess ever seen). The persistent E–F coupling.
- `warmstart.py` (74) — the **read** side of `best_plan`: reconstructs a
  runnable DAG for REUSE, gated on task-identity.
- `handbook.py` (132) — planning handbook: topology/depth vote tallies with
  symmetric decay; CEO's prior source.
- `escalation.py` (246) — case-based escalation: nearest precedent by weighted
  cosine over the excess-vector, role-gated; LLM fallback below floor.
- `bootstrap.py` (173) — cold-start seeding so run 1 isn't empty.
- `reflection.py` (93) — periodic between-run reflection (never mid-run).
- `runlog.py` (167) — append-only `.jsonl` per-run record (fingerprint, DAG,
  tensors, verdict) — a ready-made event log for a UI/timeline.
- `quality.py` (130) — Q-factor: generation-side RAGAS (groundedness,
  relevance), mechanical (regex + cosine + role features), **no LLM judge**.

**Config**
- `config.py` (197) — all thresholds and knobs (see §7).

---

## 5. The deterministic repair operators (`DET_OPS`)

Each is a **pure function of the graph** `(dag) -> (new_dag, changed)`.
`changed=False` means the axis is already satisfied (the caller must not loop).
`apply_move(dag, move_id, required=, partition_pairs=, req=)` routes to the right
one. Registered in `DET_OPS`:

| move_id | axis | what it does |
|---------|------|--------------|
| `scale.collapse_layer` | scale | remove a redundant layer (won't drop the sole carrier of a required role) |
| `scale.add_layer` | scale | insert a layer when too shallow |
| `flow.linearize` | flow | collapse branching into a chain (sequential) |
| `flow.add_join` | flow | add a merge node so branches converge (convergent) |
| `flow.split_to_parallel` | flow | fan out into independent branches (divergent) |
| `flow.add_merge` | flow | add a merge point |
| `flow.deepen_recursive` | flow | grow depth to approximate recursive flow (≥3) |
| `role.add_missing` | role | add a node for a required-but-absent functional role |
| `role.realign` | role | reassign a node's role to fix behavioral misalignment (topology-neutral) |
| `partition.merge_redundant` | partition | merge redundant siblings; **flow-preservation guarded** so it won't collapse a required fan-out |

There is also an **LLM move** (`partition.differentiate`) on the stochastic
path. In the descent it competes with the deterministic ops and is **rejected**
by PRESERVE if it doesn't measurably help — this is exactly the behavior to show
in the DESCEND demo.

---

## 6. Data model quick reference (for rendering)

```python
TaskFingerprint:
  information_flow  : divergent|convergent|sequential|recursive
  epistemic_stance  : retrieval|synthesis|generation|verification
  output_contract   : artifact|comparison|verification|ranking
  decomposability   : independent|coupled
  complexity        : low|medium|high          # → depth_cap(): 2/3/4
  domain_volatility : stable|evolving|contested # → requires_verifier()
  shape_string()    : the 4 shape axes as a string (the SPECIES key)

DAG:            task, task_embedding, topology, depth, nodes[], dag_id
Node:           node_id, intent, roles{structural,functional,epistemic},
                dependencies[], why{...}, expected_output_fingerprint,
                assigned_agent_id
Roles.functional: retriever|synthesizer|verifier|generic
Why:            task_type_recognized, topology_chosen, depth_chosen,
                alternatives_rejected, priors_used("warmstart" ⇒ REUSE),
                directive_received, directive_response

ExecutionTrace: dag_id, task, results[NodeResult], total_tokens, wallclock_s
NodeResult:     node_id, intent, output, output_embedding, cost_tokens,
                latency_s, role_function_match, fingerprint_match, gated, error

DeltaReport:    verdict("good"|"mixed"|"poor"), ef_tensor{4 axes},
                q_tensor{2 axes}, required_structure{...}, delta_e, ...
DeltaDirective: action("surface"|"refine"|"replan"), reason, replan_hint,
                refinement_targets[], primary_dim
```

---

## 7. Key thresholds (`config.py`)

```
ef_partition_threshold = 0.70    ef_flow_threshold  = 0.25
ef_role_threshold      = 0.35    ef_scale_threshold = 0.34
ef_mixed_margin        = 0.15    # one axis this far over → "mixed", not "poor"
llm_temperature        = 0.0     # deterministic by default (0.7 only in CONVERGE test)
```

---

## 8. UI-Builder Guide  ← read this to build the demo UI

### 8.0 Prerequisites & hard constraints (read first)

**You are building a data-visualization front-end, not running the model.**

1. **Do NOT try to run CEO-Delta.** It needs a self-hosted GPU + vLLM
   (Qwen-family) server and the `sentence-transformers` model. You almost
   certainly do not have that. You do not need it.
2. **Build against JSON.** Two files, identical schema:
   - `scripts/evidence_results.sample.json` — a labeled, representative sample.
     **Develop against this now.** (It has an `_README` marking it illustrative.)
   - `scripts/evidence_results.json` — the real measured output, produced by the
     presenter running `python3 -u scripts/evidence.py` on the model host. Swap
     it in when it exists. Same shape, so nothing in your UI needs to change.
3. **Ship a single self-contained artifact.** Prefer one HTML file with all
   CSS/JS inline and **no external network calls / CDNs** — it must render on the
   presenter's laptop, possibly offline, in front of a CEO.
   **Data loading (required, so re-runs are seamless):**
   - The primary loader MUST be a **drag-and-drop zone + file-picker** for a
     local `.json`. Do **not** `fetch()` a path — an offline `file://` page is
     blocked by the browser from reading local files that way.
   - The presenter re-runs `python3 -u scripts/evidence.py` (it always
     overwrites the same file, `scripts/evidence_results.json`), then drops that
     file onto the page. New data renders instantly — **no rebuild, no edit.**
   - Embed `evidence_results.sample.json` inline as the default view so the page
     shows something on first open, but let a dropped file replace it live.
   - Show the loaded file's `generated_at` (as a readable timestamp) and
     `elapsed_s` somewhere visible, so the presenter can confirm they're viewing
     a fresh run and not the sample. If the loaded JSON has an `_README` key,
     show a small "SAMPLE DATA" badge.
4. **Do not invent fields.** Render only what's in the JSON (schema in §8.2).
   If a field is missing, degrade gracefully — never crash the page.
5. **Theme:** clean, confident, executive. This is shown to a CEO. Legible type,
   generous spacing, a clear PASS/CHECK state per claim, one strong visual per
   claim (not a wall of numbers).

**Priority order** (build in this order; the first two are the emotional beats):
`2 GENERALIZE` (the climax) → `3 DESCEND` (the proof) → `1 CAPTURE` →
`4 REUSE` → `5 CONVERGE`.

### 8.1 Where the data comes from

Run `python3 -u scripts/evidence.py`. It:
1. Prints a live, human-readable trace of all 5 claims.
2. Writes **`scripts/evidence_results.json`** — the machine-readable payload.
   **Build the UI against this JSON.** You do not need to run the model to
   render; re-run `evidence.py` to refresh the data.

`RunResult` (returned by `Orchestrator.run()`) is the richest live object if you
want a real-time UI instead of the static JSON. Fields worth rendering:
`answer`, `report.verdict`, `report.ef_tensor`, `report.q_tensor`,
`report.required_structure`, `iterations`, `ef_trace`,
`ef_incumbent_trace` (**monotone curve**), `ef_move_log`, `dag.nodes`.

### 8.2 `evidence_results.json` schema

```jsonc
{
  "generated_at": <epoch>,
  "elapsed_s": <float>,
  "thresholds": { "partition":0.70, "flow":0.25, "role":0.35, "scale":0.34 },
  "claims": [
    {
      "id": "1_capture", "name": "CAPTURE", "passed": true,
      "headline": "3 distinct species across 3 task-types",   // verbose, for detail views
      "summary": "3 distinct species / 3 tasks",               // short label, for the summary strip
      "payload": { "tasks": [ {"task": "...", "species": "information_flow:..."} ] }
    },
    {
      "id": "2_generalize", "name": "GENERALIZE", "passed": true,
      "headline": "cosine=0.27 (unrelated) yet same class=true",
      "payload": { "task_a","task_b","text_cosine","species_a","species_b","same_class" }
    },
    {
      "id": "3_descend", "name": "DESCEND", "passed": true,
      "headline": "monotone=true, 1 deterministic repair, 1 bad move rejected, ...",
      "payload": {
        "task", "verdict", "iterations",
        "incumbent_trace": [ {partition,flow,role,scale}, ... ],  // the monotone curve
        "worst_excess_curve": [ 0.28, 0.04, 0.04 ],               // headline line chart
        "raw_trace": [...],                                        // trials (diagnostic)
        "move_log": [ {axis,move,before,after,improved}, ... ],
        "monotone", "improved",
        "deterministic_repairs": [ ...moves kept... ],
        "llm_rejected": [ ...moves rejected... ]
      }
    },
    {
      "id": "4_reuse", "name": "REUSE", "passed": true,
      "headline": "run1 cold, run2 warm=true iters=1",
      "payload": { "task", "run1":{verdict,iters,warm}, "run2":{verdict,iters,warm} }
    },
    {
      "id": "5_converge", "name": "CONVERGE", "passed": true,
      "headline": "verdicts=[good,good,mixed] all in satisfying region=true",
      "payload": { "task", "reps": [ {rep,verdict,iters,ef} ] }
    }
  ]
}
```

### 8.3 Suggested visualization per claim

- **CAPTURE** — three task cards, each with its `species` string rendered as
  four labeled chips (flow / stance / contract / decomposability). Emphasize
  that these came from raw sentences with no keywords in common.
- **GENERALIZE** — two very different sentences side by side, a big
  `text cosine = 0.27` "these are unrelated" badge, then an arrow collapsing
  both into **one identical species** card. This is the showstopper — make it
  the visual climax.
- **DESCEND** — a **line chart of `worst_excess_curve`** (y-axis: worst-axis
  excess; a dashed line at y=0 = "satisfying region"). Overlay the `move_log`
  as events on the timeline: green check for a kept deterministic repair, red X
  for a rejected LLM move. Caption: "the delivered-plan quality only ever
  improves — a guarantee, not luck." Optionally animate the DAG mutating.
- **REUSE** — two run rows; run 2 shows a **"warm-started ⚡"** badge and
  `iterations: 1`. Caption: "second time it saw this shape, it skipped planning."
- **CONVERGE** — three seeds → three verdict pills all landing inside a shaded
  "satisfying region" band. Caption: "different starts, same region — stable."

### 8.4 Reliability notes for the presenter

- Claims **1, 2, 5** are deterministic/near-deterministic — safe.
- Claim **3 (DESCEND)** asserts only architecturally-guaranteed facts
  (monotone incumbent + a deterministic op fired + a worse move rejected), so
  it does not depend on lucky convergence. The *magnitude* of improvement can
  vary run to run; the monotonicity cannot.
- Claim **4 (REUSE)** uses a stable-domain task specifically so the cached plan
  is reproducible and the warm-start marker sticks.

### 8.5 Honest limitations (say these plainly if asked)

- The `partition` axis is measured from execution outputs, so on tasks with
  similar-format-but-distinct-content parallel branches (e.g. flight vs hotel
  retrievers) it can read a false redundancy. PRESERVE contains the damage (the
  bad merge is rejected), but the axis itself is a known imperfection — the
  principled fix is to measure partition from plan-time intent embeddings.
- Grounding is a seam; the retrieval backend is swappable (built for a NANDA
  agent-directory backend, demoed with a web-search backend).
- This is a research prototype: single machine, self-hosted model, no
  multi-tenant hardening.

---

## 9. How to run

```bash
# full evidence battery (writes scripts/evidence_results.json)
python3 -u scripts/evidence.py

# one task, programmatically
python3 -c "from ceo_delta import Orchestrator, Config; \
r = Orchestrator(Config(), workdir='.demo').run('Explain how a C compiler works'); \
print(r.report.verdict, r.report.ef_tensor)"
```
