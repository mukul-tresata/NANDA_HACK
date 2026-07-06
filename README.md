# CEO-Delta Architecture

A self-improving multi-agent system that **plans a full computation DAG before
any execution fires**, executes it with an intent-gated kernel, then **audits
the run against a deterministic error tensor and repairs it via coordinate
descent** — no LLM in the measurement or decision path, only in authoring
content and (rarely) an earned structural fix.

```
User → Research (parse) → CEO (plan) → Kernel (execute) → Delta (measure + repair) → Handbook (remember)
```

## The loop

| Stage | Agent | Job |
|-------|-------|-----|
| Parse | **Research** (`research.py`) | Stateless parser. Converts raw intent into a `TaskFingerprint` (structural species: flow/stance/contract/decomposability, embedded) plus `TaskSpecifics` (situational details, not embedded). Owns no handbook and does not learn. |
| Plan | **CEO** (`ceo.py`) | Queries the single handbook keyed on fingerprint embedding, forces a reasoning chain, emits a DAG with separate WHY for topology and depth. Code-level gate forces a verifier node when `domain_volatility` requires one. |
| Execute | **Kernel** (`kernel.py`) | Dependency-aware parallel dispatch + intent-gated execution. Retriever-role nodes call the grounding seam (`grounding.py`) instead of hallucinating. Emits a per-node execution trace. |
| Measure + repair | **Delta** (`delta.py`, `ef.py`, `descent.py`) | Computes the EF error tensor, decides good/mixed/poor as a pure function of it, and runs deterministic coordinate descent to repair the worst axis. |
| Remember | **Handbook** (`handbook.py`) | Single vector DB (no more CEO/Research split). Topology & depth tracked as separate vote tallies; multi-way conflicts resolved explicitly. `ef_store.py` separately persists per-fingerprint move history and irreducibility counters. |

## v3.0 — EF-driven planning adaptation

The core idea: the error tensor **E is the dual of the task fingerprint F**.
Both live in one shared basis: `[partition, flow, role, scale]`.

- **F** (`ef.compute_required`) — the structure the fingerprint *requires*,
  derived deterministically from `TaskFingerprint` (category → target). No LLM.
- **E** (`ef.compute_ef`) — how far the *realized* plan + execution deviates
  from F, per axis. Computed from graph topology and output embeddings
  (`partition`, `flow`, `scale`) and behavioral bands (`role`). No LLM in the
  measurement path.
- **Verdict** (`ef.verdict_from_ef`) is a pure function of E: all axes below
  threshold → `good`; one axis over by less than `ef_mixed_margin` → `mixed`;
  otherwise `poor`. There is no second, parallel error model to disagree with
  it — this is what v3.0 replaced (the old scalar Δe / tensor split that could
  produce a verdict and a directive that contradicted each other).

`drift` (output-vs-intent-string cosine geometry) was deliberately dropped
from the planning tensor — it measured execution-quality noise, not planning
error. It still exists as one axis of the older `error_tensor.py` (v2.1
fallback path, `cfg.use_ef_tensor=False`).

### Deterministic coordinate descent (`descent.py`, `moves.py`)

Because E and F share a basis, repair is literal coordinate descent with an
**incumbent** (the best plan seen this run) and **uphill-step rejection**:

1. Lock onto the incumbent's single worst over-threshold axis.
2. Apply a **table-lookup** structural move for that axis — chosen from the
   moves whose structural direction can actually *reduce* that axis given the
   current graph signature and required F (`moves.applicable_moves`). Opposite-
   direction moves (e.g. `scale.collapse_layer` when the plan is too shallow,
   or `flow.linearize` on a divergent task) are never candidates, so the loop
   cannot thrash between contradictory repairs.
3. Re-measure. If the new plan improved the worst axis it becomes the new
   incumbent; if it regressed, it is **rejected** and the descent continues
   from the incumbent — so one bad LLM replan can't poison the rest of the run.
   The incumbent's worst-axis error is monotone non-increasing by construction.
4. A move that drives its axis below threshold is written back as a **learned
   move** for that fingerprint class (prioritized next run) and **resets** that
   axis's irreducibility counter (the axis is currently solved).
5. When an axis exhausts its *applicable* moves without converging, its
   irreducibility counter is bumped — persisted across runs.

**Earned LLM escalation** unlocks once an `(F, axis)` pair's irreducibility
counter reaches `ef_irreducible_escalate_threshold` (default 2). Because the
counter resets on every success, a high count means "the current repertoire is
failing this axis *right now*" — so escalation re-opens whenever a
once-working move stops working, instead of being permanently disabled by a
single early success. A successful escalation folds back into the deterministic
repertoire — the LLM's contribution becomes a permanent table-lookup move
(b feeds a). The only non-deterministic step in the whole loop is the wording
the LLM authors inside an earned escalation; the decision to escalate is a
deterministic counter test (`escalation.py`, `ef_store.py`).

### Determinism (v3.2)

The system's identity is determinism, so every model call runs at
`temperature = 0.0` (`config.llm_temperature`). The fingerprint (from
Research), the plan (from CEO), and execution are therefore reproducible
run-to-run up to backend nondeterminism — the same task yields the same F,
the same F yields the same descent. The *only* nondeterminism the architecture
wants is the content an escalation LLM authors, and even that fires on a
deterministic counter.

### Retrieval grounding (v3.1, `grounding.py`)

Retriever-role nodes call `retrieve()` instead of generating plausible-sounding
facts from the model's own weights. Backend today is Claude's server-side web
search tool; the seam is backend-agnostic so it can later swap to a
NANDA-discovered retrieval agent without touching planner, kernel, or EF code.

## The four original limitations — explicitly resolved

1. **Cold start** (`bootstrap.py`, `config.cold_start_runs`): handbook is
   seeded with low-confidence synthetic entries, and CEO runs in exploratory
   mode for the first `N` runs.
2. **Reflection trigger/budget** (`reflection.py`): fires between
   interactions when `run_count % reflection_interval == 0` **or** contested
   entries ≥ `reflection_contested_trigger`. Exits when the entry resolves,
   after `reflection_max_explorations`, or when `reflection_token_budget` is
   spent.
3. **Replan threshold** (`research.py`, `config.replan_threshold = 0.70`):
   Research's secondary pass replans iff post-plan intent drifts too far from
   the structured intent.
4. **Multi-way conflict** (`handbook.py`): not a boolean. Each entry keeps
   vote tallies over topology/depth options; a winner is declared only when
   it beats the runner-up by `conflict_dominance_margin` **and** the
   distribution's normalized entropy is below `conflict_entropy_threshold` —
   so a 5-3-2 split is correctly flagged contested, not silently "won" by
   plurality.

## LLM backend

Talks to the real Anthropic API (`llm.py`), default model
`claude-haiku-4-5-20251001` via `ANTHROPIC_API_KEY`. Calls request generous
`max_tokens` and JSON is extracted defensively (fenced, prose-wrapped, or
raw). If the API call fails (no key, network, etc.) and
`cfg.llm_allow_stub=True`, a deterministic offline stub keeps the pipeline
runnable — used by the test suite. Embeddings are a dependency-free hashing
bag-of-ngrams (`embeddings.py`).

Override via env: `ANTHROPIC_API_KEY`, `CEO_LLM_MODEL`.

The project previously ran against a self-hosted vLLM/Qwen server; that
backend is preserved commented-out in `llm.py` in case of a switch back.

## Empirical validation

`scripts/perturbation_harness.py` injects known structural defects
(overlapping sibling intents, mismatched role labels, wrong flow shape, wrong
depth) at the assignment level — never at the metric's direct input — and
measures whether the EF tensor recovers them. This is a non-circular check
for the partition/role axes (the LLM could still differentiate overlapping
siblings on its own); flow/scale are documented as consistency checks rather
than validity tests, since injecting them manipulates exactly the graph
property the metric reads.

```bash
python scripts/perturbation_harness.py --trials 60   # writes perturbation_results.jsonl
```

`scripts/calibrate_ef.py` and `scripts/role_retest.py` support recalibrating
the hand-seeded thresholds in `config.py` once enough real run volume exists.

## Usage

```bash
python3 cli.py run "Compare KAIJU and POLARIS approaches to agent planning"
python3 cli.py run "task" --satisfaction 0.9   # standard run + meta signal
python3 cli.py demo                            # multi-run learning demo
python3 cli.py reflect                         # force a reflection session
python3 cli.py handbook                         # dump handbook state

python3 -m pytest tests/ -q                    # offline test suite (stubbed LLM)
```

State persists to `.ceo_delta/handbook.json`, `.ceo_delta/ef_cases.json`, and
`.ceo_delta/escalations.jsonl` between runs — that persistence *is* the
cross-interaction learning.

## Layout

```
ceo_delta/
  config.py         all thresholds (v2.1 tensor + v3.0 EF thresholds have one explicit home)
  embeddings.py     deterministic hashing embeddings + cosine
  llm.py            Anthropic API client + offline stub
  schemas.py        Node / DAG / ExecutionTrace / HandbookEntry / TaskFingerprint
  handbook.py       single vector DB + multi-way conflict resolution
  bootstrap.py      cold-start seeding
  ceo.py            planner (reasoning chain, WHY, replan, verifier gate)
  research.py       stateless intent parser -> (TaskFingerprint, TaskSpecifics)
  kernel.py         intent-gated parallel execution
  grounding.py      retrieval seam (web search today, NANDA-agent-ready)
  error_tensor.py   v2.1 scalar-replacing 5D tensor (drift/echo/cascade/role/resource) -- fallback path
  ef.py             v3.0 E/F tensor (partition/flow/role/scale) + verdict -- primary path
  role_features.py  behavioral profiling (citation density / compression / structure) feeding role axis
  moves.py          per-axis repair repertoire + applicability gating (direction-correct moves only)
  descent.py        coordinate-descent controller: incumbent + uphill rejection, lock -> apply -> re-measure
  ef_store.py       F-keyed trajectory store: learned moves, escalation moves, irreducibility counters
  escalation.py     case store + LLM prompt/parsing for earned escalations (v2.1 path)
  delta.py          orchestrates measurement + repair, writes handbook votes
  reflection.py     reflection-mode trigger / budget / exit
  orchestrator.py   the loop + 2 communication modes (standard, meta-feedback) + reflection
cli.py              command-line entry point
tests/test_core.py  offline tests for the loop + 4 limitations
scripts/            perturbation harness, EF calibration, role re-testing
```
