# Conductor-Delta

A self-improving multi-agent system that **plans a full computation DAG before
any execution fires**, executes it with an intent-gated kernel, then **audits
the run against a deterministic error tensor and repairs it via coordinate
descent** — no LLM in the measurement or decision path, only in authoring
content and (rarely) an earned structural fix.

```
User → Research (parse) → Conductor (plan) → Kernel (execute) → Delta (measure + repair) → Handbook (remember)
```

## What's in this repo

| Piece | Path | What it is |
|-------|------|------------|
| **Engine** | `ceo_delta/` | The planner/executor/measure-repair loop (pure-Python; only `sentence-transformers` for embeddings). |
| **Hosted service** | `service/` | A thin FastAPI wrapper exposing the engine over HTTP for agents. |
| **Agent contract** | `SKILL.md` | How another agent calls the service — the NANDA-facing entry point. |
| **Demo** | `demo/evidence_console.html` | A self-contained page that renders the measured evidence for each claim. |
| **CLI** | `cli.py` | Local command-line entry point. |
| **Tests** | `tests/` | Offline suite (stubbed LLM, no server needed). |
| **Reproducible evidence** | `scripts/evidence.py` | (Re)generates the numbers the demo renders. |
| **Docs** | `docs/` | `PROJECT_OVERVIEW.md` (deep dive) + `architecture.svg`. |

Quick links: **[SKILL.md](SKILL.md)** (call the service) · **[service/README.md](service/README.md)** (run & deploy) · **[docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)** (deep dive) · **[docs/architecture.svg](docs/architecture.svg)** (diagram).

## The loop

| Stage | Agent | Job |
|-------|-------|-----|
| Parse | **Research** (`research.py`) | Stateless parser. Converts raw intent into a `TaskFingerprint` (structural species: flow/stance/contract/decomposability, embedded) plus `TaskSpecifics` (situational details, not embedded). Owns no handbook and does not learn. |
| Plan | **Conductor** (`ceo.py`) | Queries the single handbook keyed on fingerprint embedding, forces a reasoning chain, emits a DAG with separate WHY for topology and depth. Code-level gate forces a verifier node when `domain_volatility` requires one. |
| Execute | **Kernel** (`kernel.py`) | Dependency-aware parallel dispatch + intent-gated execution. Retriever-role nodes call the grounding seam (`grounding.py`) instead of hallucinating. Emits a per-node execution trace. |
| Measure + repair | **Delta** (`delta.py`, `ef.py`, `descent.py`) | Computes the EF error tensor, decides good/mixed/poor as a pure function of it, and runs deterministic coordinate descent to repair the worst axis. |
| Remember | **Handbook** (`handbook.py`) | Single vector DB. Topology & depth tracked as separate vote tallies; multi-way conflicts resolved explicitly. `ef_store.py` separately persists per-fingerprint move history and irreducibility counters. |

## Core idea — E is the dual of F

Both live in one shared basis: `[partition, flow, role, scale]`.

- **F** (`ef.compute_required`) — the structure the fingerprint *requires*,
  derived deterministically from `TaskFingerprint` (category → target). No LLM.
- **E** (`ef.compute_ef`) — how far the *realized* plan + execution deviates
  from F, per axis. Computed from graph topology and output embeddings
  (`partition`, `flow`, `scale`) and behavioral bands (`role`). No LLM in the
  measurement path.
- **Verdict** (`ef.verdict_from_ef`) is a pure function of E: all axes below
  threshold → `good`; one axis over by less than `ef_mixed_margin` → `mixed`;
  otherwise `poor`. There is no second, parallel error model to contradict it.

`drift` (output-vs-intent cosine geometry) was deliberately dropped from the
planning tensor — it measured execution-quality noise, not planning error. It
still exists as one axis of the older `error_tensor.py` fallback path
(`cfg.use_ef_tensor=False`).

### Deterministic coordinate descent (`descent.py`, `moves.py`)

Because E and F share a basis, repair is literal coordinate descent with an
**incumbent** (the best plan seen this run) and **uphill-step rejection**:

1. Lock onto the incumbent's single worst over-threshold axis.
2. Apply a **table-lookup** structural move for that axis — only from the moves
   whose structural direction can actually *reduce* that axis given the current
   graph signature and required F (`moves.applicable_moves`). Opposite-direction
   moves are never candidates, so the loop cannot thrash between contradictory
   repairs.
3. Re-measure. If the new plan improved the worst axis it becomes the new
   incumbent; if it regressed, it is **rejected** and descent continues from the
   incumbent — so one bad LLM replan can't poison the run. The incumbent's
   worst-axis error is monotone non-increasing by construction.
4. A move that drives its axis below threshold is written back as a **learned
   move** for that fingerprint class (prioritized next run) and resets that
   axis's irreducibility counter.
5. When an axis exhausts its *applicable* moves without converging, its
   irreducibility counter is bumped — persisted across runs.

**Earned LLM escalation** unlocks once an `(F, axis)` pair's irreducibility
counter reaches `ef_irreducible_escalate_threshold`. A successful escalation
folds back into the deterministic repertoire — the LLM's contribution becomes a
permanent table-lookup move. The only non-deterministic step in the whole loop
is the wording an escalation LLM authors; the decision to escalate is a
deterministic counter test (`escalation.py`, `ef_store.py`).

Every model call runs at `temperature = 0.0` (`config.llm_temperature`), so the
same task yields the same F and the same F yields the same descent, up to
backend nondeterminism.

## The four original limitations — explicitly resolved

1. **Cold start** (`bootstrap.py`, `config.cold_start_runs`): handbook is seeded
   with low-confidence synthetic entries; Conductor runs exploratory for the first `N`
   runs.
2. **Reflection trigger/budget** (`reflection.py`): fires between interactions
   on `run_count % reflection_interval == 0` **or** contested entries ≥
   `reflection_contested_trigger`; exits on resolution, `reflection_max_explorations`,
   or spent `reflection_token_budget`.
3. **Replan threshold** (`research.py`, `config.replan_threshold`): Research's
   secondary pass replans iff post-plan intent drifts too far.
4. **Multi-way conflict** (`handbook.py`): vote tallies per entry; a winner is
   declared only when it beats the runner-up by `conflict_dominance_margin`
   **and** normalized entropy is below `conflict_entropy_threshold` — a 5-3-2
   split is flagged contested, not silently won by plurality.

## LLM backend

The live backend is a **self-hosted vLLM OpenAI-compatible server** running a
Qwen-family model (`llm.py`). Configure via env:

| env var | default | meaning |
|---------|---------|---------|
| `CEO_LLM_URL` | `http://localhost:8000/v1` | vLLM/OpenAI-compatible base URL |
| `CEO_LLM_MODEL` | `Qwen/Qwen3.6-35B-A3B` | model name served by that endpoint |

Calls request generous `max_tokens`; JSON is extracted defensively (fenced,
prose-wrapped, or raw). If the server is unreachable and `cfg.llm_allow_stub=True`,
a deterministic offline stub keeps the pipeline runnable — this is what the test
suite and cold-start demos use, so **you can run the CLI and tests with no
server at all** (the embedding model still loads). Embeddings use
`sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim unit vectors, CPU-friendly,
~80 MB; `embeddings.py`).

The Anthropic API is preserved as a **dormant, commented-out alternative** in
`llm.py`; it is only reached if you enable the `web_search` grounding backend
(a Claude-only server-side tool) and set `ANTHROPIC_API_KEY`.

## Grounding

Retriever-role nodes call `retrieve()` (`grounding.py`) instead of inventing
facts. The seam is backend-agnostic and **off by default**:

| `CEO_DELTA_GROUNDING_BACKEND` | effect |
|-------------------------------|--------|
| `none` (default) | no outward fetch; reasons from model knowledge |
| `web_search` | Claude server-side web search (needs `ANTHROPIC_API_KEY`) |
| `nanda` | reserved seam for a NANDA-discovered retrieval agent |

## Quickstart

```bash
pip install -r requirements.txt

# CLI — works fully offline against the deterministic stub (no server needed)
python3 cli.py run "Compare Python, Rust, and Go on performance, safety, concurrency"
python3 cli.py demo          # multi-run learning demo
python3 cli.py reflect       # force a reflection session
python3 cli.py handbook      # dump handbook state

python3 -m pytest tests/ -q  # offline test suite (stubbed LLM)
```

To run against a real model, point `CEO_LLM_URL` at your vLLM server.

### Run it as a service (for agents / NANDA)

```bash
pip install -r service/requirements.txt
python -m service.serve                 # serves on 0.0.0.0:6001
curl -s localhost:6001/health
```

Expose it publicly and register it per **[service/README.md](service/README.md)**;
the agent-facing contract is **[SKILL.md](SKILL.md)**.

### See the evidence

```bash
python3 -u scripts/evidence.py   # (re)writes scripts/evidence_results.json
```

Open `demo/evidence_console.html` in a browser and drop that JSON onto the page —
or just open it to see the embedded sample run.

## Empirical validation

`scripts/research/perturbation_harness.py` injects known structural defects
(overlapping sibling intents, mismatched role labels, wrong flow shape, wrong
depth) at the assignment level — never at the metric's direct input — and measures
whether the EF tensor recovers them. This is a non-circular check for the
partition/role axes; flow/scale are documented as consistency checks rather than
validity tests, since injecting them manipulates exactly the graph property the
metric reads.

```bash
python -m scripts.research.perturbation_harness --trials 60   # writes perturbation_results.jsonl
```

`scripts/research/calibrate_ef.py` and `scripts/research/role_retest.py` support
recalibrating the hand-seeded thresholds in `config.py` once enough real run
volume exists. (`scripts/research/` holds the exploratory harnesses; the top of
`scripts/` is just the reproducible `evidence.py`.)

## Environment variables

| var | default | purpose |
|-----|---------|---------|
| `CEO_LLM_URL` | `http://localhost:8000/v1` | vLLM/OpenAI-compatible base URL (core/CLI) |
| `CEO_DELTA_LLM_BASE_URL` | *(unset)* | same base URL, but the **service**'s override name |
| `CEO_LLM_MODEL` | `Qwen/Qwen3.6-35B-A3B` | served model name |
| `ANTHROPIC_API_KEY` | *(empty)* | only for `web_search` grounding |
| `CEO_DELTA_GROUNDING_BACKEND` | `none` | `none` / `web_search` / `nanda` |
| `CEO_DELTA_WORKDIR` | `.ceo_delta` | persistent handbook / EF state dir (service) |
| `HOST` / `PORT` | `0.0.0.0` / `6001` | service bind address |

State persists to `$CEO_DELTA_WORKDIR/handbook.json`, `ef_cases.json`, and
`escalations.jsonl` between runs — that persistence *is* the cross-interaction
learning. (These are generated at runtime and git-ignored.)

## Layout

```
ceo_delta/
  config.py         all thresholds + backend/grounding config (one home)
  embeddings.py     sentence-transformers (all-MiniLM-L6-v2) embeddings + cosine
  llm.py            vLLM client + offline stub (Anthropic path dormant)
  schemas.py        Node / DAG / ExecutionTrace / HandbookEntry / TaskFingerprint
  research.py       stateless intent parser -> (TaskFingerprint, TaskSpecifics)
  ceo.py            planner (reasoning chain, WHY, replan, verifier gate, compose)
  kernel.py         intent-gated parallel execution
  grounding.py      retrieval seam (none by default, web_search / nanda optional)
  ef.py             E/F tensor (partition/flow/role/scale) + verdict  [primary]
  error_tensor.py   older 5D tensor (drift/echo/cascade/role/resource) [fallback]
  role_features.py  behavioral profiling feeding the role axis
  moves.py          per-axis repair repertoire + applicability gating
  delta_rules.py    move-selection rules
  descent.py        coordinate-descent controller: incumbent + uphill rejection
  ef_store.py       F-keyed store: learned moves, escalation moves, irreducibility
  escalation.py     case store + LLM prompt/parsing for earned escalations
  warmstart.py      best-plan reuse: reconstructs a runnable DAG for same-species tasks
  handbook.py       single vector DB + multi-way conflict resolution
  bootstrap.py      cold-start seeding
  reflection.py     reflection-mode trigger / budget / exit
  quality.py        Q-factor (groundedness, relevance), mechanical, no LLM judge
  runlog.py         append-only JSONL per-run record
  delta.py          orchestrates measurement + repair, writes handbook votes
  orchestrator.py   the run() loop; RunResult
service/            FastAPI wrapper (app, serve) + its own README
cli.py              command-line entry point
tests/              offline suite (stubbed LLM)
scripts/
  evidence.py       reproduces the demo's measured evidence
  research/         exploratory harnesses (perturbation, calibration, ablation, plots)
docs/
  PROJECT_OVERVIEW.md   architecture deep dive
  architecture.svg      system diagram
demo/
  evidence_console.html self-contained evidence demo page
SKILL.md            agent-facing contract for the hosted service
LICENSE  ·  .env.example  ·  requirements.txt
```
