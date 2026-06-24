# CEO-Delta Architecture

A self-improving multi-agent system that **plans a full computation DAG before
any execution fires**, executes it with an intent-gated kernel, then **audits
the run and writes the learnings back into a vector handbook** so the planner
gets better at each task class over time.

Learning happens at the *reasoning* level (topology / depth / WHY), not just at
the cost level.

```
User → CEO → Research → CEO → Kernel → Delta → Handbook → CEO
```

## The loop

| Stage | Agent | Job |
|-------|-------|-----|
| Plan | **CEO** (`ceo.py`) | Queries handbook, forces a 6-step reasoning chain, emits a DAG with a separate WHY for topology and depth. |
| Inform | **Research** (`research.py`) | Gathers what the plan needs, returns a brief. Triggers a CEO replan if the task understanding materially shifts. |
| Execute | **Kernel** (`kernel.py`) | KAIJU-style dependency-aware parallel dispatch + Intent-Gated Execution. Emits a per-node execution trace. |
| Audit | **Delta** (`delta.py`) | Computes structural / runtime / failure / semantic / satisfaction metrics, writes votes to both handbooks. |
| Remember | **Handbook** (`handbook.py`) | Vector DB; topology & depth tracked as separate vote tallies; multi-way conflicts resolved explicitly. |

## Communication modes (`orchestrator.py`)

- **Standard** — `orch.run(task)`. One pass, user gets the answer, Delta audits after.
- **Reflection** — CEO⇄Delta only, no user. Resolves contested handbook entries with exploratory DAGs under a hard token budget.
- **Meta/Feedback** — `orch.meta_feedback(task, satisfaction)`. User gives Delta an explicit satisfaction signal in `[0,1]`.

## The four limitations — explicitly resolved

1. **Cold start** (`bootstrap.py`, `config.cold_start_runs`): handbook is seeded
   with low-confidence synthetic entries distilled from the four inspiring
   papers, *and* CEO runs in exploratory mode (WHY flagged low-confidence) for
   the first `N` runs. The first demo run is not empty.
2. **Reflection trigger/budget** (`reflection.py`): fires **between** interactions
   when `run_count % reflection_interval == 0` **or** contested entries ≥
   `reflection_contested_trigger`. Exits when the entry resolves, after
   `reflection_max_explorations`, or when `reflection_token_budget` is spent.
3. **Replan threshold** (`research.py`, `config.replan_threshold = 0.70`): CEO
   replans iff `cosine(original_task_emb, post_brief_intent_emb) < 0.70`.
4. **Multi-way conflict** (`handbook.py`): not a boolean. Each entry keeps vote
   tallies over topology/depth options; a winner is declared only when it beats
   the runner-up by `conflict_dominance_margin` **and** the distribution's
   normalized entropy is below `conflict_entropy_threshold` — so a 5-3-2 split
   is correctly flagged contested, not silently "won" by the plurality.

## LLM backend

Talks to a vLLM OpenAI-compatible server (default `http://10.8.0.23:8001/v1`,
model `my-model`). The served model is a reasoning model, so calls request
generous `max_tokens` and JSON is extracted defensively. If the server is
unreachable, a deterministic stub keeps the pipeline runnable offline (used by
the tests). Embeddings are a dependency-free hashing bag-of-ngrams (the server
exposes no `/embeddings` endpoint).

Override via env: `CEO_LLM_URL`, `CEO_LLM_MODEL`.

## Usage

```bash
python3 cli.py run "Compare KAIJU and POLARIS approaches to agent planning"
python3 cli.py run "task" --satisfaction 0.9   # standard run + meta signal
python3 cli.py demo                            # 6-run learning demo
python3 cli.py reflect                         # force a reflection session
python3 cli.py handbook                         # dump handbook state

python3 -m pytest tests/ -q                    # offline test suite (stubbed LLM)
```

State persists to `.ceo_delta/{ceo,research}_handbook.json` between runs — that
persistence *is* the cross-interaction learning.

## Layout

```
ceo_delta/
  config.py        all thresholds (the 4 limitations have one explicit home)
  embeddings.py    deterministic hashing embeddings + cosine
  llm.py           vLLM client (reasoning-model aware) + offline stub
  schemas.py       Node / DAG / ExecutionTrace / HandbookEntry
  handbook.py      vector DB + multi-way conflict resolution
  bootstrap.py     cold-start seeding from the papers
  ceo.py           planner (6-step chain, WHY, replan)
  research.py      research agent + replan threshold
  kernel.py        intent-gated parallel execution
  delta.py         metrics + handbook writes
  reflection.py    reflection-mode trigger / budget / exit
  orchestrator.py  the loop + 3 communication modes
cli.py             command-line entry point
tests/test_core.py offline tests for the loop + 4 limitations
```
# NANDA_HACK
