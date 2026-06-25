# Run 001 — Benchmark Analysis

**Date:** 2026-06-25  
**Model:** Qwen3-35B-A3B (MoE) via vLLM at 10.8.0.23:8001  
**Setup:** 3 tasks × 5 runs, fresh Orchestrator per benchmark (handbook starts clean)  
**Total LLM calls:** 108 across 15 runs

## Tasks

| Label | Task |
|---|---|
| A_research | What are the key factors that make a distributed database system resilient to network partitions? |
| B_planning | Design a step-by-step migration plan to move a monolithic e-commerce application to microservices |
| C_verification | Compare and verify the trade-offs between batch processing and stream processing for real-time analytics |

## Headline Metrics

| Task | Runs | Topology | fp_mean range | Verdict | Replanned |
|---|---|---|---|---|---|
| A_research | 5 | fan-out/2 (all) | 0.190–0.258 | all good | 0 |
| B_planning | 5 | fan-out/2 (all) | 0.180–0.304 | all good | 0 |
| C_verification | 5 | fan-out/2 (all) | 0.233–0.324 | all good | 0 |

fp_mean shows no consistent upward trend across runs for any task — movements are noise (~±0.05), not learning.

## Failure Modes

### FM-1: CEO re-reasons from scratch despite strong handbook priors

By run 4 of task A, the handbook shows `sim=0.83` with `"keep topo=fan-out depth=2 (clean run)"` as the prior. CEO reads this signal and then writes a fresh 150–200 word justification for fan-out anyway — identical in structure to run 1's justification. The prior is appended to the prompt but there is no mechanism that *changes* the planning behaviour when prior confidence is high. The handbook functions as confirmation rather than steering.

**Root cause:** `CEO._format_priors()` formats prior entries as prose context but the prompt has no instruction that correlates prior confidence to planning constraint strength. A sim=0.83 conf=3 prior looks the same to the LLM as a sim=0.03 seed entry.

**Fix direction:** Add a prompt instruction: if any prior exceeds `sim > 0.7` and `conf >= 3`, treat its topology/depth as the default and require an explicit reason to deviate. Also consider a `force_topology` fast-path that skips LLM planning entirely when confidence is above a threshold.

---

### FM-2: Research drift is real but the replan trigger is permanently broken

`structured_brief.drift` = 0.79 on run 1 of task A — Research substantially rewrote the task (added "CAP theorem, PACELC, quorum systems, vector clocks" as explicit scope). This is genuine value.

However `secondary_brief.drift` = 0.0 on every single run because `embed()` in `embeddings.py` is a random-vector stub. The cosine between any two embeddings is near-zero by construction, so `1 - cosine` is always ~1.0, which means the replan threshold (`cfg.replan_threshold = 0.70`) is never crossed in the direction that triggers replan — the secondary investigate pass finds `sim ≈ 0` (low similarity), which means `drift ≈ 1.0 > 0.70`, which *should* trigger replan but the logic checks `sim < threshold` not `drift > threshold`.

**Root cause:** Two separate bugs:
1. `embeddings.py` returns random unit vectors — all semantic distances are meaningless noise.
2. `Research.investigate()` computes `triggers = sim < self.cfg.replan_threshold` where `sim` is cosine similarity. With random embeddings sim ≈ 0, so `0 < 0.70` is always True — but looking at the data, `triggers_replan` is always False. Re-reading the code: `sim = cosine(base_emb, embed(refined))` — both `base_emb` and the new embed are random, so their cosine is random and sometimes above 0.70 by chance.

**Fix direction:** Replace `embeddings.py` stub with a real sentence embedding (even a small model like `all-MiniLM-L6-v2` via sentence-transformers). Until then, all semantic signals (fp_mean, drift, fingerprint_match, echo detection) are noise.

---

### FM-3: Synthesizer node has the lowest fingerprint match

Across task A runs, the synthesis node (n4, the final answer) consistently scores lower fp than the retrieval nodes:

```
Run 1: n1=0.232  n2=0.292  n3=0.301  n4=0.139
```

The node doing final synthesis — whose output becomes the answer — diverges most from its expected fingerprint. This is partly expected (synthesis should abstract away from retrieval content) but the scoring treats any divergence as underperformance. Delta penalises the synthesizer and the verdict could flip to "mixed" if fp_mean weighting were stronger.

**Root cause:** `expected_output_fingerprint` for a synthesizer node is `embed(intent)` — the embedding of the node's own intent string. A synthesizer's intent is something like "synthesize findings into a cohesive analysis", which is semantically distant from the actual synthesis output. The fingerprint check was designed for retrieval nodes where intent ≈ output topic; it breaks for synthesis.

**Fix direction:** Either (a) don't fingerprint-check synthesis nodes (role=synthesizer), or (b) set `expected_output_fingerprint` to the embedding of the *parent task* rather than the node intent.

---

### FM-4: `agent_generic` monopolises node assignments, defeating role specialisation

After 15 runs, agent confidence:

```
agent_retriever:   conf=41  trust=0.682
agent_synthesizer: conf=2   trust=0.665
agent_verifier:    conf=0   trust=0.500
agent_generic:     conf=20  trust=0.498
```

`agent_generic` supports all roles. Once it accumulates runs it enters the "confident pool" for every functional role. Even though `agent_retriever` has higher trust (0.682 vs 0.498), `agent_generic` wins for synthesizer and verifier nodes because it's the only card that supports those roles above the `agent_card_min_confidence=3` threshold.

The 41 retriever confidence on `agent_retriever` is suspicious given only 15 runs × ~3 retriever nodes = 45 max — consistent with it being selected for retrieval nodes correctly. But synthesizer nodes (which appear once per run as the final node) only ever assigned to `agent_generic`.

**Root cause:** `AgentRegistry._seed_default_agents()` only seeds one synthesizer card (`agent_synthesizer`) and it starts at `confidence=0`. It takes 3 runs before it clears `agent_card_min_confidence`. In those first 3 runs, `agent_generic` (also cold-start) wins by default. After that `agent_generic` has accumulated more runs than `agent_synthesizer` and out-ranks it on task-class score.

**Fix direction:** Seed specialized agents with `confidence=cfg.agent_card_min_confidence` so they're immediately in the confident pool, or lower `agent_card_min_confidence` to 1.

---

### FM-5: Topology space is never explored

Zero replans, zero exploratory runs, zero `hierarchical` or `join` or `linear` topology selections across 15 runs. The handbook converges to fan-out/2 after run 2 and never deviates. For task B (migration planning), a `hierarchical` topology would be a natural fit — plan phases at depth 1, implement at depth 2, verify at depth 3. This is present as a seed entry but never gets above `conf=1` because it's never selected.

The reflection mechanism (designed to trigger at run 5) fired for task A run 5 (run_count=5 hits `reflection_interval=5`) but with random embeddings there are no genuinely contested entries to resolve.

**Root cause:** Combines FM-1 (handbook doesn't steer away from fan-out) and FM-2 (drift signal is broken, so replan that might produce different topologies never fires). The system has positive feedback locked into fan-out from the seed prior.

**Fix direction:** Add temperature to handbook resolution — when a topology has won N consecutive runs, introduce a small probability of forced exploration of the next-ranked topology. This breaks the positive feedback loop without abandoning learned priors.

## What Is Working

- **Research.clarify produces genuine value.** The structured_intent rewrites add concrete technical scope (protocols, theorems, patterns) that CEO uses as retrieval targets. Node intents in runs 2–5 are more specific than run 1.
- **Kernel execution is clean.** 0% error rate, 100% plan-exec alignment across all 15 runs. Every planned node executed.
- **Node outputs are substantive.** Retrieval nodes write 500–2000 token structured analyses with sections and subsections — the LLM is actually working.
- **The pipeline topology (Research → CEO → Kernel → Delta → Handbook) is correct.** The feedback loop wiring is sound; the problems are in the signal quality feeding it.

## Priority Fixes

1. **Replace `embeddings.py` stub** — this unblocks FM-2, FM-3, FM-5, and makes fp_mean a real signal.
2. **Add prior-confidence instruction to CEO prompt** (FM-1) — cheap change, high leverage.
3. **Fix synthesizer fingerprint target** (FM-3) — use parent task embedding, not node intent embedding.
4. **Seed specialized agents at `min_confidence`** (FM-4) — one-line fix.
