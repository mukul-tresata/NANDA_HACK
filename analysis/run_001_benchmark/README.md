# run_001_benchmark

First instrumented benchmark run. 3 tasks × 5 runs, Qwen3-35B-A3B.

| File | Contents |
|---|---|
| `findings.md` | Full failure mode analysis with root causes and fix directions |
| `runs.jsonl` | Raw run log — one JSON record per run, includes all LLM call prompts/responses |
| `benchmark_results.json` | Per-run metrics table (fp_mean, verdict, topology, latency, etc.) |

Load `runs.jsonl` into the dashboard (`run_benchmark.py` output → drag into CEO·DELTA Observatory).
