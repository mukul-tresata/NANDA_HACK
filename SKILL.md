# Conductor-Delta — Self-Correcting Planning Agent

## What it does

Conductor-Delta takes a natural-language task or question and returns a composed
answer. Under the hood it does more than prompt an LLM: it classifies the
task's structure, plans a DAG of specialized sub-agents (retriever /
synthesizer / verifier), and runs a deterministic self-correction loop over the
plan — one that can only hold the plan still or improve it, never make it
structurally worse — before composing the final answer. You get back the answer
plus the task's structural signature and a verdict on how cleanly the plan
converged, so you can judge how much to trust the result.

- **Good for:** "explain how X works", "compare A / B / C on these axes", "given
  these options, recommend one and justify it", multi-step research syntheses.
- **Not for:** real-time data lookups (this deployment does not fetch live web
  data — see *Limitations*), or trivial one-liners where planning is overkill.

## Base URL

```
BASE_URL = https://cac-production-52ab.up.railway.app
```

Hosted on Railway — a permanent public address, always on, no tunnel.

---

## How to call it — READ THIS FIRST: the service is asynchronous

A single task runs a multi-agent pipeline that makes several LLM calls, so a
run takes **roughly 30 seconds to 5 minutes**. To avoid any connection timing
out while it works, the service does **not** answer in one call. Instead:

> **You submit the task and get a `job_id` immediately, then poll a status URL
> until the answer is ready. This is two steps. The first response is a ticket,
> NOT the answer.**

### Step 1 — submit the task

`POST {BASE_URL}/plan` with a JSON body `{"task": "<your task>"}`.

It returns in well under a second with a job id:

```json
{ "job_id": "6608f8818d4247d4", "status": "running", "poll_url": "/plan/6608f8818d4247d4" }
```

### Step 2 — poll until the job is done

`GET {BASE_URL}/plan/{job_id}` (i.e. the base URL followed by `poll_url`).
Call it repeatedly, about **once every 5 seconds**, and read the `status` field:

- `status` == `"running"` → not ready yet. **Wait and poll again.**
- `status` == `"done"` → finished. The answer is in `result.answer`, with all
  telemetry in the rest of `result` (see the table below).
- `status` == `"error"` → the run failed. `detail` explains why.

> **CRITICAL FOR AUTOMATED AGENTS — do not give up early.** A run can legitimately
> take up to ~5 minutes, so you will see `"running"` many times before `"done"`.
> That is normal and expected, not a failure. Keep polling. A safe policy is:
> **poll every 5 seconds for up to 6 minutes (about 72 attempts) before treating
> the job as failed.** Do not conclude the service is broken just because the
> first few polls return `"running"`.

### Copy-paste example — bash

```bash
BASE_URL="https://cac-production-52ab.up.railway.app"

# 1. Submit the task, capture the job_id.
JOB=$(curl -s -X POST "$BASE_URL/plan" \
  -H "Content-Type: application/json" \
  -d '{"task": "Compare Python, Rust, and Go on performance, safety, and concurrency."}' \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['job_id'])")

# 2. Poll every 5s (up to ~6 min) until done, then print the answer.
for i in $(seq 1 72); do
  RESP=$(curl -s "$BASE_URL/plan/$JOB")
  STATUS=$(echo "$RESP" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")
  if [ "$STATUS" = "done" ]; then
    echo "$RESP" | python3 -c "import sys, json; print(json.load(sys.stdin)['result']['answer'])"
    break
  fi
  if [ "$STATUS" = "error" ]; then echo "FAILED: $RESP"; break; fi
  sleep 5
done
```

### Copy-paste example — Python

```python
import time, requests

BASE_URL = "https://cac-production-52ab.up.railway.app"

# 1. Submit the task.
job = requests.post(
    f"{BASE_URL}/plan",
    json={"task": "Compare Python, Rust, and Go on performance, safety, and concurrency."},
).json()
job_id = job["job_id"]

# 2. Poll until done (every 5s, up to ~6 minutes). Keep going while "running".
for _ in range(72):
    r = requests.get(f"{BASE_URL}/plan/{job_id}").json()
    if r["status"] == "done":
        print(r["result"]["answer"])
        break
    if r["status"] == "error":
        raise RuntimeError(r["detail"])
    time.sleep(5)
else:
    raise TimeoutError("job did not finish within 6 minutes")
```

---

## The result object

When `status` is `"done"`, everything is under the `result` field:

| field         | type   | description                                                              |
|---------------|--------|--------------------------------------------------------------------------|
| `answer`      | string | **The composed deliverable — the thing to use.**                         |
| `verdict`     | string | `good` \| `mixed` \| `poor` — structural verdict on the delivered plan.   |
| `iterations`  | int    | How many self-correction iterations ran.                                 |
| `species`     | string | The task's structural signature (its class).                             |
| `signature`   | object | The classified axes: `information_flow`, `epistemic_stance`, `output_contract`, `decomposability`, `complexity`, `domain_volatility`. |
| `ef_tensor`   | object | Measured structural error per axis `[partition, flow, role, scale]` — lower is better. |
| `q_tensor`    | object | Content-quality tensor `[groundedness, relevance]`.                       |
| `clarification` | string \| null | Set (with an empty `answer`) only if the task was too vague to plan and `auto_clarify` was `false`. |
| `elapsed_s`   | number | Wall-clock seconds the run took.                                         |
| `mode`        | string | Grounding backend in effect (`none` here).                               |

### Example — a completed poll (`GET {BASE_URL}/plan/{job_id}`)

Captured from a real run of the task in the examples above:

```json
{
  "job_id": "6608f8818d4247d4",
  "status": "done",
  "result": {
    "task": "Compare Python, Rust, and Go on performance, safety, and concurrency.",
    "answer": "Python, Rust, and Go represent three distinct philosophies in systems and application programming. While they share the goal of enabling developers to build software, they diverge ... (full multi-paragraph comparison)",
    "verdict": "good",
    "iterations": 1,
    "species": "information_flow:convergent epistemic_stance:synthesis output_contract:comparison decomposability:independent",
    "signature": {
      "information_flow": "convergent",
      "epistemic_stance": "synthesis",
      "output_contract": "comparison",
      "decomposability": "independent",
      "complexity": "medium",
      "domain_volatility": "evolving"
    },
    "ef_tensor": {"partition": 0.5683, "flow": 0.0, "role": 0.2645, "scale": 0.0},
    "q_tensor": {"groundedness": 0.0736, "relevance": 0.26},
    "clarification": null,
    "elapsed_s": 276.76,
    "mode": "none"
  },
  "detail": null
}
```

### Optional request field

`POST /plan` also accepts `"auto_clarify"` (boolean, default `true`). Leave it
out for normal use. If you set it to `false`, a task too vague to classify comes
back with a `clarification` message and an empty `answer` instead of a plan.

---

## Simpler one-shot alternative — `POST /plan/sync`

If your client can hold a connection open for minutes and controls its own
timeout, `POST {BASE_URL}/plan/sync` with the same `{"task": "..."}` body runs
the pipeline and returns the **result object directly** in one blocking call —
no `job_id`, no polling. Only use this if you can tolerate a request that stays
open for the full run; otherwise use the async `POST /plan` + poll flow above.

```bash
curl -s -X POST "$BASE_URL/plan/sync" \
  -H "Content-Type: application/json" \
  -d '{"task": "Explain how a modern C compiler turns C source into an executable."}'
```

## Other endpoints

- `GET /health` — liveness. Returns `{"status": "ok", ...}`. Cheap; safe to poll before submitting.
- `GET /skill.md` — this contract, served live as plain text.
- `GET /docs` — interactive OpenAPI documentation.

## Notes & limitations

- **Asynchronous by design.** Use `POST /plan` then poll `GET /plan/{job_id}`
  (see above). Runs take ~30s–5min; keep polling until `status` is `done`.
- **One task at a time.** Jobs are executed serially server-side, so under
  concurrent load a job may sit in `running` a little longer before it starts.
- **No live data / grounding is off** (`mode: "none"`): the agent reasons from
  the model's own knowledge and does not fetch current web data. Do not rely on
  it for prices, news, or today's facts.
- **No authentication.** Send only non-sensitive tasks.
- A `verdict` of `mixed` or `poor` means the plan did not fully converge to the
  ideal structure; the answer is still returned, but treat it with more caution.
