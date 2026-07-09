# CEO-Delta — Self-Correcting Planning Agent

## What it does

CEO-Delta takes a natural-language task or question and returns a composed
answer. Under the hood it does more than prompt an LLM: it **classifies the
task's structure**, plans a DAG of specialized sub-agents (retriever /
synthesizer / verifier), and runs a **deterministic coordinate descent** over
the plan that is guaranteed to either hold the plan still or improve it —
never make it structurally worse — before composing the final answer.

Give it any task that benefits from being decomposed and checked: explanations,
comparisons, rankings, multi-step syntheses. You get back the answer **plus** the
task's structural signature and a verdict on how cleanly the plan converged, so
you can decide how much to trust the result.

- **Good for:** "explain how X works", "compare A/B/C on these axes", "given
  these options, recommend one and justify it", multi-step research syntheses.
- **Not for:** real-time data lookups (grounding is off by default in this
  deployment — see *Limitations*), or trivial one-liners where planning is overkill.

## Web address

```
BASE_URL = https://cac-production-52ab.up.railway.app
```

Hosted on Railway — a permanent address, no tunnel, stays up independent of
any local machine.

## Endpoints

### `POST /plan` — plan and answer a task

**Request body** (JSON):

| field          | type    | required | description                                                        |
|----------------|---------|----------|--------------------------------------------------------------------|
| `task`         | string  | yes      | The task or question to plan and answer.                           |
| `auto_clarify` | boolean | no       | Default `true`. If `false`, a too-vague task returns a `clarification` instead of an answer. |

**Response** (JSON):

| field         | type   | description                                                            |
|---------------|--------|------------------------------------------------------------------------|
| `answer`      | string | The composed deliverable — the thing to use.                           |
| `verdict`     | string | `good` \| `mixed` \| `poor` — structural verdict on the delivered plan. |
| `iterations`  | int    | How many descent iterations ran.                                       |
| `species`     | string | The task's structural signature (its class).                           |
| `signature`   | object | The six classified axes (information_flow, epistemic_stance, …).        |
| `ef_tensor`   | object | Measured structural error per axis `[partition, flow, role, scale]`.    |
| `q_tensor`    | object | Content-quality tensor `[groundedness, relevance]` when measured.       |
| `elapsed_s`   | number | Wall-clock seconds for the run.                                        |
| `mode`        | string | Grounding backend in effect (`none` \| `web_search` \| `nanda`).       |

### `GET /health` — liveness

Returns `{"status": "ok", ...}`. Cheap; safe to poll before sending a task.

### `GET /skill.md` — this contract, served live

Returns this document as plain text, so you can fetch the contract directly
from the running service.

### `GET /docs` — interactive OpenAPI documentation

## How to call it

Minimal working call (one POST, no auth):

```bash
curl -s -X POST "$BASE_URL/plan" \
  -H "Content-Type: application/json" \
  -d '{"task": "Explain how a modern C compiler turns C source into an executable."}'
```

Example response — captured verbatim from this live deployment:

```json
{
  "task": "Explain how a modern C compiler turns C source into an executable.",
  "answer": "stub response (LLM server unreachable)",
  "verdict": "poor",
  "iterations": 1,
  "species": "information_flow:sequential epistemic_stance:synthesis output_contract:artifact decomposability:coupled",
  "signature": {
    "information_flow": "sequential",
    "epistemic_stance": "synthesis",
    "output_contract": "artifact",
    "decomposability": "coupled",
    "complexity": "medium",
    "domain_volatility": "stable"
  },
  "ef_tensor": {"partition": 0.0, "flow": 0.0, "role": 0.19, "scale": 0.1667},
  "q_tensor": {"groundedness": 0.0, "relevance": 1.0},
  "elapsed_s": 4.55,
  "mode": "none"
}
```

> **Current deployment status:** this instance's content-composition backend
> is being finalized, so `answer` currently returns a deterministic stub
> placeholder rather than a generated answer. Everything else in the
> response — task classification (`species`/`signature`), the structural
> descent (`iterations`, `ef_tensor`), and the content-quality tensor
> (`q_tensor`) — is fully live and reflects a real run of the pipeline
> against this exact task.

Python:

```python
import requests
r = requests.post(f"{BASE_URL}/plan", json={"task": "Compare Python, Rust, and Go on performance, safety, and concurrency."})
print(r.json()["answer"])
```

## Notes & limitations

- **One task at a time.** Calls are serialized server-side; expect tens of
  seconds per task (it runs a real multi-agent plan, not a single completion).
- **Grounding is off by default** (`mode: "none"`): the agent reasons from the
  model's own knowledge and does not fetch live web data in this deployment. Do
  not rely on it for current facts (prices, news, today's data).
- **No authentication.** Send only non-sensitive tasks.
- A `verdict` of `mixed` or `poor` means the plan did not fully converge to the
  ideal structure; the answer is still returned, but treat it with more caution.
- **`answer` is currently a stub placeholder** (see status note above) — the
  content-composition backend is being finalized. All other fields are live.
