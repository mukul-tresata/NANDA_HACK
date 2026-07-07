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
BASE_URL = https://YOUR-PUBLIC-URL
```

> Replace `https://YOUR-PUBLIC-URL` with the live base URL (printed when the
> service is exposed — see `service/README.md`). All endpoints below are
> relative to it.

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
| `verdict`     | string | `good` \| `mixed` \| `bad` — structural verdict on the delivered plan.  |
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

Example response (truncated):

```json
{
  "task": "Explain how a modern C compiler turns C source into an executable.",
  "answer": "A modern C compiler proceeds in distinct stages ...",
  "verdict": "good",
  "iterations": 2,
  "species": "information_flow:sequential epistemic_stance:synthesis output_contract:artifact decomposability:coupled",
  "signature": {
    "information_flow": "sequential",
    "epistemic_stance": "synthesis",
    "output_contract": "artifact",
    "decomposability": "coupled",
    "complexity": "medium",
    "domain_volatility": "stable"
  },
  "ef_tensor": {"partition": 0.30, "flow": 0.0, "role": 0.20, "scale": 0.0},
  "q_tensor": {"groundedness": 0.0, "relevance": 0.0},
  "elapsed_s": 42.7,
  "mode": "none"
}
```

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
- A `verdict` of `mixed` or `bad` means the plan did not fully converge to the
  ideal structure; the answer is still returned, but treat it with more caution.
