"""Conductor-Delta as a NANDA-discoverable HTTP service.

One agent-facing endpoint -- POST /plan -- wraps `Orchestrator.run(task)` and
returns the composed answer plus the structural signature/verdict the model
produced getting there. A cold agent that has only read SKILL.md can make a
working call with a single POST; that is exactly the hackathon's main-track
judging bar ("agents succeed using only your SKILL.md", "little effort to a
working call").

Design notes
------------
* The heavy object is the `Orchestrator` (loads the handbook, opens the LLM
  client, seeds the agent registry). We build ONE at startup and reuse it.
* `Orchestrator.run` mutates per-run state (run_count, handbook, EF stores) and
  is not written to be re-entrant, so we serialize calls behind a lock and run
  them in a worker thread to avoid blocking the event loop. For a demo/eval
  service this is the honest, safe choice -- correctness over throughput.
* NANDA vs. local is a runtime toggle, not a code fork (see _build_config): the
  same service runs fully offline for testing (grounding backend "none") and
  flips to outward grounding/discovery via env when a NANDA directory is live.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ceo_delta.config import Config, DEFAULT
from ceo_delta.orchestrator import Orchestrator, RunResult


# --------------------------------------------------------------------------- #
# Configuration / mode toggle
# --------------------------------------------------------------------------- #

def _build_config() -> Config:
    """Construct the runtime Config, applying the local<->NANDA toggle.

    Everything is driven by env so the SAME image runs in both modes:

      CEO_DELTA_GROUNDING_BACKEND = none | web_search | nanda   (default: none)
        "none"       -- pure local test mode. No outward calls beyond the LLM.
        "web_search" -- Anthropic web-search grounding (needs ANTHROPIC_API_KEY).
        "nanda"      -- grounding/retrieval served by a NANDA-discovered agent
                        (enable once a NANDA directory endpoint is live).
      CEO_DELTA_WORKDIR           -- persistent state dir (default: .ceo_delta)
      CEO_DELTA_LLM_BASE_URL      -- override the vLLM/OpenAI-compatible base URL
      CEO_DELTA_MAX_ITER          -- cap descent iterations (default: Config's own,
                                     currently 5). Descent is monotone -- capping it
                                     lower only trades "fully converged" for "still
                                     correct, just less optimized" -- so this is safe
                                     to lower for a hosted deployment sitting behind
                                     a request timeout.

    Defaulting grounding to "none" means a fresh clone serves correctly with
    zero external wiring -- you turn NANDA on deliberately, not by accident.
    """
    backend = os.getenv("CEO_DELTA_GROUNDING_BACKEND", "none").strip().lower()
    overrides = dict(grounding_backend=backend, grounding_enabled=backend != "none")

    max_iter = os.getenv("CEO_DELTA_MAX_ITER", "").strip()
    if max_iter:
        overrides["max_ceo_eval_iterations"] = int(max_iter)

    base_url = os.getenv("CEO_DELTA_LLM_BASE_URL", "").strip()
    if base_url:
        overrides["llm_base_url"] = base_url

    # replace() returns a fresh Config; DEFAULT (the shared singleton) is untouched.
    return replace(DEFAULT, **overrides)


WORKDIR = os.getenv("CEO_DELTA_WORKDIR", ".ceo_delta")
GROUNDING_BACKEND = os.getenv("CEO_DELTA_GROUNDING_BACKEND", "none").strip().lower()

# The service reads SKILL.md straight off disk so GET /skill.md always serves
# whatever is committed -- an agent can fetch the contract live, no drift.
_SKILL_PATH = Path(__file__).resolve().parent.parent / "SKILL.md"


# --------------------------------------------------------------------------- #
# Request / response models  (also serve as inline API docs at /docs)
# --------------------------------------------------------------------------- #

class PlanRequest(BaseModel):
    task: str = Field(
        ...,
        min_length=1,
        description="A natural-language task or question to plan and answer.",
        examples=["Explain how a modern C compiler turns C source into an executable."],
    )
    auto_clarify: bool = Field(
        True,
        description="If false, the model returns a clarification request instead of "
                    "planning when the task is too vague to classify confidently.",
    )


class Signature(BaseModel):
    information_flow: str
    epistemic_stance: str
    output_contract: str
    decomposability: str
    complexity: str
    domain_volatility: str


class PlanResponse(BaseModel):
    task: str
    answer: str = Field(..., description="The composed deliverable -- the thing to use.")
    verdict: str = Field(..., description="Structural verdict: good | mixed | bad.")
    iterations: int = Field(..., description="Coordinate-descent iterations run.")
    species: str = Field(..., description="Structural signature string (the task's class).")
    signature: Signature
    ef_tensor: Optional[dict] = Field(
        None, description="Measured structural error per axis [partition,flow,role,scale]."
    )
    q_tensor: Optional[dict] = Field(
        None, description="Content-quality tensor [groundedness,relevance] when measured."
    )
    clarification: Optional[str] = Field(
        None, description="Set (with empty answer) when auto_clarify=false and the task is vague."
    )
    elapsed_s: float
    mode: str = Field(..., description="Grounding backend in effect: none | web_search | nanda.")


class JobHandle(BaseModel):
    """Returned by POST /plan -- the ticket for an accepted, still-running job."""
    job_id: str
    status: str = Field(..., description="Always 'running' at submit time; poll poll_url until 'done'.")
    poll_url: str = Field(..., description="GET this path to retrieve status and, once ready, the result.")


class JobStatus(BaseModel):
    """Returned by GET /plan/{job_id}."""
    job_id: str
    status: str = Field(..., description="running | done | error")
    result: Optional[PlanResponse] = Field(None, description="Populated only when status == 'done'.")
    detail: Optional[str] = Field(None, description="Error message, only when status == 'error'.")


# --------------------------------------------------------------------------- #
# App + single shared orchestrator
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Conductor-Delta Planning Agent",
    version="3.6",
    description=(
        "A self-correcting multi-agent planner. Give it a task; it classifies the "
        "task's structure, plans a DAG of sub-agents, and runs a deterministic "
        "coordinate descent that can only hold the plan still or improve it before "
        "composing the answer. See SKILL.md (GET /skill.md) for the agent contract."
    ),
)

_orch: Optional[Orchestrator] = None
_orch_lock = threading.Lock()  # serialize run() -- it mutates per-run state


def get_orchestrator() -> Orchestrator:
    global _orch
    if _orch is None:
        with _orch_lock:
            if _orch is None:  # double-checked: only build once
                _orch = Orchestrator(cfg=_build_config(), workdir=WORKDIR)
    return _orch


@app.on_event("startup")
def _warm() -> None:
    # Build the orchestrator eagerly so the first real /plan call isn't paying
    # handbook-load + registry-seed latency on top of the model run.
    get_orchestrator()


@app.get("/health")
def health() -> dict:
    """Liveness + which mode the service is in. Cheap; safe to poll."""
    return {
        "status": "ok",
        "service": "ceo-delta",
        "version": "3.6",
        "grounding_backend": GROUNDING_BACKEND,
        "model_ready": _orch is not None,
    }


@app.get("/", response_class=PlainTextResponse)
def root() -> str:
    return (
        "Conductor-Delta planning agent.\n"
        "  POST /plan          {\"task\": \"...\"}  -> {job_id} immediately (non-blocking)\n"
        "  GET  /plan/{job_id}                    -> poll: 'running' then the full result\n"
        "  POST /plan/sync     {\"task\": \"...\"}  -> blocks, returns the result in one call\n"
        "  GET  /skill.md                         -> the agent contract (SKILL.md)\n"
        "  GET  /health                           -> liveness\n"
        "  GET  /docs                             -> interactive OpenAPI docs\n"
    )


@app.get("/skill.md", response_class=PlainTextResponse)
def skill() -> str:
    """Serve the committed SKILL.md so an agent can fetch the contract live."""
    if not _SKILL_PATH.exists():
        raise HTTPException(status_code=404, detail="SKILL.md not found on server.")
    return _SKILL_PATH.read_text(encoding="utf-8")


def _to_response(task: str, r: RunResult, elapsed: float) -> PlanResponse:
    fp = r.fingerprint
    rep = r.report
    return PlanResponse(
        task=task,
        answer=r.answer,
        verdict=getattr(rep, "verdict", "unknown"),
        iterations=r.iterations,
        species=fp.shape_string(),
        signature=Signature(
            information_flow=fp.information_flow,
            epistemic_stance=fp.epistemic_stance,
            output_contract=fp.output_contract,
            decomposability=fp.decomposability,
            complexity=fp.complexity,
            domain_volatility=fp.domain_volatility,
        ),
        ef_tensor=getattr(rep, "ef_tensor", None),
        q_tensor=getattr(rep, "q_tensor", None),
        clarification=r.clarification,
        elapsed_s=round(elapsed, 2),
        mode=GROUNDING_BACKEND,
    )


def _compute_plan(task: str, auto_clarify: bool) -> PlanResponse:
    """Run the full EFQ pipeline for one task. Blocking, serialized; raises on failure.

    Serialize: Orchestrator.run mutates shared per-run state and is not
    re-entrant. The lock makes concurrent callers queue rather than corrupt.
    """
    orch = get_orchestrator()
    start = time.time()
    with _orch_lock:
        r = orch.run(task, auto_clarify=auto_clarify)
    elapsed = time.time() - start
    return _to_response(task, r, elapsed)


# --------------------------------------------------------------------------- #
# Async job store  (why: a full run can take minutes on a multi-LLM pipeline;
# a synchronous call risks the caller's HTTP client timing out before the
# answer exists. POST /plan returns a job_id in <1s and does the work in a
# background thread; the caller polls GET /plan/{job_id}. No single request is
# ever held open during the slow part, so no client/proxy timeout can fire.)
# In-memory is safe here: serve.py runs a single uvicorn worker on purpose.
# --------------------------------------------------------------------------- #

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
# max_workers=1 preserves the "one run at a time" invariant the orchestrator
# requires -- jobs queue and execute serially, same as the _orch_lock did.
_job_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="plan-job")

_JOB_TTL_S = 1800   # forget finished jobs after 30 min
_JOB_MAX = 200      # hard cap on retained jobs


def _prune_jobs_locked() -> None:
    """Drop old finished jobs so the store can't grow without bound. Caller holds _jobs_lock."""
    now = time.time()
    for jid in [j for j, v in _jobs.items()
                if v.get("finished_at") and now - v["finished_at"] > _JOB_TTL_S]:
        _jobs.pop(jid, None)
    if len(_jobs) > _JOB_MAX:
        oldest = sorted(_jobs.items(), key=lambda kv: kv[1].get("finished_at") or float("inf"))
        for jid, _ in oldest[: len(_jobs) - _JOB_MAX]:
            _jobs.pop(jid, None)


def _job_worker(job_id: str, task: str, auto_clarify: bool) -> None:
    """Run one job in the background and record its outcome in the store."""
    try:
        resp = _compute_plan(task, auto_clarify)
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].update(status="done", result=resp, finished_at=time.time())
    except Exception as exc:  # capture, don't crash the worker thread
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].update(status="error", detail=str(exc), finished_at=time.time())


@app.post("/plan", response_model=JobHandle, status_code=202)
def plan(req: PlanRequest) -> JobHandle:
    """Submit a task; returns a job_id immediately. Poll GET /plan/{job_id} for the result.

    Non-blocking on purpose: the pipeline can take minutes, so we never hold the
    request open while it runs. The work happens in a background thread; this
    call returns in well under a second, inside any client/proxy timeout.
    """
    task = req.task.strip()
    if not task:
        raise HTTPException(status_code=422, detail="task must be non-empty.")

    job_id = uuid.uuid4().hex[:16]
    with _jobs_lock:
        _prune_jobs_locked()
        _jobs[job_id] = {"status": "running", "task": task, "submitted_at": time.time()}
    _job_pool.submit(_job_worker, job_id, task, req.auto_clarify)
    return JobHandle(job_id=job_id, status="running", poll_url=f"/plan/{job_id}")


@app.get("/plan/{job_id}", response_model=JobStatus)
def plan_status(job_id: str) -> JobStatus:
    """Poll a submitted job. 'running' until finished, then 'done' (with result) or 'error'."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job_id (expired or never existed).")
        job = dict(job)  # snapshot so we don't hold the lock while building the response

    status = job["status"]
    if status == "done":
        return JobStatus(job_id=job_id, status="done", result=job["result"])
    if status == "error":
        return JobStatus(job_id=job_id, status="error", detail=job.get("detail"))
    return JobStatus(job_id=job_id, status="running")


@app.post("/plan/sync", response_model=PlanResponse)
def plan_sync(req: PlanRequest) -> PlanResponse:
    """Blocking variant: runs the pipeline and returns the full result in one call.

    Kept for the demo and for fast tasks where the caller controls the timeout.
    For an unattended agent, prefer POST /plan + GET /plan/{job_id}.
    """
    task = req.task.strip()
    if not task:
        raise HTTPException(status_code=422, detail="task must be non-empty.")
    try:
        return _compute_plan(task, req.auto_clarify)
    except Exception as exc:  # surface real failures, don't swallow them
        raise HTTPException(status_code=500, detail=f"planning failed: {exc}") from exc
