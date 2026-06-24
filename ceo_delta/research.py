"""Research Agent.

Separate from CEO so CEO specializes purely in planning. Given CEO's DAG it
gathers the information the nodes will need and returns a structured brief.

Limitation #3 — the replan condition is made concrete: we embed the original
task and the brief-refined intent, and CEO replans iff
cosine(original, refined) < cfg.replan_threshold (default 0.70).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import Config, DEFAULT
from .embeddings import cosine, embed
from .handbook import Handbook
from .llm import LLMClient
from .schemas import DAG

_SYSTEM = ("You are the Research agent. Given a planned DAG and a task, gather "
           "the information the plan will need and report a tight brief. You do "
           "not execute the plan.")

_PROMPT = """TASK: {task}

PLANNED DAG INTENTS:
{intents}

RESEARCH HANDBOOK PRIORS:
{priors}

Produce a structured brief. Output ONLY this JSON:
{{
  "summary": "2-3 sentence brief of what matters for this task",
  "key_findings": ["..."],
  "refined_intent": "one sentence restating the task as research clarified it",
  "materiality": "low|medium|high"
}}"""


@dataclass
class Brief:
    summary: str
    key_findings: List[str]
    refined_intent: str
    materiality: str
    drift: float            # 1 - cosine(original, refined)
    triggers_replan: bool


class Research:
    def __init__(self, handbook: Handbook, llm: LLMClient, cfg: Config | None = None):
        self.hb = handbook
        self.llm = llm
        self.cfg = cfg or DEFAULT

    def investigate(self, dag: DAG) -> Brief:
        intents = "\n".join(f"- {n.node_id}: {n.intent}" for n in dag.nodes)
        priors = self._priors(dag.task_embedding)
        data = self.llm.chat_json([
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _PROMPT.format(
                task=dag.task, intents=intents, priors=priors)},
        ])
        refined = str(data.get("refined_intent") or dag.task)
        sim = cosine(dag.task_embedding, embed(refined))
        drift = 1.0 - sim
        triggers = sim < self.cfg.replan_threshold
        return Brief(
            summary=str(data.get("summary", "")),
            key_findings=[str(x) for x in (data.get("key_findings") or [])],
            refined_intent=refined,
            materiality=str(data.get("materiality", "low")),
            drift=drift,
            triggers_replan=triggers,
        )

    def _priors(self, task_emb) -> str:
        matches = self.hb.query(task_emb)
        if not matches:
            return "(none)"
        return "\n".join(f"- sim={s:.2f}: {e.revision or e.task_summary[:80]}"
                         for e, s in matches)
