"""Research Agent — upstream intent clarifier and resource mapper.

Design change from v0.1: Research now runs BEFORE CEO plans, not after.
The flow is:

    User intent -> Research.clarify() -> StructuredBrief -> CEO.plan()

Research translates raw user intent into a structured problem statement:
  - what is the actual goal (not just what the user said)
  - what constraints exist
  - what resources / knowledge the executor will need
  - what task class this looks like

CEO then plans against the structured brief, not the raw user string.

The old investigate() method (post-DAG brief) is kept as a secondary pass
for cases where execution context materially changes understanding — but it
is no longer the primary Research contribution.

Drift is still computed: cosine(original_intent, structured_intent).
If drift exceeds cfg.replan_threshold after the secondary pass, CEO replans.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .config import Config, DEFAULT
from .embeddings import cosine, embed
from .handbook import Handbook
from .llm import LLMClient
from .schemas import DAG

# -- upstream clarification (primary pass) ------------------------------------

_CLARIFY_SYSTEM = (
    "You are the Research agent. Your job is to translate a raw user intent "
    "into a structured, workable problem statement that a planning agent can "
    "act on precisely. You do NOT plan. You do NOT execute. You clarify, "
    "constrain, and resource-map."
)

_CLARIFY_PROMPT = """RAW USER INTENT: {task}

RESEARCH HANDBOOK PRIORS (similar past tasks):
{priors}

Your job:
1. Identify the REAL goal behind the stated intent (what does success look like?)
2. Identify constraints (time, scope, format, depth)
3. Identify what knowledge/resources the executor will need
4. Classify the task type (research-heavy / reasoning / tool-use / verification / simple)
5. Restate the intent as a single clear, workable problem sentence

Output ONLY this JSON:
{{
  "real_goal": "what success actually looks like for this task",
  "constraints": ["list of constraints, or empty if none"],
  "resources_needed": ["what the executor needs to know or have access to"],
  "task_class": "research-heavy|reasoning|tool-use|verification|simple",
  "structured_intent": "one precise sentence the planner should act on",
  "complexity": "low|medium|high",
  "notes": "anything the planner should know that doesn't fit above"
}}"""


# -- secondary pass (post-DAG, legacy investigate behavior) -------------------

_INVESTIGATE_SYSTEM = (
    "You are the Research agent doing a secondary check. A plan has been made. "
    "Check whether execution context changes the problem understanding. "
    "Be terse."
)

_INVESTIGATE_PROMPT = """ORIGINAL TASK: {task}
STRUCTURED INTENT (from clarification): {structured_intent}

PLANNED DAG INTENTS:
{intents}

Does the plan reveal anything that materially changes task understanding?
Output ONLY this JSON:
{{
  "summary": "2-3 sentence brief of what matters for execution",
  "key_findings": ["..."],
  "refined_intent": "updated intent if understanding changed, else repeat structured_intent",
  "materiality": "low|medium|high"
}}"""


@dataclass
class StructuredBrief:
    """Primary output of Research.clarify() — fed to CEO before planning."""
    real_goal: str
    constraints: List[str]
    resources_needed: List[str]
    task_class: str                  # research-heavy / reasoning / tool-use / verification / simple
    structured_intent: str           # the single sentence CEO plans against
    complexity: str                  # low / medium / high
    notes: str
    original_task: str               # kept for drift computation
    drift: float = 0.0               # cosine drift from raw intent to structured intent
    triggers_replan: bool = False    # set by secondary pass if needed


@dataclass
class Brief:
    """Secondary output of Research.investigate() — legacy post-DAG pass.
    Kept for backward compatibility with orchestrator and delta."""
    summary: str
    key_findings: List[str]
    refined_intent: str
    materiality: str
    drift: float
    triggers_replan: bool
    structured_brief: Optional[StructuredBrief] = None  # backref to primary pass


class Research:
    def __init__(self, handbook: Handbook, llm: LLMClient, cfg: Config | None = None):
        self.hb = handbook
        self.llm = llm
        self.cfg = cfg or DEFAULT

    # -- PRIMARY PASS: upstream clarification (runs before CEO) ---------------

    def clarify(self, task: str) -> StructuredBrief:
        """Translate raw user intent into a structured problem statement.
        This is the main Research contribution — CEO receives this, not the
        raw task string.
        """
        task_emb = embed(task)
        priors = self._priors(task_emb)
        data = self.llm.chat_json([
            {"role": "system", "content": _CLARIFY_SYSTEM},
            {"role": "user", "content": _CLARIFY_PROMPT.format(
                task=task, priors=priors)},
        ])

        structured_intent = str(data.get("structured_intent") or task)
        structured_emb = embed(structured_intent)
        drift = 1.0 - cosine(task_emb, structured_emb)

        return StructuredBrief(
            real_goal=str(data.get("real_goal", "")),
            constraints=[str(x) for x in (data.get("constraints") or [])],
            resources_needed=[str(x) for x in (data.get("resources_needed") or [])],
            task_class=str(data.get("task_class", "reasoning")),
            structured_intent=structured_intent,
            complexity=str(data.get("complexity", "medium")),
            notes=str(data.get("notes", "")),
            original_task=task,
            drift=drift,
            triggers_replan=False,  # primary pass never triggers replan
        )

    # -- SECONDARY PASS: post-DAG check (runs after CEO, replaces investigate) -

    def investigate(self, dag: DAG,
                    structured_brief: Optional[StructuredBrief] = None) -> Brief:
        """Secondary pass — checks whether the plan reveals anything that
        materially changes task understanding. Triggers CEO replan if so.

        structured_brief is passed in from the primary pass so Research can
        compare against it rather than the raw task string.
        """
        intents = "\n".join(f"- {n.node_id}: {n.intent}" for n in dag.nodes)
        si = structured_brief.structured_intent if structured_brief else dag.task

        data = self.llm.chat_json([
            {"role": "system", "content": _INVESTIGATE_SYSTEM},
            {"role": "user", "content": _INVESTIGATE_PROMPT.format(
                task=dag.task, structured_intent=si, intents=intents)},
        ])

        refined = str(data.get("refined_intent") or si)
        # drift measured against structured_intent, not raw task
        base_emb = embed(si)
        sim = cosine(base_emb, embed(refined))
        drift = 1.0 - sim
        triggers = sim < self.cfg.replan_threshold

        return Brief(
            summary=str(data.get("summary", "")),
            key_findings=[str(x) for x in (data.get("key_findings") or [])],
            refined_intent=refined,
            materiality=str(data.get("materiality", "low")),
            drift=drift,
            triggers_replan=triggers,
            structured_brief=structured_brief,
        )

    # -- helpers --------------------------------------------------------------

    def _priors(self, task_emb) -> str:
        matches = self.hb.query(task_emb)
        if not matches:
            return "(none)"
        return "\n".join(
            f"- sim={s:.2f} [{e.task_class if hasattr(e, 'task_class') else 'unknown'}]: "
            f"{e.revision or e.task_summary[:80]}"
            for e, s in matches
        )