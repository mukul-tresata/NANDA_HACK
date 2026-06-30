"""Research Agent — upstream intent parser.

v2.0 — Species/Specifics split, single-handbook architecture.

Research is a PARSER, not a learner. It does one job: convert raw,
unstructured user intent into two distinct outputs:

  1. TaskFingerprint ("task species") — the structural invariants that
     determine what shape of plan is optimal. Four shape axes (embedded,
     drive handbook retrieval) + two modifier axes (complexity, volatility
     — never embedded, used as direct scalar/gate signals downstream).

  2. TaskSpecifics ("task specifics") — the concrete, situational details
     (real goal, constraints, resources) that don't generalize across
     tasks but matter for refinement if something goes wrong mid-run.

Research does NOT learn across runs. It has no handbook of its own.
The single CEO handbook now stores entries keyed on fingerprint.embedding
(structural similarity), not on raw task-string embeddings.

The old dual-handbook design (ceo_hb + research_hb) is removed. Research
never queried priors meaningfully different from CEO's own queries — it
was duplicated state with no distinct purpose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .config import Config, DEFAULT
from .embeddings import cosine, embed
from .llm import LLMClient
from .schemas import DAG, TaskFingerprint

# -- upstream parsing (primary pass) ------------------------------------------

_CLARIFY_SYSTEM = (
    "You are the Research agent. You are a PARSER, not a planner. Your job is "
    "to convert raw, unstructured user intent into a structured task signature "
    "along explicit invariant axes, plus the concrete specifics of this "
    "particular task. You do NOT plan. You do NOT execute. Reason carefully "
    "about each axis independently before answering."
)

_CLARIFY_PROMPT = """RAW USER INTENT: {task}

Classify this task along SIX axes. Reason through each one independently —
do not let one axis bias another.

SHAPE AXES (determine what plan topology/depth fits):

1. information_flow — how does information need to move through the plan?
   - divergent: must gather/retrieve from multiple independent sources before combining
   - convergent: multiple distinct inputs must be merged into one judgment
   - sequential: each step strictly depends on the previous step's output
   - recursive: the problem decomposes into smaller versions of itself

2. epistemic_stance — what is the dominant cognitive act required?
   - retrieval: finding/recalling existing facts or sources
   - synthesis: combining multiple existing facts/sources into a coherent whole
   - generation: producing genuinely novel content (not just recombination)
   - verification: checking/auditing claims against evidence

3. output_contract — what shape is the final deliverable?
   - artifact: a single coherent piece of output (report, essay, explanation)
   - comparison: a structured comparison between 2+ named things
   - verification: a verdict/audit report on claims
   - ranking: an ordered list with justified ordering

4. decomposability — can subtasks run truly independently?
   - independent: subtasks share no real dependency; genuine parallel fan-out is safe
   - coupled: subtasks look separable on the surface but actually need to share
     context/findings with each other to avoid redundant or contradictory work

MODIFIER AXES (scale/gate the plan, do NOT change its shape):

5. complexity — low | medium | high (depth of reasoning required, not task length)

6. domain_volatility — how stable is the ground truth?
   - stable: settled facts, unlikely to be disputed (math, established history)
   - evolving: an active research area where claims update frequently
   - contested: claims are actively disputed or no consensus exists

SPECIFICS (do not generalize across tasks, but matter for this run):
- real_goal: what success actually looks like for this specific task
- constraints: time/scope/format/depth constraints, or empty list
- resources_needed: what the executor needs to know or access

Output ONLY this JSON, no other text:
{{
  "information_flow": "divergent|convergent|sequential|recursive",
  "epistemic_stance": "retrieval|synthesis|generation|verification",
  "output_contract": "artifact|comparison|verification|ranking",
  "decomposability": "independent|coupled",
  "complexity": "low|medium|high",
  "domain_volatility": "stable|evolving|contested",
  "real_goal": "...",
  "constraints": ["..."],
  "resources_needed": ["..."],
  "structured_intent": "one precise sentence the planner should act on",
  "notes": "anything that doesn't fit above"
}}"""


# -- secondary pass (post-DAG, unchanged role, just retargeted) --------------

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
class TaskSpecifics:
    """Concrete, situational details. Do not generalize across tasks.
    Preserved for refinement if Delta later flags an issue mid-run."""
    real_goal: str
    constraints: List[str]
    resources_needed: List[str]
    structured_intent: str
    notes: str
    original_task: str
    drift: float = 0.0


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


class Research:
    """Pure parser. No handbook. No learning. One-to-many mapping from
    unstructured intent to (TaskFingerprint, TaskSpecifics)."""

    def __init__(self, llm: LLMClient, cfg: Config | None = None):
        self.llm = llm
        self.cfg = cfg or DEFAULT

    # -- PRIMARY PASS: parse into species + specifics -------------------------

    def clarify(self, task: str) -> tuple[TaskFingerprint, TaskSpecifics]:
        """Convert raw user intent into (fingerprint, specifics).
        No handbook query — Research does not learn or retrieve priors.
        This is a stateless parse.
        """
        task_emb = embed(task)
        data = self.llm.chat_json([
            {"role": "system", "content": _CLARIFY_SYSTEM},
            {"role": "user", "content": _CLARIFY_PROMPT.format(task=task)},
        ], tag="research.clarify")

        fingerprint = TaskFingerprint(
            information_flow=str(data.get("information_flow", "sequential")),
            epistemic_stance=str(data.get("epistemic_stance", "synthesis")),
            output_contract=str(data.get("output_contract", "artifact")),
            decomposability=str(data.get("decomposability", "coupled")),
            complexity=str(data.get("complexity", "medium")),
            domain_volatility=str(data.get("domain_volatility", "stable")),
        )
        # embed ONLY the shape axes -- modifiers never enter the embedding
        fingerprint.embedding = embed(fingerprint.shape_string())

        structured_intent = str(data.get("structured_intent") or task)
        structured_emb = embed(structured_intent)
        drift = 1.0 - cosine(task_emb, structured_emb)

        specifics = TaskSpecifics(
            real_goal=str(data.get("real_goal", "")),
            constraints=[str(x) for x in (data.get("constraints") or [])],
            resources_needed=[str(x) for x in (data.get("resources_needed") or [])],
            structured_intent=structured_intent,
            notes=str(data.get("notes", "")),
            original_task=task,
            drift=drift,
        )
        return fingerprint, specifics

    # -- SECONDARY PASS: post-DAG check (unchanged behavior) ------------------

    def investigate(self, dag: DAG, specifics: Optional[TaskSpecifics] = None) -> Brief:
        intents = "\n".join(f"- {n.node_id}: {n.intent}" for n in dag.nodes)
        si = specifics.structured_intent if specifics else dag.task

        data = self.llm.chat_json([
            {"role": "system", "content": _INVESTIGATE_SYSTEM},
            {"role": "user", "content": _INVESTIGATE_PROMPT.format(
                task=dag.task, structured_intent=si, intents=intents)},
        ], tag="research.investigate")

        refined = str(data.get("refined_intent") or si)
        base_emb = embed(si)
        sim = cosine(base_emb, embed(refined))
        drift = 1.0 - sim
        triggers = sim < (1.0 - self.cfg.replan_threshold)

        return Brief(
            summary=str(data.get("summary", "")),
            key_findings=[str(x) for x in (data.get("key_findings") or [])],
            refined_intent=refined,
            materiality=str(data.get("materiality", "low")),
            drift=drift,
            triggers_replan=triggers,
        )