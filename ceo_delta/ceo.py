"""CEO Planner.

Receives user intent -> queries the handbook -> plans the full DAG before any
execution. Forces the 6-step reasoning chain and emits a separate WHY for
topology vs depth. Handles:
  * cold start: exploratory flag for first N runs OR low retrieval similarity;
  * clarifying question when retrieval similarity is below threshold;
  * replanning when Research's brief materially shifts the task (limitation #3).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from .config import Config, DEFAULT
from .embeddings import embed
from .handbook import Handbook
from .llm import LLMClient
from .schemas import DAG, Node, Roles, Why

_SYSTEM = (
    "You are the CEO planning agent in a multi-agent system. You ONLY plan; you "
    "never execute. You build a full computation DAG before anything runs, "
    "optimizing globally. Follow the reasoning chain exactly and be terse."
)

_PROMPT = """Plan a DAG for this task.

TASK: {task}

HANDBOOK PRIORS (similar past tasks, may be empty/low-confidence):
{priors}

{mode_note}

Reason through these steps, THEN output JSON:
1. What kind of task is this?
2. What do the priors say about similar tasks?
3. What topology shape fits? (linear / fan-out / join / hierarchical / hub)
4. What depth is warranted? (integer 1-4; shallower is cheaper)
5. What are the nodes and dependencies?
6. WHY per decision.

Output ONLY a JSON object of this exact shape:
{{
  "task_type": "...",
  "topology": "fan-out",
  "depth": 2,
  "why_topology": "why this shape, and what shape you rejected",
  "why_depth": "why this depth",
  "alternatives_rejected": "...",
  "nodes": [
    {{"node_id":"n1","intent":"...","dependencies":[],
      "structural":"fan-out","functional":"retriever","epistemic":"specialist"}}
  ]
}}"""


class CEO:
    def __init__(self, handbook: Handbook, llm: LLMClient, cfg: Config | None = None):
        self.hb = handbook
        self.llm = llm
        self.cfg = cfg or DEFAULT

    # -- clarifying gate ------------------------------------------------------
    def needs_clarification(self, task: str) -> Tuple[bool, float]:
        emb = embed(task)
        _, sim = self.hb.best_match(emb)
        return (sim < self.cfg.clarify_similarity_threshold and len(self.hb.entries) > 0), sim

    # -- planning -------------------------------------------------------------
    def plan(self, task: str, run_index: int = 0, *, force_exploratory: bool = False
             ) -> DAG:
        task_emb = embed(task)
        priors = self._format_priors(task_emb)
        exploratory = force_exploratory or run_index < self.cfg.cold_start_runs
        mode_note = ("EXPLORATORY MODE: priors are weak/cold-start. Make a "
                     "reasonable plan and we will flag WHY as low-confidence."
                     if exploratory else
                     "STANDARD MODE: lean on the priors where confident.")
        prompt = _PROMPT.format(task=task, priors=priors, mode_note=mode_note)
        data = self.llm.chat_json([
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ])
        return self._build_dag(task, task_emb, data, exploratory, priors_used=self._priors_id(task_emb))

    def replan(self, task: str, brief_summary: str, run_index: int = 0) -> DAG:
        """Re-plan after a materially-changing research brief."""
        enriched = f"{task}\n\n[research brief]: {brief_summary}"
        dag = self.plan(enriched, run_index=run_index, force_exploratory=False)
        dag.task = task  # keep original task label
        return dag

    # -- helpers --------------------------------------------------------------
    def _build_dag(self, task, task_emb, data, exploratory, priors_used) -> DAG:
        topology = str(data.get("topology", "linear"))
        depth = int(data.get("depth", 1) or 1)
        nodes_raw = data.get("nodes") or []
        nodes: List[Node] = []
        for nr in nodes_raw:
            intent = str(nr.get("intent", "")).strip() or "unspecified"
            why = Why(
                task_type_recognized=str(data.get("task_type", "")),
                topology_chosen=topology,
                depth_chosen=depth,
                alternatives_rejected=str(data.get("alternatives_rejected", "")),
                priors_used=priors_used,
                exploratory=exploratory,
            )
            nodes.append(Node(
                node_id=str(nr.get("node_id") or f"n{len(nodes)+1}"),
                intent=intent,
                roles=Roles(
                    structural=str(nr.get("structural", topology)),
                    functional=str(nr.get("functional", "generic")),
                    epistemic=str(nr.get("epistemic", "generalist")),
                ),
                dependencies=[str(d) for d in (nr.get("dependencies") or [])],
                why=why,
                expected_output_fingerprint=embed(intent),
            ))
        if not nodes:  # guarantee at least one node
            nodes = [Node(node_id="n1", intent=task,
                          expected_output_fingerprint=embed(task),
                          why=Why(exploratory=exploratory, priors_used=priors_used))]
        return DAG(
            task=task, task_embedding=task_emb, topology=topology, depth=depth,
            nodes=nodes, why_topology=str(data.get("why_topology", "")),
            why_depth=str(data.get("why_depth", "")), exploratory=exploratory,
        )

    def _format_priors(self, task_emb) -> str:
        matches = self.hb.query(task_emb)
        if not matches:
            return "(none — cold start)"
        lines = []
        for e, sim in matches:
            tag = "CONTESTED" if e.contested else f"conf={e.confidence}"
            lines.append(
                f"- sim={sim:.2f} [{tag}] topo={e.topology_chosen} depth={e.depth_chosen} "
                f":: {e.revision or e.task_summary[:80]}")
        return "\n".join(lines)

    def _priors_id(self, task_emb) -> str:
        e, sim = self.hb.best_match(task_emb)
        if not e or sim < 0.3:
            return "none"
        return f"{e.entry_id}(sim={sim:.2f})"
