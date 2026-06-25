"""CEO Planner.

Receives user intent -> queries handbook -> resolves agents via AgentRegistry
-> plans the full DAG before any execution fires.

v0.3 — AgentCard integration.

After building the DAG structure, CEO calls AgentRegistry.resolve_for_node()
for each node. The registry filters by:
    1. functional role match
    2. security clearance >= task security requirement
    3. confidence floor (cold-start guard)
    4. trust score ranking, task-class-specific where possible

The resolved agent_id is stored in node.assigned_agent_id.
Kernel reads this to dispatch to the right agent and inject card context.

If no agent passes the filters, node.assigned_agent_id stays None and
Kernel falls back to the generic LLM — same behavior as before.

v0.4 changes:
    - Prior-confidence steering: high-confidence handbook priors now lock in
      topology/depth by default; LLM must justify deviation explicitly.
    - Synthesizer fingerprint fix: synthesizer nodes fingerprint against the
      parent task embedding, not the node intent string.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from .config import Config, DEFAULT
from .embeddings import embed
from .handbook import Handbook
from .llm import LLMClient
from .schemas import AgentRegistry, DAG, Node, Roles, Why

_SYSTEM = (
    "You are the CEO planning agent in a multi-agent system. You ONLY plan; you "
    "never execute. You build a full computation DAG before anything runs, "
    "optimizing globally. Follow the reasoning chain exactly and be terse."
)

_PROMPT = """Plan a DAG for this task.

TASK: {task}

HANDBOOK PRIORS (similar past tasks, may be empty/low-confidence):
{priors}

AVAILABLE AGENT ROLES (from registry):
{agent_summary}

{mode_note}

{prior_lock}

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
    def __init__(self, handbook: Handbook, llm: LLMClient,
                 registry: Optional[AgentRegistry] = None,
                 cfg: Config | None = None):
        self.hb = handbook
        self.llm = llm
        self.registry = registry or AgentRegistry(cfg)
        self.cfg = cfg or DEFAULT

    # -- clarifying gate ------------------------------------------------------
    def needs_clarification(self, task: str) -> Tuple[bool, float]:
        emb = embed(task)
        _, sim = self.hb.best_match(emb)
        return (sim < self.cfg.clarify_similarity_threshold and len(self.hb.entries) > 0), sim

    # -- planning -------------------------------------------------------------
    def plan(self, task: str, run_index: int = 0, *,
             force_exploratory: bool = False,
             task_class: str = "reasoning") -> DAG:
        task_emb = embed(task)
        priors = self._format_priors(task_emb)
        agent_summary = self._format_agent_summary(task_class)
        exploratory = force_exploratory or run_index < self.cfg.cold_start_runs
        mode_note = (
            "EXPLORATORY MODE: priors are weak/cold-start. Make a reasonable "
            "plan and flag WHY as low-confidence."
            if exploratory else
            "STANDARD MODE: lean on the priors where confident."
        )
        prior_lock = self._prior_lock_note(task_emb, exploratory)
        prompt = _PROMPT.format(
            task=task, priors=priors,
            agent_summary=agent_summary, mode_note=mode_note,
            prior_lock=prior_lock,
        )
        data = self.llm.chat_json([
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ], tag="ceo.replan" if getattr(self, "_in_replan", False) else "ceo.plan")
        dag = self._build_dag(
            task, task_emb, data, exploratory,
            priors_used=self._priors_id(task_emb),
        )
        # resolve agents for each node AFTER DAG structure is set
        self._resolve_agents(dag, task_class)
        return dag

    def replan(self, task: str, brief_summary: str,
               run_index: int = 0, task_class: str = "reasoning") -> DAG:
        self._in_replan = True
        try:
            enriched = f"{task}\n\n[research brief]: {brief_summary}"
            dag = self.plan(enriched, run_index=run_index,
                            force_exploratory=False, task_class=task_class)
            dag.task = task
        finally:
            self._in_replan = False
        return dag

    # -- agent resolution -----------------------------------------------------
    def _resolve_agents(self, dag: DAG, task_class: str) -> None:
        """Assign the best available agent to each node."""
        for node in dag.nodes:
            card = self.registry.resolve_for_node(
                functional_role=node.roles.functional,
                task_class=task_class,
            )
            node.assigned_agent_id = card.agent_id if card else None

    # -- helpers --------------------------------------------------------------
    def _build_dag(self, task, task_emb, data, exploratory, priors_used) -> DAG:
        topology = str(data.get("topology", "linear"))
        depth = int(data.get("depth", 1) or 1)
        nodes_raw = data.get("nodes") or []
        nodes: List[Node] = []
        for nr in nodes_raw:
            intent = str(nr.get("intent", "")).strip() or "unspecified"
            functional_role = str(nr.get("functional", "generic"))
            why = Why(
                task_type_recognized=str(data.get("task_type", "")),
                topology_chosen=topology,
                depth_chosen=depth,
                alternatives_rejected=str(data.get("alternatives_rejected", "")),
                priors_used=priors_used,
                exploratory=exploratory,
            )
            # Synthesizer nodes fingerprint against the parent task, not their
            # own intent string — synthesis output should reflect task fidelity,
            # not intent-string similarity (which is always low by design).
            fingerprint_target = task_emb if functional_role == "synthesizer" else embed(intent)
            nodes.append(Node(
                node_id=str(nr.get("node_id") or f"n{len(nodes)+1}"),
                intent=intent,
                roles=Roles(
                    structural=str(nr.get("structural", topology)),
                    functional=functional_role,
                    epistemic=str(nr.get("epistemic", "generalist")),
                ),
                dependencies=[str(d) for d in (nr.get("dependencies") or [])],
                why=why,
                expected_output_fingerprint=fingerprint_target,
            ))
        if not nodes:
            nodes = [Node(
                node_id="n1", intent=task,
                expected_output_fingerprint=task_emb,
                why=Why(exploratory=exploratory, priors_used=priors_used),
            )]
        return DAG(
            task=task, task_embedding=task_emb, topology=topology, depth=depth,
            nodes=nodes, why_topology=str(data.get("why_topology", "")),
            why_depth=str(data.get("why_depth", "")), exploratory=exploratory,
        )

    def _prior_lock_note(self, task_emb, exploratory: bool) -> str:
        """If a high-confidence prior exists, tell the LLM to lock it in.

        Only fires in standard mode (not exploratory) — during cold start we
        want free exploration, not premature convergence.
        """
        if exploratory:
            return ""
        entry, sim = self.hb.best_match(task_emb)
        if (entry
                and sim >= 0.7
                and entry.confidence >= self.cfg.cold_start_runs
                and not entry.contested):
            return (
                f"HIGH-CONFIDENCE PRIOR (sim={sim:.2f}, conf={entry.confidence}): "
                f"topology={entry.topology_chosen}, depth={entry.depth_chosen}. "
                f"Treat these as the default. Only deviate if you can state a "
                f"concrete structural reason why this task differs from the prior — "
                f"'seems different' is not sufficient."
            )
        return ""

    def _format_priors(self, task_emb) -> str:
        matches = self.hb.query(task_emb)
        if not matches:
            return "(none — cold start)"
        lines = []
        for e, sim in matches:
            tag = "CONTESTED" if e.contested else f"conf={e.confidence}"
            lines.append(
                f"- sim={sim:.2f} [{tag}] topo={e.topology_chosen} "
                f"depth={e.depth_chosen} :: {e.revision or e.task_summary[:80]}"
            )
        return "\n".join(lines)

    def _format_agent_summary(self, task_class: str) -> str:
        """Show the LLM what agents exist and their trust levels.
        This lets CEO make role assignments that are grounded in reality.
        """
        summary = self.registry.summary()
        if not summary:
            return "(no agents registered)"
        lines = []
        for s in summary:
            conf_note = f"conf={s['confidence']}" if s['confidence'] > 0 else "cold-start"
            lines.append(
                f"- {s['agent_id']} roles={s['roles']} "
                f"trust={s['trust']:.2f} [{conf_note}] "
                f"sec={s['security']}"
            )
        return "\n".join(lines)

    def _priors_id(self, task_emb) -> str:
        e, sim = self.hb.best_match(task_emb)
        if not e or sim < 0.3:
            return "none"
        return f"{e.entry_id}(sim={sim:.2f})"