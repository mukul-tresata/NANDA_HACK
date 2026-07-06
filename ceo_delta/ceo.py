"""CEO Planner.

Receives user intent -> queries handbook -> resolves agents via AgentRegistry
-> plans the full DAG before any execution fires.

v2.0 — TaskFingerprint integration.

CEO now receives a TaskFingerprint from Research (via Orchestrator) for
each plan() call. The fingerprint's shape-axis embedding (information_flow,
epistemic_stance, output_contract, decomposability) replaces the raw task-
string embedding as the handbook retrieval key -- this is what makes
structural generalization (M2) possible: two tasks with different surface
phrasing but identical shape axes now retrieve the same prior.

The fingerprint's two modifier axes (complexity, domain_volatility) are
NEVER embedded and never enter retrieval. They are rendered as direct
prompt context (complexity -> depth-cap suggestion) or enforced in code
after planning (domain_volatility -> force_verifier()).

v0.4 changes (retained):
    - AgentCard integration via AgentRegistry.resolve_for_node()
    - Prior-confidence steering: high-confidence handbook priors lock in
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
from .schemas import AgentRegistry, DAG, Node, Roles, TaskFingerprint, Why

_CANONICAL_ROLES = {"retriever", "synthesizer", "verifier", "generic"}


def _coerce_functional_role(raw) -> str:
    """The LLM occasionally emits a list for `functional` when a node blends
    roles (e.g. ["retriever", "verifier"]). str(list) used to mangle that into
    an unmatched literal string ("['retriever', 'verifier']") that silently
    fell through to the zero-weight 'generic' behavioral band -- the role axis
    went blind on that node without any signal that it had. Take the first
    element for a list; anything outside the canonical set explicitly becomes
    'generic' rather than an unmatched garbage string that behaves the same
    way but looks like real data in logs."""
    if isinstance(raw, list) and raw:
        raw = raw[0]
    role = str(raw or "generic").strip().lower()
    return role if role in _CANONICAL_ROLES else "generic"

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

{directive_context}

Reason through these steps, THEN output JSON:
1. What kind of task is this?
2. What do the priors say about similar tasks?
3. What topology shape fits? Choose EXACTLY ONE from this list - no combinations, no hypens, no compound names: linear, fan-out, join, hierarchical, hub
4. What depth is warranted? (integer 1-4; shallower is cheaper)
5. What are the nodes and dependencies?
6. WHY per decision.

Output ONLY a JSON object of this exact shape. Do not output any text before or after the JSON block. No reasoning, no explanation, no markdown headers:
{{
  "task_type": "...",
  "topology": "fan-out",
  "depth": 2,
  "why_topology": "why this shape, and what shape you rejected",
  "directive_response": "if a Delta directive was received, how did you respond to it and why",
  "why_depth": "why this depth",
  "alternatives_rejected": "...",
  "nodes": [
    {{"node_id":"n1","intent":"...","dependencies":[],
      "structural":"fan-out", "functional":"retriever","epistemic":"specialist"}}
  ]
}}"""

# v3.x — composition step. The CEO does not just plan; it closes the loop by
# speaking as the single accountable voice of the org. Verifier output is
# internal machinery (an audit of the org's own work) and must be CONSUMED
# here, not forwarded raw -- the user should never see [UNVERIFIED] tags,
# node ids, or "upstream context" language.
_COMPOSE_SYSTEM = (
    "You are the CEO -- the single coherent voice of a multi-agent system that "
    "has just completed its internal work (planning, retrieval, synthesis, "
    "verification) for one task. Deliver ONE coherent, professional answer to "
    "the user's original request, written as if from a single expert author. "
    "You have access to your own internal findings, including verification "
    "notes about what could and could not be confirmed. Speak in one voice. "
    "Do NOT mention nodes, plans, agents, 'upstream context', or internal "
    "structure. Do NOT emit raw [UNVERIFIED] tags. Where something could not "
    "be verified, express appropriate confidence or caveats in natural "
    "language instead."
)

_COMPOSE_PROMPT = """The user asked:

{task}

Your organization's internal work on this task (raw notes, findings, and any
verification audit) is below. Use it to compose the final deliverable -- do
not just repeat it.

INTERNAL MATERIAL:
{material}

Produce ONLY the final deliverable, in your own voice, as the complete answer
to the user's request above. No preamble, no meta-commentary about the process
that produced it."""


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
             task_class: str = "reasoning",
             directive=None,
             fingerprint: Optional[TaskFingerprint] = None) -> DAG:
        # Retrieval key: structural fingerprint embedding when available
        # (shape axes only -- information_flow, epistemic_stance,
        # output_contract, decomposability). Falls back to raw task
        # embedding only if no fingerprint was supplied (e.g. internal
        # replan() calls that don't re-derive one).
        task_emb = embed(task)
        retrieval_emb = (
            fingerprint.embedding if (fingerprint and fingerprint.embedding) else task_emb
        )

        priors = self._format_priors(retrieval_emb)
        agent_summary = self._format_agent_summary(task_class)
        exploratory = force_exploratory or run_index < self.cfg.cold_start_runs
        mode_note = (
            "EXPLORATORY MODE: priors are weak/cold-start. Make a reasonable "
            "plan and flag WHY as low-confidence."
            if exploratory else
            "STANDARD MODE: lean on the priors where confident."
        )
        # Prior/directive arbitration: while a corrective directive is active,
        # the prior lock is suppressed — otherwise the same prompt says both
        # "keep this topology (high-confidence prior)" and "restructure it
        # (directive)", and the lock wins (observed live: conf=10 prior pinned
        # CEO to a topology the directive was trying to fix). Within a
        # correction loop the measured error signal outranks the prior.
        in_correction = directive is not None and getattr(directive, "action", "surface") != "surface"
        prior_lock = "" if in_correction else self._prior_lock_note(retrieval_emb, exploratory)
        directive_context = self._format_directive(directive)
        modifier_context = self._format_modifiers(fingerprint)
        full_context = "\n".join(filter(None, [directive_context, modifier_context]))

        prompt = _PROMPT.format(
            task=task, priors=priors,
            agent_summary=agent_summary, mode_note=mode_note,
            prior_lock=prior_lock,
            directive_context=full_context,
        )
        data = self.llm.chat_json([
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ], tag="ceo.replan" if getattr(self, "_in_replan", False) else "ceo.plan")
        dag = self._build_dag(
            task, retrieval_emb, data, exploratory,
            priors_used=self._priors_id(retrieval_emb),
            directive=directive,
        )
        # resolve agents for each node AFTER DAG structure is set
        self._resolve_agents(dag, task_class)
        return dag

    def replan(self, task: str, brief_summary: str,
               run_index: int = 0, task_class: str = "reasoning",
               fingerprint: Optional[TaskFingerprint] = None) -> DAG:
        self._in_replan = True
        try:
            enriched = f"{task}\n\n[research brief]: {brief_summary}"
            dag = self.plan(enriched, run_index=run_index,
                            force_exploratory=False, task_class=task_class,
                            fingerprint=fingerprint)
            dag.task = task
        finally:
            self._in_replan = False
        return dag

    # -- composition ------------------------------------------------------------
    def compose(self, task: str, fingerprint, dag, trace) -> str:
        """The CEO's closing act: compose ONE coherent deliverable from the org's
        internal work. Verifier findings are CONSUMED as the entity's own awareness
        of uncertainty (rendered as natural-language caveats), never leaked as raw
        [UNVERIFIED] tags or node/plan talk. This is the machinery->output boundary."""
        by_id = {r.node_id: r for r in trace.results}
        blocks = []
        for n in dag.nodes:
            r = by_id.get(n.node_id)
            if r is None or r.error:
                continue
            blocks.append(f"[{n.roles.functional}] {r.output[:4000]}")
        material = "\n\n".join(blocks)
        if not material:
            return ""
        prompt = _COMPOSE_PROMPT.format(task=task, material=material)
        return self.llm.chat([
            {"role": "system", "content": _COMPOSE_SYSTEM},
            {"role": "user", "content": prompt},
        ], tag="ceo.compose").strip()

    # -- mandatory verifier enforcement ----------------------------------------
    def force_verifier(self, dag: DAG) -> DAG:
        """Append a mandatory verifier node when domain_volatility required
        one and CEO's plan omitted it. Code-level guarantee, not a prompt
        suggestion -- this is what makes verification non-optional for
        volatile-domain tasks regardless of what the LLM chose to plan.
        """
        terminal_ids = [
            n.node_id for n in dag.nodes
            if not any(n.node_id in other.dependencies for other in dag.nodes)
        ]
        verifier = Node(
            node_id=f"n{len(dag.nodes) + 1}_verify",
            intent=(
                "Audit all upstream claims for unverified statistics, figures, "
                "or causal claims lacking cited sources."
            ),
            roles=Roles(structural="join", functional="verifier", epistemic="specialist"),
            dependencies=terminal_ids,
            why=Why(
                task_type_recognized="forced-verifier",
                topology_chosen=dag.topology,
                depth_chosen=dag.depth,
                alternatives_rejected="none -- structurally mandated by domain_volatility",
            ),
            expected_output_fingerprint=dag.task_embedding,
        )
        dag.nodes.append(verifier)
        dag.depth += 1
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
    def _build_dag(self, task, task_emb, data, exploratory, priors_used, directive=None) -> DAG:
        topology = str(data.get("topology", "linear"))
        depth = int(data.get("depth", 1) or 1)
        nodes_raw = data.get("nodes") or []
        nodes: List[Node] = []
        for nr in nodes_raw:
            intent = str(nr.get("intent", "")).strip() or "unspecified"
            functional_role = _coerce_functional_role(nr.get("functional", "generic"))
            why = Why(
                task_type_recognized=str(data.get("task_type", "")),
                topology_chosen=topology,
                depth_chosen=depth,
                alternatives_rejected=str(data.get("alternatives_rejected", "")),
                priors_used=priors_used,
                exploratory=exploratory,
                directive_received=directive.reason if directive and directive.action != "surface" else "",
                directive_response=str(data.get("directive_response", ""))[:200],
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
                why=Why(exploratory=exploratory,
                        priors_used=priors_used,
                        directive_received=directive.reason if directive and directive.action != "surface" else ""),
            )]
        return DAG(
            task=task, task_embedding=task_emb, topology=topology, depth=depth,
            nodes=nodes, why_topology=str(data.get("why_topology", "")),
            why_depth=str(data.get("why_depth", "")), exploratory=exploratory,
        )

    def _prior_lock_note(self, retrieval_emb, exploratory: bool) -> str:
        """If a high-confidence prior exists, tell the LLM to lock it in.

        Only fires in standard mode (not exploratory) — during cold start we
        want free exploration, not premature convergence.
        """
        if exploratory:
            return ""
        entry, sim = self.hb.best_match(retrieval_emb)
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

    def _format_priors(self, retrieval_emb) -> str:
        matches = self.hb.query(retrieval_emb)
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

    def _format_modifiers(self, fingerprint: Optional[TaskFingerprint]) -> str:
        """Render complexity + decomposability + volatility as direct prompt
        context. These are deterministic outputs from Research, not things
        the LLM should re-infer from the raw task string."""
        if fingerprint is None:
            return ""
        lines = [
            f"COMPLEXITY (from Research): {fingerprint.complexity} "
            f"-> suggested max depth is {fingerprint.depth_cap()}. "
            f"Do not exceed this without a concrete structural reason.",
        ]
        if fingerprint.decomposability == "coupled":
            lines.append(
                "DECOMPOSABILITY: coupled -- subtasks that look independent "
                "may actually need shared context. If you use fan-out, make sure "
                "sibling node intents are differentiated enough to avoid echo, "
                "or consider an earlier join."
            )
        if fingerprint.requires_verifier():
            lines.append(
                f"DOMAIN VOLATILITY: {fingerprint.domain_volatility} -- a verifier "
                f"node is MANDATORY in this plan. If you omit it, it will be "
                f"force-inserted after planning."
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

    def _priors_id(self, retrieval_emb) -> str:
        e, sim = self.hb.best_match(retrieval_emb)
        if not e or sim < 0.3:
            return "none"
        return f"{e.entry_id}(sim={sim:.2f})"

    def _format_directive(self, directive) -> str:
        """Render Delta's directive as prompt context for CEO."""
        if directive is None or directive.action == "surface":
            return ""
        lines = [
            f"DELTA DIRECTIVE ({directive.action.upper()}) — iteration {directive.iteration}:",
            f"Reason: {directive.reason}",
        ]
        if directive.replan_hint:
            lines.append(f"Guidance: {directive.replan_hint}")
        if directive.refinement_targets:
            lines.append(f"Focus on: {', '.join(directive.refinement_targets)}")
        lines.append(
            "Make the MINIMAL structural change that fixes the flagged PRIMARY AXIS. "
            "If the guidance lists PRESERVE constraints or a CURRENT PLAN, treat that plan "
            "as your baseline and edit it incrementally — keep every node and edge that is "
            "not implicated by the primary axis, and do NOT rebuild from scratch (rebuilding "
            "regresses axes that were already correct). "
            "In directive_response, state exactly what you changed and what you deliberately kept."
        )
        return "\n".join(lines)