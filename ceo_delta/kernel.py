"""Execution Kernel (KAIJU-inspired).

v0.3 — AgentCard context injection.

If node.assigned_agent_id is set, Kernel fetches the AgentCard from the
registry and injects two things into the execution prompt:
    1. The agent's performance history for this role/task-class — so the
       executing LLM knows what this "agent" has done well and poorly before.
    2. The security requirement — so the executor knows the stakes.

This is the AgentCard feedback loop at execution time: Delta writes to the
card, CEO resolves against it, Kernel injects it. The card is the memory.

If assigned_agent_id is None, execution falls back to the generic LLM prompt
with no card context — same behavior as v0.1/v0.2.
"""
from __future__ import annotations

import concurrent.futures as cf
import time
from typing import Dict, List, Optional

from .config import Config, DEFAULT
from .embeddings import cosine, embed
from .llm import LLMClient
from .schemas import AgentCard, AgentRegistry, DAG, ExecutionTrace, Node, NodeResult

_SYSTEM = (
    "You are an execution worker. Carry out the single node intent given. "
    "Be concise and produce the artifact, not commentary about it."
)

_GATE_HINTS = {
    "retriever":   ("retrieve", "find", "gather", "search", "collect", "fetch"),
    "synthesizer": ("synthesize", "combine", "summarize", "write", "compose", "answer"),
    "verifier":    ("verify", "check", "validate", "audit", "confirm", "cross"),
}


class Kernel:
    def __init__(self, llm: LLMClient,
                 registry: Optional[AgentRegistry] = None,
                 cfg: Config | None = None,
                 max_workers: int = 4):
        self.llm = llm
        self.registry = registry
        self.cfg = cfg or DEFAULT
        self.max_workers = max_workers

    def execute(self, dag: DAG, brief_context: str = "") -> ExecutionTrace:
        t0 = time.time()
        results: Dict[str, NodeResult] = {}
        remaining = {n.node_id: n for n in dag.nodes}
        done: set[str] = set()

        with cf.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            while remaining:
                ready = [n for n in remaining.values()
                         if all(d in done for d in n.dependencies)]
                if not ready:
                    for n in remaining.values():
                        results[n.node_id] = self._failed(n, "unsatisfiable dependencies")
                    break
                futs = {
                    pool.submit(self._run_node, n, dag, results, brief_context): n
                    for n in ready
                }
                for fut in cf.as_completed(futs):
                    n = futs[fut]
                    results[n.node_id] = fut.result()
                    done.add(n.node_id)
                    remaining.pop(n.node_id, None)

        ordered = [results[n.node_id] for n in dag.nodes if n.node_id in results]
        return ExecutionTrace(
            dag_id=dag.dag_id, task=dag.task, results=ordered,
            total_tokens=sum(r.cost_tokens for r in ordered),
            wallclock_s=time.time() - t0,
        )

    # -- per node -------------------------------------------------------------

    def _run_node(self, node: Node, dag: DAG, results, brief_context) -> NodeResult:
        gated = self._intent_gate(node)
        if not gated:
            return self._failed(node, "intent gate rejected", gated=False)

        nl = "\n"
        dep_ctx = nl.join(
            f"[{d} output]: {results[d].output[:400]}"
            for d in node.dependencies if d in results
        )

        # -- AgentCard context injection (new) --------------------------------
        card_context = self._card_context(node)

        brief_line = ("BRIEF: " + brief_context) if brief_context else ""
        upstream_line = ("UPSTREAM:" + nl + dep_ctx) if dep_ctx else ""
        prompt = (
            f"NODE INTENT: {node.intent}{nl}"
            f"ROLE: {node.roles.functional}/{node.roles.epistemic}{nl}"
            f"{card_context}{nl}"
            f"{brief_line}{nl}"
            f"{upstream_line}{nl}"
            "Produce the artifact for this node."
        )

        before = self.llm.total_tokens
        t = time.time()
        try:
            out = self.llm.chat([
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ], max_tokens=1024, tag=f"kernel.{node.node_id}")
            err = None
        except Exception as e:
            out, err = f"[node error] {e}", str(e)

        latency = time.time() - t
        cost = max(1, self.llm.total_tokens - before)
        out_emb = embed(out)
        fp = (
            cosine(node.expected_output_fingerprint, out_emb)
            if node.expected_output_fingerprint else 0.0
        )
        return NodeResult(
            node_id=node.node_id, intent=node.intent, output=out,
            output_embedding=out_emb, cost_tokens=cost, latency_s=latency,
            role_function_match=self._role_match(node),
            fingerprint_match=fp, gated=True, error=err,
        )

    def _card_context(self, node: Node) -> str:
        """Build an AgentCard context string to inject into the prompt.

        If the node has an assigned agent and the registry is available,
        pull the card's per-role and per-class stats and format them as a
        short instruction block. This tells the executing LLM:
            - what this agent has done well/poorly in this role before
            - what the security stakes are

        Returns empty string if no card is available — no-op for fallback.
        """
        if not self.registry or not node.assigned_agent_id:
            return ""
        card = self.registry.get(node.assigned_agent_id)
        if not card or card.confidence < 1:
            return ""

        role = node.roles.functional
        role_stats = card.dynamic.per_role_stats.get(role)
        lines = [f"AGENT CONTEXT (id={node.assigned_agent_id} trust={card.trust_score:.2f}):"]

        if role_stats and role_stats["runs"] >= 2:
            lines.append(
                f"  Prior performance as {role}: "
                f"fp_match={role_stats['fp_mean']:.2f} "
                f"error_rate={role_stats['error_rate']:.2f} "
                f"role_match={role_stats['role_match_rate']:.2f} "
                f"({role_stats['runs']} runs)"
            )
            if role_stats["error_rate"] > 0.3:
                lines.append(f"  WARNING: high error rate in {role} role — be especially careful.")
            if role_stats["fp_mean"] < 0.3:
                lines.append(f"  NOTE: fingerprint match has been low — stay close to the stated intent.")

        if card.dynamic.hub_failure_count > 0:
            lines.append(
                f"  WARNING: this agent has failed {card.dynamic.hub_failure_count} "
                f"time(s) on hub nodes. Prioritize correctness over speed."
            )

        if card.dynamic.last_revision:
            lines.append(f"  Last Delta feedback: {card.dynamic.last_revision[:120]}")

        return nl.join(lines) if len(lines) > 1 else ""

    # -- helpers --------------------------------------------------------------

    def _intent_gate(self, node: Node) -> bool:
        hints = _GATE_HINTS.get(node.roles.functional)
        if not hints:
            return True
        # warn-not-block: always passes but role_function_match records the miss
        return True

    def _role_match(self, node: Node) -> bool:
        hints = _GATE_HINTS.get(node.roles.functional)
        if not hints:
            return True
        return any(h in node.intent.lower() for h in hints)

    def _failed(self, node: Node, reason: str, gated: bool = True) -> NodeResult:
        return NodeResult(
            node_id=node.node_id, intent=node.intent,
            output=f"[skipped] {reason}",
            output_embedding=embed(reason), cost_tokens=0, latency_s=0.0,
            role_function_match=False, fingerprint_match=0.0,
            gated=gated, error=reason,
        )


nl = "\n"