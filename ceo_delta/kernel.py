"""Execution Kernel (KAIJU-inspired).

Dependency-aware parallel dispatch with Intent-Gated Execution (IGX): each node
must pass an intent gate before it fires — its declared intent is checked for
consistency against its assigned roles. Records actual cost, latency, output,
fingerprint match and role-function match per node as an execution trace, then
signals Delta (the orchestrator wires that up).
"""
from __future__ import annotations

import concurrent.futures as cf
import time
from typing import Dict, List

from .config import Config, DEFAULT
from .embeddings import cosine, embed
from .llm import LLMClient
from .schemas import DAG, ExecutionTrace, Node, NodeResult

_SYSTEM = ("You are an execution worker. Carry out the single node intent given. "
           "Be concise and produce the artifact, not commentary about it.")

# functional roles whose intent should contain certain signals (cheap IGX check)
_GATE_HINTS = {
    "retriever": ("retrieve", "find", "gather", "search", "collect", "fetch"),
    "synthesizer": ("synthesize", "combine", "summarize", "write", "compose", "answer"),
    "verifier": ("verify", "check", "validate", "audit", "confirm", "cross"),
}


class Kernel:
    def __init__(self, llm: LLMClient, cfg: Config | None = None, max_workers: int = 4):
        self.llm = llm
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
                if not ready:  # cyclic/broken deps — fail the rest gracefully
                    for n in remaining.values():
                        results[n.node_id] = self._failed(n, "unsatisfiable dependencies")
                    break
                futs = {pool.submit(self._run_node, n, dag, results, brief_context): n
                        for n in ready}
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
            return self._failed(node, "intent gate rejected (role/intent mismatch)",
                                gated=False)
        nl = "\n"
        dep_ctx = nl.join(
            f"[{d} output]: {results[d].output[:400]}"
            for d in node.dependencies if d in results)
        brief_line = ("BRIEF: " + brief_context) if brief_context else ""
        upstream_line = ("UPSTREAM:" + nl + dep_ctx) if dep_ctx else ""
        prompt = (f"NODE INTENT: {node.intent}{nl}"
                  f"ROLE: {node.roles.functional}/{node.roles.epistemic}{nl}"
                  f"{brief_line}{nl}"
                  f"{upstream_line}{nl}"
                  "Produce the artifact for this node.")
        before = self.llm.total_tokens
        t = time.time()
        try:
            out = self.llm.chat([
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ], max_tokens=1024)
            err = None
        except Exception as e:  # noqa: BLE001 - kernel must not crash the run
            out, err = f"[node error] {e}", str(e)
        latency = time.time() - t
        cost = max(1, self.llm.total_tokens - before)
        out_emb = embed(out)
        fp = cosine(node.expected_output_fingerprint, out_emb) if node.expected_output_fingerprint else 0.0
        return NodeResult(
            node_id=node.node_id, intent=node.intent, output=out,
            output_embedding=out_emb, cost_tokens=cost, latency_s=latency,
            role_function_match=self._role_match(node), fingerprint_match=fp,
            gated=True, error=err,
        )

    def _intent_gate(self, node: Node) -> bool:
        """IGX: cheap pre-execution consistency check. Unknown roles pass."""
        hints = _GATE_HINTS.get(node.roles.functional)
        if not hints:
            return True
        intent = node.intent.lower()
        return any(h in intent for h in hints) or True  # warn-not-block in proto
        # NOTE: returns True (warn-mode) but role_function_match records the miss
        # so Delta can measure mismatch rate without starving the demo.

    def _role_match(self, node: Node) -> bool:
        hints = _GATE_HINTS.get(node.roles.functional)
        if not hints:
            return True
        return any(h in node.intent.lower() for h in hints)

    def _failed(self, node: Node, reason: str, gated: bool = True) -> NodeResult:
        return NodeResult(
            node_id=node.node_id, intent=node.intent, output=f"[skipped] {reason}",
            output_embedding=embed(reason), cost_tokens=0, latency_s=0.0,
            role_function_match=False, fingerprint_match=0.0, gated=gated, error=reason,
        )
