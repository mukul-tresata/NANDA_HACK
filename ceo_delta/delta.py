"""Delta Agent.

v0.3 — AgentCard feedback loop.

After auditing a run, Delta writes a PerformanceRecord to each node's
assigned AgentCard via AgentRegistry.record_performance(). This closes the
feedback loop:

    CEO resolves agents from cards -> Kernel executes with card context
    -> Delta writes performance back to cards -> CEO resolves better next run

PerformanceRecord carries:
    - fingerprint_match, role_function_match, error (from NodeResult)
    - centrality_weight (from DAG — hub nodes matter more)
    - task_class (from Research's structured brief)

All failure and semantic metrics remain centrality-weighted (v0.2).
hub_failures list is now also used to update AgentCard.hub_failure_count
via the record (centrality_weight > 1.0 + error triggers the count).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import Config, DEFAULT
from .embeddings import cosine
from .handbook import Handbook
from .schemas import (
    AgentRegistry, DAG, ExecutionTrace, NodeResult, PerformanceRecord,
)


@dataclass
class DeltaReport:
    structural: Dict[str, float]
    runtime: Dict[str, float]
    failure: Dict[str, float]
    semantic: Dict[str, float]
    satisfaction: Dict[str, float]
    centrality: Dict[str, float]
    hub_failures: List[str]
    verdict: str
    ceo_feedback: str
    research_feedback: str
    granular_entries: List[str] = field(default_factory=list)


class Delta:
    def __init__(self, ceo_hb: Handbook, research_hb: Handbook,
                 registry: Optional[AgentRegistry] = None,
                 cfg: Config | None = None):
        self.ceo_hb = ceo_hb
        self.research_hb = research_hb
        self.registry = registry
        self.cfg = cfg or DEFAULT

    def audit(self, dag: DAG, trace: ExecutionTrace,
              brief_drift: float = 0.0,
              user_satisfaction: Optional[float] = None,
              task_class: str = "reasoning") -> DeltaReport:

        weights = dag.centrality_weights()

        structural = self._structural(dag, trace)
        runtime = self._runtime(trace)
        failure = self._failure(dag, trace, weights)
        semantic = self._semantic(dag, trace, brief_drift, weights)
        satisfaction = {"signal": user_satisfaction if user_satisfaction is not None else -1.0}
        hub_failures = self._hub_failures(dag, trace, weights)

        verdict, good = self._verdict(structural, failure, semantic, satisfaction)
        granular = self._granular(dag, trace, failure, weights, hub_failures)
        revision = self._make_revision(dag, structural, failure, verdict, hub_failures)

        # write to handbooks
        for hb in (self.ceo_hb, self.research_hb):
            hb.upsert_votes(
                task_embedding=dag.task_embedding,
                task_summary=dag.task[:120],
                topology=dag.topology, depth=dag.depth,
                outcome_good=good, revision=revision,
            )

        # write performance records to AgentCards (new)
        self._update_agent_cards(dag, trace, weights, task_class)

        return DeltaReport(
            structural=structural, runtime=runtime, failure=failure,
            semantic=semantic, satisfaction=satisfaction,
            centrality=weights, hub_failures=hub_failures,
            verdict=verdict,
            ceo_feedback=self._ceo_feedback(structural, failure, verdict, hub_failures),
            research_feedback=self._research_feedback(brief_drift, semantic),
            granular_entries=granular,
        )

    # -- AgentCard update (new) -----------------------------------------------

    def _update_agent_cards(self, dag: DAG, trace: ExecutionTrace,
                            weights: Dict[str, float],
                            task_class: str) -> None:
        """Write one PerformanceRecord per node to the assigned agent's card.

        Only fires if a registry is attached and the node has an assigned agent.
        Nodes with no assigned_agent_id were executed by the generic LLM fallback
        and don't update any card.
        """
        if not self.registry:
            return

        by_id = {r.node_id: r for r in trace.results}

        for node in dag.nodes:
            if not node.assigned_agent_id:
                continue
            result = by_id.get(node.node_id)
            if not result:
                continue

            record = PerformanceRecord(
                run_id=dag.dag_id,
                node_id=node.node_id,
                task_class=task_class,
                functional_role=node.roles.functional,
                fingerprint_match=result.fingerprint_match,
                role_function_match=result.role_function_match,
                error=result.error,
                latency_s=result.latency_s,
                cost_tokens=result.cost_tokens,
                centrality_weight=weights.get(node.node_id, 1.0),
            )
            self.registry.record_performance(node.assigned_agent_id, record)

        # update last_revision on cards that had hub failures
        hub_fail_ids = {
            dag.node(nid).assigned_agent_id
            for nid in self._hub_failures(dag, trace, weights)
            if dag.node(nid) and dag.node(nid).assigned_agent_id
        }
        for aid in hub_fail_ids:
            card = self.registry.get(aid)
            if card:
                card.dynamic.last_revision = (
                    f"hub node failure on dag={dag.dag_id} task_class={task_class}"
                )

    # -- metric categories ----------------------------------------------------

    def _structural(self, dag: DAG, trace: ExecutionTrace) -> Dict[str, float]:
        deps = sum(len(n.dependencies) for n in dag.nodes)
        roots = sum(1 for n in dag.nodes if not n.dependencies)
        fan_out = roots / max(1, len(dag.nodes))
        crit = self._critical_path(dag)
        executed = len(trace.results)
        align = executed / max(1, len(dag.nodes))
        return {
            "fan_out_ratio": round(fan_out, 3),
            "critical_path": float(crit),
            "edges": float(deps),
            "plan_exec_alignment": round(align, 3),
        }

    def _runtime(self, trace: ExecutionTrace) -> Dict[str, float]:
        lat = sorted(r.latency_s for r in trace.results) or [0.0]
        def pct(p):
            if len(lat) == 1:
                return lat[0]
            k = min(len(lat) - 1, int(round(p / 100 * (len(lat) - 1))))
            return lat[k]
        throughput = len(trace.results) / max(1e-6, trace.wallclock_s)
        return {
            "p50_latency": round(pct(50), 3),
            "p95_latency": round(pct(95), 3),
            "throughput": round(throughput, 3),
            "wallclock": round(trace.wallclock_s, 3),
        }

    def _failure(self, dag: DAG, trace: ExecutionTrace,
                 weights: Dict[str, float]) -> Dict[str, float]:
        n = len(trace.results) or 1
        errs = sum(1 for r in trace.results if r.error)
        role_miss = sum(1 for r in trace.results if not r.role_function_match)
        echo = self._echo_rate(dag, trace)
        cascade = self._cascade_rate(dag, trace)

        total_weight = sum(weights.get(r.node_id, 1.0) for r in trace.results) or 1.0
        weighted_err = sum(
            weights.get(r.node_id, 1.0) for r in trace.results if r.error
        ) / total_weight
        weighted_role_miss = sum(
            weights.get(r.node_id, 1.0) for r in trace.results if not r.role_function_match
        ) / total_weight
        weighted_echo = self._weighted_echo_rate(dag, trace, weights)
        weighted_cascade = self._weighted_cascade_rate(dag, trace, weights)

        return {
            "error_rate": round(errs / n, 3),
            "role_mismatch_rate": round(role_miss / n, 3),
            "echoing_rate": round(echo, 3),
            "cascade_rate": round(cascade, 3),
            "weighted_error_rate": round(weighted_err, 3),
            "weighted_role_mismatch_rate": round(weighted_role_miss, 3),
            "weighted_echo_rate": round(weighted_echo, 3),
            "weighted_cascade_rate": round(weighted_cascade, 3),
        }

    def _semantic(self, dag: DAG, trace: ExecutionTrace,
                  drift: float, weights: Dict[str, float]) -> Dict[str, float]:
        fps = [r.fingerprint_match for r in trace.results if r.fingerprint_match]
        fp_mean = statistics.fmean(fps) if fps else 0.0
        total_weight = sum(weights.get(r.node_id, 1.0) for r in trace.results) or 1.0
        weighted_fp = sum(
            r.fingerprint_match * weights.get(r.node_id, 1.0)
            for r in trace.results
        ) / total_weight
        why_explore = statistics.fmean(
            [1.0 if n.why.exploratory else 0.0 for n in dag.nodes]
        )
        return {
            "fingerprint_match_mean": round(fp_mean, 3),
            "weighted_fingerprint_match": round(weighted_fp, 3),
            "brief_drift": round(drift, 3),
            "why_exploratory_frac": round(why_explore, 3),
            "prior_calibration": round(1.0 - why_explore, 3),
        }

    # -- centrality helpers ---------------------------------------------------

    def _hub_failures(self, dag: DAG, trace: ExecutionTrace,
                      weights: Dict[str, float]) -> List[str]:
        deg = dag.in_degree()
        by_id = {r.node_id: r for r in trace.results}
        return [
            nid for nid, w in weights.items()
            if deg.get(nid, 0) > 0
            and by_id.get(nid) is not None
            and by_id[nid].error is not None
        ]

    def _weighted_echo_rate(self, dag: DAG, trace: ExecutionTrace,
                            weights: Dict[str, float]) -> float:
        by_id = {r.node_id: r for r in trace.results}
        nodes = dag.nodes
        total_w, echo_w = 0.0, 0.0
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if set(nodes[i].dependencies) == set(nodes[j].dependencies):
                    a = by_id.get(nodes[i].node_id)
                    b = by_id.get(nodes[j].node_id)
                    if a and b:
                        pair_w = max(
                            weights.get(nodes[i].node_id, 1.0),
                            weights.get(nodes[j].node_id, 1.0),
                        )
                        total_w += pair_w
                        if cosine(a.output_embedding, b.output_embedding) > self.cfg.echo_cosine_threshold:
                            echo_w += pair_w
        return echo_w / total_w if total_w > 0 else 0.0

    def _weighted_cascade_rate(self, dag: DAG, trace: ExecutionTrace,
                               weights: Dict[str, float]) -> float:
        by_id = {r.node_id: r for r in trace.results}
        errored = {nid for nid, r in by_id.items() if r.error}
        if not errored:
            return 0.0
        total_w = sum(weights.get(nid, 1.0) for nid in errored) or 1.0
        propagated_w = 0.0
        for n in dag.nodes:
            r = by_id.get(n.node_id)
            if r and r.error:
                errored_parents = [d for d in n.dependencies if d in errored]
                if errored_parents:
                    propagated_w += max(weights.get(p, 1.0) for p in errored_parents)
        return propagated_w / total_w

    def _echo_rate(self, dag: DAG, trace: ExecutionTrace) -> float:
        by_id = {r.node_id: r for r in trace.results}
        sibs: list = []
        nodes = dag.nodes
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if set(nodes[i].dependencies) == set(nodes[j].dependencies):
                    a, b = by_id.get(nodes[i].node_id), by_id.get(nodes[j].node_id)
                    if a and b:
                        sibs.append((a, b))
        if not sibs:
            return 0.0
        echoes = sum(
            1 for a, b in sibs
            if cosine(a.output_embedding, b.output_embedding) > self.cfg.echo_cosine_threshold
        )
        return echoes / len(sibs)

    def _cascade_rate(self, dag: DAG, trace: ExecutionTrace) -> float:
        by_id = {r.node_id: r for r in trace.results}
        errored = {nid for nid, r in by_id.items() if r.error}
        if not errored:
            return 0.0
        propagated = 0
        for n in dag.nodes:
            r = by_id.get(n.node_id)
            if r and r.error and any(d in errored for d in n.dependencies):
                propagated += 1
        return propagated / len(errored)

    def _critical_path(self, dag: DAG) -> int:
        memo: Dict[str, int] = {}
        def depth(nid: str) -> int:
            if nid in memo:
                return memo[nid]
            node = dag.node(nid)
            if not node or not node.dependencies:
                memo[nid] = 1
            else:
                memo[nid] = 1 + max(depth(d) for d in node.dependencies)
            return memo[nid]
        return max((depth(n.node_id) for n in dag.nodes), default=0)

    def _verdict(self, structural, failure, semantic, satisfaction):
        score = 1.0
        score -= failure["weighted_error_rate"] * 0.4
        score -= failure["weighted_echo_rate"] * 0.3
        score -= failure["weighted_cascade_rate"] * 0.3
        score -= failure["weighted_role_mismatch_rate"] * 0.1
        score += (semantic["weighted_fingerprint_match"] - 0.3) * 0.3
        if satisfaction["signal"] >= 0:
            score = 0.6 * score + 0.4 * satisfaction["signal"]
        good = score >= 0.55
        verdict = "good" if good else ("mixed" if score >= 0.4 else "poor")
        return verdict, good

    def _granular(self, dag: DAG, trace: ExecutionTrace, failure,
                  weights: Dict[str, float], hub_failures: List[str]) -> List[str]:
        out: List[str] = ["per-run entry written"]
        costs = [r.cost_tokens for r in trace.results] or [1]
        planned = statistics.fmean(costs)
        for r in trace.results:
            if r.cost_tokens > self.cfg.surprise_factor * planned:
                w = weights.get(r.node_id, 1.0)
                out.append(
                    f"per-node entry: {r.node_id} cost {r.cost_tokens} >> {planned:.0f} "
                    f"(centrality={w:.2f})"
                )
        if failure["role_mismatch_rate"] > 0:
            out.append("per-role entry: role-function mismatch detected")
        if failure["cascade_rate"] > 0:
            out.append("compound entry: error cascade captured as a unit")
        if hub_failures:
            out.append(f"hub-failure entry: {hub_failures} — structurally critical nodes failed")
        return out

    def _make_revision(self, dag, structural, failure, verdict, hub_failures) -> str:
        if verdict == "good" and not hub_failures:
            return f"keep topo={dag.topology} depth={dag.depth} (clean run)"
        tips = []
        if hub_failures:
            tips.append(f"hub nodes {hub_failures} failed — add verification upstream of each hub")
        if failure["weighted_echo_rate"] > 0.3:
            tips.append("high weighted echo — differentiate sibling intents or reduce fan-out")
        if failure["weighted_cascade_rate"] > 0:
            tips.append("hub-origin cascade — isolate hub nodes with guard nodes")
        if structural["critical_path"] > dag.depth + 1:
            tips.append("critical path longer than planned depth; flatten plan")
        if not tips:
            tips.append("outcome weak; try alternative topology next time")
        return f"revise topo={dag.topology} depth={dag.depth}: " + "; ".join(tips)

    def _ceo_feedback(self, structural, failure, verdict, hub_failures) -> str:
        hub_note = f" HUB_FAILURES={hub_failures}" if hub_failures else ""
        return (
            f"[{verdict}] align={structural['plan_exec_alignment']} "
            f"w_echo={failure['weighted_echo_rate']} "
            f"w_cascade={failure['weighted_cascade_rate']} "
            f"w_role_miss={failure['weighted_role_mismatch_rate']}"
            f"{hub_note}"
        )

    def _research_feedback(self, drift, semantic) -> str:
        flag = "HIGH" if drift > 0.3 else "ok"
        return (
            f"brief drift={drift:.2f} [{flag}] "
            f"fp={semantic['fingerprint_match_mean']} "
            f"w_fp={semantic['weighted_fingerprint_match']}"
        )