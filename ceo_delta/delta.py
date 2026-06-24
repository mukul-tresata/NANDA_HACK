"""Delta Agent.

Audits an execution trace against the plan, computes the metric categories, and
writes learnings to both handbooks. Emits separate feedback signals for CEO and
Research. Handbook entry granularity is error-magnitude driven.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import Config, DEFAULT
from .embeddings import cosine
from .handbook import Handbook
from .schemas import DAG, ExecutionTrace, NodeResult


@dataclass
class DeltaReport:
    structural: Dict[str, float]
    runtime: Dict[str, float]
    failure: Dict[str, float]
    semantic: Dict[str, float]
    satisfaction: Dict[str, float]
    verdict: str                      # good / mixed / poor
    ceo_feedback: str
    research_feedback: str
    granular_entries: List[str] = field(default_factory=list)


class Delta:
    def __init__(self, ceo_hb: Handbook, research_hb: Handbook, cfg: Config | None = None):
        self.ceo_hb = ceo_hb
        self.research_hb = research_hb
        self.cfg = cfg or DEFAULT

    def audit(self, dag: DAG, trace: ExecutionTrace,
              brief_drift: float = 0.0, user_satisfaction: Optional[float] = None
              ) -> DeltaReport:
        structural = self._structural(dag, trace)
        runtime = self._runtime(trace)
        failure = self._failure(dag, trace)
        semantic = self._semantic(dag, trace, brief_drift)
        satisfaction = {"signal": user_satisfaction if user_satisfaction is not None else -1.0}

        verdict, good = self._verdict(structural, failure, semantic, satisfaction)
        granular = self._granular(dag, trace, failure)

        revision = self._make_revision(dag, structural, failure, verdict)
        # write to BOTH handbooks (CEO + Research), as votes (multi-way safe)
        for hb in (self.ceo_hb, self.research_hb):
            hb.upsert_votes(
                task_embedding=dag.task_embedding,
                task_summary=dag.task[:120],
                topology=dag.topology, depth=dag.depth,
                outcome_good=good, revision=revision,
            )
        return DeltaReport(
            structural=structural, runtime=runtime, failure=failure,
            semantic=semantic, satisfaction=satisfaction, verdict=verdict,
            ceo_feedback=self._ceo_feedback(structural, failure, verdict),
            research_feedback=self._research_feedback(brief_drift, semantic),
            granular_entries=granular,
        )

    # -- metric categories ----------------------------------------------------
    def _structural(self, dag: DAG, trace: ExecutionTrace) -> Dict[str, float]:
        deps = sum(len(n.dependencies) for n in dag.nodes)
        roots = sum(1 for n in dag.nodes if not n.dependencies)
        fan_out = roots / max(1, len(dag.nodes))
        crit = self._critical_path(dag)
        executed = len(trace.results)
        align = executed / max(1, len(dag.nodes))   # plan-to-execution alignment
        return {"fan_out_ratio": round(fan_out, 3), "critical_path": float(crit),
                "edges": float(deps), "plan_exec_alignment": round(align, 3)}

    def _runtime(self, trace: ExecutionTrace) -> Dict[str, float]:
        lat = sorted(r.latency_s for r in trace.results) or [0.0]
        def pct(p):
            if len(lat) == 1:
                return lat[0]
            k = min(len(lat) - 1, int(round(p / 100 * (len(lat) - 1))))
            return lat[k]
        throughput = len(trace.results) / max(1e-6, trace.wallclock_s)
        return {"p50_latency": round(pct(50), 3), "p95_latency": round(pct(95), 3),
                "throughput": round(throughput, 3), "wallclock": round(trace.wallclock_s, 3)}

    def _failure(self, dag: DAG, trace: ExecutionTrace) -> Dict[str, float]:
        n = len(trace.results) or 1
        errs = sum(1 for r in trace.results if r.error)
        role_miss = sum(1 for r in trace.results if not r.role_function_match)
        echo = self._echo_rate(dag, trace)
        cascade = self._cascade_rate(dag, trace)
        return {"error_rate": round(errs / n, 3),
                "role_mismatch_rate": round(role_miss / n, 3),
                "echoing_rate": round(echo, 3),
                "cascade_rate": round(cascade, 3)}

    def _semantic(self, dag: DAG, trace: ExecutionTrace, drift: float) -> Dict[str, float]:
        fps = [r.fingerprint_match for r in trace.results if r.fingerprint_match]
        fp_mean = statistics.fmean(fps) if fps else 0.0
        # WHY stability proxy: how exploratory was the plan (1 == fully exploratory)
        why_explore = statistics.fmean([1.0 if n.why.exploratory else 0.0 for n in dag.nodes])
        return {"fingerprint_match_mean": round(fp_mean, 3),
                "brief_drift": round(drift, 3),
                "why_exploratory_frac": round(why_explore, 3),
                "prior_calibration": round(1.0 - why_explore, 3)}

    # -- helpers --------------------------------------------------------------
    def _echo_rate(self, dag: DAG, trace: ExecutionTrace) -> float:
        """Fraction of sibling pairs (shared deps) whose outputs are near-duplicate."""
        by_id = {r.node_id: r for r in trace.results}
        sibs: List[tuple] = []
        nodes = dag.nodes
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if set(nodes[i].dependencies) == set(nodes[j].dependencies):
                    a, b = by_id.get(nodes[i].node_id), by_id.get(nodes[j].node_id)
                    if a and b:
                        sibs.append((a, b))
        if not sibs:
            return 0.0
        echoes = sum(1 for a, b in sibs
                     if cosine(a.output_embedding, b.output_embedding) > self.cfg.echo_cosine_threshold)
        return echoes / len(sibs)

    def _cascade_rate(self, dag: DAG, trace: ExecutionTrace) -> float:
        """Errors that have at least one errored upstream dependency."""
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
        score -= failure["error_rate"] * 0.4
        score -= failure["echoing_rate"] * 0.3
        score -= failure["cascade_rate"] * 0.3
        score -= failure["role_mismatch_rate"] * 0.1
        score += (semantic["fingerprint_match_mean"] - 0.3) * 0.3
        if satisfaction["signal"] >= 0:
            score = 0.6 * score + 0.4 * satisfaction["signal"]
        good = score >= 0.55
        verdict = "good" if good else ("mixed" if score >= 0.4 else "poor")
        return verdict, good

    def _granular(self, dag: DAG, trace: ExecutionTrace, failure) -> List[str]:
        out: List[str] = ["per-run entry written"]
        costs = [r.cost_tokens for r in trace.results] or [1]
        planned = statistics.fmean(costs)
        for r in trace.results:
            if r.cost_tokens > self.cfg.surprise_factor * planned:
                out.append(f"per-node entry: {r.node_id} cost {r.cost_tokens} >> {planned:.0f}")
        if failure["role_mismatch_rate"] > 0:
            out.append("per-role entry: role-function mismatch detected")
        if failure["cascade_rate"] > 0:
            out.append("compound entry: error cascade captured as a unit")
        return out

    def _make_revision(self, dag, structural, failure, verdict) -> str:
        if verdict == "good":
            return f"keep topo={dag.topology} depth={dag.depth} (clean run)"
        tips = []
        if failure["echoing_rate"] > 0.3:
            tips.append("reduce sibling fan-out or differentiate sibling intents (echoing)")
        if failure["cascade_rate"] > 0:
            tips.append("add verification node before hub to stop cascade")
        if structural["critical_path"] > dag.depth + 1:
            tips.append("critical path longer than planned depth; flatten plan")
        if not tips:
            tips.append("outcome weak; try alternative topology next time")
        return f"revise topo={dag.topology} depth={dag.depth}: " + "; ".join(tips)

    def _ceo_feedback(self, structural, failure, verdict) -> str:
        return (f"[{verdict}] align={structural['plan_exec_alignment']} "
                f"echo={failure['echoing_rate']} cascade={failure['cascade_rate']} "
                f"role_miss={failure['role_mismatch_rate']}")

    def _research_feedback(self, drift, semantic) -> str:
        flag = "HIGH" if drift > 0.3 else "ok"
        return f"brief drift={drift:.2f} [{flag}] fingerprint_fit={semantic['fingerprint_match_mean']}"
