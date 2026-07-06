"""Delta Agent — v2.1, tensor error.

Replaces the scalar Δe with the 5-dim ErrorTensor (drift, echo, cascade,
role, resource). No weighted collapse anywhere -- directive construction
matches on the unreduced vector via gates -> dedup -> hysteresis ->
magnitude ranking -> escalation-store retrieval (no more hand-written
delta_rules.py table).

Fallback: set cfg.use_tensor_error=False to run the old scalar path
unmodified (kept intact below) -- safety net while the new path proves
itself on real runs.
"""
from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import Config, DEFAULT
from .embeddings import cosine
from .error_tensor import (
    ErrorTensor, HysteresisTracker, apply_gates,
    cluster_primary_secondary, compute_cascade, compute_drift, compute_echo,
    compute_resource, compute_role, rank_violations,
)
from .escalation import EscalationCase, EscalationStore, _case_id
from .escalation import build_escalation_prompt, parse_escalation_response
from .bootstrap import ensure_escalations_seeded
from .handbook import Handbook
from .ef import compute_required, compute_ef, verdict_from_ef, EF_AXES
from .ef_store import EFStore
from .descent import Descent
from .schemas import (
    AgentRegistry, DAG, DeltaDirective, ExecutionTrace, NodeResult,
    PerformanceRecord, TaskFingerprint,
)

try:
    from .config import ROLE_BANDS
except ImportError:
    ROLE_BANDS = {"generic": {"cite": (0, None), "comp": (0, None), "struct": (0, None),
                               "_weights": {"cite": 0, "comp": 0, "struct": 0}}}

# legacy scalar weights -- kept for fallback path only
_W1_FP = 0.4
_W2_MISMATCH = 0.3
_W3_ECHO = 0.3


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
    delta_e: float = 0.0                       # legacy scalar (fallback path)
    error_tensor: Optional[Dict[str, float]] = None  # v2.1 vector
    granular_entries: List[str] = field(default_factory=list)
    ef_tensor: Optional[Dict[str, float]] = None     # v3.0 [partition,flow,role,scale]
    required_structure: Optional[Dict] = None        # v3.0 F, for display
    q_tensor: Optional[Dict[str, float]] = None      # v3.5 [groundedness,relevance] content quality

    def comparison_key(self) -> Tuple:
        """For Pareto-style best-answer selection (orchestrator). v3.0 uses
        the EF tensor; falls back to the v2.1 tensor, then scalar delta_e.
        Keys are read in a stable sorted order so dominance zips correctly."""
        t = self.ef_tensor or self.error_tensor
        if t:
            return tuple(t[k] for k in sorted(t.keys()))
        return (self.delta_e,)


def dominates(a: "DeltaReport", b: "DeltaReport") -> bool:
    """True if report a is no worse than b on every dimension and strictly
    better on at least one. Falls back to scalar comparison if no tensor."""
    ta = a.ef_tensor or a.error_tensor
    tb = b.ef_tensor or b.error_tensor
    if not ta or not tb:
        return a.delta_e < b.delta_e
    ka, kb = a.comparison_key(), b.comparison_key()
    no_worse = all(x <= y for x, y in zip(ka, kb))
    strictly_better = any(x < y for x, y in zip(ka, kb))
    return no_worse and strictly_better


class Delta:
    def __init__(
        self,
        hb: Handbook,
        registry: Optional[AgentRegistry] = None,
        cfg: Config | None = None,
        escalation_log_path: str = ".ceo_delta/escalations.jsonl",
        llm=None,  # LLMClient, needed for escalation calls in tensor path
    ):
        self.hb = hb
        self.registry = registry
        self.cfg = cfg or DEFAULT
        self.escalation_log_path = escalation_log_path
        self._surface_threshold = self.cfg.delta_surface_threshold
        self.llm = llm

        self.use_tensor = getattr(self.cfg, "use_tensor_error", True)
        self.role_bands = ROLE_BANDS
        self.thresholds = {
            "drift": getattr(self.cfg, "error_drift_threshold", 0.40),
            "echo": getattr(self.cfg, "error_echo_threshold", 0.45),
            "cascade": getattr(self.cfg, "error_cascade_threshold", 0.30),
            "role": getattr(self.cfg, "error_role_threshold", 0.35),
            "resource": 1.0,  # gate handles this separately, not via rank
        }
        self.escalations = EscalationStore(escalation_log_path)
        ensure_escalations_seeded(self.escalations)
        self.hysteresis = HysteresisTracker(getattr(self.cfg, "hysteresis_margin", 0.15))

        # v3.0 EF-driven adaptation: F-keyed trajectory store + per-run descent.
        self.use_ef = getattr(self.cfg, "use_ef_tensor", True)
        ef_path = os.path.join(os.path.dirname(escalation_log_path) or ".", "ef_cases.json")
        self.ef_store = EFStore(ef_path)
        self._descent: Optional[Descent] = None

    # -------------------------------------------------------------------
    # audit() -- unchanged signature, branches internally on use_tensor
    # -------------------------------------------------------------------

    def audit(
        self,
        dag: DAG,
        trace: ExecutionTrace,
        fingerprint: Optional[TaskFingerprint] = None,
        brief_drift: float = 0.0,
        user_satisfaction: Optional[float] = None,
        task_class: str = "reasoning",
        iteration: int = 0,
        prev_delta_e: Optional[float] = None,
        prev_directive: Optional[DeltaDirective] = None,
    ) -> Tuple[DeltaReport, DeltaDirective]:

        hb_key = (
            fingerprint.embedding if (fingerprint and fingerprint.embedding)
            else dag.task_embedding
        )
        weights = dag.centrality_weights()

        structural = self._structural(dag, trace)
        runtime = self._runtime(trace)
        failure = self._failure(dag, trace, weights)  # legacy metrics, still used for verdict
        semantic = self._semantic(dag, trace, brief_drift, weights)
        satisfaction = {"signal": user_satisfaction if user_satisfaction is not None else -1.0}
        hub_failures = self._hub_failures(dag, trace, weights)

        # -- v3.0 EF path: one error model, verdict derived from E ---------
        if self.use_ef and self._descent is not None:
            ef = compute_ef(dag, trace, self._descent.req, weights, self.role_bands)
            verdict, good = verdict_from_ef(
                ef, self._descent.thresholds(), self.cfg.ef_mixed_margin
            )
            granular = self._granular(dag, trace, failure, weights, hub_failures)
            revision = self._make_revision(dag, structural, failure, verdict, hub_failures)
            # keep vote-based handbook fed for prior_calibration display
            self.hb.upsert_votes(
                task_embedding=hb_key, task_summary=dag.task[:120],
                topology=dag.topology, depth=dag.depth,
                outcome_good=good, revision=revision,
            )
            directive = self._descent.step(dag, trace, ef, iteration)
            self._update_agent_cards(dag, trace, weights, task_class)
            ef_d = ef.as_dict()
            report = DeltaReport(
                structural=structural, runtime=runtime, failure=failure,
                semantic=semantic, satisfaction=satisfaction, centrality=weights,
                hub_failures=hub_failures, verdict=verdict,
                ceo_feedback=self._ceo_feedback_ef(ef_d, verdict),
                research_feedback=self._research_feedback(brief_drift, semantic),
                delta_e=round(max(ef_d.values()), 4),   # worst-axis, for display/tiebreak
                error_tensor=ef_d, ef_tensor=ef_d,
                required_structure=self._descent.req.as_dict(),
                granular_entries=granular,
            )
            return report, directive

        verdict, good = self._verdict(structural, failure, semantic, satisfaction)
        granular = self._granular(dag, trace, failure, weights, hub_failures)
        revision = self._make_revision(dag, structural, failure, verdict, hub_failures)

        self.hb.upsert_votes(
            task_embedding=hb_key, task_summary=dag.task[:120],
            topology=dag.topology, depth=dag.depth,
            outcome_good=good, revision=revision,
        )

        if self.use_tensor:
            tensor = self._compute_tensor(dag, trace, weights, runtime)
            directive = self._generate_directive_tensor(
                dag, trace, tensor, fingerprint, iteration
            )
            delta_e_legacy = self._error_vector(failure, semantic)  # kept for outcome tracking continuity
            if prev_directive and prev_delta_e is not None:
                self._record_directive_outcome(hb_key, prev_directive, prev_delta_e, delta_e_legacy)
            self._update_agent_cards(dag, trace, weights, task_class)

            report = DeltaReport(
                structural=structural, runtime=runtime, failure=failure,
                semantic=semantic, satisfaction=satisfaction, centrality=weights,
                hub_failures=hub_failures, verdict=verdict,
                ceo_feedback=self._ceo_feedback(structural, failure, verdict, hub_failures),
                research_feedback=self._research_feedback(brief_drift, semantic),
                delta_e=round(delta_e_legacy, 4),
                error_tensor=tensor.as_dict(),
                granular_entries=granular,
            )
            return report, directive

        # -- legacy scalar path (fallback) -----------------------------
        delta_e = self._error_vector(failure, semantic)
        if prev_directive and prev_delta_e is not None:
            self._record_directive_outcome(hb_key, prev_directive, prev_delta_e, delta_e)
        self._update_agent_cards(dag, trace, weights, task_class)
        directive = self._generate_directive_legacy(
            failure, semantic, structural, dag.topology, dag.depth,
            delta_e, verdict, hub_failures, iteration,
        )
        report = DeltaReport(
            structural=structural, runtime=runtime, failure=failure,
            semantic=semantic, satisfaction=satisfaction, centrality=weights,
            hub_failures=hub_failures, verdict=verdict,
            ceo_feedback=self._ceo_feedback(structural, failure, verdict, hub_failures),
            research_feedback=self._research_feedback(brief_drift, semantic),
            delta_e=round(delta_e, 4),
            granular_entries=granular,
        )
        return report, directive

    # -------------------------------------------------------------------
    # v3.0 EF run lifecycle
    # -------------------------------------------------------------------

    def begin_run(self, fingerprint) -> None:
        """Open a fresh coordinate-descent for this task. Loads the F-keyed
        move memory (learned + escalation moves) for this fingerprint class."""
        if not self.use_ef:
            return
        required = compute_required(fingerprint)
        self._descent = Descent(
            fingerprint, required, self.ef_store, self.cfg, self.llm, self.role_bands
        )

    def end_run(self, best_dag) -> None:
        """Persist the convergence trajectory for this fingerprint class."""
        if self.use_ef and self._descent is not None:
            self._descent.finalize(best_dag)
            self.ef_store.save()

    def _ceo_feedback_ef(self, ef_d: Dict[str, float], verdict: str) -> str:
        return (
            f"[{verdict}] partition={ef_d['partition']} flow={ef_d['flow']} "
            f"role={ef_d['role']} scale={ef_d['scale']}"
        )

    # -------------------------------------------------------------------
    # v2.1 tensor computation
    # -------------------------------------------------------------------

    def _compute_tensor(self, dag, trace, weights, runtime) -> ErrorTensor:
        drift, node_drift = compute_drift(dag, trace, weights)
        echo = compute_echo(dag, trace, weights, self.cfg.echo_cosine_threshold)
        cascade = compute_cascade(dag, trace, weights, node_drift)
        role, node_role_excess = compute_role(dag, trace, weights, self.role_bands)
        resource = compute_resource(
            trace.total_tokens, trace.wallclock_s,
            getattr(self.cfg, "error_resource_cost_max_usd", 0.5),
            getattr(self.cfg, "error_resource_latency_max_s", 180.0),
        )
        tensor = ErrorTensor(
            drift=drift, echo=echo, cascade=cascade, role=role, resource=resource,
            node_role_excess=node_role_excess, node_drift=node_drift,
        )
        apply_gates(
            tensor,
            cost_max=getattr(self.cfg, "error_resource_cost_max_usd", 0.5),
            latency_max=getattr(self.cfg, "error_resource_latency_max_s", 180.0),
            cascade_threshold=self.thresholds["cascade"],
            drift_node_threshold=getattr(self.cfg, "cascade_drift_dedup_node_threshold", 0.35),
        )
        return tensor

    # -------------------------------------------------------------------
    # v2.1 directive construction
    # -------------------------------------------------------------------

    def _generate_directive_tensor(
        self, dag: DAG, trace: ExecutionTrace, tensor: ErrorTensor,
        fingerprint: Optional[TaskFingerprint], iteration: int,
    ) -> DeltaDirective:

        # 1. resource hard gate -- checked first, standalone
        if tensor.resource > 1.0:  # already normalized: >1.0 means over ceiling
            self.hysteresis.advance_iteration()
            return DeltaDirective(
                action="replan", reason="budget_exceeded",
                replan_hint=f"resource budget exceeded (excess={tensor.resource:.3f}); simplify topology to cut cost/latency",
                confidence=1.0, iteration=iteration, escalated=False,
            )

        # 2. rank remaining real violations (post gate/dedup), filtered by hysteresis
        primary_node = self._worst_node(tensor)
        violations = rank_violations(
            tensor, self.thresholds, hysteresis=self.hysteresis, primary_node=primary_node
        )
        self.hysteresis.advance_iteration()

        if not violations:
            return DeltaDirective(
                action="surface",
                reason=f"all tensor dims below threshold — output quality sufficient",
                confidence=1.0, iteration=iteration,
            )

        primary, secondary = cluster_primary_secondary(
            violations, getattr(self.cfg, "directive_cluster_ratio", 0.6)
        )

        # 3. retrieve escalation precedent for the primary-violated dimension's
        #    implicated node/role
        node_id, role = self._implicated_node(dag, trace, primary.dim, tensor)
        excess_dict = tensor.as_dict()
        case, sim = self.escalations.best_match(
            excess_dict, role,
            min_cases=getattr(self.cfg, "escalation_min_cases_per_role", 5),
            sim_floor=getattr(self.cfg, "escalation_sim_floor", 0.6),
        )

        if case:
            self.hysteresis.mark_refined(node_id, primary.dim)
            constraint = case.directive_issued.get("constraint", "")
            hint = self._render_directive_hint(primary, secondary, constraint)
            return DeltaDirective(
                action=case.directive_issued.get("action", "refine"),
                reason=f"matched_case:{case.case_id}",
                replan_hint=hint,
                refinement_targets=[node_id] if node_id else [],
                confidence=round(sim, 3),
                iteration=iteration,
                escalated=False,
                primary_dim=primary.dim,
            )

        # 4. no precedent -- LLM escalation, build directive + new case
        return self._escalate_tensor(
            dag, trace, tensor, fingerprint, node_id, role, primary, secondary, sim, iteration
        )

    def _render_directive_hint(self, primary, secondary, constraint: str) -> str:
        parts = [f"PRIMARY: {primary.dim} violation (excess={primary.excess:.3f}) — {constraint}"]
        for s in secondary:
            parts.append(f"SECONDARY: also address {s.dim} (excess={s.excess:.3f})")
        return " | ".join(parts)

    def _worst_node(self, tensor: ErrorTensor) -> Optional[str]:
        if tensor.node_role_excess:
            return max(
                tensor.node_role_excess,
                key=lambda nid: sum(tensor.node_role_excess[nid].values()),
            )
        if tensor.node_drift:
            return max(tensor.node_drift, key=tensor.node_drift.get)
        return None

    def _implicated_node(self, dag, trace, dim: str, tensor: ErrorTensor) -> Tuple[str, str]:
        node_id = None
        if dim == "role" and tensor.node_role_excess:
            node_id = max(tensor.node_role_excess, key=lambda n: sum(tensor.node_role_excess[n].values()))
        elif dim == "drift" and tensor.node_drift:
            node_id = max(tensor.node_drift, key=tensor.node_drift.get)
        else:
            node_id = dag.nodes[0].node_id if dag.nodes else None
        node = dag.node(node_id) if node_id else None
        role = node.roles.functional if node else "generic"
        return node_id or "", role

    # -- LLM escalation (tensor path) ---------------------------------------

    def _escalate_tensor(
        self, dag, trace, tensor, fingerprint, node_id, role, primary, secondary, sim, iteration,
    ) -> DeltaDirective:
        node = dag.node(node_id) if node_id else None
        by_id = {r.node_id: r for r in trace.results}
        result = by_id.get(node_id)
        output = result.output if result else ""
        intent = node.intent if node else ""
        deps = node.dependencies if node else []

        excess_dict = tensor.as_dict()
        case_data, err = None, None
        if self.llm is not None and fingerprint is not None:
            prompt = build_escalation_prompt(
                fingerprint, node_id, role, deps, output, intent, excess_dict, sim
            )
            try:
                raw = self.llm.complete(prompt)
                case_data, err = parse_escalation_response(raw)
            except Exception as e:
                err = f"llm_call_failed: {e}"

        if case_data is None:
            # schema/LLM failure -- log meta-failure, fall back to generic replan
            self._log_schema_violation(err or "llm_unavailable", excess_dict)
            directive_payload = {
                "action": "replan", "target": node_id or "topology",
                "constraint": "reconsider topology and node decomposition from first principles",
                "why": f"escalation schema/LLM failure: {err}",
            }
            case_summary = "schema_violation_fallback"
        else:
            directive_payload = case_data["directive"]
            case_summary = case_data.get("case_summary", "")

        new_case = EscalationCase(
            case_id=_case_id(), timestamp=__import__("time").time(),
            run_id=dag.dag_id,
            task_species={
                "flow": fingerprint.information_flow if fingerprint else "",
                "epistemic_stance": fingerprint.epistemic_stance if fingerprint else "",
                "output_contract": fingerprint.output_contract if fingerprint else "",
                "complexity": fingerprint.complexity if fingerprint else "",
            },
            node_context={"node_id": node_id or "", "role": role, "deps": str(deps)},
            e_tensor_raw=excess_dict,
            e_tensor_excess=excess_dict,
            gates_applied={"resource_gate_fired": False, "cascade_drift_dedup_applied": False},
            directive_issued=directive_payload,
            llm_escalation_meta={"raw_response_valid_schema": case_data is not None},
            case_summary=case_summary,
        )
        self.escalations.append(new_case)
        self.hysteresis.mark_refined(node_id, primary.dim)

        hint = self._render_directive_hint(primary, secondary, directive_payload.get("constraint", ""))
        return DeltaDirective(
            action=directive_payload.get("action", "replan"),
            reason=f"escalated_new_case:{new_case.case_id}",
            replan_hint=hint,
            refinement_targets=[node_id] if node_id else [],
            confidence=0.5, iteration=iteration, escalated=True,
            primary_dim=primary.dim,
        )

    def _log_schema_violation(self, reason: str, excess: Dict) -> None:
        try:
            import os
            os.makedirs(".ceo_delta", exist_ok=True)
            with open(".ceo_delta/schema_violations.jsonl", "a") as f:
                f.write(json.dumps({"reason": reason, "excess": excess}) + "\n")
        except Exception:
            pass

    def backfill_last_escalation(self, case_id: str, dim: str, new_excess: float,
                                  collateral: Optional[Dict[str, float]] = None) -> None:
        """Call from orchestrator after the NEXT iteration's tensor is known,
        passing reason string parsed out of prev_directive.reason."""
        self.escalations.backfill_outcome(case_id, dim, new_excess, collateral)

    # -------------------------------------------------------------------
    # legacy scalar path (fallback, cfg.use_tensor_error=False)
    # -------------------------------------------------------------------

    def _error_vector(self, failure: Dict, semantic: Dict) -> float:
        fp_miss = 1.0 - semantic["weighted_fingerprint_match"]
        mismatch = failure["weighted_role_mismatch_rate"]
        echo = failure["weighted_echo_rate"]
        return _W1_FP * fp_miss + _W2_MISMATCH * mismatch + _W3_ECHO * echo

    def _generate_directive_legacy(
        self, failure, semantic, structural, topology, depth,
        delta_e, verdict, hub_failures, iteration,
    ) -> DeltaDirective:
        from .delta_rules import lookup_hint
        hint = lookup_hint(failure, semantic, structural, topology, depth)
        if hint:
            action = self._action_from_hint(failure, semantic, delta_e)
            targets = self._refinement_targets(failure, structural)
            return DeltaDirective(
                action=action, reason=self._reason_slug(failure, semantic),
                replan_hint=hint, refinement_targets=targets,
                confidence=min(1.0, delta_e), iteration=iteration, escalated=False,
            )
        if delta_e < self._surface_threshold and verdict != "poor":
            return DeltaDirective(
                action="surface",
                reason=f"Δe={delta_e:.3f} below threshold — output quality sufficient",
                confidence=1.0 - delta_e, iteration=iteration,
            )
        return self._escalate_legacy(failure, semantic, structural, topology, depth, delta_e, verdict, iteration)

    def _action_from_hint(self, failure, semantic, delta_e) -> str:
        if failure["weighted_cascade_rate"] > 0.1:
            return "replan"
        if failure["weighted_echo_rate"] > 0.3:
            return "replan"
        if failure["weighted_error_rate"] > 0.2:
            return "replan"
        if failure["weighted_role_mismatch_rate"] > 0.4:
            return "refine"
        if semantic["weighted_fingerprint_match"] < 0.4:
            return "refine"
        return "replan" if delta_e > 0.5 else "refine"

    def _reason_slug(self, failure, semantic) -> str:
        if failure["weighted_cascade_rate"] > 0.1:
            return "hub_cascade"
        if failure["weighted_echo_rate"] > 0.2 and failure["weighted_role_mismatch_rate"] > 0.2:
            return "combined_echo_mismatch"
        if failure["weighted_echo_rate"] > 0.3:
            return "high_echo"
        if failure["weighted_role_mismatch_rate"] > 0.4:
            return "role_mismatch"
        if semantic["weighted_fingerprint_match"] < 0.4:
            return "low_fingerprint"
        if failure["weighted_error_rate"] > 0.2:
            return "high_error_rate"
        return "low_quality"

    def _refinement_targets(self, failure, structural) -> List[str]:
        targets = []
        if failure["weighted_echo_rate"] > 0.3:
            targets.append("sibling nodes with identical dependency sets")
        if failure["weighted_role_mismatch_rate"] > 0.4:
            targets.append("nodes with mismatched structural/functional roles")
        if structural["critical_path"] > 3:
            targets.append("deep dependency chains")
        return targets

    def _escalate_legacy(self, failure, semantic, structural, topology, depth, delta_e, verdict, iteration) -> DeltaDirective:
        entry = {
            "delta_e": round(delta_e, 4), "verdict": verdict, "failure": failure,
            "semantic": {k: round(v, 4) for k, v in semantic.items()},
            "structural": structural, "topology": topology, "depth": depth, "iteration": iteration,
        }
        try:
            import os
            os.makedirs(".ceo_delta", exist_ok=True)
            with open(self.escalation_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
        return DeltaDirective(
            action="replan", reason="novel_failure",
            replan_hint=f"Delta encountered an unclassified failure pattern (Δe={delta_e:.3f}, verdict={verdict}). Reconsider topology and node decomposition from first principles.",
            confidence=0.5, iteration=iteration, escalated=True,
        )

    # -------------------------------------------------------------------
    # shared metric/helper methods -- UNCHANGED from v2.0, copied verbatim.
    # -------------------------------------------------------------------

    def _record_directive_outcome(
        self, hb_key: List[float], prev_directive: DeltaDirective,
        prev_delta_e: float, curr_delta_e: float,
    ) -> None:
        key = f"{prev_directive.action}:{prev_directive.reason}"
        self.hb.upsert_directive_outcome(hb_key, key, prev_delta_e, curr_delta_e)

    def _update_agent_cards(self, dag: DAG, trace: ExecutionTrace, weights: Dict[str, float], task_class: str) -> None:
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
                run_id=dag.dag_id, node_id=node.node_id, task_class=task_class,
                functional_role=node.roles.functional,
                fingerprint_match=result.fingerprint_match,
                role_function_match=result.role_function_match,
                error=result.error, latency_s=result.latency_s,
                cost_tokens=result.cost_tokens,
                centrality_weight=weights.get(node.node_id, 1.0),
            )
            self.registry.record_performance(node.assigned_agent_id, record)

        hub_fail_ids = {
            dag.node(nid).assigned_agent_id
            for nid in self._hub_failures(dag, trace, weights)
            if dag.node(nid) and dag.node(nid).assigned_agent_id
        }
        for aid in hub_fail_ids:
            card = self.registry.get(aid)
            if card:
                card.dynamic.last_revision = f"hub node failure on dag={dag.dag_id} task_class={task_class}"

    def _structural(self, dag: DAG, trace: ExecutionTrace) -> Dict[str, float]:
        deps = sum(len(n.dependencies) for n in dag.nodes)
        roots = sum(1 for n in dag.nodes if not n.dependencies)
        fan_out = roots / max(1, len(dag.nodes))
        crit = self._critical_path(dag)
        executed = len(trace.results)
        align = executed / max(1, len(dag.nodes))
        return {
            "fan_out_ratio": round(fan_out, 3), "critical_path": float(crit),
            "edges": float(deps), "plan_exec_alignment": round(align, 3),
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
            "p50_latency": round(pct(50), 3), "p95_latency": round(pct(95), 3),
            "throughput": round(throughput, 3), "wallclock": round(trace.wallclock_s, 3),
        }

    def _failure(self, dag: DAG, trace: ExecutionTrace, weights: Dict[str, float]) -> Dict[str, float]:
        n = len(trace.results) or 1
        errs = sum(1 for r in trace.results if r.error)
        role_miss = sum(1 for r in trace.results if not r.role_function_match)
        echo = self._echo_rate(dag, trace)
        cascade = self._cascade_rate(dag, trace)

        total_weight = sum(weights.get(r.node_id, 1.0) for r in trace.results) or 1.0
        weighted_err = sum(weights.get(r.node_id, 1.0) for r in trace.results if r.error) / total_weight
        weighted_role_miss = sum(
            weights.get(r.node_id, 1.0) for r in trace.results if not r.role_function_match
        ) / total_weight
        weighted_echo = self._weighted_echo_rate(dag, trace, weights)
        weighted_cascade = self._weighted_cascade_rate(dag, trace, weights)

        return {
            "error_rate": round(errs / n, 3), "role_mismatch_rate": round(role_miss / n, 3),
            "echoing_rate": round(echo, 3), "cascade_rate": round(cascade, 3),
            "weighted_error_rate": round(weighted_err, 3),
            "weighted_role_mismatch_rate": round(weighted_role_miss, 3),
            "weighted_echo_rate": round(weighted_echo, 3),
            "weighted_cascade_rate": round(weighted_cascade, 3),
        }

    def _semantic(self, dag: DAG, trace: ExecutionTrace, drift: float, weights: Dict[str, float]) -> Dict[str, float]:
        fps = [r.fingerprint_match for r in trace.results if r.fingerprint_match]
        fp_mean = statistics.fmean(fps) if fps else 0.0
        total_weight = sum(weights.get(r.node_id, 1.0) for r in trace.results) or 1.0
        weighted_fp = sum(
            r.fingerprint_match * weights.get(r.node_id, 1.0) for r in trace.results
        ) / total_weight
        why_explore = statistics.fmean([1.0 if n.why.exploratory else 0.0 for n in dag.nodes])
        return {
            "fingerprint_match_mean": round(fp_mean, 3), "weighted_fingerprint_match": round(weighted_fp, 3),
            "brief_drift": round(drift, 3), "why_exploratory_frac": round(why_explore, 3),
            "prior_calibration": round(1.0 - why_explore, 3),
        }

    def _hub_failures(self, dag: DAG, trace: ExecutionTrace, weights: Dict[str, float]) -> List[str]:
        deg = dag.in_degree()
        by_id = {r.node_id: r for r in trace.results}
        return [
            nid for nid, w in weights.items()
            if deg.get(nid, 0) > 0 and by_id.get(nid) is not None and by_id[nid].error is not None
        ]

    def _weighted_echo_rate(self, dag: DAG, trace: ExecutionTrace, weights: Dict[str, float]) -> float:
        by_id = {r.node_id: r for r in trace.results}
        nodes = dag.nodes
        total_w, echo_w = 0.0, 0.0
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if set(nodes[i].dependencies) == set(nodes[j].dependencies):
                    a = by_id.get(nodes[i].node_id)
                    b = by_id.get(nodes[j].node_id)
                    if a and b:
                        pair_w = max(weights.get(nodes[i].node_id, 1.0), weights.get(nodes[j].node_id, 1.0))
                        total_w += pair_w
                        if cosine(a.output_embedding, b.output_embedding) > self.cfg.echo_cosine_threshold:
                            echo_w += pair_w
        return echo_w / total_w if total_w > 0 else 0.0

    def _weighted_cascade_rate(self, dag: DAG, trace: ExecutionTrace, weights: Dict[str, float]) -> float:
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
                    a = by_id.get(nodes[i].node_id)
                    b = by_id.get(nodes[j].node_id)
                    if a and b:
                        sibs.append((a, b))
        if not sibs:
            return 0.0
        echoes = sum(1 for a, b in sibs if cosine(a.output_embedding, b.output_embedding) > self.cfg.echo_cosine_threshold)
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

    def _granular(self, dag: DAG, trace: ExecutionTrace, failure, weights: Dict[str, float], hub_failures: List[str]) -> List[str]:
        out: List[str] = ["per-run entry written"]
        costs = [r.cost_tokens for r in trace.results] or [1]
        planned = statistics.fmean(costs)
        for r in trace.results:
            if r.cost_tokens > self.cfg.surprise_factor * planned:
                w = weights.get(r.node_id, 1.0)
                out.append(f"per-node entry: {r.node_id} cost {r.cost_tokens} >> {planned:.0f} (centrality={w:.2f})")
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
            f"w_echo={failure['weighted_echo_rate']} w_cascade={failure['weighted_cascade_rate']} "
            f"w_role_miss={failure['weighted_role_mismatch_rate']}{hub_note}"
        )

    def _research_feedback(self, drift, semantic) -> str:
        flag = "HIGH" if drift > 0.3 else "ok"
        return f"brief drift={drift:.2f} [{flag}] fp={semantic['fingerprint_match_mean']} w_fp={semantic['weighted_fingerprint_match']}"