"""v3.0 — Coordinate descent between E and F.

Because E and F share a basis, the directive loop is literal coordinate
descent: lock onto the worst over-threshold axis, apply a deterministic
repair move, re-measure the SAME axis next iteration, confirm it dropped.

Discipline that separates descent from the old random-walk thrashing:
  - The loop commits to a (axis, move) target and only re-picks the axis
    once the current one is driven below threshold — it does NOT re-scan
    all dims every iteration.
  - Intra-loop memory: a move already tried this run is never retried, so
    the loop searches instead of resampling.
  - A move that drops its axis below threshold is written back as a LEARNED
    move for this fingerprint class (prioritized next run).
  - When an axis exhausts its deterministic moves, its irreducibility
    counter is bumped. LLM escalation (b) unlocks ONLY once that counter
    crosses the configured threshold — escalation is earned by repeated
    proven irreducibility, not triggered on one failure. A successful
    escalation is folded back into the deterministic repertoire (b -> a).

The only non-deterministic step is the CONTENT the LLM authors inside an
earned escalation. The DECISION to escalate is a deterministic counter test.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

from .ef import EF_AXES, EFTensor, verdict_from_ef
from .moves import Move, applicable_moves, select_move
from .schemas import DeltaDirective


_ESC_SYSTEM = (
    "You are Delta's escalation planner. Deterministic structural repairs have "
    "repeatedly FAILED to reduce one error axis for this class of task. Propose ONE "
    "new structural repair move. Be concrete and structural (about nodes, dependencies, "
    "topology), never about content wording. Output ONLY JSON."
)

# Deterministic, static description of what each axis MEASURES and which
# structural levers move it. Without this the escalation LLM guesses at the
# metric and authors misdiagnosed moves (observed live: it added an output-side
# aggregation node for an input-side flow violation).
_AXIS_SEMANTICS = {
    "partition": (
        "partition-E = MEAN COSINE SIMILARITY among sibling nodes' OUTPUTS (siblings = "
        "identical dependency set). High = siblings produced near-duplicate content. "
        "Levers: rewrite sibling intents to cover strictly DISJOINT sub-topics, or merge "
        "redundant siblings into one node."
    ),
    "flow": (
        "flow-E measures GRAPH SHAPE ONLY (edges, never content) against the required flow. "
        "divergent: needs parallel gathering (>=2 independent roots OR one node fanning out "
        "to >=2 branches) AND a join node that merges them. convergent: needs a merge node "
        "with in-degree >=2 and a single sink. sequential: needs one single chain (single "
        "root, no branching). recursive: needs critical-path depth >= 3."
    ),
    "role": (
        "role-E = per-node behavioral band excess (citation density / compression ratio / "
        "structural density vs the node's declared functional role) + 0.3 for each REQUIRED "
        "role entirely absent from the plan. Levers: add the missing role node, or "
        "relabel/split nodes whose declared role mismatches their actual task."
    ),
    "scale": (
        "scale-E = (realized critical-path depth - target_depth)/target_depth when over "
        "(harsh), half that ratio when under (mild). Levers: remove an intermediate layer "
        "to reduce depth, or add a decomposition layer to increase it — toward target_depth."
    ),
}

_ESC_PROMPT = """FINGERPRINT SHAPE: {shape}
REQUIRED STRUCTURE (the target F): {required}
STUCK AXIS: {axis}  (current deviation this axis = {value})
HOW THIS AXIS IS MEASURED (deterministic): {semantics}
CURRENT PLAN: topology={topology} depth={depth} nodes={n_nodes}
GRAPH SIGNATURE: {sig}
DETERMINISTIC MOVES ALREADY TRIED on this axis (this fingerprint, prior runs included): {tried}

Propose ONE structural move that directly reduces the {axis} deviation AS MEASURED ABOVE.
Do not touch parts of the plan the measurement ignores. Output ONLY:
{{
  "constraint": "one concrete structural instruction to the planner (about nodes/edges/topology)",
  "action": "replan|refine"
}}"""


class Descent:
    def __init__(self, fingerprint, required, store, cfg, llm, role_bands, task_raw=None):
        self.fp = fingerprint
        self.req = required
        self.store = store
        self.cfg = cfg
        self.llm = llm
        self.role_bands = role_bands
        self.shape_key = fingerprint.shape_string()
        # v3.6 robustness: the raw run() task string, the one deterministic
        # identity per run (structured_intent is re-derived and can drift).
        # Used only as the warm-start task_embedding key in finalize() below;
        # falls back to best_dag.task if not supplied (e.g. old call sites).
        self.task_raw = task_raw

        # register this species so counters/moves have a home
        self.store.ensure(self.shape_key, fingerprint.embedding)

        # per-species move memory, loaded once. The APPLICABLE candidate list
        # is recomputed each step from the incumbent's live signature (a move's
        # validity depends on current graph shape, not just its axis), so we
        # keep the raw memory here and let moves.applicable_moves assemble the
        # ordered candidate set per iteration.
        depr_threshold = self.cfg.ef_side_effect_deprioritize_threshold
        self._deprioritized = self.store.deprioritized_moves(self.shape_key, depr_threshold)
        self._priority: Dict[str, List[str]] = {}
        self._extra: Dict[str, List[Move]] = {}
        for ax in EF_AXES:
            self._extra[ax] = [Move(**d) for d in self.store.escalation_moves_for(self.shape_key, ax)]
            self._priority[ax] = self.store.learned_moves_for(self.shape_key, ax)

        # earned-escalation eligibility, decided from PRIOR runs at run start:
        # eligible iff the axis has been irreducible >= threshold times. The
        # counter now RESETS whenever a move converges the axis (see
        # ef_store.reset_irreducible), so a high count means "the current
        # repertoire is failing this axis RIGHT NOW", not "it failed once, ever"
        # -- which is exactly when earned escalation should re-open. The old
        # `and not learned` clause is gone: it let one early success disable
        # escalation permanently even as the axis kept failing.
        case = self.store.get(self.shape_key) or {}
        irr = case.get("irreducible_counts", {})
        thr = self.cfg.ef_irreducible_escalate_threshold
        self._eligible: Dict[str, bool] = {
            ax: irr.get(ax, 0) >= thr for ax in EF_AXES
        }
        self._escalated_this_run: set = set()

        # per-run state
        self.locked_axis: Optional[str] = None
        self.tried: Dict[str, set] = {ax: set() for ax in EF_AXES}
        self.axis_trace: List[Dict] = []     # raw measured ef.as_dict() per iteration
        self.detail_trace: List[Dict] = []   # per-iteration diagnostics (sig, role excess, worst pair)
        self.move_log: List[Dict] = []
        self._worked_axes: set = set()       # axes that converged this run
        self._scored_axes: set = set()       # axes with >=1 APPLIED+MEASURED move
        self._pending_escalations: Dict[str, Dict] = {}  # move_id -> desc; persisted only on success
        self._last_axis: Optional[str] = None

        # incumbent: the best plan seen this run. Directives are ALWAYS rendered
        # from the incumbent, and a move that fails to improve the axis is
        # REJECTED (we do not descend from a worse plan). This is what makes the
        # descent monotone: incumbent worst-axis excess never increases across
        # iterations.
        self._incumbent_dag = None
        self._incumbent_ef = None            # EFTensor
        self._incumbent_worst: float = float("inf")
        self.incumbent_trace: List[Dict] = []  # incumbent ef.as_dict() per iteration (the honest curve)
        # PUBLIC, single source of truth for "did THIS iteration's plan just
        # become the best one seen this run?" orchestrator.py mirrors this
        # flag verbatim instead of re-deriving its own answer with a second,
        # independent comparison -- see the orchestrator.run() loop. Two
        # mechanisms computing "which plan is best" (this one, threshold-
        # relative; and the old Pareto+raw-max tiebreak in orchestrator.py)
        # is exactly what let the delivered answer disagree with the
        # descent's own notion of progress (observed live: a trip-planning
        # run delivered iteration 2's answer -- excess 0.186 -- when
        # iteration 3 -- excess 0.101 -- was strictly better by every
        # threshold-aware measure; the old tiebreak compared raw axis
        # magnitudes across DIFFERENT thresholds, apples to oranges).
        self.incumbent_changed: bool = False
        self._last_move: Optional[str] = None
        self._last_ef: Optional[Dict] = None

        # v3.5 Q-factor: content quality tracked as a LEXICOGRAPHIC secondary
        # to the structural worst-excess. Structure is always primary; Q only
        # breaks a structural tie and can NEVER cause a structurally-worse plan
        # to be accepted. Inert (identical to pre-Q behavior) when use_q_tensor
        # is off. Q measurement is best-effort and never raises into the descent.
        self._use_q = getattr(self.cfg, "use_q_tensor", False)
        self._q_thresholds = {
            "groundedness": getattr(self.cfg, "q_groundedness_threshold", 1.0),
            "relevance": getattr(self.cfg, "q_relevance_threshold", 1.0),
        }
        self._incumbent_q_worst: float = float("inf")
        self._incumbent_q: Dict = {}
        self.q_trace: List[Dict] = []            # per-iteration measured Q (observability)
        self.incumbent_q_trace: List[Dict] = []  # incumbent Q per iteration

    # -- config -------------------------------------------------------------
    def thresholds(self) -> Dict[str, float]:
        return {
            "partition": self.cfg.ef_partition_threshold,
            "flow": self.cfg.ef_flow_threshold,
            "role": self.cfg.ef_role_threshold,
            "scale": self.cfg.ef_scale_threshold,
        }

    # -- the descent step ---------------------------------------------------
    def _worst_excess(self, ef_d: Dict, th: Dict) -> float:
        """Max over-threshold excess across axes (negative if all satisfied).
        This scalar is used ONLY to order incumbents (lower is better) -- the
        verdict itself is still the full per-axis gate, never this number."""
        return max((ef_d[ax] - th[ax] for ax in EF_AXES), default=0.0)

    def _measure_q(self, dag, trace) -> Dict:
        """Per-iteration content quality from the executed DAG (best-effort).
        Uses the verifier output as the audit (groundedness is inherited from
        it, never from a composed surface -- composition hasn't happened yet in
        the loop anyway) and the terminal non-verifier output as the answer
        proxy for relevance. Returns {} when Q is off or on any error."""
        if not self._use_q:
            return {}
        try:
            from .quality import compute_quality
            by_id = {r.node_id: r for r in trace.results}
            verifier_out, content_texts, answer_proxy = [], [], ""
            for n in dag.nodes:
                r = by_id.get(n.node_id)
                if r is None or r.error:
                    continue
                if n.roles.functional == "verifier":
                    verifier_out.append(r.output)
                else:
                    content_texts.append(r.output)
                    answer_proxy = r.output   # last non-verifier output = synthesis proxy
            audit = "\n".join(verifier_out)
            # dag.task_embedding is the STRUCTURAL fingerprint embedding, not a
            # semantic embedding of the task text -- see the same fix/note in
            # orchestrator.py. Use a genuine semantic embedding of dag.task
            # (the structured intent text) for the relevance axis instead.
            from .embeddings import embed as _embed
            task_semantic_emb = _embed(dag.task)
            q = compute_quality(task_semantic_emb, answer_proxy, audit, content_texts, self.cfg,
                                 verifier_expected=self.fp.requires_verifier(),
                                 epistemic_stance=self.fp.epistemic_stance)
            return q.as_dict()
        except Exception:
            return {}

    def _q_worst(self, q_d: Dict) -> float:
        """Threshold-relative worst-excess over the content axes (0.0 if none)."""
        if not q_d:
            return 0.0
        return max((q_d.get(ax, 0.0) - self._q_thresholds.get(ax, 1.0) for ax in q_d), default=0.0)

    def step(self, dag, trace, ef: EFTensor, iteration: int) -> DeltaDirective:
        ef_d = ef.as_dict()
        self.axis_trace.append(ef_d)      # raw measured plan (diagnostics)
        self._capture_detail(ef)
        th = self.thresholds()

        # (A) score the previous move against the incumbent it was told to edit.
        self._score_previous(ef_d, th)

        # (B) incumbent update with UPHILL REJECTION. The just-measured plan
        # becomes the new incumbent only if it is no worse (by worst-axis
        # excess) than the best plan seen this run; otherwise it is rejected and
        # we keep descending from the incumbent. This is the guarantee that the
        # incumbent's worst axis is monotone non-increasing across iterations --
        # one bad LLM replan can no longer poison the rest of the descent.
        measured_worst = self._worst_excess(ef_d, th)
        q_d = self._measure_q(dag, trace)          # {} when Q off -> fully inert
        self.q_trace.append(q_d)
        q_worst = self._q_worst(q_d)
        # Lexicographic incumbent: STRUCTURE primary (unchanged), Q secondary.
        # When Q is off (q_worst==0, _incumbent_q_worst starts inf then 0), the
        # structural-tie branch reduces EXACTLY to the original `<=` rule.
        if self._incumbent_dag is None:
            self.incumbent_changed = True
        elif measured_worst < self._incumbent_worst - 1e-9:
            self.incumbent_changed = True                       # strictly better structure
        elif measured_worst <= self._incumbent_worst + 1e-9:
            # structural tie -> Q breaks it; NEVER regress structure for Q
            self.incumbent_changed = (q_worst <= self._incumbent_q_worst + 1e-9)
        else:
            self.incumbent_changed = False
        if self.incumbent_changed:
            self._incumbent_dag = dag
            self._incumbent_ef = ef
            self._incumbent_worst = measured_worst
            self._incumbent_q_worst = q_worst
            self._incumbent_q = q_d
        inc_ef = self._incumbent_ef
        inc_d = inc_ef.as_dict()
        self.incumbent_trace.append(inc_d)
        self.incumbent_q_trace.append(dict(self._incumbent_q))

        # (C) convergence is judged on the INCUMBENT (the plan we would deliver)
        _, good = verdict_from_ef(inc_ef, th, self.cfg.ef_mixed_margin)
        if good:
            self.locked_axis = None
            return DeltaDirective(
                action="surface",
                reason="ef_converged: all axes below threshold",
                confidence=1.0, iteration=iteration,
            )

        # (D) worst-first over-threshold axes OF THE INCUMBENT, honoring lock
        over = sorted(
            [ax for ax in EF_AXES if inc_d[ax] > th[ax]],
            key=lambda ax: inc_d[ax] - th[ax], reverse=True,
        )
        if self.locked_axis in over:
            over.remove(self.locked_axis)
            over.insert(0, self.locked_axis)

        present_roles = {n.roles.functional for n in self._incumbent_dag.nodes}
        sig = inc_ef.signature

        # (E) walk over-threshold axes worst-first. For each: the next UNTRIED
        # APPLICABLE deterministic move (applicability is derived from the
        # incumbent's live signature + required F, so only direction-correct
        # moves are ever candidates -- no more split-vs-linearize thrash).
        # Escalation is the LAST resort WITHIN an axis: it fires only once the
        # deterministic repertoire for that axis is spent this run AND the axis
        # is escalation-eligible (earned across prior runs).
        for ax in over:
            cands = applicable_moves(
                ax, sig, self.req, present_roles,
                extra=self._extra.get(ax), priority_ids=self._priority.get(ax),
                deprioritized_ids=self._deprioritized,
            )
            move = select_move(ax, self.tried[ax], cands)
            if move is not None:
                self.locked_axis = ax
                self.tried[ax].add(move.move_id)
                self._stash(ax, move.move_id, inc_d)
                node_id = inc_ef.worst_node(ax)
                return DeltaDirective(
                    action=move.action,
                    reason=f"ef_move:{move.move_id}",
                    replan_hint=self._render(ax, inc_ef, th, move.constraint, self._incumbent_dag),
                    refinement_targets=[node_id] if node_id else [],
                    confidence=0.9, iteration=iteration,
                    primary_dim=ax, escalated=False,
                )
            if (self._eligible.get(ax) and ax not in self._escalated_this_run
                    and self.llm is not None):
                self._escalated_this_run.add(ax)
                desc = self._escalate(self._incumbent_dag, inc_ef, ax)
                if desc and desc["move_id"] not in self.tried[ax]:
                    # NOT persisted yet — b feeds a ONLY on success (see
                    # _score_previous). Session-local until it converges.
                    self._pending_escalations[desc["move_id"]] = desc
                    new_move = Move(**desc)
                    self.tried[ax].add(new_move.move_id)
                    self.locked_axis = ax
                    self._stash(ax, new_move.move_id, inc_d)
                    node_id = inc_ef.worst_node(ax)
                    return DeltaDirective(
                        action=new_move.action,
                        reason=f"ef_escalated:{new_move.move_id}",
                        replan_hint=self._render(ax, inc_ef, th, new_move.constraint, self._incumbent_dag),
                        refinement_targets=[node_id] if node_id else [],
                        confidence=0.5, iteration=iteration,
                        primary_dim=ax, escalated=True,
                    )

        # (F) every over-threshold axis has exhausted its APPLICABLE repertoire
        # (and escalation, if eligible) this run. Surface honestly; irreducibility
        # is tallied at finalize() so it accumulates ACROSS runs.
        axis = over[0]
        return DeltaDirective(
            action="surface",
            reason=f"ef_exhausted:{axis} (repertoire spent this run)",
            confidence=0.4, iteration=iteration, primary_dim=axis,
        )

    # -- helpers ------------------------------------------------------------
    def _score_previous(self, ef_d: Dict, th: Dict) -> None:
        if self._last_axis is None or self._last_ef is None:
            return
        ax, mid = self._last_axis, self._last_move
        before_ef = self._last_ef
        before = before_ef.get(ax, 0.0)
        after = ef_d.get(ax, 0.0)
        converged = after <= th[ax]
        dropped = after < before - 1e-6
        entry = {
            "axis": ax, "move": mid,
            "before": round(before, 3), "after": round(after, 3),
            "converged": bool(converged), "improved": bool(dropped),
        }
        self._scored_axes.add(ax)   # this axis had a move APPLIED and MEASURED
        if converged:
            self.store.add_learned_move(self.shape_key, ax, mid)
            # the axis is reducible after all -> it is no longer "currently
            # stuck", so its irreducibility counter resets. This is half of the
            # escalation fix: the counter tracks CONSECUTIVE-recent failures, so
            # a move that stops working later will let the count climb and
            # re-arm escalation instead of being blocked forever.
            self.store.reset_irreducible(self.shape_key, ax)
            self._worked_axes.add(ax)
            # b feeds a — an escalated move earns permanence ONLY here, once
            # it has demonstrably converged its axis.
            if mid in self._pending_escalations:
                self.store.add_escalation_move(
                    self.shape_key, ax, self._pending_escalations.pop(mid)
                )
            if self.locked_axis == ax:
                self.locked_axis = None

        # -- side-effect capture: did this move ripple into a DIFFERENT axis
        # that was previously fine and is now newly violated? Record it
        # regardless of whether the primary axis converged -- collateral
        # damage matters even when the intended fix worked.
        side_effects = []
        for other_ax in EF_AXES:
            if other_ax == ax:
                continue
            was_ok = before_ef.get(other_ax, 0.0) <= th[other_ax]
            now_bad = ef_d.get(other_ax, 0.0) > th[other_ax]
            if was_ok and now_bad:
                count = self.store.bump_side_effect(self.shape_key, mid, other_ax)
                side_effects.append({"axis": other_ax, "count": count})
        if side_effects:
            entry["side_effects"] = side_effects

        self.move_log.append(entry)
        self._last_axis = None  # consume

    def _stash(self, axis: str, move_id: str, ef_d: Dict) -> None:
        self._last_axis = axis
        self._last_move = move_id
        self._last_ef = ef_d

    def _render(self, axis: str, ef, th: Dict, constraint: str, dag) -> str:
        """Full directive hint: the primary-axis fix + PRESERVE constraints for
        every already-satisfied axis + the current plan as an editable baseline.
        This is what makes the descent multi-axis monotonic: fixing one axis no
        longer rebuilds the whole DAG and regresses another."""
        ef_d = ef.as_dict()
        parts = [f"PRIMARY AXIS: {axis} (E={ef_d[axis]:.3f} > threshold {th[axis]:.2f}). {constraint}"]
        preserve = self._preserve_clause(ef, th, axis)
        if preserve:
            parts.append(preserve)
        parts.append(self._plan_snapshot(dag))
        return " ".join(parts)

    def _preserve_clause(self, ef, th: Dict, primary_axis: str) -> str:
        ef_d = ef.as_dict()
        sig = ef.signature or {}
        keep: List[str] = []
        for ax in EF_AXES:
            if ax == primary_axis or ef_d[ax] > th[ax]:
                continue  # the axis we're fixing, or one that isn't satisfied
            if ax == "flow":
                if sig.get("root_count", 0) >= 2:
                    shape = f"{sig['root_count']} parallel root nodes"
                elif sig.get("max_out", 0) >= 2:
                    shape = f"a fan-out point spreading into {sig['max_out']} parallel branches"
                else:
                    shape = "the current chain"
                if sig.get("has_join"):
                    shape += ", converging into a join"
                keep.append(f"FLOW is already correct — keep this shape ({shape}); "
                            f"do not collapse or re-route it.")
            elif ax == "partition":
                keep.append("PARTITION is already correct — sibling nodes are well-"
                            "differentiated; keep their disjoint scopes.")
            elif ax == "scale":
                keep.append(f"SCALE is already correct — keep depth={sig.get('depth', '?')}; "
                            f"do not add or remove layers.")
            elif ax == "role":
                keep.append("ROLE assignment is already correct — keep each node's "
                            "functional role.")
        if not keep:
            return ""
        return ("PRESERVE (these axes are already satisfied — do NOT change them, edit ONLY "
                "for the primary axis): " + " ".join(keep))

    def _plan_snapshot(self, dag) -> str:
        lines = [f"{n.node_id}[{n.roles.functional}] deps={n.dependencies}: {n.intent[:50]}"
                 for n in dag.nodes]
        return ("CURRENT PLAN (edit this incrementally — keep untouched nodes as-is, do not "
                "rebuild from scratch):\n" + "\n".join(lines))

    def _escalate(self, dag, ef: EFTensor, axis: str) -> Optional[Dict]:
        sig = ef.signature
        # what already failed = everything tried this run + the persisted repertoire
        tried_hist = sorted(set(self.tried[axis])
                            | set(self.store.learned_moves_for(self.shape_key, axis)))
        try:
            data = self.llm.chat_json([
                {"role": "system", "content": _ESC_SYSTEM},
                {"role": "user", "content": _ESC_PROMPT.format(
                    shape=self.shape_key, required=self.req.as_dict(),
                    axis=axis, value=round(ef.as_dict()[axis], 3),
                    semantics=_AXIS_SEMANTICS.get(axis, ""),
                    topology=dag.topology, depth=dag.depth, n_nodes=len(dag.nodes),
                    sig=sig, tried=tried_hist,
                )},
            ], tag="ef.escalate")
        except Exception:
            return None
        constraint = str(data.get("constraint", "")).strip()
        if not constraint:
            return None
        action = str(data.get("action", "replan"))
        if action not in ("replan", "refine"):
            action = "replan"
        # content-hashed id: unique per distinct constraint, so failed unpersisted
        # escalations never collide with later ones in side-effect bookkeeping
        h = hashlib.sha1(constraint.encode()).hexdigest()[:6]
        return {
            "move_id": f"{axis}.esc.{h}",
            "axis": axis,
            "constraint": constraint,
            "action": action,
        }

    def _capture_detail(self, ef: EFTensor) -> None:
        """Per-iteration diagnostics for the evidence log (JSON-safe)."""
        worst = None
        if ef.partition_pairs:
            (a, b), sim = max(ef.partition_pairs.items(), key=lambda kv: kv[1])
            worst = {"pair": [a, b], "sim": round(sim, 4)}
        self.detail_trace.append({
            "signature": dict(ef.signature or {}),
            "role_excess": {nid: round(sum(v.values()), 3)
                            for nid, v in (ef.role_excess or {}).items()},
            "partition_worst": worst,
        })

    # -- end of run ---------------------------------------------------------
    def finalize(self, best_dag) -> None:
        if not self.axis_trace:
            return
        # Irreducibility accounting: any axis that ended the run still over
        # threshold, was actually worked (a move was tried), and did NOT
        # converge, gets its per-(F,axis) counter bumped. This accumulates
        # across runs and is what unlocks earned escalation on a later run.
        # "Ended the run" is judged on the INCUMBENT (the plan actually
        # delivered), not the raw last iteration -- a rejected uphill plan
        # must not count against an axis the incumbent already had in hand.
        th = self.thresholds()
        final = (self._incumbent_ef.as_dict() if self._incumbent_ef is not None
                 else self.axis_trace[-1])
        for ax in EF_AXES:
            # Only axes whose moves were actually APPLIED AND MEASURED count
            # toward irreducibility. A directive issued on the run's final
            # iteration is never applied — counting it would unfairly bump the
            # escalation counter for a move that never got its chance.
            worked = ax in self._scored_axes
            if worked and ax not in self._worked_axes and final.get(ax, 0.0) > th[ax]:
                self.store.bump_irreducible(self.shape_key, ax)

        best_plan = {}
        if best_dag is not None:
            # DAG-exhaustive: everything needed to rebuild the graph WITHOUT
            # re-planning (see warmstart.dag_from_best_plan), EXCEPT runtime-
            # derivable data (embeddings, assigned agents) which get
            # recomputed on reconstruction rather than persisted stale.
            from .embeddings import embed
            best_plan = {
                "topology": best_dag.topology,
                "depth": best_dag.depth,
                "why_topology": best_dag.why_topology,
                "why_depth": best_dag.why_depth,
                "nodes": [
                    {
                        "node_id": n.node_id,
                        "intent": n.intent,
                        "structural": n.roles.structural,
                        "functional": n.roles.functional,
                        "epistemic": n.roles.epistemic,
                        "dependencies": list(n.dependencies),
                    }
                    for n in best_dag.nodes
                ],
                # Semantic identity of the task this plan solves. shape_key is
                # purely STRUCTURAL (topology/depth/coupling) and is shared
                # across many distinct tasks -- without this, warm-start would
                # retrieve a task-SPECIFIC plan (e.g. "the four compilation
                # stages") by a structural key alone, a category error (a
                # semantic artifact fetched by a non-semantic key). Stored
                # alongside best_plan so it is overwritten in lockstep with
                # the plan it describes.
                "task_embedding": embed(self.task_raw or best_dag.task),
            }
        self.store.upsert_case(
            shape_key=self.shape_key,
            shape_embedding=self.fp.embedding,
            best_plan=best_plan,
            initial_ef=self.axis_trace[0],
            final_ef=final,
            iterations=len(self.axis_trace),
            # threshold-relative, from the SAME incumbent-tracking this run
            # used to pick directives -- ef_store must not re-derive "best"
            # from raw axis magnitude (see moves.py/orchestrator.py history:
            # that comparison is invalid across axes with different
            # thresholds and was the exact bug that let a worse iteration
            # get delivered as the answer).
            worst_excess=self._incumbent_worst,
            final_q=(self._incumbent_q or None),
        )
