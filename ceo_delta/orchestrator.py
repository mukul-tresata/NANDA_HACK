"""The Loop — wires CEO -> Research -> Kernel -> Delta -> CEO -> ...

v2.0 — Single-handbook architecture, TaskFingerprint threading.

Research no longer owns a handbook -- it is a stateless parser. The single
remaining handbook (self.hb) is queried by CEO and written to by Delta,
both keyed on the SAME structural fingerprint embedding (shape axes only:
information_flow, epistemic_stance, output_contract, decomposability).
This is what makes M2 (structural generalization across surface-different
tasks) possible -- retrieval and writing finally happen in the same space.

The fingerprint's two modifier axes (complexity, domain_volatility) never
enter the embedding. complexity is rendered to CEO as a depth-cap prompt
hint; domain_volatility is enforced in code via CEO.force_verifier() after
planning, not left to prompt compliance.

Directive history is recorded in RunResult for inspection and demo.

v2.1 changes:
    - Delta now receives the LLM client (needed for escalation calls
      that produce a directive + structured case in one shot).
    - Best-answer selection across directive-loop iterations switched
      from scalar `delta_e <` comparison to Pareto dominance over the
      5-dim error tensor (falls back to scalar automatically when
      cfg.use_tensor_error=False, see delta.dominates()).
    - Escalation case outcomes get backfilled after the iteration
      following a matched/escalated directive, using DeltaDirective's
      new primary_dim field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .ceo import CEO
from .config import Config, DEFAULT
from .delta import Delta, DeltaReport, dominates
from .handbook import Handbook
from .kernel import Kernel
from .llm import LLMClient
from .reflection import Reflection, ReflectionLog, should_reflect
from .research import Brief, Research, TaskFingerprint, TaskSpecifics
from .runlog import RunLogger
from .schemas import AgentRegistry, DAG, DeltaDirective, ExecutionTrace
from . import bootstrap
from . import graph_ops


@dataclass
class RunResult:
    task: str
    answer: str
    dag: DAG
    fingerprint: TaskFingerprint
    specifics: TaskSpecifics
    brief: Brief
    replanned: bool
    trace: ExecutionTrace
    report: DeltaReport
    clarification: Optional[str] = None
    reflection: Optional[ReflectionLog] = None
    raw_answer: str = ""   # pre-composition last-node output (transparency/debug)
    audit: str = ""        # verifier findings, consumed by composition; Q will read this
    # directive loop fields
    iterations: int = 1
    directive_history: List[DeltaDirective] = field(default_factory=list)
    final_delta_e: float = 0.0
    # v3.0 EF descent trace
    ef_trace: List[dict] = field(default_factory=list)     # raw measured ef per iteration
    ef_incumbent_trace: List[dict] = field(default_factory=list)  # best-so-far ef per iteration (monotone)
    ef_move_log: List[dict] = field(default_factory=list)  # which move ran, did its axis drop
    ef_detail: List[dict] = field(default_factory=list)    # per-iteration sig/role-excess/worst-pair


class Orchestrator:
    def __init__(
        self,
        cfg: Config | None = None,
        workdir: str = ".ceo_delta",
        run_log_path: str | None = None,
    ):
        self.cfg = cfg or DEFAULT
        self.llm = LLMClient(self.cfg)

        # Single handbook. Research no longer learns or owns state -- it's
        # a stateless parser (see research.py v2.0). The old research_hb
        # stored near-duplicate data with no distinct purpose.
        self.hb = Handbook("planning", self.cfg, path=f"{workdir}/handbook.json")
        if self.cfg.seed_handbook:
            bootstrap.ensure_seeded(self.hb)

        self.run_logger = RunLogger(run_log_path or f"{workdir}/runs.jsonl")

        self.registry = AgentRegistry(self.cfg)
        self.ceo = CEO(self.hb, self.llm, self.registry, self.cfg)
        self.research = Research(self.llm, self.cfg)
        self.kernel = Kernel(self.llm, self.registry, self.cfg)
        self.delta = Delta(
            self.hb, self.registry, self.cfg,
            escalation_log_path=f"{workdir}/escalations.jsonl",
            llm=self.llm,  # v2.1: needed for escalation directive+case calls
        )
        self.reflection = Reflection(
            self.ceo, self.delta, self.kernel, self.llm, self.cfg
        )
        self._run_count_path = f"{workdir}/run_count.json"
        self.run_count = self._load_run_count()

    # -- standard mode -------------------------------------------------------

    def run(
        self,
        task: str,
        *,
        auto_clarify: bool = True,
        user_satisfaction: Optional[float] = None,
        task_label: str = "",
    ) -> RunResult:

        self.llm.reset_call_log()
        clarification = None
        needs, sim = self.ceo.needs_clarification(task)
        if needs and not auto_clarify:
            clarification = (
                f"Low handbook similarity ({sim:.2f}). Clarify scope before I plan."
            )

        # Research parses raw intent into (species, specifics). Stateless --
        # no handbook query happens here.
        fingerprint, specifics = self.research.clarify(task)
        # task_class (used for AgentRegistry bucketing) is now derived from
        # epistemic_stance, a controlled vocabulary, rather than a free-text
        # field the LLM invented independently.
        task_class = fingerprint.epistemic_stance

        # v3.0: open a fresh EF coordinate-descent for this task. Loads the
        # F-keyed move memory for this fingerprint class (learned + escalation
        # moves), so repeated species start their descent closer to F.
        self.delta.begin_run(fingerprint, task_raw=task)

        # --- iterative directive loop ---------------------------------------
        max_iter = self.cfg.max_ceo_eval_iterations
        directive_history: List[DeltaDirective] = []
        prev_delta_e: Optional[float] = None
        prev_directive: Optional[DeltaDirective] = None
        current_task = specifics.structured_intent
        best_answer = ""
        best_report = None
        best_dag = None
        best_trace = None
        replanned = False
        brief = None

        for iteration in range(max_iter):
            # v3.3: build the next plan. For a deterministic flow/scale move we
            # APPLY the repair as a pure graph transform on the incumbent DAG
            # (no LLM) -- this is the actual descent step the loop was missing.
            # Everything else (partition/role moves, escalations, iteration 0)
            # falls back to the stochastic CEO (re)plan below.
            dag, used_det_op = self._next_dag(
                current_task, prev_directive, task_class, fingerprint, task_raw=task,
            )

            # Structural gate: domain_volatility forces a verifier node.
            # Code-level guarantee, not a prompt suggestion -- CEO's plan
            # is checked and patched if it skipped verification on a
            # volatile-domain task.
            if fingerprint.requires_verifier() and not any(
                n.roles.functional == "verifier" for n in dag.nodes
            ):
                dag = self.ceo.force_verifier(dag)

            # secondary Research pass on first iteration only
            if iteration == 0:
                # A warm-started plan is a deliberately reused known-good plan
                # for THIS exact task -- the secondary research replan must not
                # discard it (that clobber is why reuse never stuck). Still run
                # investigate() for the brief context; just don't let it replan
                # a warm plan away. If the reused plan is actually wrong, the
                # descent repairs it downstream like any other incumbent.
                is_warm = any(getattr(n.why, "priors_used", "") == "warmstart"
                              for n in dag.nodes)
                brief = self.research.investigate(dag, specifics)
                if brief.triggers_replan and not is_warm:
                    dag = self.ceo.replan(
                        task,
                        brief.summary,
                        run_index=self.run_count,
                        task_class=task_class,
                        fingerprint=fingerprint,
                    )
                    replanned = True

            trace = self.kernel.execute(dag, brief_context=brief.summary if brief else "")
            trace.iteration = iteration
            answer = self._final_answer(trace)

            # Delta audit — returns (report, directive). hb writes are keyed
            # on fingerprint.embedding, same space CEO retrieved from.
            report, directive = self.delta.audit(
                dag, trace,
                fingerprint=fingerprint,
                brief_drift=brief.drift if brief else 0.0,
                user_satisfaction=user_satisfaction if iteration == max_iter - 1 else None,
                task_class=task_class,
                iteration=iteration,
                prev_delta_e=prev_delta_e,
                prev_directive=prev_directive,
            )

            # v2.1: backfill the PREVIOUS iteration's escalation case outcome
            # now that we know the current tensor. Only applies when the
            # previous directive came from a matched case or a fresh
            # escalation (reason prefix set in delta.py), and only when
            # the tensor path is active.
            if (
                prev_directive
                and prev_directive.primary_dim
                and prev_directive.reason.startswith(("matched_case:", "escalated_new_case:"))
                and report.error_tensor
            ):
                case_id = prev_directive.reason.split(":", 1)[1]
                dim = prev_directive.primary_dim
                self.delta.backfill_last_escalation(
                    case_id, dim, report.error_tensor.get(dim, 0.0)
                )

            directive_history.append(directive)

            # track best answer across iterations.
            # v3.2: when the EF/descent path is active, Descent ALREADY made
            # this exact decision (threshold-relative incumbent tracking,
            # with uphill-step rejection) in order to decide what to build
            # the next directive from. Mirroring that flag here, rather than
            # re-deriving "which plan is best" a second time, is what
            # guarantees the delivered answer can never disagree with the
            # descent loop's own notion of progress -- there is now exactly
            # ONE place in the codebase that decides "is this plan better",
            # not two. (The old second copy -- Pareto dominance + a raw
            # max(tensor.values()) tiebreak -- compared axes with DIFFERENT
            # thresholds head-to-head, which is invalid; it once selected a
            # strictly worse iteration, by the architecture's own metric, as
            # the delivered answer. Kept below ONLY as the fallback for the
            # legacy non-EF paths, where no Descent object exists at all.)
            descent = self.delta._descent
            if descent is not None:
                if descent.incumbent_changed:
                    best_answer = answer
                    best_report = report
                    best_dag = dag
                    best_trace = trace
            elif best_report is None or dominates(report, best_report):
                best_answer = answer
                best_report = report
                best_dag = dag
                best_trace = trace
            elif not dominates(best_report, report):
                # neither dominates -- genuinely incomparable (e.g. better
                # on role, worse on drift). Tiebreak: prefer the run with
                # the lower worst-case single dimension, rather than
                # averaging across incommensurable axes.
                if report.error_tensor and best_report.error_tensor:
                    report_worst = max(report.error_tensor.values())
                    best_worst = max(best_report.error_tensor.values())
                    if report_worst < best_worst:
                        best_answer = answer
                        best_report = report
                        best_dag = dag
                        best_trace = trace

            prev_delta_e = report.delta_e
            prev_directive = directive

            # termination check
            if directive.action == "surface":
                break

            # prepare next iteration — the directive (with its hint, PRESERVE
            # clause and plan snapshot) is injected once via CEO's directive
            # context; duplicating the hint into the task string was double-
            # injection and is removed.
            current_task = specifics.structured_intent

        # -------------------------------------------------------------------

        # v3.0: persist this fingerprint class's convergence trajectory and
        # any newly-learned/escalated moves, then read back the descent trace.
        self.delta.end_run(best_dag)
        ef_trace = list(self.delta._descent.axis_trace) if self.delta._descent else []
        ef_incumbent_trace = list(self.delta._descent.incumbent_trace) if self.delta._descent else []
        ef_move_log = list(self.delta._descent.move_log) if self.delta._descent else []
        ef_detail = list(self.delta._descent.detail_trace) if self.delta._descent else []

        # v3.x — CEO composition step. The winning node's raw output (often
        # the verifier's AUDIT on evolving-domain tasks, since it's forced
        # terminal) is NOT the deliverable. The CEO consumes its own internal
        # work -- including the audit -- and composes ONE coherent answer in
        # a single voice. Never fail the run on this: fall back to the raw
        # answer if composition errors or produces nothing.
        audit = ""
        raw_answer = best_answer
        if best_dag is not None and best_trace is not None:
            by_id = {r.node_id: r for r in best_trace.results}
            audit = "\n".join(
                by_id[n.node_id].output
                for n in best_dag.nodes
                if n.roles.functional == "verifier"
                and by_id.get(n.node_id) and not by_id[n.node_id].error
            )

        composed = best_answer
        if best_dag is not None and best_trace is not None:
            try:
                result_text = self.ceo.compose(task, fingerprint, best_dag, best_trace)
                if result_text:
                    composed = result_text
            except Exception:
                composed = best_answer   # never fail the run on the voice step

        # v3.5 Q-factor: measure content quality (RAGAS generation-side) on the
        # DELIVERED answer, fold into the verdict so it reflects answer quality
        # not just plan shape. Groundedness is inherited from the audit (a
        # confident composed surface cannot launder ungrounded claims).
        q_dict = None
        if self.cfg.use_q_tensor and best_dag is not None and best_trace is not None and best_report is not None and best_report.ef_tensor:
            from .quality import compute_quality
            from .ef import final_verdict
            by_id_q = {r.node_id: r for r in best_trace.results}
            content_texts = [
                by_id_q[n.node_id].output
                for n in best_dag.nodes
                if n.roles.functional != "verifier"
                and by_id_q.get(n.node_id) and not by_id_q[n.node_id].error
            ]
            # v3.5.1 fix: dag.task_embedding is the STRUCTURAL fingerprint
            # embedding (categorical flow/epistemic/decomposability axes),
            # not a semantic embedding of the task text -- see ceo.plan(),
            # which passes retrieval_emb (fingerprint.embedding) into
            # _build_dag's task_emb slot for handbook-retrieval consistency.
            # That's the right vector for prior-matching, but answer-relevancy
            # needs a genuine semantic embedding of the actual task, so we
            # compute one fresh here rather than reusing dag.task_embedding.
            from .embeddings import embed as _embed
            task_semantic_emb = _embed(task)
            q = compute_quality(task_semantic_emb, composed, audit, content_texts, self.cfg,
                                 verifier_expected=fingerprint.requires_verifier(),
                                 epistemic_stance=fingerprint.epistemic_stance)
            q_dict = q.as_dict()
            ef_th = {"partition": self.cfg.ef_partition_threshold, "flow": self.cfg.ef_flow_threshold,
                     "role": self.cfg.ef_role_threshold, "scale": self.cfg.ef_scale_threshold}
            q_th = {"groundedness": self.cfg.q_groundedness_threshold, "relevance": self.cfg.q_relevance_threshold}
            fv, _ = final_verdict(best_report.ef_tensor, q_dict, ef_th, q_th, self.cfg.q_mixed_margin)
            best_report.verdict = fv
            best_report.q_tensor = q_dict

        self.run_count += 1
        self._persist()

        refl = None
        ok, _ = should_reflect(self.run_count, self.hb, self.cfg)
        if ok:
            refl = self.reflection.run(self.run_count)
            self._persist()

        result = RunResult(
            task=task,
            answer=composed,
            dag=best_dag,
            fingerprint=fingerprint,
            specifics=specifics,
            brief=brief,
            replanned=replanned,
            trace=best_trace,
            report=best_report,
            clarification=clarification,
            reflection=refl,
            iterations=len(directive_history),
            directive_history=directive_history,
            final_delta_e=best_report.delta_e if best_report else 0.0,
            ef_trace=ef_trace,
            ef_incumbent_trace=ef_incumbent_trace,
            ef_move_log=ef_move_log,
            ef_detail=ef_detail,
            raw_answer=raw_answer,
            audit=audit,
        )
        result._run_index = self.run_count
        result._task_label = task_label

        llm_calls = self.llm.flush_call_log()
        self.run_logger.write(result, llm_calls)

        return result

    # -- meta / feedback mode ------------------------------------------------

    def meta_feedback(self, task: str, satisfaction: float, note: str = "") -> str:
        from .embeddings import embed
        emb = embed(task)
        entry, sim = self.hb.best_match(emb)
        good = satisfaction >= 0.6
        if entry:
            entry.revision = (
                f"[user-satisfaction={satisfaction:.2f}] {note} | "
                + (entry.revision or "")
            )[:300]
            self.hb.upsert_votes(
                emb, task[:120],
                entry.topology_chosen or "linear",
                entry.depth_chosen or 1,
                outcome_good=good,
            )
        self._persist()
        return (
            f"Delta recorded satisfaction={satisfaction:.2f} for task region "
            f"(match sim={sim:.2f}). good={good}."
        )

    # -- reflection mode -----------------------------------------------------

    def reflect_now(self) -> ReflectionLog:
        log = self.reflection.run(max(self.run_count, self.cfg.reflection_interval))
        self._persist()
        return log

    # -- helpers -------------------------------------------------------------

    def _next_dag(self, current_task, prev_directive, task_class, fingerprint, task_raw=None):
        """Produce the next plan for the directive loop.

        If the previous directive is a deterministic flow/scale move and we have
        an incumbent to edit, apply the move as a pure DAG->DAG transform (the
        real coordinate-descent step) and only re-resolve agents -- NO planning
        LLM call, so the structural axis moves the provably-correct direction and
        the trajectory is reproducible. Otherwise fall back to the stochastic
        CEO (re)plan (partition/role moves, LLM escalations, iteration 0, or a
        move whose target shape the incumbent already has).

        Returns (dag, used_deterministic_op).
        """
        descent = getattr(self.delta, "_descent", None)

        # -- warm-start (v3.6, "learning consume-side"): iteration 0 only
        # (prev_directive is None). If this fingerprint species already has a
        # cached best_plan whose best_worst_excess < 0 (threshold-relative --
        # negative means EVERY axis was satisfied, not just "better than some
        # other plan"), skip the planning LLM entirely and reconstruct that
        # plan directly. It still flows through the loop's normal
        # force_verifier/execute/measure gates below, so if it's genuinely
        # still satisfying for this task the descent converges in 1 iteration;
        # if not, the loop repairs it exactly like any other incumbent.
        if prev_directive is None and descent is not None:
            store = getattr(descent, "store", None)
            shape_key = fingerprint.shape_string()
            case = store.get(shape_key) if store is not None else None
            if case and case.get("best_worst_excess", float("inf")) <= self.cfg.ef_mixed_margin and case.get("best_plan", {}).get("nodes"):
                from .embeddings import embed, cosine
                from .warmstart import dag_from_best_plan
                # Task-IDENTITY gate (v3.6 fix): shape_key is purely STRUCTURAL
                # (topology/depth/coupling) and is shared across many distinct
                # tasks, but best_plan's nodes carry task-SPECIFIC intent (e.g.
                # "the four compilation stages"). Retrieving a semantically
                # task-specific artifact via a structural key alone is a
                # category error -- it once reused a C-compiler plan to answer
                # "explain the water cycle" because both happened to share a
                # sequential/synthesis/artifact/coupled shape. A semantic
                # artifact must be retrieved by a semantic key: only warm-start
                # if the cached plan's task_embedding is (near-)identical to
                # the incoming task's embedding. Missing embedding (old case,
                # predates this fix) safely falls through to cold-plan below.
                cached_emb = case["best_plan"].get("task_embedding")
                if cached_emb:
                    identity_task = task_raw if task_raw is not None else current_task
                    sim = cosine(embed(identity_task), cached_emb)
                    if sim < self.cfg.warm_start_similarity_threshold:
                        cached_emb = None   # different task, same shape -> do NOT warm-start
                if cached_emb:
                    # Use the STRUCTURAL fingerprint embedding as the DAG's
                    # task_embedding, exactly as ceo.plan does (retrieval_emb =
                    # fingerprint.embedding), so a warm-started DAG is byte-
                    # consistent with a cold-planned one -- same value flows into
                    # synthesizer nodes' expected_output_fingerprint and the
                    # handbook-key fallback. Fall back to semantic embed() only if
                    # the fingerprint carries no embedding.
                    task_emb = fingerprint.embedding if getattr(fingerprint, "embedding", None) else embed(current_task)
                    warm_dag = dag_from_best_plan(case["best_plan"], current_task, task_emb)
                    if warm_dag is not None:
                        self.ceo._resolve_agents(warm_dag, task_class)
                        return warm_dag, False

        move_id = graph_ops.move_id_from_directive(prev_directive) if prev_directive else None
        if (descent is not None and move_id
                and graph_ops.is_deterministic_move(move_id)
                and descent._incumbent_dag is not None):
            required = getattr(descent, "req", None)  # the RequiredStructure this descent targets
            required_roles = getattr(required, "required_roles", None) if required else None
            partition_pairs = getattr(descent._incumbent_ef, "partition_pairs", None)
            new_dag, changed = graph_ops.apply_move(
                descent._incumbent_dag, move_id, required=required_roles,
                partition_pairs=partition_pairs, req=required,
            )
            if changed:
                new_dag.dag_id = f"{descent._incumbent_dag.dag_id}+{move_id}"
                self.ceo._resolve_agents(new_dag, task_class)
                return new_dag, True

        dag = self.ceo.plan(
            current_task,
            run_index=self.run_count,
            task_class=task_class,
            directive=prev_directive,
            fingerprint=fingerprint,
        )
        return dag, False

    def _final_answer(self, trace: ExecutionTrace) -> str:
        if not trace.results:
            return "(no output)"
        for r in reversed(trace.results):
            if not r.error:
                return r.output
        return trace.results[-1].output

    def _persist(self) -> None:
        self.hb.save()
        import json, os
        os.makedirs(os.path.dirname(self._run_count_path) or ".", exist_ok=True)
        with open(self._run_count_path, "w") as f:
            json.dump({"run_count": self.run_count}, f)

    def _load_run_count(self) -> int:
        import json
        try:
            with open(self._run_count_path) as f:
                return int(json.load(f).get("run_count", 0))
        except (FileNotFoundError, Exception):
            return 0