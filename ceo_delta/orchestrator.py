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
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .ceo import CEO
from .config import Config, DEFAULT
from .delta import Delta, DeltaReport
from .handbook import Handbook
from .kernel import Kernel
from .llm import LLMClient
from .reflection import Reflection, ReflectionLog, should_reflect
from .research import Brief, Research, TaskFingerprint, TaskSpecifics
from .runlog import RunLogger
from .schemas import AgentRegistry, DAG, DeltaDirective, ExecutionTrace
from . import bootstrap


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
    # directive loop fields
    iterations: int = 1
    directive_history: List[DeltaDirective] = field(default_factory=list)
    final_delta_e: float = 0.0


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
        )
        self.reflection = Reflection(
            self.ceo, self.delta, self.kernel, self.llm, self.cfg
        )
        self.run_count = 0

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
            # CEO plans (fingerprint drives retrieval + modifier context;
            # directive hint injected if available)
            dag = self.ceo.plan(
                current_task,
                run_index=self.run_count,
                task_class=task_class,
                directive=prev_directive,
                fingerprint=fingerprint,
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
                brief = self.research.investigate(dag, specifics)
                if brief.triggers_replan:
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

            directive_history.append(directive)

            # track best answer by lowest delta_e
            if best_report is None or report.delta_e < best_report.delta_e:
                best_answer = answer
                best_report = report
                best_dag = dag
                best_trace = trace

            prev_delta_e = report.delta_e
            prev_directive = directive

            # termination check
            if directive.action == "surface":
                break

            # prepare next iteration
            if directive.action == "refine" and directive.replan_hint:
                # enrich the task with gap information
                current_task = (
                    f"{specifics.structured_intent}\n\n"
                    f"[DELTA REFINEMENT REQUEST]: {directive.replan_hint}"
                )
            elif directive.action == "replan":
                # full reset to original intent — CEO will see directive in prompt
                current_task = specifics.structured_intent

        # -------------------------------------------------------------------

        self.run_count += 1
        self._persist()

        refl = None
        ok, _ = should_reflect(self.run_count, self.hb, self.cfg)
        if ok:
            refl = self.reflection.run(self.run_count)
            self._persist()

        result = RunResult(
            task=task,
            answer=best_answer,
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

    def _final_answer(self, trace: ExecutionTrace) -> str:
        if not trace.results:
            return "(no output)"
        for r in reversed(trace.results):
            if not r.error:
                return r.output
        return trace.results[-1].output

    def _persist(self) -> None:
        self.hb.save()