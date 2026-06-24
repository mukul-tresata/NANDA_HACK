"""The Loop — wires CEO -> Research -> Kernel -> Delta -> Handbook and exposes
the three communication modes.

  Standard  : full task pass. User sees the result. Delta audits after.
  Reflection: CEO<->Delta only, resolve contested entries within a token budget.
  Meta      : user talks to Delta directly, supplies explicit satisfaction.

Across interactions, Delta's writes accumulate so CEO plans better over time.
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
from .research import Brief, Research
from .schemas import DAG, ExecutionTrace
from . import bootstrap


@dataclass
class RunResult:
    task: str
    answer: str
    dag: DAG
    brief: Brief
    replanned: bool
    trace: ExecutionTrace
    report: DeltaReport
    clarification: Optional[str] = None
    reflection: Optional[ReflectionLog] = None


class Orchestrator:
    def __init__(self, cfg: Config | None = None, workdir: str = ".ceo_delta"):
        self.cfg = cfg or DEFAULT
        self.llm = LLMClient(self.cfg)
        self.ceo_hb = Handbook("ceo", self.cfg, path=f"{workdir}/ceo_handbook.json")
        self.research_hb = Handbook("research", self.cfg, path=f"{workdir}/research_handbook.json")
        if self.cfg.seed_handbook:
            bootstrap.ensure_seeded(self.ceo_hb)
            bootstrap.ensure_seeded(self.research_hb)
        self.ceo = CEO(self.ceo_hb, self.llm, self.cfg)
        self.research = Research(self.research_hb, self.llm, self.cfg)
        self.kernel = Kernel(self.llm, self.cfg)
        self.delta = Delta(self.ceo_hb, self.research_hb, self.cfg)
        self.reflection = Reflection(self.ceo, self.delta, self.kernel, self.llm, self.cfg)
        self.run_count = 0

    # -- standard mode --------------------------------------------------------
    def run(self, task: str, *, auto_clarify: bool = True,
            user_satisfaction: Optional[float] = None) -> RunResult:
        clarification = None
        needs, sim = self.ceo.needs_clarification(task)
        if needs and not auto_clarify:
            clarification = (f"Low handbook similarity ({sim:.2f}). Clarify scope "
                             "before I plan.")

        dag = self.ceo.plan(task, run_index=self.run_count)
        brief = self.research.investigate(dag)
        replanned = False
        if brief.triggers_replan:
            dag = self.ceo.replan(task, brief.summary, run_index=self.run_count)
            replanned = True

        trace = self.kernel.execute(dag, brief_context=brief.summary)
        answer = self._final_answer(trace)
        report = self.delta.audit(dag, trace, brief_drift=brief.drift,
                                  user_satisfaction=user_satisfaction)

        self.run_count += 1
        self._persist()

        # between-interaction reflection check
        refl = None
        ok, _ = should_reflect(self.run_count, self.ceo_hb, self.cfg)
        if ok:
            refl = self.reflection.run(self.run_count)
            self._persist()

        return RunResult(task=task, answer=answer, dag=dag, brief=brief,
                         replanned=replanned, trace=trace, report=report,
                         clarification=clarification, reflection=refl)

    # -- meta / feedback mode -------------------------------------------------
    def meta_feedback(self, task: str, satisfaction: float, note: str = "") -> str:
        """User talks to Delta directly with an explicit satisfaction signal in
        [0,1]. Delta folds it into the most recent entry for the task region."""
        from .embeddings import embed
        emb = embed(task)
        entry, sim = self.ceo_hb.best_match(emb)
        good = satisfaction >= 0.6
        for hb in (self.ceo_hb, self.research_hb):
            e, _ = hb.best_match(emb)
            if e:
                e.revision = (f"[user-satisfaction={satisfaction:.2f}] {note} | " + (e.revision or ""))[:300]
                hb.upsert_votes(emb, task[:120], e.topology_chosen or "linear",
                                e.depth_chosen or 1, outcome_good=good)
        self._persist()
        return (f"Delta recorded satisfaction={satisfaction:.2f} for task region "
                f"(match sim={sim:.2f}). good={good}.")

    # -- reflection mode (manual trigger) -------------------------------------
    def reflect_now(self) -> ReflectionLog:
        log = self.reflection.run(max(self.run_count, self.cfg.reflection_interval))
        self._persist()
        return log

    # -- helpers --------------------------------------------------------------
    def _final_answer(self, trace: ExecutionTrace) -> str:
        if not trace.results:
            return "(no output)"
        # the answer is the last non-errored node output (sink of the DAG)
        for r in reversed(trace.results):
            if not r.error:
                return r.output
        return trace.results[-1].output

    def _persist(self) -> None:
        self.ceo_hb.save()
        self.research_hb.save()
