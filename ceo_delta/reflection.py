"""Reflection mode (limitation #2) — made fully explicit.

TRIGGER (checked only *between* interactions, never mid single-pass):
  * standard-run counter reaches cfg.reflection_interval, OR
  * number of contested handbook entries >= cfg.reflection_contested_trigger.

WHAT IT DOES:
  CEO and Delta talk directly (no user). For each contested entry, CEO runs
  exploratory DAGs over the contested task region and Delta votes the outcomes
  back into the handbook until the conflict resolves.

EXIT (any of):
  * the targeted contested entry is no longer contested, OR
  * cfg.reflection_max_explorations exploratory DAGs have run, OR
  * the token budget cfg.reflection_token_budget is exhausted (runaway guard).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .config import Config, DEFAULT
from .ceo import CEO
from .delta import Delta
from .handbook import Handbook
from .kernel import Kernel
from .llm import LLMClient


@dataclass
class ReflectionLog:
    triggered: bool
    reason: str
    explorations: int = 0
    tokens_spent: int = 0
    resolved: List[str] = field(default_factory=list)
    still_contested: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def should_reflect(run_count: int, ceo_hb: Handbook, cfg: Config) -> tuple[bool, str]:
    contested = ceo_hb.contested_entries()
    if len(contested) >= cfg.reflection_contested_trigger:
        return True, f"{len(contested)} contested entries >= {cfg.reflection_contested_trigger}"
    if run_count > 0 and run_count % cfg.reflection_interval == 0:
        return True, f"run_count {run_count} hit interval {cfg.reflection_interval}"
    return False, ""


class Reflection:
    def __init__(self, ceo: CEO, delta: Delta, kernel: Kernel, llm: LLMClient,
                 cfg: Config | None = None):
        self.ceo = ceo
        self.delta = delta
        self.kernel = kernel
        self.llm = llm
        self.cfg = cfg or DEFAULT

    def run(self, run_count: int) -> ReflectionLog:
        ok, reason = should_reflect(run_count, self.ceo.hb, self.cfg)
        log = ReflectionLog(triggered=ok, reason=reason)
        if not ok:
            return log

        start_tokens = self.llm.total_tokens
        contested = self.ceo.hb.contested_entries()
        for entry in contested:
            while True:
                # budget + exploration ceilings (runaway guard)
                if self.llm.total_tokens - start_tokens >= self.cfg.reflection_token_budget:
                    log.notes.append("exited: token budget exhausted")
                    log.still_contested.append(entry.entry_id)
                    return self._finalize(log, start_tokens)
                if log.explorations >= self.cfg.reflection_max_explorations:
                    log.notes.append("exited: max explorations reached")
                    log.still_contested.append(entry.entry_id)
                    return self._finalize(log, start_tokens)

                task = f"[reflection] resolve plan for: {entry.task_summary}"
                dag = self.ceo.plan(task, run_index=0, force_exploratory=True)
                trace = self.kernel.execute(dag)
                self.delta.audit(dag, trace)
                log.explorations += 1

                if not entry.contested:
                    log.resolved.append(entry.entry_id)
                    log.notes.append(f"resolved {entry.entry_id} -> {entry.topology_chosen}")
                    break
        return self._finalize(log, start_tokens)

    def _finalize(self, log: ReflectionLog, start_tokens: int) -> ReflectionLog:
        log.tokens_spent = self.llm.total_tokens - start_tokens
        return log
