"""Persistent run logger.

Writes one JSON record per run to a .jsonl file (append-only).
Each record captures everything needed to diagnose failure modes:
  - task fingerprint (Research's structural parse: species + specifics)
  - DAG (topology, depth, nodes with why-reasoning)
  - all node outputs with metrics
  - all LLM calls (prompt, response, latency, tag)
  - Delta report (verdict, failure metrics, hub failures)
  - reflection log if it fired

v2.0 — updated for TaskFingerprint/TaskSpecifics split (replaces the old
StructuredBrief single-object log entry).

Usage:
    logger = RunLogger("path/to/runs.jsonl")
    logger.write(run_result, llm_calls)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .schemas import dag_to_dict


def _safe_asdict(obj) -> Any:
    """dataclass -> dict, falling back gracefully for non-dataclass types."""
    try:
        return asdict(obj)
    except TypeError:
        return str(obj)


class RunLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def write(self, result, llm_calls: List[Dict]) -> None:
        """Serialize a RunResult + captured LLM calls and append to the log."""
        record = {
            "ts": time.time(),
            "run_global_index": getattr(result, "_run_index", None),
            "task": result.task,
            "task_label": getattr(result, "_task_label", None),

            # Research upstream parse — species (TaskFingerprint) + specifics
            "fingerprint": {
                "information_flow":   result.fingerprint.information_flow,
                "epistemic_stance":   result.fingerprint.epistemic_stance,
                "output_contract":    result.fingerprint.output_contract,
                "decomposability":    result.fingerprint.decomposability,
                "complexity":         result.fingerprint.complexity,
                "domain_volatility":  result.fingerprint.domain_volatility,
                "depth_cap":          result.fingerprint.depth_cap(),
                "requires_verifier":  result.fingerprint.requires_verifier(),
            },
            "specifics": {
                "real_goal":         result.specifics.real_goal,
                "constraints":       result.specifics.constraints,
                "resources_needed":  result.specifics.resources_needed,
                "structured_intent": result.specifics.structured_intent,
                "notes":             result.specifics.notes,
                "drift":             result.specifics.drift,
            },

            # Secondary brief + replan
            "secondary_brief": {
                "summary":        result.brief.summary,
                "key_findings":   result.brief.key_findings,
                "refined_intent": result.brief.refined_intent,
                "materiality":    result.brief.materiality,
                "drift":          result.brief.drift,
                "triggers_replan": result.brief.triggers_replan,
            },
            "replanned": result.replanned,

            # DAG: topology choice + per-node why-reasoning
            "dag": {
                "dag_id":       result.dag.dag_id,
                "topology":     result.dag.topology,
                "depth":        result.dag.depth,
                "why_topology": result.dag.why_topology,
                "why_depth":    result.dag.why_depth,
                "exploratory":  result.dag.exploratory,
                "nodes": [
                    {
                        "node_id":    n.node_id,
                        "intent":     n.intent,
                        "roles":      _safe_asdict(n.roles),
                        "deps":       n.dependencies,
                        "why":        _safe_asdict(n.why),
                        "assigned_agent": n.assigned_agent_id,
                    }
                    for n in result.dag.nodes
                ],
            },

            # Execution trace: per-node outputs + metrics
            "trace": {
                "total_tokens": result.trace.total_tokens,
                "wallclock_s":  result.trace.wallclock_s,
                "results": [
                    {
                        "node_id":            r.node_id,
                        "intent":             r.intent,
                        "output":             r.output,
                        "fingerprint_match":  r.fingerprint_match,
                        "role_function_match": r.role_function_match,
                        "cost_tokens":        r.cost_tokens,
                        "latency_s":          r.latency_s,
                        "error":              r.error,
                        "gated":              r.gated,
                    }
                    for r in result.trace.results
                ],
            },

            # Delta report
            "delta": {
                "verdict":            result.report.verdict,
                "ceo_feedback":       result.report.ceo_feedback,
                "research_feedback":  result.report.research_feedback,
                "structural":         result.report.structural,
                "runtime":            result.report.runtime,
                "failure":            result.report.failure,
                "semantic":           result.report.semantic,
                "satisfaction":       result.report.satisfaction,
                "centrality":         result.report.centrality,
                "hub_failures":       result.report.hub_failures,
                "granular_entries":   result.report.granular_entries,
            },

            # Reflection (if it fired this run)
            "reflection": (
                {
                    "triggered":       result.reflection.triggered,
                    "reason":          result.reflection.reason,
                    "resolved":        result.reflection.resolved,
                    "still_contested": result.reflection.still_contested,
                    "explorations":    result.reflection.explorations,
                    "tokens_spent":    result.reflection.tokens_spent,
                    "notes":           result.reflection.notes,
                }
                if result.reflection else None
            ),

            # Every LLM call made during this run
            "llm_calls": llm_calls,
        }

        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")