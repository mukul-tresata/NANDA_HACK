"""Escalation case store + LLM prompt/parsing — one cohesive unit.

Replaces the hand-written rule table (old delta_rules.py). Delta retrieves
the nearest precedent case by weighted cosine similarity over the
excess-vector, gated by node role. Below MIN_CASES (per role-bucket) or
below the similarity floor, falls through to LLM escalation, which both
issues a directive AND structures the new case for storage.

Kept as one file (not split into store/prompt) because these three things
share a single concern -- the lifecycle of a case -- store defines the
schema, prompt asks the LLM to fill that exact schema, parser validates
the fill before it's allowed to become a stored case. Splitting them
across files would separate steps of one pipeline rather than separate
genuinely independent responsibilities.

outcome_quality is tracked but NOT used in retrieval scoring -- too few
observations to trust as a weight; logged for inspection/future use only.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# Fixed dimension order -- must match error_tensor.ErrorTensor.DIMS exactly.
TENSOR_DIMS = ("drift", "echo", "cascade", "role", "resource")

MIN_CASES_PER_ROLE = 5
SIM_FLOOR = 0.6


def _case_id() -> str:
    return f"esc_{uuid.uuid4().hex[:8]}"


# ===========================================================================
# Case schema + store
# ===========================================================================

@dataclass
class EscalationCase:
    case_id: str
    timestamp: float
    run_id: str
    task_species: Dict[str, str]
    node_context: Dict[str, str]
    e_tensor_raw: Dict[str, float]
    e_tensor_excess: Dict[str, float]
    gates_applied: Dict[str, bool]
    directive_issued: Dict[str, str]
    llm_escalation_meta: Dict[str, object] = field(default_factory=dict)
    outcome: Dict[str, object] = field(default_factory=lambda: {
        "status": "pending", "dim_checked": None, "delta_next_iter": None,
        "collateral_damage": None, "outcome_quality": None, "n_observations": 0,
    })
    case_summary: str = ""
    synthetic: bool = False

    def excess_vector(self) -> List[float]:
        return [self.e_tensor_excess.get(d, 0.0) for d in TENSOR_DIMS]


def _cosine_vec(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)


class EscalationStore:
    def __init__(self, path: str = ".ceo_delta/escalations.jsonl"):
        self.path = path
        self._cases: List[EscalationCase] = []
        self._load()

    # -- persistence ----------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    self._cases.append(EscalationCase(**d))
                except Exception:
                    continue  # never let a corrupt line break loading

    def append(self, case: EscalationCase) -> None:
        self._cases.append(case)
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(asdict(case)) + "\n")
        except Exception:
            pass  # never let logging break the run

    def rewrite_all(self) -> None:
        """Used after outcome backfill mutates an existing case."""
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w") as f:
                for c in self._cases:
                    f.write(json.dumps(asdict(c)) + "\n")
        except Exception:
            pass

    def seed_synthetic(self, cases: List[EscalationCase]) -> None:
        for c in cases:
            c.synthetic = True
            self.append(c)

    # -- retrieval --------------------------------------------------------

    def cases_for_role(self, role: str) -> List[EscalationCase]:
        return [c for c in self._cases if c.node_context.get("role") == role]

    def best_match(
        self, excess_vector: Dict[str, float], role: str,
        min_cases: int = MIN_CASES_PER_ROLE, sim_floor: float = SIM_FLOOR,
    ) -> Tuple[Optional[EscalationCase], float]:
        pool = self.cases_for_role(role)
        if len(pool) < min_cases:
            return None, 0.0  # cold start -- force LLM escalation

        query = [excess_vector.get(d, 0.0) for d in TENSOR_DIMS]
        best, best_sim = None, 0.0
        for c in pool:
            sim = _cosine_vec(query, c.excess_vector())
            if sim > best_sim:
                best, best_sim = c, sim

        if best_sim < sim_floor:
            return None, best_sim
        return best, best_sim

    # -- outcome backfill ---------------------------------------------------

    def backfill_outcome(
        self, case_id: str, dim_checked: str, delta_next_iter: float,
        collateral_damage: Optional[Dict[str, float]] = None,
    ) -> None:
        for c in self._cases:
            if c.case_id == case_id:
                c.outcome["status"] = "resolved"
                c.outcome["dim_checked"] = dim_checked
                c.outcome["delta_next_iter"] = round(delta_next_iter, 4)
                c.outcome["collateral_damage"] = collateral_damage
                # outcome_quality computed + stored for inspection only,
                # never fed back into retrieval scoring.
                improvement = max(0.0, -delta_next_iter)  # negative delta = improvement
                c.outcome["outcome_quality"] = round(min(1.0, improvement), 4)
                c.outcome["n_observations"] = c.outcome.get("n_observations", 0) + 1
                self.rewrite_all()
                return


# ===========================================================================
# Escalation prompt construction + structured-response validation.
#
# The escalation LLM call does double duty: it returns a firm CEO-facing
# directive AND a structured case ready to drop into escalations.jsonl.
# The schema is fixed here -- the LLM fills it, never invents it.
# ===========================================================================

ESCALATION_PROMPT_TEMPLATE = """You are Delta, diagnosing an unclassified execution failure.

TASK SPECIES: {flow}/{epistemic_stance}/{output_contract}, complexity={complexity}

FAILING NODE: {node_id} (role={role}), deps={deps}

ERROR TENSOR (excess over threshold, 0=compliant):
  drift={drift:.3f}  echo={echo:.3f}  cascade={cascade:.3f}
  role={role_err:.3f}    resource={resource:.3f}

NODE OUTPUT (truncated): {output_snippet}
NODE INTENT: {intent}

No existing precedent matched this failure shape closely enough (best similarity={best_sim:.2f}, below retrieval floor or insufficient case history).

Return ONLY valid JSON matching this exact schema, no text outside the JSON object:
{{
  "directive": {{
    "action": "refine" | "replan",
    "target": "<node_id or 'topology'>",
    "constraint": "<specific, measurable, imperative correction>",
    "why": "<one sentence, diagnostic, not advisory>"
  }},
  "case_summary": "<one sentence describing this failure shape for future retrieval>"
}}

The directive must be a firm correction with a concrete target value, not a suggestion.
"""


def build_escalation_prompt(
    fingerprint, node_id: str, role: str, deps, output: str, intent: str,
    excess: Dict[str, float], best_sim: float,
) -> str:
    return ESCALATION_PROMPT_TEMPLATE.format(
        flow=fingerprint.information_flow,
        epistemic_stance=fingerprint.epistemic_stance,
        output_contract=fingerprint.output_contract,
        complexity=fingerprint.complexity,
        node_id=node_id, role=role, deps=deps,
        drift=excess.get("drift", 0.0), echo=excess.get("echo", 0.0),
        cascade=excess.get("cascade", 0.0), role_err=excess.get("role", 0.0),
        resource=excess.get("resource", 0.0),
        output_snippet=(output or "")[:600],
        intent=intent, best_sim=best_sim,
    )


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_escalation_response(raw: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Returns (parsed_dict, error). Coerces against the fixed schema --
    never trusts LLM-invented structure beyond these two top-level keys."""
    match = _JSON_BLOCK.search(raw or "")
    if not match:
        return None, "no_json_found"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return None, f"json_decode_error: {e}"

    directive = data.get("directive")
    if not isinstance(directive, dict):
        return None, "missing_directive"
    required = {"action", "target", "constraint", "why"}
    if not required.issubset(directive.keys()):
        return None, f"directive_missing_fields: {required - directive.keys()}"
    if directive["action"] not in ("refine", "replan"):
        return None, f"invalid_action: {directive['action']}"
    if "case_summary" not in data:
        data["case_summary"] = ""

    return data, None