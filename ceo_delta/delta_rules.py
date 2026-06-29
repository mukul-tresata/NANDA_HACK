"""Deterministic rule table for Delta directive generation.

Each rule is a function that takes the computed metrics and returns a
hint string if the rule fires, or None if it doesn't apply.

Rules are evaluated in priority order. First match wins.
When no rule fires, Delta escalates to LLM fallback.

Adding new rules: write a function with signature
    (failure, semantic, structural, dag_topology, dag_depth) -> Optional[str]
and append it to RULES at the bottom of this file.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

RuleFunc = Callable[
    [Dict, Dict, Dict, str, int],
    Optional[str]
]


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------

def rule_hub_cascade(failure, semantic, structural, topology, depth) -> Optional[str]:
    """Cascade originating from a hub node — structurally critical failure."""
    if failure["weighted_cascade_rate"] > 0.1:
        return (
            f"cascade detected (weighted_cascade={failure['weighted_cascade_rate']:.2f}) — "
            f"isolate hub nodes with guard/verifier nodes upstream; "
            f"consider reducing fan-out to limit blast radius"
        )
    return None


def rule_combined_echo_mismatch(failure, semantic, structural, topology, depth) -> Optional[str]:
    """Both echo and role mismatch present simultaneously — structural confusion."""
    if failure["weighted_echo_rate"] > 0.2 and failure["weighted_role_mismatch_rate"] > 0.2:
        return (
            f"combined echo+mismatch signal "
            f"(echo={failure['weighted_echo_rate']:.2f}, mismatch={failure['weighted_role_mismatch_rate']:.2f}) — "
            f"DAG structure may be fundamentally misaligned with task type; "
            f"reconsider topology from scratch rather than patching individual nodes"
        )
    return None


def rule_high_echo(failure, semantic, structural, topology, depth) -> Optional[str]:
    """Sibling nodes producing near-identical outputs."""
    if failure["weighted_echo_rate"] > 0.3:
        return (
            f"high echo rate (weighted_echo={failure['weighted_echo_rate']:.2f}) — "
            f"sibling nodes are converging on the same output; "
            f"differentiate intents more aggressively or reduce fan-out width"
        )
    return None


def rule_role_mismatch(failure, semantic, structural, topology, depth) -> Optional[str]:
    """Nodes executing in the wrong functional role."""
    if failure["weighted_role_mismatch_rate"] > 0.4:
        return (
            f"high role-function mismatch (weighted_mismatch={failure['weighted_role_mismatch_rate']:.2f}) — "
            f"structural roles must be exactly one of: fan-out, join, linear, hierarchical, hub; "
            f"review node role assignments in next plan"
        )
    return None


def rule_low_fp_high_drift(failure, semantic, structural, topology, depth) -> Optional[str]:
    """Research brief drifted significantly AND fingerprint is low."""
    if semantic["brief_drift"] > 0.15 and semantic["weighted_fingerprint_match"] < 0.5:
        return (
            f"brief drift + low fingerprint (drift={semantic['brief_drift']:.2f}, "
            f"fp={semantic['weighted_fingerprint_match']:.2f}) — "
            f"research brief materially changed task understanding but DAG wasn't replanned; "
            f"force replan on high-drift briefs"
        )
    return None


def rule_low_fingerprint(failure, semantic, structural, topology, depth) -> Optional[str]:
    """Outputs semantically far from intended fingerprints."""
    if semantic["weighted_fingerprint_match"] < 0.4:
        return (
            f"low fingerprint match (weighted_fp={semantic['weighted_fingerprint_match']:.2f}) — "
            f"node outputs are diverging from intended semantics; "
            f"sharpen node intent strings and reduce ambiguity in synthesizer targets"
        )
    return None


def rule_critical_path_overrun(failure, semantic, structural, topology, depth) -> Optional[str]:
    """Execution critical path significantly longer than planned depth."""
    if structural["critical_path"] > depth + 1:
        return (
            f"critical path overrun (path={structural['critical_path']:.0f} vs depth={depth}) — "
            f"plan depth underestimates true execution chain; "
            f"either increase planned depth or flatten dependency graph"
        )
    return None


def rule_high_error_rate(failure, semantic, structural, topology, depth) -> Optional[str]:
    """Significant fraction of nodes erroring out."""
    if failure["weighted_error_rate"] > 0.2:
        return (
            f"high error rate (weighted_error={failure['weighted_error_rate']:.2f}) — "
            f"multiple nodes failed during execution; "
            f"simplify node intents and reduce dependency complexity"
        )
    return None


# ---------------------------------------------------------------------------
# Rule registry — evaluated in order, first match wins
# ---------------------------------------------------------------------------

RULES: list[RuleFunc] = [
    rule_hub_cascade,
    rule_combined_echo_mismatch,
    rule_high_echo,
    rule_role_mismatch,
    rule_low_fp_high_drift,   # moved up to avoid shadowing
    rule_low_fingerprint,
    rule_critical_path_overrun,
    rule_high_error_rate,
]


def lookup_hint(
    failure: Dict,
    semantic: Dict,
    structural: Dict,
    dag_topology: str,
    dag_depth: int,
) -> Optional[str]:
    """Evaluate rules in priority order. Return first matching hint or None."""
    for rule in RULES:
        hint = rule(failure, semantic, structural, dag_topology, dag_depth)
        if hint:
            return hint
    return None