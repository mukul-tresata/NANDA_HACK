"""Functional Behavioral Alignment (E_role) — feature extraction + banded error.

Replaces the old keyword-scan `role_function_match: bool` with a deterministic,
text-derived behavioral profile compared against per-role BANDS (not point
archetypes — see config CHANGELOG for why point-distance was dropped).

Pure functions only. No LLM calls, no state. Computed entirely from
NodeResult.output text + upstream parent outputs already available at
audit() time.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_ANCHOR_PATTERN = re.compile(r"\[n\d+\]|https?://\S+|\(\s*n\d+\s*\)")
_STRUCT_TOKEN_PATTERN = re.compile(
    r"^\s*[-*]\s|^\s*\d+\.\s|^\s*\|.*\|\s*$|^#{1,6}\s|^\s*```|:\s*$", re.MULTILINE
)
_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-zA-Z0-9\-]{2,}\b|\b\d[\d,.]*\b")


def _tokenize_count(text: str) -> int:
    return max(1, len(text.split()))


def _line_count(text: str) -> int:
    return max(1, len(text.splitlines()))


def _entity_set(text: str) -> set:
    return set(_ENTITY_PATTERN.findall(text or ""))


def citation_density(output: str) -> float:
    """f_cite: grounded anchors per token."""
    anchors = len(_ANCHOR_PATTERN.findall(output or ""))
    return anchors / _tokenize_count(output)


def entropy_compression(output: str, parent_outputs: Iterable[str]) -> float:
    """f_comp: unique entities in output vs combined unique entities in parents.

    Unbounded above (a retriever expanding far past 1.0 is GOOD, not bad —
    this is exactly why role error uses one-sided bands, not point distance).
    Falls back to 1.0 (neutral) if there are no parents (root node).
    """
    parent_text = " ".join(p for p in parent_outputs if p)
    parent_entities = _entity_set(parent_text)
    if not parent_entities:
        return 1.0
    out_entities = _entity_set(output)
    return len(out_entities) / max(1, len(parent_entities))


def structural_density(output: str) -> float:
    """f_struct: density of structural layout markers per line."""
    matches = len(_STRUCT_TOKEN_PATTERN.findall(output or ""))
    return matches / _line_count(output)


def behavioral_profile(output: str, parent_outputs: Iterable[str]) -> Tuple[float, float, float]:
    """sigma(O_n) = [f_cite, f_comp, f_struct]"""
    return (
        citation_density(output),
        entropy_compression(output, parent_outputs),
        structural_density(output),
    )


# ---------------------------------------------------------------------------
# Band-based error (replaces point-distance archetype matrix)
# ---------------------------------------------------------------------------

# band = (lo, hi); hi=None means unbounded above (never penalize overshoot)
RoleBand = Dict[str, Tuple[float, float | None]]


def _excess(value: float, band: Tuple[float, float | None]) -> float:
    lo, hi = band
    if value < lo:
        return lo - value
    if hi is not None and value > hi:
        return value - hi
    return 0.0


def role_error(
    sigma: Tuple[float, float, float],
    bands: Dict[str, RoleBand],
    role: str,
) -> Tuple[float, Dict[str, float]]:
    """Returns (scalar role error for this node, per-feature excess breakdown).

    Error is a weighted SUM of per-dimension band-excess, not a Euclidean
    distance to a point — see CHANGELOG. Zero inside the band on every
    dimension means zero error, regardless of how "centered" the node is.
    """
    role_bands = bands.get(role, bands.get("generic"))
    if role_bands is None:
        return 0.0, {}

    f_cite, f_comp, f_struct = sigma
    excess = {
        "cite": _excess(f_cite, role_bands["cite"]),
        "comp": _excess(f_comp, role_bands["comp"]),
        "struct": _excess(f_struct, role_bands["struct"]),
    }
    weights = role_bands.get("_weights", {"cite": 1.0, "comp": 1.0, "struct": 1.0})
    err = sum(excess[k] * weights.get(k, 1.0) for k in excess)
    return err, excess