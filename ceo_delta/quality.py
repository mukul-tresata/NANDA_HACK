"""v3.5 — Q-factor (content quality). Generation-side RAGAS, mechanically.

This is the GENERATION side of RAGAS, computed with NO LLM judge anywhere in
this module — regex, cosine, and existing role_features functions only:

  groundedness  ~= RAGAS faithfulness. v1 proxy: the verifier's [UNVERIFIED]
    flags in the audit ARE the non-entailed claims. The denominator is the
    count of distinct claim-like entities (numbers, named things) in the
    UPSTREAM content the verifier was auditing -- not a literal count of the
    word "claim" in the audit's own prose, which the verifier's prompt
    (kernel.py's AUDIT MODE instructions) never actually writes; it only
    emits "[UNVERIFIED: <claim>]" lines for flagged items and stays silent
    on ones that check out, so a word-count denominator floors to ~0 and the
    ratio saturates to 1.0 the moment 2+ items are flagged. This is a proxy,
    not a real NLI check -- the upgrade path is to replace the regex count
    with an actual NLI entailment model in a later phase, without changing
    the QTensor shape or the verdict wiring.

  relevance     ~= RAGAS answer-relevancy, embedding-similarity variant:
    1 - cosine(embed(composed_answer), task_embedding). This only holds for
    non-generative stances (retrieval/synthesis/verification), where the
    answer echoes the task's domain. For `generation` stance the deliverable
    is a NEW artifact (story/design/code) that is legitimately semantically
    distant from the instruction describing it, so the cosine proxy is
    invalid there and relevance error is forced to 0 instead (mirrors the
    `verifier_expected` gate on groundedness: condition the metric on the
    fingerprint axis that decides whether it even applies).

The retrieval-side RAGAS metrics (context precision / context recall) are
DEFERRED until a real grounding/retrieval backend exists (see grounding.py) --
there is no retrieved-context set to score against yet.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from .embeddings import cosine, embed
from .role_features import citation_density, _entity_set

_UNVERIFIED_RE = re.compile(r"\[UNVERIFIED", re.I)


@dataclass
class QTensor:
    groundedness: float = 0.0        # ERROR in [0,1], higher=worse (= 1 - faithfulness proxy)
    relevance: float = 0.0           # ERROR in [0,1], higher=worse (= 1 - answer-relevancy)
    faithfulness_score: float = 1.0  # raw display [0,1] = 1 - groundedness
    relevancy_score: float = 1.0     # raw display [0,1] = 1 - relevance

    def as_dict(self) -> Dict[str, float]:
        return {"groundedness": round(self.groundedness, 4), "relevance": round(self.relevance, 4)}


Q_AXES = ("groundedness", "relevance")


def compute_quality(task_embedding, composed_answer: str, audit: str,
                     content_texts: List[str], cfg, verifier_expected: bool = False,
                     epistemic_stance: str = "synthesis") -> QTensor:
    """Measures content quality of the DELIVERED answer.

    Groundedness is deliberately derived from `audit`, NOT `composed_answer`
    -- the CEO's composition step gives the deliverable a clean, confident
    voice, and a clean surface must not be able to launder ungrounded claims.

    If there is no audit, `verifier_expected` decides what "no audit" means:
    - False (stable-domain / self-contained reasoning task -- no verifier was
      ever forced, e.g. explaining an algorithm): RAGAS faithfulness is not
      applicable -- there is no external context these claims were meant to
      be grounded against, so groundedness error is 0, not penalized. Falling
      back to a citation-density heuristic here would conflate "nothing to
      cite because the answer is self-contained" with "hiding sources" --
      exactly the false-fire an entity-dense-but-legitimate explanation
      (e.g. quicksort, naming Big-O/pivot/Lomuto) would trigger.
    - True (evolving-domain task that SHOULD have produced an audit but
      didn't -- an execution gap, not a domain choice): fall back to a
      citation-density shortfall gated by an assertion-density factor, as a
      degraded-but-real signal that something citable went unaudited.
    """
    # -- groundedness (faithfulness proxy) -----------------------------------
    # For generation stance, faithfulness-to-retrieved-context is category-
    # inapplicable: the deliverable is produced from the model's parametric
    # knowledge, not retrieved context, so there is nothing for its claims to
    # be "faithful to". A verifier that happens to audit a generated artifact
    # (e.g. flagging every sentence of a story as [UNVERIFIED]) yields a
    # VACUOUS signal, not a quality measurement -- so do not penalize. This is
    # the same stance-conditioning as the relevance gate below and the
    # verifier_expected gate: RAGAS faithfulness is DEFINED relative to a
    # retrieved-context set, which generation does not have.
    if epistemic_stance == "generation":
        groundedness = 0.0
    elif (audit or "").strip():
        u = len(_UNVERIFIED_RE.findall(audit or ""))
        upstream_text = "\n".join(content_texts)
        total_claims = max(len(_entity_set(upstream_text)), u, 1)
        groundedness = min(1.0, u / total_claims)
    elif not verifier_expected:
        groundedness = 0.0
    else:
        text = "\n".join(content_texts)
        cd = citation_density(text)
        ent = len(_entity_set(text))
        assertion_factor = min(1.0, ent / max(1, cfg.q_assertion_entity_floor))
        shortfall = max(0.0, 1.0 - cd / max(1e-9, cfg.q_citation_target))
        groundedness = shortfall * assertion_factor

    # -- relevance (answer-relevancy proxy, embedding variant) --------------
    # For generation stance the answer is a NEW artifact (story/design/code)
    # that is legitimately distant from the instruction describing it, so
    # answer-vs-task cosine is not a valid relevance proxy -- mechanically
    # measuring topical relevance here would need an LLM judge (deliberately
    # excluded). Do not penalize. Mirrors the verifier_expected gate on
    # groundedness: condition the metric on the fingerprint axis that decides
    # whether it even applies.
    if epistemic_stance == "generation":
        relevance = 0.0
    elif (composed_answer or "").strip():
        relevance = 1.0 - cosine(embed(composed_answer), task_embedding)
    else:
        relevance = 1.0

    groundedness = max(0.0, min(1.0, groundedness))
    relevance = max(0.0, min(1.0, relevance))

    return QTensor(
        groundedness=groundedness, relevance=relevance,
        faithfulness_score=1.0 - groundedness, relevancy_score=1.0 - relevance,
    )
