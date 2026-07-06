"""v3.5 Q-factor tests -- content quality, generation-side RAGAS proxies.

Offline-only, no LLM judge (quality.py has none by design). embed() loads a
local sentence-transformers model, same as test_composition.py; no network
backend is touched.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ceo_delta.config import Config
from ceo_delta.embeddings import cosine, embed
from ceo_delta.ef import final_verdict
from ceo_delta.quality import QTensor, compute_quality

cfg = Config()


# ---------------------------------------------------------------------------
# groundedness -- derived from the AUDIT, never the composed surface
# ---------------------------------------------------------------------------

def test_groundedness_high_when_audit_flags_unverified():
    # Denominator is distinct claim-like entities (numbers, named things) in
    # the UPSTREAM content the verifier audited -- not a literal count of the
    # word "claim" in the audit's own prose. The real verifier prompt
    # (kernel.py AUDIT MODE) never writes the word "claim"; it only emits
    # "[UNVERIFIED: <claim>]" lines for flagged items and stays silent on
    # ones that check out, so a word-count denominator always floored to ~1
    # and saturated groundedness to 1.0 the moment 2+ items were flagged --
    # a metric that could never discriminate. See quality.py module docstring.
    content_texts = ["Flight costs 15000 rupees from Bangalore to Bangkok"]
    audit = (
        "[UNVERIFIED: the 15000 rupee flight estimate]\n"
        "[UNVERIFIED: the Bangalore to Bangkok routing]"
    )
    q = compute_quality([0.0] * 4, "some composed answer", audit, content_texts, cfg)
    # 2 flagged out of 4 distinct entities (Flight, 15000, Bangalore, Bangkok)
    assert abs(q.groundedness - 0.5) < 1e-6


def test_groundedness_low_when_audit_clean():
    audit = "**Claim:** a is true\n**Claim:** b is true\n**Claim:** c is true"
    q = compute_quality([0.0] * 4, "some composed answer", audit, [], cfg)
    assert q.groundedness == 0.0


def test_composition_cannot_launder_gravel():
    # ANTI-LAUNDERING GUARANTEE: a long, clean, confident composed answer with
    # zero [UNVERIFIED] tags must NOT inflate groundedness when the audit
    # (the actual verification record) is full of unverified claims. If this
    # test fails, the "clean surface" is laundering ungrounded content and the
    # whole point of Stage 1 (measure the DELIVERED answer's quality) is lost.
    composed_answer = (
        "The project succeeded across every dimension, delivering strong "
        "results with clear, confident, well-organized prose that reads "
        "as fully authoritative and complete in every respect."
    )
    audit = (
        "**Claim:** revenue grew 40% [UNVERIFIED: no source]\n"
        "**Claim:** market share doubled [UNVERIFIED: no source]\n"
        "**Claim:** customer churn fell [UNVERIFIED: no source]\n"
        "**Claim:** headcount grew\n"
    )
    q = compute_quality([0.0] * 4, composed_answer, audit, [], cfg)
    assert q.groundedness > 0.5   # gravel in the audit wins, not the clean surface


# ---------------------------------------------------------------------------
# relevance -- computed ON the composed deliverable
# ---------------------------------------------------------------------------

def test_relevance_low_for_on_topic():
    task = "Summarize the causes of the French Revolution."
    task_embedding = embed(task)
    on_topic = "The French Revolution was caused by fiscal crisis, social inequality, and Enlightenment ideas."
    off_topic = "The recipe calls for two cups of flour, a teaspoon of baking soda, and butter."

    q_on = compute_quality(task_embedding, on_topic, "", [], cfg)
    q_off = compute_quality(task_embedding, off_topic, "", [], cfg)

    assert q_on.relevance < q_off.relevance   # relative comparison -- embeddings are model-dependent


def test_relevance_zeroed_for_generation_stance():
    # A generated artifact (the story itself) is legitimately semantically
    # distant from the sentence instructing it -- cosine(story, instruction)
    # is not a valid relevance proxy for generation stance. Same inputs must
    # produce q.relevance == 0.0 under "generation" but > 0 under the default
    # "synthesis" stance, proving the gate -- not the inputs -- changed the
    # outcome.
    task = "Write a sci-fi story about a lighthouse keeper who discovers a signal from deep space."
    task_embedding = embed(task)
    story = (
        "Mara had kept the light for eleven winters when the static in her "
        "old radio finally resolved into a rhythm no storm could explain. "
        "She climbed the spiral stairs at midnight, logbook in hand, and "
        "listened as the pulses spelled out coordinates far beyond the reef."
    )

    q_generation = compute_quality(task_embedding, story, "", [], cfg, epistemic_stance="generation")
    assert q_generation.relevance == 0.0

    q_synthesis = compute_quality(task_embedding, story, "", [], cfg, epistemic_stance="synthesis")
    assert q_synthesis.relevance > 0.0


def test_groundedness_zeroed_for_generation_stance():
    # A verifier that audits a generated artifact (fiction/design/code) flags
    # its self-authored claims as [UNVERIFIED] because there is no retrieved
    # context to trace them to -- RAGAS faithfulness is category-inapplicable
    # for generation. A story-auditing verifier that flags every sentence must
    # NOT tank groundedness. Same audit, generation stance -> 0.0; non-
    # generation stance -> the audit path fires (>0), proving the stance gate,
    # not the audit, changed the outcome.
    story_audit = (
        "[UNVERIFIED: The storm on Kepler-186f did not howl; it screamed.]\n"
        "[UNVERIFIED: Elara adjusted the focus ring on the great lens.]\n"
        "[UNVERIFIED: She had been here for three years.]"
    )
    content = ["A story set on Kepler-186f with a keeper named Elara over three years."]

    q_generation = compute_quality([0.0] * 4, "the story", story_audit, content, cfg,
                                   epistemic_stance="generation")
    assert q_generation.groundedness == 0.0

    q_synthesis = compute_quality([0.0] * 4, "the story", story_audit, content, cfg,
                                  epistemic_stance="synthesis")
    assert q_synthesis.groundedness > 0.0


def test_relevance_unchanged_for_non_generation():
    # Regression guard: default (synthesis) stance still computes the
    # cosine-based relevance exactly as before this change.
    task = "Summarize the causes of the French Revolution."
    task_embedding = embed(task)
    on_topic = "The French Revolution was caused by fiscal crisis, social inequality, and Enlightenment ideas."

    q_default = compute_quality(task_embedding, on_topic, "", [], cfg)
    q_explicit_synthesis = compute_quality(task_embedding, on_topic, "", [], cfg, epistemic_stance="synthesis")
    expected = 1.0 - cosine(embed(on_topic), task_embedding)

    assert abs(q_default.relevance - expected) < 1e-6
    assert abs(q_explicit_synthesis.relevance - expected) < 1e-6


# ---------------------------------------------------------------------------
# final_verdict -- content quality can flip a structurally-good run
# ---------------------------------------------------------------------------

def test_final_verdict_flips_on_bad_groundedness():
    ef_dict = {
        "partition": 0.0, "flow": 0.0, "role": 0.0, "scale": 0.0,
    }
    ef_thresholds = {
        "partition": cfg.ef_partition_threshold, "flow": cfg.ef_flow_threshold,
        "role": cfg.ef_role_threshold, "scale": cfg.ef_scale_threshold,
    }
    q_dict = {"groundedness": 0.9, "relevance": 0.0}   # far over threshold
    q_thresholds = {
        "groundedness": cfg.q_groundedness_threshold, "relevance": cfg.q_relevance_threshold,
    }
    verdict, good = final_verdict(ef_dict, q_dict, ef_thresholds, q_thresholds, cfg.q_mixed_margin)
    assert verdict == "poor"
    assert good is False


# ---------------------------------------------------------------------------
# no-verifier fallback -- citation shortfall gated by assertion density,
# and only even consulted when a verifier was actually expected
# ---------------------------------------------------------------------------

def test_no_verifier_citation_fallback():
    # many entities/numbers, no citation anchors -> shortfall is real and NOT
    # discounted -- but ONLY when this was an evolving-domain task that
    # should have produced a verifier audit and didn't (verifier_expected=True).
    dense_content = [
        "Revenue was $4.2M in Q3 2025, up from $3.1M in Q2 2025. "
        "Acme Corp, Beta Industries, and Gamma LLC each grew headcount by 12%, 8%, and 15% respectively. "
        "The Boston, Chicago, and Denver offices all reported record numbers in 2026."
    ]
    q_dense = compute_quality([0.0] * 4, "answer", "", dense_content, cfg, verifier_expected=True)
    assert q_dense.groundedness > 0.0

    # pure low-entity reasoning, no citations -- nothing to cite, so the
    # assertion_factor discounts the shortfall down toward zero
    sparse_content = [
        "It follows that if the premise holds then the conclusion must also hold, "
        "by the basic rule of inference, so the argument is valid regardless of context."
    ]
    q_sparse = compute_quality([0.0] * 4, "answer", "", sparse_content, cfg, verifier_expected=True)
    assert q_sparse.groundedness < q_dense.groundedness
    assert q_sparse.groundedness < 0.1


def test_stable_domain_no_verifier_expected_is_never_penalized():
    # A self-contained explanation (e.g. quicksort) on a task that never
    # required a verifier: entity-dense, zero citations, but there is no
    # external context these claims were ever meant to be grounded against.
    # verifier_expected=False (the default) must skip the citation-shortfall
    # heuristic entirely rather than false-firing on entity density -- this
    # is the exact false-fire the fallback used to produce on stable-domain
    # tasks like explaining an algorithm.
    entity_dense_explanation = [
        "Quicksort selects a pivot via the Lomuto or Hoare scheme, then "
        "partitions the array so elements less than the pivot move left and "
        "elements greater move right, achieving O(n log n) average time."
    ]
    q = compute_quality([0.0] * 4, "answer", "", entity_dense_explanation, cfg)
    assert q.groundedness == 0.0
