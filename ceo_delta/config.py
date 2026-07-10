"""Central configuration and thresholds for the Conductor-Delta architecture."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- LLM backend --------------------------------------------------------
    # vLLM OpenAI-compatible server (self-hosted, default Qwen-family model).
    # To switch back to the Anthropic API: restore the ANTHROPIC BACKEND block
    # in llm.py (kept there, commented, as an exact mirror of the vLLM block)
    # and set anthropic_api_key.
    llm_base_url: str = os.environ.get("CEO_LLM_URL", "http://localhost:8000/v1")
    llm_model: str = os.environ.get("CEO_LLM_MODEL", "Qwen/Qwen3.6-35B-A3B")
    llm_max_tokens: int = 8000
    llm_timeout_s: int = 120
    llm_allow_stub: bool = True
    # only read if grounding_backend="web_search" (Anthropic-only tool, off by
    # default on the vLLM backend -- see grounding_backend below)
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    # The system's identity is determinism: the fingerprint (from Research),
    # the plan (from CEO), and execution should be as reproducible as the
    # backend allows. Sampling the fingerprint at temperature 1.0 was the
    # single largest source of UNINTENDED non-determinism -- same task ->
    # different fingerprint -> different F -> different convergence. The only
    # non-determinism the architecture WANTS is the content an escalation LLM
    # authors, and even that fires on a deterministic counter. Default 0.0.
    llm_temperature: float = 0.0

    # ---- Embeddings ---------------------------------------------------------
    embed_dim: int = 384

    # ---- Cold start ---------------------------------------------------------
    cold_start_runs: int = 3
    seed_handbook: bool = True
    clarify_similarity_threshold: float = 0.45

    # ---- Reflection mode ----------------------------------------------------
    reflection_interval: int = 5
    reflection_contested_trigger: int = 2
    reflection_token_budget: int = 8000
    reflection_max_explorations: int = 4

    # ---- Research replan ----------------------------------------------------
    replan_threshold: float = 0.70

    # ---- Multi-way conflict -------------------------------------------------
    conflict_dominance_margin: float = 0.20
    conflict_entropy_threshold: float = 0.85
    conflict_min_votes: int = 3

    # ---- Delta surprise / metrics -------------------------------------------
    surprise_factor: float = 2.0
    echo_cosine_threshold: float = 0.85

     # ---- Directive loop -----------------------------------------------------
    max_ceo_eval_iterations: int = 5
    delta_surface_threshold: float = 0.20   # Δe below this → surface immediately
    delta_echo_threshold: float = 0.3       # weighted echo above this → replan
    delta_mismatch_threshold: float = 0.4   # weighted mismatch above this → refine
    delta_fp_low_threshold: float = 0.4     # weighted fp below this → refine
    escalation_enabled: bool = True

    # ---- Handbook retrieval -------------------------------------------------
    handbook_top_k: int = 4

    # ---- AgentCard / AgentRegistry ------------------------------------------
    # CEO ignores cards with fewer runs than this (cold-start guard)
    agent_card_min_confidence: int = 3

    # CEO rejects agents whose trust_score is below this floor
    agent_card_min_trust: float = 0.35

    seed: int = 1234

    # ---- v2.1 Error tensor (replaces scalar delta_e) -------------------------
    # All thresholds below are HAND-SEEDED PROVISIONAL ESTIMATES from manual
    # inspection of a handful of node outputs in existing run logs, NOT
    # calibrated from "verdict=good" runs (that label is circular -- it comes
    # from the old scalar/rule-table pipeline this is replacing). Recalibrate
    # once escalation_min_cases_per_role real cases have accumulated per role.
    use_tensor_error: bool = True          # fallback switch -- False = old scalar path
    error_drift_threshold: float = 0.40
    error_echo_threshold: float = 0.45      # raw mean cosine sim among siblings
    error_cascade_threshold: float = 0.30
    error_role_threshold: float = 0.35
    error_resource_cost_max_usd: float = 0.50      # per-run ceiling, hand-set
    error_resource_latency_max_s: float = 180.0     # per-run ceiling, hand-set

    # gate/dedup
    cascade_drift_dedup_node_threshold: float = 0.35

    # hysteresis (one-iteration grace margin after a refine, per node+dim)
    hysteresis_margin: float = 0.15

    # directive clustering (primary/secondary multi-target grouping)
    directive_cluster_ratio: float = 0.6

    # escalation retrieval (case-based, replaces delta_rules.py)
    escalation_min_cases_per_role: int = 5
    escalation_sim_floor: float = 0.6

    # ---- v3.0 EF-driven planning adaptation ---------------------------------
    # E and F share one basis: [partition, flow, role, scale]. The verdict is
    # a pure function of E (all axes below threshold -> good) — no second
    # error model. Drift is GONE from planning error (it was execution-quality
    # geometry). These per-axis thresholds gate BOTH the verdict and the
    # coordinate-descent directive loop, so the two can never disagree again.
    use_ef_tensor: bool = True             # v3.0 master switch (False -> v2.1 tensor path)
    ef_partition_threshold: float = 0.70   # mean sibling cosine above this = redundant fan-out
    ef_flow_threshold: float = 0.25        # any realized-vs-required flow penalty over this fires
    ef_role_threshold: float = 0.35        # behavioral-band excess + missing-role penalty
    ef_scale_threshold: float = 0.34       # depth over budget by >1 level fires; mild underage tolerated
    ef_mixed_margin: float = 0.15          # single axis this far over -> "mixed" not "poor"
    # earned escalation: an (F, axis) pair must be declared irreducible this
    # many times (deterministic moves exhausted) before LLM escalation unlocks.
    ef_irreducible_escalate_threshold: int = 2
    # earned deprioritization: a move must cause collateral damage (a
    # different, previously-satisfied axis crossing its own threshold) this
    # many times for the same fingerprint before it's pushed to the back of
    # its axis's move list. Never removed -- still tried if nothing else
    # is left. Same "earn it, don't react to one incident" discipline.
    ef_side_effect_deprioritize_threshold: int = 2

    # ---- v3.5 Q-factor (content quality, RAGAS generation-side) -------------
    # ALL PROVISIONAL, to be calibrated non-circularly (same discipline as the
    # EF thresholds above -- hand-set from inspection, not from a label that
    # comes from the metric itself). Extends the EF basis with content axes
    # so the delivered verdict reflects ANSWER quality, not just plan shape.
    use_q_tensor: bool = True
    q_groundedness_threshold: float = 0.30   # PROVISIONAL: >0.30 unverified-per-claim = ungrounded
    q_relevance_threshold: float = 0.65      # PROVISIONAL: 1-cosine above this = off-topic (cosine<0.35)
    q_mixed_margin: float = 0.15
    q_assertion_entity_floor: int = 10       # entities below which citation-shortfall is discounted
    q_citation_target: float = 0.02          # target citation density for grounded factual content

    # ---- v3.6 Warm-start task-identity gate ---------------------------------
    # Warm-start may only reuse a cached best_plan when the incoming task is
    # (near-)IDENTICAL, semantically, to the task that produced it -- shape_key
    # is structural and shared across many distinct tasks, so a structural
    # match alone is not sufficient (see descent.py task_embedding and
    # orchestrator.py _next_dag identity gate).
    warm_start_similarity_threshold: float = 0.92

    # ---- v3.1 Retrieval grounding (retrieve() seam) -------------------------
    # Retriever-role nodes call the grounding seam instead of hallucinating.
    # Backend is swappable WITHOUT touching kernel/planner/EF: today web search,
    # later a NANDA-discovered retrieval agent. See grounding.py.
    # "web_search" is Claude's server-side tool -- it has no equivalent on the
    # vLLM/Qwen backend, so it is OFF by default here. Only re-enable it if
    # anthropic_api_key is set (grounding.py falls back cleanly to plain
    # generation if the key is missing).
    grounding_enabled: bool = True
    grounding_backend: str = "none"   # web_search (Anthropic-only) | none | (future) nanda
    # web_search_20260209 (dynamic filtering) needs Opus 4.8/4.7/4.6 or Sonnet 5/4.6;
    # Haiku 4.5 uses the basic variant. Only relevant if grounding_backend="web_search".
    web_search_tool_version: str = "web_search_20250305"
    web_search_max_uses: int = 5


DEFAULT = Config()


# ---------------------------------------------------------------------------
# Per-role behavioral bands (f_cite, f_comp, f_struct) -- E_role inputs.
# Kept here (not a separate file) because it's static data, same as the
# Config dataclass above -- no logic, no I/O, genuinely config-shaped.
#
# band = (lo, hi); hi=None means unbounded above, never penalized.
# PROVISIONAL -- hand-seeded from manual inspection, not statistically
# calibrated. Recalibrate once real run volume exists per role.
# ---------------------------------------------------------------------------

ROLE_BANDS = {
    "retriever": {
        "cite": (0.02, None),     # some grounding expected, no ceiling
        "comp": (1.0, None),      # must EXPAND info footprint -- one-sided
        "struct": (0.0, 0.5),     # low synthesis-formatting expected
    },
    "synthesizer": {
        "cite": (0.0, None),
        "comp": (0.0, 0.7),       # must compress upstream data
        "struct": (0.4, None),    # high structural cross-referencing
    },
    "verifier": {
        "cite": (0.04, None),     # heaviest citation requirement
        "comp": (0.0, 0.4),       # compress to audit table
        "struct": (0.3, None),
    },
    "generic": {
        "cite": (0.0, None),
        "comp": (0.0, None),
        "struct": (0.0, None),
        "_weights": {"cite": 0.0, "comp": 0.0, "struct": 0.0},  # no-op band --
        # zeroes all three excess terms in role_features.role_error, so an
        # unclassified role is never penalized on role error.
    },
}