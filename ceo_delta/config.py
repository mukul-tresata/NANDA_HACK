"""Central configuration and thresholds for the CEO-Delta architecture."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- LLM backend --------------------------------------------------------
    # llm_base_url: str = os.environ.get("CEO_LLM_URL", "http://10.8.0.23:8001/v1")
    # llm_model: str = os.environ.get("CEO_LLM_MODEL", "my-model")
    llm_base_url : str = os.environ.get("CEO_LLM_URL", "https://api.anthropic.com/v1")
    llm_model : str = os.environ.get("CEO_LLM_MODEL", "claude-haiku-4-5")
    llm_max_tokens: int = 6000
    llm_temperature: float = 0.2
    llm_timeout_s: int = 120
    llm_allow_stub: bool = True

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

    # ---- Handbook retrieval -------------------------------------------------
    handbook_top_k: int = 4

    # ---- AgentCard / AgentRegistry ------------------------------------------
    # CEO ignores cards with fewer runs than this (cold-start guard)
    agent_card_min_confidence: int = 3

    # CEO rejects agents whose trust_score is below this floor
    agent_card_min_trust: float = 0.35

    seed: int = 1234


DEFAULT = Config()