"""Central configuration and thresholds for the CEO-Delta architecture.

Every magic number that the spec left implicit lives here so the four
limitations (cold start, reflection triggers, replan threshold, multi-way
conflict) have one explicit home.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- LLM backend (vLLM, OpenAI-compatible) -------------------------------
    llm_base_url: str = os.environ.get("CEO_LLM_URL", "http://10.8.0.23:8001/v1")
    llm_model: str = os.environ.get("CEO_LLM_MODEL", "my-model")
    llm_max_tokens: int = 6000          # reasoning model spends ~2k+ in its reasoning channel before the answer
    llm_temperature: float = 0.2
    llm_timeout_s: int = 120
    llm_allow_stub: bool = True         # fall back to deterministic stub if server down

    # ---- Embeddings ----------------------------------------------------------
    embed_dim: int = 256

    # ---- Cold start (limitation #1) -----------------------------------------
    # First N runs: CEO operates in exploratory mode, all WHY flagged low-confidence.
    cold_start_runs: int = 3
    seed_handbook: bool = True          # seed synthetic entries from the papers
    # below this retrieval similarity CEO asks one clarifying question
    clarify_similarity_threshold: float = 0.45

    # ---- Reflection mode (limitation #2) ------------------------------------
    reflection_interval: int = 5        # trigger every N standard runs
    # OR trigger immediately once this many contested entries accumulate
    reflection_contested_trigger: int = 2
    # hard token ceiling for one reflection session (runaway guard)
    reflection_token_budget: int = 8000
    # max exploratory DAGs to run before forcing exit
    reflection_max_explorations: int = 4

    # ---- Research replan (limitation #3) ------------------------------------
    # cosine(original_task_emb, post_brief_task_emb) below this => CEO replans
    replan_threshold: float = 0.70

    # ---- Multi-way conflict (limitation #4) ---------------------------------
    # an option must beat the runner-up by this vote-share margin to be settled
    conflict_dominance_margin: float = 0.20
    # normalized entropy above this over the option distribution => contested
    conflict_entropy_threshold: float = 0.85
    # minimum total votes before we even attempt to declare a winner
    conflict_min_votes: int = 3

    # ---- Delta surprise / metrics -------------------------------------------
    surprise_factor: float = 2.0        # actual > factor*planned => per-node entry
    echo_cosine_threshold: float = 0.85 # sibling cosine above this == echoing

    # ---- Handbook retrieval --------------------------------------------------
    handbook_top_k: int = 4

    seed: int = 1234


DEFAULT = Config()
