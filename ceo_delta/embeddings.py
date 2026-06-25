"""Semantic text embeddings via sentence-transformers.

Replaces the bag-of-ngrams stub. all-MiniLM-L6-v2 is ~80MB, CPU-friendly,
and produces 384-dim unit vectors with real semantic geometry.

Swap embed() for a larger model later without touching callers.
Model is loaded once and cached — subsequent calls are fast.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed(text: str, dim: int | None = None) -> List[float]:
    # dim arg kept for API compatibility — MiniLM is fixed at 384
    vec = _model().encode(text or "", normalize_embeddings=True)
    return vec.tolist()


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    # vectors are already unit-normalized by encode(); clamp for float safety
    return max(-1.0, min(1.0, dot))