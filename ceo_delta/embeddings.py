"""Dependency-free deterministic text embeddings.

The vLLM server exposes no /embeddings endpoint, so the handbook's semantic
similarity is backed by a hashing bag-of-ngrams embedding. It is deterministic,
needs no model download, and gives stable cosine geometry for retrieval and for
the echoing / replan / fingerprint comparisons. Swap `embed()` for a real model
later without touching callers.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import List

from .config import DEFAULT

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    words = _WORD.findall(text.lower())
    grams = list(words)
    # word bigrams capture a little ordering
    grams += [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return grams


def _bucket(token: str, dim: int) -> int:
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % dim


def embed(text: str, dim: int | None = None) -> List[float]:
    dim = dim or DEFAULT.embed_dim
    vec = [0.0] * dim
    for tok in _tokens(text or ""):
        # signed hashing trick reduces collision bias
        b = _bucket(tok, dim)
        sign = 1.0 if _bucket(tok + "#", 2) == 0 else -1.0
        vec[b] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    # vectors are already unit-normalized in embed(); clamp for safety
    return max(-1.0, min(1.0, dot))
