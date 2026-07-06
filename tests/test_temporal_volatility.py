"""Deterministic temporal-volatility override — the ungroundable discriminator.

The LLM volatility taxonomy (stable|evolving|contested) is epistemic ("are the
claims disputed?") and misses TEMPORAL volatility: tasks that need live/current
data the model can't hold in parametric knowledge. Those read as "stable" to
the LLM, so no verifier is forced and groundedness can never fire. The
deterministic override upgrades stable -> evolving when temporal/live-data
markers are present. Offline, no LLM backend touched.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ceo_delta.research import _TEMPORAL_VOLATILITY_RE


# tasks that NEED live/current data -> must be flagged temporally volatile
TEMPORAL = [
    "Should I buy NVIDIA stock tomorrow?",
    "Summarize today's biggest AI news.",
    "What restaurants near me are currently open and have less than a 15-minute wait?",
    "What is the latest version of Python?",
    "What's the weather right now in Bangalore?",
    "Give me the current price of Bitcoin.",
    "What happened in the news this week?",
]

# self-contained tasks whose ground truth is stable -> must NOT be flagged
STABLE = [
    "Explain the entire water cycle from evaporation to groundwater recharge.",
    "Explain how quicksort partitions an array and recurses on the subarrays.",
    "Compare Python, Rust, Go, Java, and C++ across performance and safety.",
    "Explain how a modern C compiler transforms C source into an executable.",
    "Write a complete design document for a distributed rate limiter.",
]


def test_temporal_markers_flagged():
    for t in TEMPORAL:
        assert _TEMPORAL_VOLATILITY_RE.search(t), f"should flag temporal: {t!r}"


def test_stable_tasks_not_flagged():
    for t in STABLE:
        assert not _TEMPORAL_VOLATILITY_RE.search(t), f"should NOT flag stable: {t!r}"
