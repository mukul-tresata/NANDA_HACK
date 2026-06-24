"""Offline tests (force LLM stub) covering the four limitations + the loop."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ceo_delta.config import Config
from ceo_delta.embeddings import cosine, embed
from ceo_delta.handbook import Handbook
from ceo_delta.schemas import HandbookEntry
from ceo_delta.orchestrator import Orchestrator
from ceo_delta import bootstrap


def _cfg():
    c = Config()
    c.llm_base_url = "http://127.0.0.1:1"  # unreachable -> forces stub
    c.llm_allow_stub = True
    c.llm_timeout_s = 1
    return c


def _orch():
    d = tempfile.mkdtemp()
    return Orchestrator(_cfg(), workdir=d)


# ---- embeddings ----
def test_embedding_self_similarity():
    a = embed("dag optimization for agents")
    assert abs(cosine(a, a) - 1.0) < 1e-6
    b = embed("completely unrelated cooking recipe")
    assert cosine(a, b) < cosine(a, a)


# ---- limitation #1: cold start ----
def test_cold_start_seeded_and_exploratory():
    o = _orch()
    assert len(o.ceo_hb.entries) > 0, "handbook seeded on cold start"
    r = o.run("plan something novel about agents")
    assert r.dag.exploratory, "first runs are exploratory"


# ---- limitation #4: multi-way conflict ----
def test_multiway_conflict_contested():
    hb = Handbook("t", _cfg())
    e = HandbookEntry(task_embedding=embed("x"), task_summary="x",
                      topology_votes={"A": 5, "B": 3, "C": 2})
    hb.resolve(e)
    # 5-3-2 split: leader margin (5-3)/10=0.2 but high entropy -> contested
    assert e.contested, "balanced multi-way split must be contested"

    e2 = HandbookEntry(task_embedding=embed("y"), task_summary="y",
                       topology_votes={"A": 9, "B": 1})
    hb.resolve(e2)
    assert not e2.contested and e2.topology_chosen == "A"


# ---- limitation #3: replan threshold ----
def test_replan_threshold_config():
    o = _orch()
    assert o.cfg.replan_threshold == 0.70
    r = o.run("summarize agent research")
    # brief carries a concrete drift number and a boolean derived from threshold
    assert hasattr(r.brief, "drift") and isinstance(r.brief.triggers_replan, bool)


# ---- limitation #2: reflection trigger + budget ----
def test_reflection_triggers_and_budget():
    from ceo_delta.reflection import should_reflect
    o = _orch()
    # force two contested entries
    for i in range(2):
        e = o.ceo_hb.entries[i]
        e.topology_votes = {"A": 3, "B": 3}
        o.ceo_hb.resolve(e)
    ok, reason = should_reflect(1, o.ceo_hb, o.cfg)
    assert ok and "contested" in reason
    log = o.reflect_now()
    assert log.triggered
    assert log.tokens_spent <= o.cfg.reflection_token_budget + 5000  # ceiling respected-ish
    assert log.explorations <= o.cfg.reflection_max_explorations


# ---- full loop ----
def test_full_loop_runs_and_learns():
    o = _orch()
    before = len(o.ceo_hb.entries)
    r = o.run("compare two agent planning frameworks")
    assert r.answer and r.dag.nodes
    assert r.report.verdict in ("good", "mixed", "poor")
    # delta wrote something (votes added or new entry)
    assert len(o.ceo_hb.entries) >= before


def test_meta_feedback():
    o = _orch()
    o.run("research task about latency")
    msg = o.meta_feedback("research task about latency", 0.9, "great")
    assert "satisfaction=0.90" in msg


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
