from datetime import datetime, timedelta, timezone

import numpy as np

from app.embeddings import HashEmbedder
from app.runtime.retrieval.candidates import Candidate
from app.runtime.retrieval.scoring import ScoreContext, score_candidates
from app.runtime.retrieval.selector import select
from app.runtime.temporal import TemporalCue
from app.tokens import count_tokens

NOW = datetime(2026, 9, 28, tzinfo=timezone.utc)
EMB = HashEmbedder(384)


def cand(i: int, content: str, **kw) -> Candidate:
    base = dict(id=f"00000000-0000-0000-0000-{i:012d}", kind="fact", content=content, scope="project",
                is_belief=False, importance=0.6, confidence=0.9, source_reliability=0.9, stability=20,
                recall_count=0, utility=0, created_at=NOW - timedelta(days=1), valid_from=NOW - timedelta(days=1))
    base.update(kw)
    return Candidate(**base)


def run(query: str, cands: list[Candidate], budget: int = 300, **ctx_kw):
    vecs = EMB.embed([c.content for c in cands])
    ctx = ScoreContext(qvec=EMB.embed([query])[0], query=query, activation=ctx_kw.pop("activation", {}),
                       cue=ctx_kw.pop("cue", TemporalCue()), wm_entities=set(), chat_id=None, now=NOW)
    return select(score_candidates(cands, vecs, ctx), NOW, budget)


def test_irrelevant_memories_are_not_injected_even_if_important():
    sel = run("how do I write a regex for emails",
              [cand(1, "The project deadline is October 15", importance=1.0),
               cand(2, "The user prefers concise answers", importance=0.9)])
    assert sel.selected == []
    assert set(sel.rejected.values()) == {"no-relevance-signal"}


def test_relevant_memory_selected_via_lexical_match():
    sel = run("which database does the project use",
              [cand(1, "The project uses postgres database for storage"),
               cand(2, "The user likes hiking on weekends")])
    assert [s.cand.content for s in sel.selected] == ["The project uses postgres database for storage"]


def test_token_budget_is_hard():
    cands = [cand(i, f"The project database note number {i} mentions postgres tuning and database indexes")
             for i in range(20)]
    sel = run("database postgres indexes", cands, budget=60)
    assert sel.tokens <= 60
    assert count_tokens("\n".join(sel.lines)) <= 60
    assert "over-budget" in sel.rejected.values() or "duplicate" in sel.rejected.values()


def test_near_duplicates_are_collapsed():
    sel = run("postgres database", [cand(1, "The project uses postgres database"),
                                    cand(2, "The project uses postgres database")])
    assert len(sel.selected) == 1


def test_association_counts_as_relevance():
    eid = "11111111-1111-1111-1111-111111111111"
    sel = run("tell me about it", [cand(1, "Alice owns the billing service", entity_ids=[eid])],
              activation={eid: 1.0})
    assert len(sel.selected) == 1 and sel.selected[0].evidence == "association"


def test_temporal_window_counts_as_relevance():
    cue = TemporalCue(start=NOW - timedelta(days=2), end=NOW, historical=False)
    sel = run("what happened yesterday", [cand(1, "Refactored the auth middleware", importance=0.7)], cue=cue)
    assert len(sel.selected) == 1


def test_superseded_memory_rendered_as_no_longer_true():
    c = cand(1, "The project uses postgres database", valid_until=NOW - timedelta(hours=1))
    sel = run("postgres database", [c])
    assert "no longer true" in sel.lines[0]


def test_beliefs_are_marked_unconfirmed():
    sel = run("postgres database", [cand(1, "Assistant suggestion: use postgres database", is_belief=True,
                                         confidence=0.55)])
    assert "unconfirmed" in sel.lines[0]


def test_empty_pool():
    assert run("anything", []).selected == []
    np.testing.assert_equal(len(run("anything", []).lines), 0)
