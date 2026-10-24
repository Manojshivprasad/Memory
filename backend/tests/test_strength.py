from datetime import datetime, timedelta, timezone

from app.memory.strength import (OPEN_THREAD_R_FLOOR, effective_importance, initial_stability, lifecycle_status,
                                 reinforce, retrievability)

NOW = datetime(2026, 9, 28, tzinfo=timezone.utc)


def test_retrievability_decays_with_time():
    s = 10.0
    fresh = retrievability(NOW, NOW, None, s)
    week = retrievability(NOW, NOW - timedelta(days=7), None, s)
    month = retrievability(NOW, NOW - timedelta(days=30), None, s)
    assert fresh == 1.0 and fresh > week > month > 0


def test_recall_resets_the_clock():
    created = NOW - timedelta(days=30)
    assert retrievability(NOW, created, NOW - timedelta(days=1), 10) > retrievability(NOW, created, None, 10)


def test_important_memories_are_more_stable():
    assert initial_stability(0.9) > initial_stability(0.2)


def test_spacing_effect_low_r_gains_more():
    assert reinforce(10, r=0.2) > reinforce(10, r=0.9) >= 11.0


def test_open_threads_never_fade_below_floor():
    r = retrievability(NOW, NOW - timedelta(days=365), None, 2, open_thread=True)
    assert r == OPEN_THREAD_R_FLOOR
    assert lifecycle_status(0.0, 0.1, open_thread=True) == "active"


def test_lifecycle():
    assert lifecycle_status(0.9, 0.5, False) == "active"
    assert lifecycle_status(0.1, 0.5, False) == "weakening"
    assert lifecycle_status(0.01, 0.5, False) == "archived"
    assert lifecycle_status(0.01, 0.9, False) == "weakening"  # important memories are not archived


def test_utility_nudges_importance():
    assert effective_importance(0.5, 10) > 0.5 > effective_importance(0.5, -5)
