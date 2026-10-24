"""Human-like memory strength: forgetting curve + spacing effect.

* Stability ``S`` (days) is how slowly a memory fades. Important memories start more stable.
* Retrievability ``R = exp(-dt / S)`` is how easy it is to recall right now.
* A *useful* recall raises S, and raises it more when R was low (the spacing effect: a memory
  recalled just before it would have been forgotten gets consolidated the most).
* Open commitments / questions (prospective memory) keep a floor on R until they are closed.
"""

import math
from datetime import datetime

DAY = 86400.0
OPEN_THREAD_R_FLOOR = 0.5
WEAKENING_R = 0.2
ARCHIVE_R = 0.05


def initial_stability(importance: float) -> float:
    return 1.0 + 29.0 * max(0.0, min(1.0, importance))


def retrievability(now: datetime, created_at: datetime, last_recalled_at: datetime | None,
                   stability: float, open_thread: bool = False) -> float:
    ref = last_recalled_at if last_recalled_at and last_recalled_at > created_at else created_at
    dt_days = max(0.0, (now - ref).total_seconds() / DAY)
    r = math.exp(-dt_days / max(stability, 0.1))
    return max(r, OPEN_THREAD_R_FLOOR) if open_thread else r


def reinforce(stability: float, r: float, gain: float = 1.5) -> float:
    """New stability after a useful recall; at least +10%, capped at ~10 years."""
    return min(3650.0, stability * max(1.1, 1.0 + gain * (1.0 - r)))


def lifecycle_status(r: float, importance: float, open_thread: bool) -> str:
    if open_thread:
        return "active"
    if r < ARCHIVE_R and importance < 0.7:
        return "archived"
    if r < WEAKENING_R:
        return "weakening"
    return "active"


def effective_importance(importance: float, utility: float) -> float:
    """Utility (was it actually useful when recalled?) nudges importance up or down."""
    return max(0.0, min(1.0, importance + 0.1 * math.tanh(utility / 3.0)))
