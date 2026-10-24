"""Strict selection: relevance floor -> MMR diversity -> hard token budget.

Injecting nothing is a valid (and common) outcome.
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from app.config import settings
from app.runtime.retrieval.render import render_line
from app.runtime.retrieval.scoring import Scored
from app.tokens import count_tokens


@dataclass
class Selection:
    selected: list[Scored] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)  # id -> reason
    tokens: int = 0


def select(
    scored: list[Scored],
    now: datetime,
    budget_tokens: int,
    max_items: int = settings.max_memories,
    min_score: float = settings.min_score,
) -> Selection:
    sel = Selection()
    pool: list[Scored] = []
    for s in scored:
        if not s.relevant:
            sel.rejected[s.id] = "no-relevance-signal"
        elif s.score < min_score:
            sel.rejected[s.id] = "below-min-score"
        elif s.features["R"] < settings.r_min:
            sel.rejected[s.id] = "faded"
        else:
            pool.append(s)

    if not pool:
        return sel
    # MMR, vectorised: one similarity matrix + a running "max similarity to anything chosen" array.
    lam = settings.mmr_lambda
    scores = np.array([s.score for s in pool], dtype=np.float32)
    mat = np.stack([s.vec for s in pool])
    sims = mat @ mat.T
    max_sim = np.zeros(len(pool), dtype=np.float32)
    open_ = np.ones(len(pool), dtype=bool)
    while open_.any() and len(sel.selected) < max_items:
        val = np.where(open_, lam * scores - (1 - lam) * max_sim, -np.inf)
        i = int(np.argmax(val))
        open_[i] = False
        s = pool[i]
        if max_sim[i] >= settings.dedupe_sim:
            sel.rejected[s.id] = "duplicate"
            continue
        line = render_line(s, now)
        t = count_tokens(line) + 1
        if sel.tokens + t > budget_tokens:
            sel.rejected[s.id] = "over-budget"
            continue
        sel.selected.append(s)
        sel.lines.append(line)
        sel.tokens += t
        np.maximum(max_sim, sims[i], out=max_sim)
    for i in np.flatnonzero(open_):
        sel.rejected.setdefault(pool[int(i)].id, "max-items")
    return sel
