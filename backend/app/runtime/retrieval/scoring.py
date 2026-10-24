"""Memory scoring: fuse recall channels with human-like strength into one explainable score.

relevance = 1 - prod(1 - w_c * f_c)   over channels: semantic, lexical, activation, temporal, context
strength  = mean_w(retrievability, importance, confidence*reliability, scope)
score     = relevance * ((1 - mix) + mix * strength)

Any single strong recall channel (a close meaning, an exact name, an associated entity, the asked-about
time window) can carry a memory; strength only modulates it. A memory must ALSO show at least one
real relevance signal (``relevant``): being important/recent/confident alone never gets it injected.
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from app.config import ScoringWeights, settings
from app.memory.strength import effective_importance, retrievability
from app.runtime.retrieval.candidates import Candidate
from app.runtime.temporal import TemporalCue
from app.textutil import content_tokens, is_identifier

SCOPE_WEIGHT = {"same_chat": 1.0, "project": 0.8, "user": 0.6}


@dataclass
class ScoreContext:
    qvec: np.ndarray
    query: str
    activation: dict[str, float]
    cue: TemporalCue
    wm_entities: set[str]
    chat_id: str | None
    now: datetime
    weights: ScoringWeights = field(default_factory=lambda: settings.weights)
    # Metamemory: a general-knowledge prompt (no personal/project reference, no known entity, no time
    # cue) must show much stronger semantic evidence before any memory is injected.
    impersonal: bool = False

    def __post_init__(self) -> None:
        toks = content_tokens(self.query)
        self.q_weights = {t: (2.0 if is_identifier(t) else 1.0) for t in toks}
        self.q_total = sum(self.q_weights.values())


@dataclass
class Scored:
    cand: Candidate
    vec: np.ndarray
    score: float
    features: dict[str, float]
    relevant: bool
    evidence: str

    @property
    def id(self) -> str:
        return self.cand.id


def _sem_norm(cos: float) -> float:
    return float(min(1.0, max(0.0, (cos - settings.sem_lo) / (settings.sem_hi - settings.sem_lo))))


# Words our own memory phrasing uses everywhere ("The project uses ...") carry no signal.
_DOMAIN_STOP = frozenset({"project", "user", "assistant", "suggestion", "memory"})


def _query_weights(ctx: ScoreContext, token_sets: list[set[str]]) -> dict[str, float]:
    """IDF-lite over the candidate pool: tokens present in most candidates are down-weighted."""
    n = len(token_sets)
    weights = {}
    for t, w in ctx.q_weights.items():
        if t in _DOMAIN_STOP:
            continue
        df = sum(1 for s in token_sets if t in s)
        weights[t] = w * (0.25 if n >= 4 and df / n > 0.5 else 1.0)
    return weights


def score_candidates(cands: list[Candidate], vecs: np.ndarray, ctx: ScoreContext) -> list[Scored]:
    if not cands:
        return []
    w = ctx.weights
    sims = vecs @ ctx.qvec
    token_sets = [set(content_tokens(c.content)) for c in cands]
    qw = _query_weights(ctx, token_sets)
    q_total = sum(qw.values())
    out: list[Scored] = []
    for c, v, cos, toks in zip(cands, vecs, sims, token_sets):
        cos = float(cos)
        is_open = c.thread_status == "open"
        sem = _sem_norm(cos)
        lex = (sum(wt for t, wt in qw.items() if t in toks) / q_total) if q_total else 0.0
        act = max((ctx.activation.get(e, 0.0) for e in c.entity_ids), default=0.0)
        in_window = bool(ctx.cue.has_window and ctx.cue.start <= c.created_at < ctx.cue.end)
        temporal = 1.0 if (in_window or c.last_chat_hit) else 0.0
        context = (
            len(set(c.entity_ids) & ctx.wm_entities) / len(c.entity_ids) if c.entity_ids and ctx.wm_entities else 0.0
        )
        r = retrievability(ctx.now, c.created_at, c.last_recalled_at, c.stability, is_open)
        imp = effective_importance(c.importance, c.utility)
        conf = c.confidence * c.source_reliability * (0.8 if c.is_belief else 1.0)
        if c.source_chat_id and c.source_chat_id == ctx.chat_id:
            scope = SCOPE_WEIGHT["same_chat"]
        else:
            scope = SCOPE_WEIGHT["user" if c.scope == "user" else "project"]
        miss = 1.0
        attr = 1.0 if c.attr_hit else 0.0
        cons = 1.0 if c.constraint_hit else 0.0
        for weight, f in ((w.semantic, sem), (w.lexical, lex), (w.activation, act), (w.temporal, temporal),
                          (w.context, context), (w.attribute, attr), (w.constraint, cons)):
            miss *= 1.0 - weight * f
        relevance = 1.0 - miss
        strength = (w.retrievability * r + w.importance * imp + w.confidence * conf + w.scope * scope) / (
            w.retrievability + w.importance + w.confidence + w.scope)
        score = relevance * ((1 - w.strength_mix) + w.strength_mix * strength)
        # --- strict relevance evidence ---
        floor = settings.semantic_floor + (settings.impersonal_floor_boost if ctx.impersonal else 0.0)
        if cos >= floor:
            evidence = "semantic"
        elif c.constraint_hit:
            evidence = "constraint"
        elif c.attr_hit:
            evidence = "attribute"
        elif lex >= 0.5:
            evidence = "lexical"
        elif act >= 0.5:
            evidence = "association"
        elif temporal:
            evidence = "temporal"
        elif is_open and (cos >= floor - 0.07 or act > 0):
            evidence = "open-thread"
        else:
            evidence = ""
        out.append(
            Scored(
                cand=c,
                vec=v,
                score=round(score, 4),
                features={
                    "cos": round(cos, 4), "sem": round(sem, 3), "lex": round(lex, 3), "act": round(act, 3),
                    "temporal": temporal, "context": round(context, 3), "R": round(r, 3),
                    "imp": round(imp, 3), "conf": round(conf, 3), "scope": scope,
                    "attribute": attr, "constraint": cons,
                    "relevance": round(relevance, 3), "strength": round(strength, 3),
                },
                relevant=bool(evidence),
                evidence=evidence,
            )
        )
    out.sort(key=lambda s: -s.score)
    return out
