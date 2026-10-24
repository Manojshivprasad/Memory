"""Candidate generation: one SQL round trip unions several recall channels.

Channels (each a CTE, included only when it has input):
  vec      - semantic nearest neighbours (exact cosine over the btree-filtered project set)
  lex      - full-text match (names, error codes, identifiers)
  act      - memories linked to activated entities (spreading activation / priming)
  win      - memories formed inside a temporal window ("yesterday", "last week")
  lastchat - episode gists of the most recent *other* chats ("where we left off")
  opn      - open commitments / questions (prospective memory)
Scoring happens in Python (scoring.py) so the warm prefetch path can rescore without SQL.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from uuid import UUID

import numpy as np
from sqlalchemy import text

from app.config import settings
from app.db.session import tenant_session
from app.embeddings import as_array
from app.runtime.temporal import TemporalCue


@dataclass
class Candidate:
    id: str
    kind: str
    content: str
    scope: str
    is_belief: bool
    importance: float
    confidence: float
    source_reliability: float
    stability: float
    recall_count: int
    utility: float
    created_at: datetime
    valid_from: datetime
    last_recalled_at: datetime | None = None
    valid_until: datetime | None = None
    thread_status: str | None = None
    due_at: datetime | None = None
    source_chat_id: str | None = None
    status: str = "active"
    entity_ids: list[str] = field(default_factory=list)
    lex_rank: float = 0.0
    last_chat_hit: bool = False
    attr_hit: bool = False        # the prompt asks about this memory's attribute (e.g. works_at)
    constraint_hit: bool = False  # a standing constraint for the prompt's domain (e.g. allergy for food)

    def to_json(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat()
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Candidate":
        d = dict(d)
        for k in ("created_at", "valid_from", "last_recalled_at", "valid_until", "due_at"):
            if d.get(k):
                d[k] = datetime.fromisoformat(d[k])
        return cls(**d)


@dataclass
class CandidateQuery:
    tenant_id: UUID
    project_id: UUID
    user_id: UUID
    chat_id: UUID | None
    include_user_scope: bool
    qvec: np.ndarray
    tsquery: str
    entity_activation: dict[str, float]
    cue: TemporalCue
    statuses: tuple[str, ...] = ("active", "weakening")
    limit: int = settings.candidate_limit
    attr_preds: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)


_F = (
    "m.tenant_id = :tid AND (m.project_id = :pid OR (CAST(:inc_user AS boolean) AND m.scope = 'user' "
    "AND m.user_id = :uid AND m.project_id IS NULL)) "
    "AND m.status = ANY(CAST(:statuses AS text[])) "
    "AND (CAST(:historical AS boolean) OR m.valid_until IS NULL OR m.valid_until > now())"
)


def _build_sql(q: CandidateQuery) -> tuple[str, dict]:
    params: dict = {
        "tid": q.tenant_id,
        "pid": q.project_id,
        "uid": q.user_id,
        "inc_user": q.include_user_scope,
        "statuses": list(q.statuses),
        "historical": q.cue.historical,
        "qv": q.qvec,
        "k": q.limit,
    }
    ctes = [
        f"vec AS (SELECT m.id FROM memories m WHERE {_F} "
        f"ORDER BY m.embedding <=> CAST(:qv AS vector) LIMIT :k)"
    ]
    unions = ["SELECT id FROM vec"]
    lex_join = "CAST(0.0 AS float8) AS lex_rank"
    if q.tsquery:
        params["tsq"] = q.tsquery
        ctes.append(
            f"lex AS (SELECT m.id, ts_rank(m.tsv, tq) AS lex FROM memories m, "
            f"to_tsquery('english', :tsq) tq WHERE {_F} AND m.tsv @@ tq ORDER BY lex DESC LIMIT 15)"
        )
        unions.append("SELECT id FROM lex")
        lex_join = "CAST(COALESCE((SELECT lex FROM lex WHERE lex.id = m.id), 0.0) AS float8) AS lex_rank"
    if q.entity_activation:
        top = sorted(q.entity_activation.items(), key=lambda kv: -kv[1])[:30]
        params["eids"] = [UUID(e) for e, _ in top]
        params["eacts"] = [float(a) for _, a in top]
        ctes.append(
            "act AS (SELECT me.memory_id AS id, max(a.act) AS act FROM memory_entities me "
            "JOIN unnest(CAST(:eids AS uuid[]), CAST(:eacts AS float8[])) AS a(eid, act) ON a.eid = me.entity_id "
            f"JOIN memories m ON m.id = me.memory_id WHERE {_F} GROUP BY me.memory_id ORDER BY act DESC LIMIT 25)"
        )
        unions.append("SELECT id FROM act")
    if q.cue.has_window:
        params["t0"], params["t1"] = q.cue.start, q.cue.end
        ctes.append(
            f"win AS (SELECT m.id FROM memories m WHERE {_F} AND m.created_at >= :t0 AND m.created_at < :t1 "
            "ORDER BY m.importance DESC LIMIT 15)"
        )
        unions.append("SELECT id FROM win")
    last_chat_flag = "false AS last_chat_hit"
    if q.cue.last_chat:
        params["cid"] = q.chat_id
        ctes.append(
            f"lastchat AS (SELECT m.id FROM memories m WHERE {_F} AND m.kind = 'episode' "
            "AND m.source_chat_id IS DISTINCT FROM CAST(:cid AS uuid) ORDER BY m.created_at DESC LIMIT 2)"
        )
        unions.append("SELECT id FROM lastchat")
        last_chat_flag = "(m.id IN (SELECT id FROM lastchat)) AS last_chat_hit"
    ctes.append(
        f"opn AS (SELECT m.id FROM memories m WHERE {_F} AND m.thread_status = 'open' "
        "ORDER BY m.importance DESC, m.created_at DESC LIMIT 10)"
    )
    unions.append("SELECT id FROM opn")
    attr_flag = "false AS attr_hit"
    if q.attr_preds:
        params["apreds"] = q.attr_preds
        ctes.append(
            f"attr AS (SELECT m.id FROM memories m WHERE {_F} AND m.pred = ANY(CAST(:apreds AS text[])) "
            "AND m.kind <> 'episode' ORDER BY m.importance DESC, m.created_at DESC LIMIT 10)"
        )
        unions.append("SELECT id FROM attr")
        attr_flag = "(m.id IN (SELECT id FROM attr)) AS attr_hit"
    cons_flag = "false AS constraint_hit"
    if q.domains:
        params["doms"] = q.domains
        ctes.append(
            f"cons AS (SELECT m.id FROM memories m WHERE {_F} "
            "AND jsonb_exists_any(m.metadata->'constraint_domains', CAST(:doms AS text[])) "
            "ORDER BY m.importance DESC LIMIT 10)"
        )
        unions.append("SELECT id FROM cons")
        cons_flag = "(m.id IN (SELECT id FROM cons)) AS constraint_hit"

    sql = (
        "WITH " + ",\n".join(ctes) + ",\nids AS (" + " UNION ".join(unions) + ")\n"
        "SELECT m.id, m.kind, m.content, m.scope, m.is_belief, m.importance, m.confidence, "
        "m.source_reliability, m.stability, m.recall_count, m.utility, m.created_at, m.valid_from, "
        "m.last_recalled_at, m.valid_until, m.thread_status, m.due_at, m.source_chat_id, m.status, m.embedding, "
        f"{lex_join}, {last_chat_flag}, {attr_flag}, {cons_flag}, "
        "COALESCE((SELECT array_agg(me.entity_id) FROM memory_entities me WHERE me.memory_id = m.id), "
        "CAST('{}' AS uuid[])) AS entity_ids "
        "FROM ids JOIN memories m ON m.id = ids.id"
    )
    return sql, params


def _row_to_candidate(r) -> tuple[Candidate, np.ndarray]:  # noqa: ANN001
    c = Candidate(
        id=str(r["id"]),
        kind=r["kind"],
        content=r["content"],
        scope=r["scope"],
        is_belief=r["is_belief"],
        importance=float(r["importance"]),
        confidence=float(r["confidence"]),
        source_reliability=float(r["source_reliability"]),
        stability=float(r["stability"]),
        recall_count=int(r["recall_count"]),
        utility=float(r["utility"]),
        created_at=r["created_at"],
        valid_from=r["valid_from"],
        last_recalled_at=r["last_recalled_at"],
        valid_until=r["valid_until"],
        thread_status=r["thread_status"],
        due_at=r["due_at"],
        source_chat_id=str(r["source_chat_id"]) if r["source_chat_id"] else None,
        status=r["status"],
        entity_ids=[str(e) for e in (r["entity_ids"] or [])],
        lex_rank=float(r["lex_rank"] or 0.0),
        last_chat_hit=bool(r["last_chat_hit"]),
        attr_hit=bool(r.get("attr_hit", False)),
        constraint_hit=bool(r.get("constraint_hit", False)),
    )
    return c, as_array(r["embedding"])


_COLUMNS = (
    "m.id, m.kind, m.content, m.scope, m.is_belief, m.importance, m.confidence, m.source_reliability, "
    "m.stability, m.recall_count, m.utility, m.created_at, m.valid_from, m.last_recalled_at, m.valid_until, "
    "m.thread_status, m.due_at, m.source_chat_id, m.status, m.embedding, CAST(0.0 AS float8) AS lex_rank, "
    "false AS last_chat_hit, false AS attr_hit, false AS constraint_hit, "
    "COALESCE((SELECT array_agg(me.entity_id) FROM memory_entities me "
    "WHERE me.memory_id = m.id), CAST('{}' AS uuid[])) AS entity_ids"
)


async def fetch_where(
    tenant_id: UUID, project_id: UUID, user_id: UUID, include_user_scope: bool,
    where: str, params: dict, order: str = "m.created_at DESC", limit: int = 15,
) -> tuple[list[Candidate], np.ndarray]:
    """Scoped fetch with an extra (trusted, code-defined) WHERE fragment. Used by agent tools."""
    base = {
        "tid": tenant_id, "pid": project_id, "uid": user_id, "inc_user": include_user_scope,
        "statuses": ["active", "weakening"], "historical": True, "lim": limit,
    }
    sql = f"SELECT {_COLUMNS} FROM memories m WHERE {_F} AND ({where}) ORDER BY {order} LIMIT :lim"
    async with tenant_session(tenant_id) as conn:
        rows = (await conn.execute(text(sql), {**base, **params})).mappings().all()
    if not rows:
        return [], np.zeros((0, settings.embed_dim), np.float32)
    pairs = [_row_to_candidate(r) for r in rows]
    return [p[0] for p in pairs], np.stack([p[1] for p in pairs])


async def fetch_candidates(q: CandidateQuery) -> tuple[list[Candidate], np.ndarray]:
    sql, params = _build_sql(q)
    async with tenant_session(q.tenant_id) as conn:
        rows = (await conn.execute(text(sql), params)).mappings().all()
    if not rows:
        return [], np.zeros((0, settings.embed_dim), np.float32)
    pairs = [_row_to_candidate(r) for r in rows]
    return [p[0] for p in pairs], np.stack([p[1] for p in pairs])
