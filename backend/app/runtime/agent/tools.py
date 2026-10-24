"""Tools the memory agent can call. All are scoped to (tenant, project [+ user scope])."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import UUID

import numpy as np
from sqlalchemy import text

from app.db.session import tenant_session
from app.embeddings import embed_query
from app.runtime.graph import ProjectGraph
from app.runtime.retrieval.candidates import CandidateQuery, fetch_candidates, fetch_where
from app.runtime.retrieval.scoring import ScoreContext, Scored, score_candidates
from app.runtime.temporal import TemporalCue
from app.textutil import normalize, tsquery_or

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_memories",
            "description": "Semantic + keyword search over this project's memories.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_timeline",
            "description": "Memories and past-chat episodes formed between two dates (inclusive), oldest first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_entity",
            "description": "Everything linked to a named entity (person, tech, component) and its neighbours.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_verbatim_turns",
            "description": "Exact past messages matching keywords (for 'what exactly did I say'). Not selectable; "
            "quote in the note if needed.",
            "parameters": {
                "type": "object",
                "properties": {"keywords": {"type": "string"}},
                "required": ["keywords"],
            },
        },
    },
]


@dataclass
class ToolContext:
    tenant_id: UUID
    project_id: UUID
    user_id: UUID
    chat_id: UUID | None
    include_user_scope: bool
    graph: ProjectGraph
    score_ctx: ScoreContext
    seen: dict[str, Scored] = field(default_factory=dict)

    def _brief(self, scored: list[Scored], limit: int) -> list[dict]:
        out = []
        for s in scored[:limit]:
            self.seen[s.id] = s
            c = s.cand
            out.append({
                "id": s.id, "kind": c.kind, "content": c.content, "date": c.created_at.strftime("%Y-%m-%d"),
                "still_true": c.valid_until is None, "belief": c.is_belief,
            })
        return out

    def _score(self, cands, vecs) -> list[Scored]:  # noqa: ANN001
        return score_candidates(cands, vecs, self.score_ctx)

    async def search_memories(self, query: str) -> list[dict]:
        qvec = await embed_query(query)
        cands, vecs = await fetch_candidates(CandidateQuery(
            tenant_id=self.tenant_id, project_id=self.project_id, user_id=self.user_id, chat_id=self.chat_id,
            include_user_scope=self.include_user_scope, qvec=qvec, tsquery=tsquery_or(query),
            entity_activation={}, cue=TemporalCue(historical=True), limit=12,
        ))
        if not cands:
            return []
        sims = vecs @ qvec
        order = np.argsort(-sims)
        scored = self._score([cands[i] for i in order], vecs[order])
        return self._brief(scored, 8)

    async def get_timeline(self, start_date: str, end_date: str) -> list[dict]:
        t0 = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        t1 = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc) + timedelta(days=1)
        cands, vecs = await fetch_where(
            self.tenant_id, self.project_id, self.user_id, self.include_user_scope,
            "m.created_at >= :t0 AND m.created_at < :t1", {"t0": t0, "t1": t1}, order="m.created_at ASC", limit=15,
        )
        return self._brief(self._score(cands, vecs), 15) if cands else []

    async def get_entity(self, name: str) -> dict:
        eid = self.graph.names.get(normalize(name))
        if not eid:
            return {"error": f"unknown entity '{name}'", "known": list(self.graph.display.values())[:30]}
        cands, vecs = await fetch_where(
            self.tenant_id, self.project_id, self.user_id, self.include_user_scope,
            "m.id IN (SELECT memory_id FROM memory_entities WHERE entity_id = :eid)", {"eid": UUID(eid)}, limit=10,
        )
        neighbours = [self.graph.display.get(n, n) for n, _ in self.graph.adj.get(eid, [])][:15]
        return {"entity": self.graph.display.get(eid), "neighbours": neighbours,
                "memories": self._brief(self._score(cands, vecs), 10) if cands else []}

    async def get_verbatim_turns(self, keywords: str) -> list[dict]:
        tsq = tsquery_or(keywords)
        if not tsq:
            return []
        async with tenant_session(self.tenant_id) as conn:
            rows = (await conn.execute(
                text(
                    "SELECT chat_id, seq, created_at, user_msg, assistant_msg FROM turns, "
                    "to_tsquery('english', :q) tq WHERE project_id = :p AND tsv @@ tq "
                    "ORDER BY ts_rank_cd(tsv, tq) DESC LIMIT 4"
                ),
                {"q": tsq, "p": self.project_id},
            )).mappings().all()
        return [{"chat_id": str(r["chat_id"]), "date": r["created_at"].strftime("%Y-%m-%d"),
                 "user": r["user_msg"][:500], "assistant": r["assistant_msg"][:500]} for r in rows]

    async def call(self, name: str, args: dict):  # noqa: ANN201
        fn = {
            "search_memories": self.search_memories,
            "get_timeline": self.get_timeline,
            "get_entity": self.get_entity,
            "get_verbatim_turns": self.get_verbatim_turns,
        }.get(name)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return await fn(**args)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)[:200]}
