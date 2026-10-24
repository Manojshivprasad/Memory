"""Sidecar API: for apps that call the LLM themselves.

    retrieve (or prefetch while typing) -> call your LLM -> commit the turn
"""

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.auth.deps import Principal, chat_for, project_for, require_app, resolve_user
from app.cache import Keys, redis
from app.config import settings
from app.db.session import tenant_session
from app.memory.strength import retrievability
from app.runtime.brief import mark_dirty
from app.runtime.orchestrator import RetrieveRequest, prefetch, retrieve
from app.write.ops import KINDS, Op, TurnCtx, apply, validate
from app.write.stage_a import commit_turn

router = APIRouter(prefix="/v1/memory", tags=["memory"])


class RetrieveIn(BaseModel):
    project_id: UUID
    chat_id: UUID | None = None
    user_id: str | None = None
    query: str = Field(max_length=20_000)
    budget_tokens: int = Field(default=settings.budget_tokens, ge=0, le=4000)
    agent: Literal["off", "auto", "always"] = "auto"
    brief: Literal["auto", "always", "never"] = "auto"


class PrefetchIn(BaseModel):
    project_id: UUID
    chat_id: UUID
    user_id: str | None = None
    partial: str = Field(max_length=20_000)
    agent: Literal["off", "auto", "always"] = "auto"
    speculate: bool = False


class CommitIn(BaseModel):
    project_id: UUID
    chat_id: UUID
    user_id: str | None = None
    user_msg: str = Field(max_length=100_000)
    assistant_msg: str = Field(max_length=100_000)
    retrieved_ids: list[UUID] | None = Field(default=None, description="Defaults to the last retrieval in this chat")
    turn_id: UUID | None = Field(default=None, description="Idempotency key")
    model: str | None = None


class RememberIn(BaseModel):
    project_id: UUID
    user_id: str | None = None
    content: str = Field(min_length=1, max_length=400)
    kind: str = "fact"
    scope: Literal["project", "user"] = "project"
    importance: float = Field(default=0.7, ge=0, le=1)
    subj: str | None = None
    pred: str | None = None
    obj: str | None = None


async def _ctx(p: Principal, user_id: str | None, project_id: UUID, chat_id: UUID | None):  # noqa: ANN202
    uid = await resolve_user(p, user_id)
    project = await project_for(p, uid, project_id)
    if chat_id is not None:
        await chat_for(p, project, chat_id)
    return uid, project


@router.post("/retrieve")
async def retrieve_memory(body: RetrieveIn, response: Response, p: Principal = Depends(require_app)) -> dict:
    if p.kind == "subject" and not ({"prefetch", "memory:read"} & p.scopes):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "missing scope: memory:read")
    uid, project = await _ctx(p, body.user_id, body.project_id, body.chat_id)
    res = await retrieve(RetrieveRequest(p.tenant_id, uid, project, body.chat_id, body.query, body.budget_tokens,
                                         body.agent, body.brief))
    response.headers["Server-Timing"] = res.server_timing
    return {
        "context": res.context, "brief": res.brief, "memories": res.memories, "trace_id": res.trace_id,
        "gate": res.gate, "cache_hit": res.cache_hit, "agent_used": res.agent_used,
        "agent_reason": res.agent_reason, "tokens": res.tokens, "latency_ms": res.latency_ms,
    }


@router.post("/prefetch")
async def prefetch_memory(body: PrefetchIn, p: Principal = Depends(require_app)) -> dict:
    p.require_scope("prefetch")
    uid, project = await _ctx(p, body.user_id, body.project_id, body.chat_id)
    return await prefetch(RetrieveRequest(p.tenant_id, uid, project, body.chat_id, body.partial, agent=body.agent),
                          speculate=body.speculate)


@router.post("/commit", status_code=202)
async def commit(body: CommitIn, p: Principal = Depends(require_app)) -> dict:
    if p.kind == "subject" and not ({"chat", "memory:write"} & p.scopes):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "missing scope: chat")
    uid, project = await _ctx(p, body.user_id, body.project_id, body.chat_id)
    return await commit_turn(
        tenant_id=p.tenant_id, user_id=uid, project_id=project.id, chat_id=body.chat_id, isolated=project.isolated,
        user_msg=body.user_msg, assistant_msg=body.assistant_msg,
        retrieved_ids=[str(x) for x in body.retrieved_ids] if body.retrieved_ids is not None else None,
        turn_id=body.turn_id, model=body.model,
    )


@router.post("/remember", status_code=201)
async def remember(body: RememberIn, p: Principal = Depends(require_app)) -> dict:
    """Explicit write (bypasses salience; still deduped / contradiction-checked)."""
    p.require_scope("memory:write")
    uid, project = await _ctx(p, body.user_id, body.project_id, None)
    if body.kind not in KINDS - {"episode"}:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"kind must be one of {sorted(KINDS - {'episode'})}")
    ops = validate([Op(op="ADD", kind=body.kind, content=body.content, scope=body.scope, importance=body.importance,
                       confidence=0.95, subj=body.subj, pred=body.pred, obj=body.obj, evidence="explicit API call")],
                   set())
    ctx = TurnCtx(p.tenant_id, uid, project.id, None, None, project.isolated, "explicit", datetime.now(timezone.utc))
    async with tenant_session(p.tenant_id) as conn:
        report = await apply(conn, ops, ctx)
    await redis.incr(Keys.memory_version(project.id))
    if report.important:
        await mark_dirty(p.tenant_id, project.id)
    return {"added": report.added, "reinforced": report.reinforced, "invalidated": report.invalidated}


@router.get("/traces/{trace_id}")
async def get_trace(trace_id: UUID, user_id: str | None = None, p: Principal = Depends(require_app)) -> dict:
    """Memory debugger: every candidate considered for one retrieval, its score breakdown and outcome."""
    p.require_scope("memory:read")
    uid = await resolve_user(p, user_id)
    async with tenant_session(p.tenant_id) as conn:
        row = (await conn.execute(
            text("SELECT id, project_id, chat_id, query, gate, cache_hit, agent_used, candidates, selected, latency, "
                 "created_at FROM retrieval_traces WHERE id = :id"), {"id": trace_id})).mappings().first()
    if not row:
        return {"id": str(trace_id), "pending": True}  # written by the worker within ~2 s; poll again
    await project_for(p, uid, row["project_id"])
    ids = [UUID(c["id"]) for c in row["candidates"]]
    async with tenant_session(p.tenant_id) as conn:
        mems = {str(r["id"]): r for r in (await conn.execute(
            text("SELECT id, kind, content, scope, is_belief, valid_until FROM memories WHERE id = ANY(CAST(:ids AS uuid[]))"),
            {"ids": ids})).mappings().all()}
    cands = []
    for c in row["candidates"]:
        m = mems.get(c["id"])
        cands.append({**c, "kind": m["kind"] if m else None, "content": m["content"] if m else "(deleted)",
                      "scope": m["scope"] if m else None, "is_belief": m["is_belief"] if m else None,
                      "still_true": (m["valid_until"] is None) if m else None})
    return {**{k: row[k] for k in ("id", "project_id", "chat_id", "query", "gate", "cache_hit", "agent_used",
                                   "latency", "created_at")}, "selected": [str(x) for x in row["selected"]],
            "candidates": cands}


@router.get("/turns/{turn_id}/changes")
async def turn_changes(turn_id: UUID, user_id: str | None = None, p: Principal = Depends(require_app)) -> dict:
    """What memory formation did with one turn: ops (ADD / REINFORCE / INVALIDATE / ...) per memory."""
    p.require_scope("memory:read")
    uid = await resolve_user(p, user_id)
    async with tenant_session(p.tenant_id) as conn:
        turn = (await conn.execute(
            text("SELECT id, project_id, chat_id, seq, salience, processed_at FROM turns WHERE id = :id"),
            {"id": turn_id})).mappings().first()
        if not turn:
            return {"turn_id": str(turn_id), "processed": False, "changes": []}
        await project_for(p, uid, turn["project_id"])
        rows = (await conn.execute(
            text("SELECT pv.op, pv.speaker, pv.method, pv.evidence, pv.created_at, m.id AS memory_id, m.kind, "
                 "m.content, m.is_belief, m.valid_until FROM provenance pv JOIN memories m ON m.id = pv.memory_id "
                 "WHERE pv.turn_id = :t ORDER BY pv.id"), {"t": turn_id})).mappings().all()
        used = (await conn.execute(
            text("SELECT id, content FROM memories WHERE last_recalled_at >= :ts AND id = ANY("
                 "SELECT unnest(retrieved_ids) FROM turns WHERE id = :t)"),
            {"t": turn_id, "ts": turn["processed_at"] or datetime.now(timezone.utc)})).mappings().all()
    return {"turn_id": str(turn_id), "processed": turn["processed_at"] is not None, "salience": turn["salience"],
            "changes": [{**dict(r), "memory_id": str(r["memory_id"])} for r in rows],
            "reinforced_by_use": [{"id": str(u["id"]), "content": u["content"]} for u in used]}


_LIST_COLS = ("id, kind, content, scope, is_belief, subj, pred, obj, importance, confidence, stability, "
              "recall_count, utility, observed_count, status, valid_from, valid_until, superseded_by, thread_status, "
              "due_at, source_chat_id, created_at, updated_at, last_recalled_at")


@router.get("")
async def list_memories(
    project_id: UUID, user_id: str | None = None, kind: str | None = None, q: str | None = None,
    status_: str | None = Query(default=None, alias="status"), beliefs_only: bool = False,
    include_invalid: bool = False, include_archived: bool = False, include_user_scope: bool = True,
    limit: int = Query(default=100, le=1000), p: Principal = Depends(require_app),
) -> list[dict]:
    p.require_scope("memory:read")
    uid, project = await _ctx(p, user_id, project_id, None)
    where = ["(project_id = :p" + (" OR (scope = 'user' AND user_id = :u AND project_id IS NULL))"
                                   if include_user_scope and not project.isolated else ")")]
    if kind:
        where.append("kind = :k")
    if not include_invalid:
        where.append("valid_until IS NULL")
    if status_:
        where.append("status = :st")
    elif not include_archived:
        where.append("status <> 'archived'")
    if q:
        where.append("content ILIKE :q")
    if beliefs_only:
        where.append("is_belief")
    async with tenant_session(p.tenant_id) as conn:
        rows = (await conn.execute(
            text(f"SELECT {_LIST_COLS} FROM memories WHERE {' AND '.join(where)} "
                 "ORDER BY importance DESC, created_at DESC LIMIT :l"),
            {"p": project_id, "u": uid, "k": kind, "l": limit, "st": status_,
             "q": f"%{q}%" if q else None})).mappings().all()
    return [dict(r) for r in rows]


async def _owned_memory(p: Principal, user_id: str | None, memory_id: UUID) -> dict:
    uid = await resolve_user(p, user_id)
    async with tenant_session(p.tenant_id) as conn:
        row = (await conn.execute(text(f"SELECT {_LIST_COLS}, user_id, project_id, source_turn_id, metadata "
                                       "FROM memories WHERE id = :id"), {"id": memory_id})).mappings().first()
    if not row or row["user_id"] != uid:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "memory not found")
    if row["project_id"] is not None:
        p.check_project(row["project_id"])
    return dict(row)


@router.get("/{memory_id}/explain")
async def explain(memory_id: UUID, user_id: str | None = None, p: Principal = Depends(require_app)) -> dict:
    """Why does the AI believe this? Provenance, version history, strength, and recent retrievals."""
    p.require_scope("memory:read")
    mem = await _owned_memory(p, user_id, memory_id)
    async with tenant_session(p.tenant_id) as conn:
        versions = (await conn.execute(
            text("SELECT version, content, op, reason, turn_id, created_at FROM memory_versions "
                 "WHERE memory_id = :m ORDER BY id"), {"m": memory_id})).mappings().all()
        prov = (await conn.execute(
            text("SELECT op, speaker, evidence, method, chat_id, turn_id, created_at FROM provenance "
                 "WHERE memory_id = :m ORDER BY id"), {"m": memory_id})).mappings().all()
        traces = (await conn.execute(
            text("SELECT id, query, created_at FROM retrieval_traces WHERE CAST(:m AS uuid) = ANY(selected) "
                 "ORDER BY created_at DESC LIMIT 5"), {"m": memory_id})).mappings().all()
        entities = (await conn.execute(
            text("SELECT e.name FROM memory_entities me JOIN entities e ON e.id = me.entity_id "
                 "WHERE me.memory_id = :m"), {"m": memory_id})).scalars().all()
    now = datetime.now(timezone.utc)
    r = retrievability(now, mem["created_at"], mem["last_recalled_at"], mem["stability"],
                       mem["thread_status"] == "open")
    return {
        "memory": mem,
        "strength": {"retrievability": round(r, 3), "stability_days": round(mem["stability"], 1),
                     "recall_count": mem["recall_count"], "utility": mem["utility"],
                     "observed_count": mem["observed_count"]},
        "status": "fact" if not mem["is_belief"] else "belief (unconfirmed)",
        "still_true": mem["valid_until"] is None,
        "entities": entities,
        "provenance": [dict(x) for x in prov],
        "versions": [dict(x) for x in versions],
        "recent_retrievals": [dict(x) for x in traces],
    }


@router.delete("/{memory_id}", status_code=204)
async def delete_memory(memory_id: UUID, user_id: str | None = None, p: Principal = Depends(require_app)) -> None:
    """Hard delete (right to be forgotten): the memory, its versions and provenance."""
    p.require_scope("memory:write")
    mem = await _owned_memory(p, user_id, memory_id)
    async with tenant_session(p.tenant_id) as conn:
        await conn.execute(text("DELETE FROM memories WHERE id = :m"), {"m": memory_id})
    if mem["project_id"]:
        await redis.incr(Keys.memory_version(mem["project_id"]))
        await mark_dirty(p.tenant_id, mem["project_id"])
