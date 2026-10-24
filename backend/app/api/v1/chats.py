import uuid
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.auth.deps import (ChatMeta, Principal, ProjectMeta, chat_for, project_for, remember_chat, require_app,
                           user_param)
from app.cache import Keys, redis
from app.consolidation.chat_episode import consolidate_chat
from app.db.session import tenant_session

router = APIRouter(prefix="/v1", tags=["chats"])


class ChatIn(BaseModel):
    title: str = Field(default="", max_length=200)


async def create_chat(tenant_id: UUID, project_id: UUID, title: str = "") -> dict:
    cid = uuid.uuid4()
    async with tenant_session(tenant_id) as conn:
        row = (await conn.execute(
            text("INSERT INTO chats (id, tenant_id, project_id, title) VALUES (:id, :t, :p, :title) "
                 "RETURNING id, project_id, title, status, turn_count, created_at, last_active_at"),
            {"id": cid, "t": tenant_id, "p": project_id, "title": title},
        )).mappings().one()
    remember_chat(cid, project_id)
    await redis.set(Keys.chat_seq(cid), 0, nx=True)
    return dict(row)


async def owned_chat(p: Principal, user_id: UUID, chat_id: UUID) -> tuple[ProjectMeta, ChatMeta]:
    async with tenant_session(p.tenant_id) as conn:
        pid = (await conn.execute(text("SELECT project_id FROM chats WHERE id = :c"), {"c": chat_id})).scalar()
    if pid is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "chat not found")
    project = await project_for(p, user_id, pid)
    return project, await chat_for(p, project, chat_id)


@router.post("/projects/{project_id}/chats", status_code=201)
async def new_chat(project_id: UUID, body: ChatIn, p: Principal = Depends(require_app),
                   user_id: UUID = Depends(user_param)) -> dict:
    p.require_scope("chat")
    await project_for(p, user_id, project_id)
    return await create_chat(p.tenant_id, project_id, body.title)


@router.get("/projects/{project_id}/chats")
async def list_chats(project_id: UUID, include_empty: bool = False, p: Principal = Depends(require_app),
                     user_id: UUID = Depends(user_param)) -> list[dict]:
    p.require_scope("memory:read")
    await project_for(p, user_id, project_id)
    async with tenant_session(p.tenant_id) as conn:
        rows = (await conn.execute(
            text("SELECT id, project_id, title, status, turn_count, created_at, last_active_at, consolidated_at "
                 "FROM chats WHERE project_id = :p AND (CAST(:e AS boolean) OR turn_count > 0) "
                 "ORDER BY last_active_at DESC"), {"p": project_id, "e": include_empty})).mappings().all()
    return [dict(r) for r in rows]


@router.get("/chats/recent")
async def recent_chats(limit: int = Query(default=30, le=200), p: Principal = Depends(require_app),
                       user_id: UUID = Depends(user_param)) -> list[dict]:
    """The user's recent chats across all projects (sidebar)."""
    p.require_scope("memory:read")
    async with tenant_session(p.tenant_id) as conn:
        rows = (await conn.execute(
            text("SELECT c.id, c.project_id, p.name AS project_name, c.title, c.status, c.turn_count, "
                 "c.last_active_at FROM chats c JOIN projects p ON p.id = c.project_id WHERE p.user_id = :u "
                 "AND c.turn_count > 0 ORDER BY c.last_active_at DESC LIMIT :l"),
            {"u": user_id, "l": limit})).mappings().all()
    return [dict(r) for r in rows if p.project_ids is None or r["project_id"] in p.project_ids]


@router.get("/chats/{chat_id}")
async def get_chat(chat_id: UUID, p: Principal = Depends(require_app), user_id: UUID = Depends(user_param)) -> dict:
    p.require_scope("memory:read")
    await owned_chat(p, user_id, chat_id)
    async with tenant_session(p.tenant_id) as conn:
        row = (await conn.execute(
            text("SELECT c.id, c.project_id, p.name AS project_name, c.title, c.status, c.turn_count, c.created_at, "
                 "c.last_active_at, c.consolidated_at FROM chats c JOIN projects p ON p.id = c.project_id "
                 "WHERE c.id = :c"), {"c": chat_id})).mappings().one()
    return dict(row)


@router.get("/chats/{chat_id}/traces")
async def chat_traces(chat_id: UUID, p: Principal = Depends(require_app), user_id: UUID = Depends(user_param)) -> list[dict]:
    """Retrieval traces of a chat in order (lets a UI attach the memory debugger to past messages)."""
    p.require_scope("memory:read")
    await owned_chat(p, user_id, chat_id)
    async with tenant_session(p.tenant_id) as conn:
        rows = (await conn.execute(
            text("SELECT id, query, gate, cache_hit, agent_used, created_at FROM retrieval_traces "
                 "WHERE chat_id = :c ORDER BY created_at"), {"c": chat_id})).mappings().all()
    return [dict(r) for r in rows]


class ChatPatch(BaseModel):
    title: str = Field(min_length=1, max_length=200)


@router.patch("/chats/{chat_id}")
async def rename_chat(chat_id: UUID, body: ChatPatch, p: Principal = Depends(require_app),
                      user_id: UUID = Depends(user_param)) -> dict:
    p.require_scope("chat")
    await owned_chat(p, user_id, chat_id)
    async with tenant_session(p.tenant_id) as conn:
        await conn.execute(text("UPDATE chats SET title = :t WHERE id = :c"), {"t": body.title, "c": chat_id})
    return {"id": str(chat_id), "title": body.title}


@router.delete("/chats/{chat_id}", status_code=204)
async def delete_chat(chat_id: UUID, p: Principal = Depends(require_app), user_id: UUID = Depends(user_param)) -> None:
    """Deletes the chat and its verbatim turns. Memories formed from it stay (they belong to the project)."""
    p.require_scope("chat")
    await owned_chat(p, user_id, chat_id)
    async with tenant_session(p.tenant_id) as conn:
        await conn.execute(text("DELETE FROM chats WHERE id = :c"), {"c": chat_id})
    await redis.delete(Keys.wm_turns(chat_id), Keys.wm_entities(chat_id), Keys.prefetch(chat_id),
                       Keys.prefetch_vecs(chat_id))


@router.get("/chats/{chat_id}/turns")
async def list_turns(chat_id: UUID, limit: int = Query(default=50, le=500), p: Principal = Depends(require_app),
                     user_id: UUID = Depends(user_param)) -> list[dict]:
    """Verbatim archive of the chat (the exact record, independent of memory decay)."""
    p.require_scope("memory:read")
    await owned_chat(p, user_id, chat_id)
    async with tenant_session(p.tenant_id) as conn:
        rows = (await conn.execute(
            text("SELECT id, seq, user_msg, assistant_msg, model, salience, created_at, processed_at FROM turns "
                 "WHERE chat_id = :c ORDER BY seq DESC LIMIT :l"), {"c": chat_id, "l": limit})).mappings().all()
    return [dict(r) for r in reversed(rows)]


@router.post("/chats/{chat_id}/close")
async def close_chat(chat_id: UUID, wait: bool = Query(default=False, description="Consolidate inline"),
                     p: Principal = Depends(require_app), user_id: UUID = Depends(user_param)) -> dict:
    """Close a chat and consolidate it into an episode (Stage C)."""
    p.require_scope("chat")
    await owned_chat(p, user_id, chat_id)
    async with tenant_session(p.tenant_id) as conn:
        await conn.execute(text("UPDATE chats SET status = 'closed' WHERE id = :c"), {"c": chat_id})
    if wait:
        return {"status": "closed", "episode": await consolidate_chat(p.tenant_id, chat_id)}
    await redis.sadd(Keys.CONSOLIDATE_PENDING, f"{p.tenant_id}:{chat_id}")
    return {"status": "closed", "episode": "scheduled"}
