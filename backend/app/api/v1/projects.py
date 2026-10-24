import uuid
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.auth.deps import Principal, forget_project, project_for, require_app, user_param
from app.cache import Keys, redis
from app.consolidation.brief import refresh_brief
from app.consolidation.sleep import sleep_project
from app.db.session import tenant_session
from app.runtime.brief import get_brief

router = APIRouter(prefix="/v1/projects", tags=["projects"])

_COLS = "id, name, description, isolated, brief, brief_updated_at, created_at, updated_at"


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    isolated: bool = Field(default=False, description="Opt out of user-level (cross-project) memory")


class ProjectPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    isolated: bool | None = None


@router.post("", status_code=201)
async def create_project(body: ProjectIn, p: Principal = Depends(require_app),
                         user_id: UUID = Depends(user_param)) -> dict:
    p.require_scope("memory:write")
    pid = uuid.uuid4()
    async with tenant_session(p.tenant_id) as conn:
        row = (await conn.execute(
            text(f"INSERT INTO projects (id, tenant_id, user_id, name, description, isolated) "
                 f"VALUES (:id, :t, :u, :n, :d, :i) RETURNING {_COLS}"),
            {"id": pid, "t": p.tenant_id, "u": user_id, "n": body.name, "d": body.description, "i": body.isolated},
        )).mappings().one()
    return dict(row)


@router.get("")
async def list_projects(p: Principal = Depends(require_app), user_id: UUID = Depends(user_param)) -> list[dict]:
    p.require_scope("memory:read")
    async with tenant_session(p.tenant_id) as conn:
        rows = (await conn.execute(
            text(f"SELECT {_COLS} FROM projects WHERE user_id = :u ORDER BY updated_at DESC"), {"u": user_id}
        )).mappings().all()
    return [dict(r) for r in rows if p.project_ids is None or r["id"] in p.project_ids]


@router.get("/{project_id}")
async def get_project(project_id: UUID, p: Principal = Depends(require_app),
                      user_id: UUID = Depends(user_param)) -> dict:
    p.require_scope("memory:read")
    await project_for(p, user_id, project_id)
    async with tenant_session(p.tenant_id) as conn:
        row = (await conn.execute(text(f"SELECT {_COLS} FROM projects WHERE id = :id"),
                                  {"id": project_id})).mappings().one()
        stats = (await conn.execute(
            text("SELECT count(*) FILTER (WHERE valid_until IS NULL AND status <> 'archived') AS active, "
                 "count(*) AS total FROM memories WHERE project_id = :id"), {"id": project_id})).mappings().one()
    return {**dict(row), "memory_stats": dict(stats)}


@router.patch("/{project_id}")
async def update_project(project_id: UUID, body: ProjectPatch, p: Principal = Depends(require_app),
                         user_id: UUID = Depends(user_param)) -> dict:
    p.require_scope("memory:write")
    await project_for(p, user_id, project_id)
    async with tenant_session(p.tenant_id) as conn:
        row = (await conn.execute(
            text(f"UPDATE projects SET name = COALESCE(:n, name), description = COALESCE(:d, description), "
                 f"isolated = COALESCE(:i, isolated), updated_at = now() WHERE id = :id RETURNING {_COLS}"),
            {"n": body.name, "d": body.description, "i": body.isolated, "id": project_id},
        )).mappings().one()
    forget_project(project_id)
    await redis.incr(Keys.memory_version(project_id))
    return dict(row)


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: UUID, p: Principal = Depends(require_app),
                         user_id: UUID = Depends(user_param)) -> None:
    """Hard delete: the project, its chats, turns and project-scoped memories. User-level memory stays."""
    p.require_scope("memory:write")
    await project_for(p, user_id, project_id)
    async with tenant_session(p.tenant_id) as conn:
        await conn.execute(text("DELETE FROM projects WHERE id = :id"), {"id": project_id})
    forget_project(project_id)
    await redis.delete(Keys.brief(project_id), Keys.memory_version(project_id))


@router.get("/{project_id}/brief")
async def project_brief(project_id: UUID, refresh: bool = Query(default=False), p: Principal = Depends(require_app),
                        user_id: UUID = Depends(user_param)) -> dict:
    p.require_scope("memory:read")
    await project_for(p, user_id, project_id)
    brief = await refresh_brief(p.tenant_id, project_id) if refresh else await get_brief(p.tenant_id, project_id)
    return {"project_id": str(project_id), "brief": brief}


@router.post("/{project_id}/sleep")
async def run_sleep(project_id: UUID, p: Principal = Depends(require_app),
                    user_id: UUID = Depends(user_param)) -> dict:
    """Run the nightly consolidation for this project now (normally scheduled by the worker)."""
    p.require_scope("memory:write")
    await project_for(p, user_id, project_id)
    return await sleep_project(p.tenant_id, project_id)


@router.get("/{project_id}/analytics")
async def project_analytics(project_id: UUID, days: int = Query(default=14, ge=1, le=90),
                            p: Principal = Depends(require_app), user_id: UUID = Depends(user_param)) -> dict:
    """Memory health: composition, lifecycle, growth, and retrieval performance for one project."""
    p.require_scope("memory:read")
    await project_for(p, user_id, project_id)
    async with tenant_session(p.tenant_id) as conn:
        by_kind = (await conn.execute(text(
            "SELECT kind, count(*) AS n FROM memories WHERE project_id = :p AND valid_until IS NULL "
            "AND status <> 'archived' GROUP BY kind ORDER BY n DESC"), {"p": project_id})).mappings().all()
        life = (await conn.execute(text(
            "SELECT count(*) FILTER (WHERE status = 'active' AND valid_until IS NULL) AS active, "
            "count(*) FILTER (WHERE status = 'weakening' AND valid_until IS NULL) AS weakening, "
            "count(*) FILTER (WHERE status = 'archived') AS archived, "
            "count(*) FILTER (WHERE valid_until IS NOT NULL) AS superseded, "
            "count(*) FILTER (WHERE is_belief AND valid_until IS NULL) AS beliefs, "
            "count(*) FILTER (WHERE thread_status = 'open') AS open_threads, "
            "count(*) AS total FROM memories WHERE project_id = :p"), {"p": project_id})).mappings().one()
        growth = (await conn.execute(text(
            "SELECT to_char(date_trunc('day', created_at), 'YYYY-MM-DD') AS day, count(*) AS n FROM memories "
            "WHERE project_id = :p AND created_at > now() - make_interval(days => :d) GROUP BY 1 ORDER BY 1"),
            {"p": project_id, "d": days})).mappings().all()
        perf = (await conn.execute(text(
            "SELECT count(*) AS retrievals, "
            "avg(CAST(latency->>'total' AS float)) AS avg_ms, "
            "percentile_cont(0.95) WITHIN GROUP (ORDER BY CAST(latency->>'total' AS float)) AS p95_ms, "
            "avg(CASE WHEN cache_hit THEN 1 ELSE 0 END) AS cache_hit_rate, "
            "avg(CASE WHEN agent_used THEN 1 ELSE 0 END) AS agent_rate, "
            "avg(coalesce(array_length(selected, 1), 0)) AS avg_injected, "
            "avg(CASE WHEN gate <> 'retrieve' THEN 1 ELSE 0 END) AS skipped_rate "
            "FROM retrieval_traces WHERE project_id = :p AND created_at > now() - make_interval(days => :d)"),
            {"p": project_id, "d": days})).mappings().one()
        chats = (await conn.execute(text(
            "SELECT count(*) AS chats, coalesce(sum(turn_count), 0) AS turns FROM chats WHERE project_id = :p"),
            {"p": project_id})).mappings().one()
    return {"by_kind": [dict(r) for r in by_kind], "lifecycle": dict(life), "growth": [dict(r) for r in growth],
            "retrieval": {k: (float(v) if v is not None else None) for k, v in dict(perf).items()},
            "activity": {k: int(v) for k, v in dict(chats).items()}}


@router.get("/{project_id}/graph")
async def project_graph(project_id: UUID, limit: int = Query(default=80, le=300), p: Principal = Depends(require_app),
                        user_id: UUID = Depends(user_param)) -> dict:
    """Entity association graph (what the spreading-activation channel walks)."""
    p.require_scope("memory:read")
    await project_for(p, user_id, project_id)
    async with tenant_session(p.tenant_id) as conn:
        nodes = (await conn.execute(text(
            "SELECT id, name, type, mention_count FROM entities WHERE project_id = :p "
            "ORDER BY mention_count DESC LIMIT :l"), {"p": project_id, "l": limit})).mappings().all()
        ids = [n["id"] for n in nodes]
        edges = (await conn.execute(text(
            "SELECT src_entity, dst_entity, relation, weight FROM edges WHERE project_id = :p "
            "AND src_entity = ANY(CAST(:ids AS uuid[])) AND dst_entity = ANY(CAST(:ids AS uuid[])) "
            "AND (valid_until IS NULL OR valid_until > now())"), {"p": project_id, "ids": ids})).mappings().all()
    return {"nodes": [{**dict(n), "id": str(n["id"])} for n in nodes],
            "edges": [{"source": str(e["src_entity"]), "target": str(e["dst_entity"]), "relation": e["relation"],
                       "weight": e["weight"]} for e in edges]}
