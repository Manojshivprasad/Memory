"""Project brief: a living ~150-token gist of the project (goals, decisions, state, open threads).

It is the *stable* part of the injected context, so it forms a cacheable prompt prefix.
"""

from uuid import UUID

from sqlalchemy import text

from app.cache import Keys, redis
from app.db.session import tenant_session

BRIEF_TTL_S = 3600


async def get_brief(tenant_id: UUID, project_id: UUID) -> str:
    cached = await redis.get(Keys.brief(project_id))
    if cached is not None:
        return cached.decode()
    async with tenant_session(tenant_id) as conn:
        brief = (
            await conn.execute(text("SELECT brief FROM projects WHERE id = :p"), {"p": project_id})
        ).scalar_one_or_none() or ""
    await redis.set(Keys.brief(project_id), brief, ex=BRIEF_TTL_S)
    return brief


async def set_cached_brief(project_id: UUID, brief: str) -> None:
    await redis.set(Keys.brief(project_id), brief, ex=BRIEF_TTL_S)


async def mark_dirty(tenant_id: UUID, project_id: UUID) -> None:
    await redis.sadd(Keys.BRIEFS_DIRTY, f"{tenant_id}:{project_id}")
