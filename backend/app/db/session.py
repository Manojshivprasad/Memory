"""Database engines.

Two engines:
* ``engine`` connects as ``memory_app``; row-level security scopes every data table to
  ``current_setting('app.tenant_id')``. All request and per-tenant worker code uses
  :func:`tenant_session`.
* ``admin_engine`` connects as the schema owner (bypasses RLS). It is used only for auth
  tables before a tenant is known and by the scheduler to find work across tenants.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from pgvector.asyncpg import register_vector
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from app.config import settings


def _make_engine(url: str, pool_size: int):
    eng = create_async_engine(url, pool_size=pool_size, max_overflow=pool_size, pool_recycle=1800)

    @event.listens_for(eng.sync_engine, "connect")
    def _register_vector(dbapi_conn, _record):  # noqa: ANN001
        dbapi_conn.run_async(register_vector)

    return eng


engine = _make_engine(settings.database_url, settings.db_pool_size)
admin_engine = _make_engine(settings.database_admin_url, 5)

_SET_TENANT = text("SELECT set_config('app.tenant_id', :t, true)")


@asynccontextmanager
async def tenant_session(tenant_id: UUID | str) -> AsyncIterator[AsyncConnection]:
    """One transaction with RLS bound to ``tenant_id``. Commits on success."""
    async with engine.begin() as conn:
        await conn.execute(_SET_TENANT, {"t": str(tenant_id)})
        yield conn


@asynccontextmanager
async def auth_session() -> AsyncIterator[AsyncConnection]:
    """App role, no tenant bound: only the non-RLS auth tables are visible."""
    async with engine.begin() as conn:
        yield conn


@asynccontextmanager
async def admin_session() -> AsyncIterator[AsyncConnection]:
    async with admin_engine.begin() as conn:
        yield conn


async def dispose() -> None:
    await engine.dispose()
    await admin_engine.dispose()
