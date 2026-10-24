"""Retrieval traces (the data behind the future memory debugger).

The API only RPUSHes the trace to Redis (fire-and-forget); the worker flushes them to Postgres in
batches, so tracing never competes with the hot path for database connections.
"""

import asyncio
import logging
import time
import uuid
from uuid import UUID

import orjson
from sqlalchemy import text

from app.cache import redis
from app.db.session import tenant_session

log = logging.getLogger(__name__)
_background: set[asyncio.Task] = set()
PENDING = "traces:pending"


class Timer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._last = self._t0
        self.marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.marks[name] = round((now - self._last) * 1000, 2)
        self._last = now

    def total(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 2)

    def server_timing(self) -> str:
        parts = [f"{k};dur={v}" for k, v in self.marks.items()]
        parts.append(f"memory;dur={self.total()}")
        return ", ".join(parts)


def spawn(coro) -> None:  # noqa: ANN001
    """Fire-and-forget with a strong reference (so the task isn't GC'd mid-flight)."""
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


async def _push(payload: bytes) -> None:
    try:
        await redis.rpush(PENDING, payload)
    except Exception:  # noqa: BLE001
        log.exception("failed to queue retrieval trace")


def record(tenant_id: UUID, project_id: UUID, chat_id: UUID | None, query: str, gate: str, cache_hit: bool,
           agent_used: bool, candidates: list[dict], selected: list[str], latency: dict) -> str:
    trace_id = str(uuid.uuid4())
    spawn(_push(orjson.dumps({
        "id": trace_id, "t": str(tenant_id), "p": str(project_id), "c": str(chat_id) if chat_id else None,
        "q": query[:2000], "g": gate, "ch": cache_hit, "ag": agent_used, "cands": candidates[:40],
        "sel": selected, "lat": latency,
    })))
    return trace_id


async def flush(max_items: int = 500) -> int:
    """Worker side: move queued traces into Postgres, grouped per tenant (RLS)."""
    pipe = redis.pipeline(transaction=True)
    pipe.lrange(PENDING, 0, max_items - 1)
    pipe.ltrim(PENDING, max_items, -1)
    raw, _ = await pipe.execute()
    if not raw:
        return 0
    by_tenant: dict[str, list[dict]] = {}
    for item in raw:
        d = orjson.loads(item)
        by_tenant.setdefault(d["t"], []).append({
            "id": UUID(d["id"]), "t": UUID(d["t"]), "p": UUID(d["p"]), "c": UUID(d["c"]) if d["c"] else None,
            "q": d["q"], "g": d["g"], "ch": d["ch"], "ag": d["ag"], "cands": orjson.dumps(d["cands"]).decode(),
            "sel": [UUID(s) for s in d["sel"]], "lat": orjson.dumps(d["lat"]).decode(),
        })
    for tenant, rows in by_tenant.items():
        async with tenant_session(tenant) as conn:
            await conn.execute(
                text(
                    "INSERT INTO retrieval_traces (id, tenant_id, project_id, chat_id, query, gate, cache_hit, "
                    "agent_used, candidates, selected, latency) VALUES (:id, :t, :p, :c, :q, :g, :ch, :ag, "
                    "CAST(:cands AS jsonb), CAST(:sel AS uuid[]), CAST(:lat AS jsonb)) ON CONFLICT (id) DO NOTHING"
                ),
                rows,
            )
    return len(raw)
