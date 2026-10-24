"""Memory worker: Stage B consumer + Stage C/D scheduler.

    python -m app.workers.worker

* Turns are consumed from a Redis Stream consumer group. Different chats run concurrently; turns of
  the SAME chat are applied strictly in order (in-process lock + Redis lock across processes + seq check).
* Failed messages stay pending and are reclaimed; after 5 attempts they go to a dead-letter stream.
* The scheduler consolidates idle chats (Stage C), rebuilds dirty briefs, and runs nightly sleep (Stage D).
"""

import asyncio
import logging
import os
import signal
import socket
import uuid
from collections import defaultdict
from uuid import UUID

import orjson
from redis.exceptions import ResponseError
from sqlalchemy import text

from app.cache import Keys, redis
from app.config import settings
from app.consolidation.brief import refresh_brief
from app.consolidation.chat_episode import consolidate_chat
from app.consolidation.sleep import sleep_project
from app.db.session import admin_session, dispose
from app.embeddings import warmup
from app.observability import trace
from app.write.pipeline import TurnJob, process_turn

log = logging.getLogger("memory.worker")
MAX_ATTEMPTS = 5
ATTEMPTS_KEY = "mem:turns:attempts"
_RELEASE = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"


async def ensure_group() -> None:
    try:
        await redis.xgroup_create(settings.stream_turns, settings.stream_group, id="0", mkstream=True)
    except ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


class TurnConsumer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def _acquire(self, chat: str, token: str) -> bool:
        for _ in range(100):  # up to ~10 s
            if await redis.set(Keys.chat_lock(chat), token, nx=True, px=60_000):
                return True
            await asyncio.sleep(0.1)
        return False

    async def _wait_order(self, chat: str, seq: int) -> None:
        for _ in range(30):  # up to ~3 s for an earlier turn of this chat to land
            applied = await redis.get(Keys.chat_applied(chat))
            if applied is None or seq <= int(applied) + 1:
                return
            await asyncio.sleep(0.1)

    async def process(self, msg_id: bytes, fields: dict) -> bool:
        job = TurnJob.from_dict(orjson.loads(fields[b"data"]))
        chat = str(job.chat_id)
        token = uuid.uuid4().hex
        if not await self._acquire(chat, token):
            return False
        try:
            await self._wait_order(chat, job.seq)
            result = await process_turn(job)
            log.info("turn %s chat=%s seq=%s -> %s", job.turn_id, chat, job.seq,
                     {k: (len(v) if isinstance(v, list) else v) for k, v in result.items()})
        except Exception:  # noqa: BLE001
            log.exception("turn %s failed", job.turn_id)
            attempts = await redis.hincrby(ATTEMPTS_KEY, msg_id, 1)
            if attempts >= MAX_ATTEMPTS:
                await redis.xadd(Keys.DEAD_LETTER, fields)
                await redis.xack(settings.stream_turns, settings.stream_group, msg_id)
                await redis.hdel(ATTEMPTS_KEY, msg_id)
            return False
        finally:
            await redis.eval(_RELEASE, 1, Keys.chat_lock(chat), token)
        await redis.xack(settings.stream_turns, settings.stream_group, msg_id)
        await redis.hdel(ATTEMPTS_KEY, msg_id)
        return True

    async def _chat_batch(self, chat: str, items: list) -> None:
        async with self.locks[chat]:
            for msg_id, fields in items:
                if not await self.process(msg_id, fields):
                    break  # keep order: the rest stays pending and is reclaimed later

    async def handle(self, entries: list) -> None:
        by_chat: dict[str, list] = defaultdict(list)
        for msg_id, fields in entries:
            by_chat[fields.get(b"chat", b"").decode()].append((msg_id, fields))
        await asyncio.gather(*(self._chat_batch(c, items) for c, items in by_chat.items()))

    async def read_once(self, block_ms: int = 2000) -> int:
        resp = await redis.xreadgroup(settings.stream_group, self.name, {settings.stream_turns: ">"},
                                      count=64, block=block_ms)
        if not resp:
            return 0
        entries = resp[0][1]
        await self.handle(entries)
        return len(entries)

    async def reclaim_once(self, min_idle_ms: int = 60_000) -> int:
        res = await redis.xautoclaim(settings.stream_turns, settings.stream_group, self.name,
                                     min_idle_time=min_idle_ms, start_id="0-0", count=32)
        entries = [e for e in res[1] if e and e[1]]
        if entries:
            await self.handle(entries)
        return len(entries)

    async def run(self, stop: asyncio.Event) -> None:
        last_reclaim = 0.0
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            try:
                await self.read_once()
                if loop.time() - last_reclaim > 15:
                    last_reclaim = loop.time()
                    await self.reclaim_once()
            except Exception:  # noqa: BLE001
                log.exception("consumer loop error")
                await asyncio.sleep(1)


# ------------------------------------------------------------------------------ scheduler

async def _pop(key: str, n: int) -> list[tuple[UUID, UUID]]:
    items = await redis.spop(key, n) or []
    out = []
    for it in items:
        t, x = it.decode().split(":", 1)
        out.append((UUID(t), UUID(x)))
    return out


async def scheduler_tick() -> dict:
    done = {"consolidated": 0, "briefs": 0, "slept": 0}
    # Stage C: explicit requests (chat closed / every N turns) + chats that went idle
    targets = set(await _pop(Keys.CONSOLIDATE_PENDING, 20))
    async with admin_session() as conn:
        rows = (await conn.execute(
            text(
                "SELECT tenant_id, id FROM chats WHERE status = 'open' "
                "AND last_active_at < now() - make_interval(secs => CAST(:idle AS double precision)) "
                "AND (consolidated_at IS NULL OR consolidated_at < last_active_at) LIMIT 50"
            ),
            {"idle": settings.chat_idle_s},
        )).all()
        targets |= {(r[0], r[1]) for r in rows}
        sleepers = (await conn.execute(
            text(
                "SELECT p.tenant_id, p.id FROM projects p WHERE COALESCE(p.last_sleep_at, p.created_at) < "
                "now() - make_interval(secs => CAST(:iv AS double precision)) "
                "AND EXISTS (SELECT 1 FROM memories m WHERE m.project_id = p.id) LIMIT 20"
            ),
            {"iv": settings.sleep_interval_s},
        )).all()
    for tenant_id, chat_id in targets:
        try:
            if await consolidate_chat(tenant_id, chat_id):
                done["consolidated"] += 1
        except Exception:  # noqa: BLE001
            log.exception("consolidation failed for chat %s", chat_id)
    for tenant_id, project_id in await _pop(Keys.BRIEFS_DIRTY, 50):
        try:
            await refresh_brief(tenant_id, project_id)
            done["briefs"] += 1
        except Exception:  # noqa: BLE001
            log.exception("brief refresh failed for project %s", project_id)
    for tenant_id, project_id in sleepers:
        try:
            stats = await sleep_project(tenant_id, project_id)
            log.info("sleep project=%s %s", project_id, stats)
            done["slept"] += 1
        except Exception:  # noqa: BLE001
            log.exception("sleep failed for project %s", project_id)
    return done


async def scheduler(stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            done = await scheduler_tick()
            if any(done.values()):
                log.info("scheduler %s", done)
        except Exception:  # noqa: BLE001
            log.exception("scheduler error")
        try:
            await asyncio.wait_for(stop.wait(), timeout=settings.scheduler_tick_s)
        except asyncio.TimeoutError:
            pass


async def trace_flusher(stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await trace.flush()
        except Exception:  # noqa: BLE001
            log.exception("trace flush error")
        try:
            await asyncio.wait_for(stop.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    await asyncio.to_thread(warmup)
    await ensure_group()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows
            signal.signal(sig, lambda *_: stop.set())
    consumer = TurnConsumer(f"{socket.gethostname()}-{os.getpid()}")
    log.info("memory worker %s started", consumer.name)
    try:
        await asyncio.gather(consumer.run(stop), scheduler(stop), trace_flusher(stop))
    finally:
        await dispose()


if __name__ == "__main__":
    asyncio.run(main())
