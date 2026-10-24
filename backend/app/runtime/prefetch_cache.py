"""Type-ahead candidate pool cache (Redis).

While the user types, the pool of candidates for the partial prompt is stored per chat. On
submit, if the final prompt is semantically close to the last partial (and the entity set,
temporal cue and project memory version are unchanged), the pool is simply *rescored* with the
final query vector: no SQL on the critical path.
"""

import time
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np
import orjson

from app.cache import Keys, redis
from app.config import settings
from app.runtime.retrieval.candidates import Candidate


@dataclass
class Pool:
    partial: str
    qvec: np.ndarray
    cands: list[Candidate]
    vecs: np.ndarray
    mver: str
    entity_key: str
    cue_key: str
    ts: float = field(default_factory=time.time)
    agent: dict | None = None  # speculative agent result: {"query", "ids", "note"}


def entity_key(ids: list[str]) -> str:
    return ",".join(sorted(ids))


async def save(chat_id: UUID, pool: Pool) -> None:
    meta = {
        "partial": pool.partial,
        "cands": [c.to_json() for c in pool.cands],
        "mver": pool.mver,
        "entity_key": pool.entity_key,
        "cue_key": pool.cue_key,
        "ts": pool.ts,
        "agent": pool.agent,
        "n": len(pool.cands),
    }
    blob = np.vstack([pool.qvec[None, :], pool.vecs]).astype(np.float32).tobytes() if len(pool.cands) else (
        pool.qvec.astype(np.float32).tobytes()
    )
    pipe = redis.pipeline(transaction=False)
    pipe.set(Keys.prefetch(chat_id), orjson.dumps(meta), ex=settings.prefetch_ttl_s)
    pipe.set(Keys.prefetch_vecs(chat_id), blob, ex=settings.prefetch_ttl_s)
    await pipe.execute()


async def load(chat_id: UUID) -> Pool | None:
    pipe = redis.pipeline(transaction=False)
    pipe.get(Keys.prefetch(chat_id))
    pipe.get(Keys.prefetch_vecs(chat_id))
    meta_raw, blob = await pipe.execute()
    if not meta_raw or not blob:
        return None
    meta = orjson.loads(meta_raw)
    mat = np.frombuffer(blob, dtype=np.float32).reshape(-1, settings.embed_dim)
    if mat.shape[0] != meta["n"] + 1:
        return None
    return Pool(
        partial=meta["partial"],
        qvec=mat[0],
        cands=[Candidate.from_json(c) for c in meta["cands"]],
        vecs=mat[1:],
        mver=meta["mver"],
        entity_key=meta["entity_key"],
        cue_key=meta["cue_key"],
        ts=meta["ts"],
        agent=meta.get("agent"),
    )


async def attach_agent(chat_id: UUID, agent: dict) -> None:
    raw = await redis.get(Keys.prefetch(chat_id))
    if not raw:
        return
    meta = orjson.loads(raw)
    meta["agent"] = agent
    await redis.set(Keys.prefetch(chat_id), orjson.dumps(meta), ex=settings.prefetch_ttl_s)
