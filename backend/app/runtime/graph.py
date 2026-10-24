"""Per-project entity index + association graph, cached in-process.

Used on the hot path with zero DB round trips:
* ``match(text)``  - find known project entities mentioned in a prompt (n-gram dictionary lookup)
* ``spread(seeds)`` - spreading activation over the entity graph (1-2 hops, decaying per hop)

The cache is keyed by the project's memory version (bumped by the write path) and a short TTL.
"""

import time
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import text

from app.cache import Keys, redis
from app.config import settings
from app.db.session import tenant_session
from app.textutil import normalize, words

HOP_DECAY = 0.5
MAX_NGRAM = 4


@dataclass
class ProjectGraph:
    names: dict[str, str] = field(default_factory=dict)            # norm name/alias -> entity id
    display: dict[str, str] = field(default_factory=dict)          # entity id -> name
    adj: dict[str, list[tuple[str, float]]] = field(default_factory=dict)

    def match(self, text_: str) -> list[str]:
        toks = [normalize(w) for w in words(text_)]
        found: list[str] = []
        for n in range(MAX_NGRAM, 0, -1):
            for i in range(len(toks) - n + 1):
                eid = self.names.get(" ".join(toks[i : i + n]))
                if eid and eid not in found:
                    found.append(eid)
        return found

    def spread(self, seeds: dict[str, float], hops: int = 2) -> dict[str, float]:
        act = dict(seeds)
        frontier = dict(seeds)
        for _ in range(hops):
            nxt: dict[str, float] = {}
            for node, a in frontier.items():
                for nb, w in self.adj.get(node, ()):
                    v = a * HOP_DECAY * w
                    if v > act.get(nb, 0.0) and v > 0.05:
                        nxt[nb] = max(nxt.get(nb, 0.0), v)
            for k, v in nxt.items():
                act[k] = max(act.get(k, 0.0), v)
            frontier = nxt
            if not frontier:
                break
        return act


_cache: dict[UUID, tuple[float, bytes | None, ProjectGraph]] = {}


async def _load(tenant_id: UUID, project_id: UUID) -> ProjectGraph:
    g = ProjectGraph()
    async with tenant_session(tenant_id) as conn:
        ents = (
            await conn.execute(
                text("SELECT id, name, norm_name, aliases FROM entities WHERE project_id = :p"), {"p": project_id}
            )
        ).all()
        edges = (
            await conn.execute(
                text(
                    "SELECT src_entity, dst_entity, weight FROM edges WHERE project_id = :p "
                    "AND (valid_until IS NULL OR valid_until > now())"
                ),
                {"p": project_id},
            )
        ).all()
    for eid, name, norm, aliases in ents:
        sid = str(eid)
        g.display[sid] = name
        g.names[norm] = sid
        for a in aliases or []:
            g.names.setdefault(normalize(a), sid)
    for src, dst, w in edges:
        s, d = str(src), str(dst)
        g.adj.setdefault(s, []).append((d, float(w)))
        g.adj.setdefault(d, []).append((s, float(w)))
    return g


async def get_graph(tenant_id: UUID, project_id: UUID, mver: bytes | None) -> ProjectGraph:
    now = time.monotonic()
    hit = _cache.get(project_id)
    if hit and hit[1] == mver and now - hit[0] < settings.graph_cache_ttl_s:
        return hit[2]
    g = await _load(tenant_id, project_id)
    _cache[project_id] = (now, mver, g)
    return g


async def memory_version(project_id: UUID) -> bytes | None:
    return await redis.get(Keys.memory_version(project_id))
