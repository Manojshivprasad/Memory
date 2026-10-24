"""Stage D: nightly "sleep" consolidation for a project.

1. forgetting curve   - recompute retrievability; active -> weakening -> archived (soft, provenance kept)
2. merge              - collapse near-identical memories (keeps history via superseded_by)
3. generalise         - beliefs observed repeatedly become facts
4. prospective hygiene- open threads untouched for 30 days are flagged stale and fade
5. promote            - the same preference in 2+ projects becomes a user-level preference
6. refresh the brief
"""

import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

import numpy as np
import orjson
from sqlalchemy import text

from app.cache import Keys, redis
from app.consolidation.brief import refresh_brief
from app.db.session import tenant_session
from app.embeddings import as_array
from app.memory.polarity import contradicts
from app.memory.strength import lifecycle_status, retrievability
from app.write.ops import ApplyReport, Op, TurnCtx, insert_memory

log = logging.getLogger(__name__)
MERGE_SIM = 0.95
PROMOTE_SIM = 0.9
STALE_DAYS = 30


async def _prov(conn, tenant_id: UUID, mid, op: str, evidence: str) -> None:  # noqa: ANN001
    await conn.execute(
        text("INSERT INTO provenance (tenant_id, memory_id, op, speaker, evidence, method) "
             "VALUES (:t, :m, :op, 'system', :e, 'sleep')"),
        {"t": tenant_id, "m": mid, "op": op, "e": evidence[:500]},
    )


async def sleep_project(tenant_id: UUID, project_id: UUID) -> dict:
    now = datetime.now(timezone.utc)
    stats = {"weakening": 0, "archived": 0, "reactivated": 0, "merged": 0, "promoted_beliefs": 0, "stale": 0,
             "user_prefs": 0}
    async with tenant_session(tenant_id) as conn:
        proj = (await conn.execute(text("SELECT user_id, isolated FROM projects WHERE id = :p"),
                                   {"p": project_id})).mappings().first()
        if not proj:
            return stats
        rows = (await conn.execute(
            text(
                "SELECT id, kind, importance, utility, stability, created_at, last_recalled_at, thread_status, "
                "status, is_belief, observed_count, confidence, embedding, valid_until, content FROM memories "
                "WHERE project_id = :p AND status <> 'archived'"
            ),
            {"p": project_id},
        )).mappings().all()

        # 1) forgetting curve
        for r in rows:
            is_open = r["thread_status"] == "open"
            R = retrievability(now, r["created_at"], r["last_recalled_at"], r["stability"], is_open)
            new = lifecycle_status(R, r["importance"], is_open)
            if new != r["status"]:
                await conn.execute(text("UPDATE memories SET status = :s, updated_at = now() WHERE id = :id"),
                                   {"s": new, "id": r["id"]})
                stats[new if new != "active" else "reactivated"] += 1

        # 2) merge near-duplicates (same kind, both still true)
        live = [r for r in rows if r["valid_until"] is None and r["kind"] != "episode"]
        archived: set = set()
        by_kind: dict[str, list] = {}
        for r in live:
            by_kind.setdefault(r["kind"], []).append(r)
        for group in by_kind.values():
            if len(group) < 2:
                continue
            mat = np.stack([as_array(r["embedding"]) for r in group])
            sims = mat @ mat.T
            for i in range(len(group)):
                if group[i]["id"] in archived:
                    continue
                for j in range(i + 1, len(group)):
                    if group[j]["id"] in archived or sims[i, j] < MERGE_SIM:
                        continue
                    a, b = group[i], group[j]
                    keep, drop = (a, b) if (a["observed_count"], -a["created_at"].timestamp()) >= (
                        b["observed_count"], -b["created_at"].timestamp()) else (b, a)
                    await conn.execute(
                        text("UPDATE memories SET status = 'archived', superseded_by = :k, updated_at = now() "
                             "WHERE id = :d"), {"k": keep["id"], "d": drop["id"]})
                    await conn.execute(
                        text("UPDATE memories SET observed_count = observed_count + :n, "
                             "importance = GREATEST(importance, :imp), updated_at = now() WHERE id = :k"),
                        {"n": drop["observed_count"], "imp": drop["importance"], "k": keep["id"]})
                    await _prov(conn, tenant_id, keep["id"], "MERGE", drop["content"])
                    archived.add(drop["id"])
                    stats["merged"] += 1

        # 2b) preference reversals already in memory: the newest attitude toward a topic wins
        attitudes = sorted((r for r in live if r["id"] not in archived and r["kind"] in ("preference", "fact", "goal")),
                           key=lambda r: r["created_at"], reverse=True)
        retired: set = set()
        for i, newer in enumerate(attitudes):
            if newer["id"] in retired:
                continue
            nv = as_array(newer["embedding"])
            for older in attitudes[i + 1:]:
                if older["id"] in retired or float(nv @ as_array(older["embedding"])) < 0.45:
                    continue
                topic = contradicts(newer["content"], older["content"])
                if topic:
                    await conn.execute(
                        text("UPDATE memories SET valid_until = now(), superseded_by = :n, version = version + 1, "
                             "updated_at = now() WHERE id = :o AND valid_until IS NULL"),
                        {"n": newer["id"], "o": older["id"]})
                    await _prov(conn, tenant_id, older["id"], "INVALIDATE", f"reversed attitude toward '{topic}'")
                    retired.add(older["id"])
                    stats["reversed"] = stats.get("reversed", 0) + 1

        # 3) generalise: repeatedly observed beliefs become facts
        res = await conn.execute(
            text("UPDATE memories SET is_belief = false, source_reliability = 0.8, updated_at = now() "
                 "WHERE project_id = :p AND is_belief AND observed_count >= 3 AND confidence >= 0.6 "
                 "AND status <> 'archived' RETURNING id"),
            {"p": project_id})
        for (mid,) in res.all():
            await _prov(conn, tenant_id, mid, "PROMOTE", "observed repeatedly")
            stats["promoted_beliefs"] += 1

        # 4) stale open threads fade (not deleted)
        res = await conn.execute(
            text(
                "UPDATE memories SET importance = importance * 0.8, "
                "metadata = metadata || CAST(:m AS jsonb), updated_at = now() "
                "WHERE project_id = :p AND thread_status = 'open' AND created_at < :cut "
                "AND (last_recalled_at IS NULL OR last_recalled_at < :cut) AND metadata->>'stale' IS NULL RETURNING id"
            ),
            {"p": project_id, "cut": now - timedelta(days=STALE_DAYS), "m": orjson.dumps({"stale": True}).decode()})
        stats["stale"] = len(res.all())

        # 5) promote cross-project preferences to user scope
        if not proj["isolated"]:
            local = [r for r in live if r["kind"] == "preference" and r["id"] not in archived]
            if local:
                others = (await conn.execute(
                    text(
                        "SELECT m.id, m.embedding, m.importance, m.confidence FROM memories m "
                        "JOIN projects p ON p.id = m.project_id WHERE m.user_id = :u AND m.kind = 'preference' "
                        "AND m.scope = 'project' AND m.project_id <> :p AND NOT p.isolated "
                        "AND m.valid_until IS NULL AND m.status <> 'archived'"
                    ),
                    {"u": proj["user_id"], "p": project_id})).mappings().all()
                user_prefs = (await conn.execute(
                    text("SELECT embedding FROM memories WHERE user_id = :u AND scope = 'user' "
                         "AND kind = 'preference' AND valid_until IS NULL"), {"u": proj["user_id"]})).all()
                umat = np.stack([as_array(e[0]) for e in user_prefs]) if user_prefs else None
                ctx = TurnCtx(tenant_id, proj["user_id"], project_id, None, None, False, "sleep", now)
                for r in local:
                    v = as_array(r["embedding"])
                    match = next((o for o in others if float(as_array(o["embedding"]) @ v) >= PROMOTE_SIM),
                                 None)
                    if not match or (umat is not None and float((umat @ v).max()) >= PROMOTE_SIM):
                        continue
                    new_id = await insert_memory(conn, ctx, Op(
                        op="ADD", kind="preference", content=r["content"], scope="user",
                        importance=max(r["importance"], match["importance"]),
                        confidence=max(r["confidence"], match["confidence"]), evidence="seen in 2+ projects"),
                        v, ApplyReport(), {"promoted_from": [str(r["id"]), str(match["id"])]})
                    for old in (r["id"], match["id"]):
                        await conn.execute(text("UPDATE memories SET status = 'archived', superseded_by = :n "
                                                "WHERE id = :o"), {"n": UUID(new_id), "o": old})
                    umat = v[None, :] if umat is None else np.vstack([umat, v])
                    stats["user_prefs"] += 1

        await conn.execute(text("UPDATE projects SET last_sleep_at = now() WHERE id = :p"), {"p": project_id})
    await redis.incr(Keys.memory_version(project_id))
    await refresh_brief(tenant_id, project_id)
    return stats
