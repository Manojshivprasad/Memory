"""Backfill for memories stored before the harness-driven fixes:

* re-embed every memory with first-person embedding text (how users ask: "where do I live?")
* tag standing constraints (allergies, injuries, response-style preferences) with their domains
* canonicalise predicates (location/moved_to -> lives_in, ...)

    python -m app.tools.reembed            # all tenants (runs as the schema owner)
"""

import asyncio

import orjson
from sqlalchemy import text

from app.db.session import admin_session, dispose
from app.embeddings import embed_passages, warmup
from app.runtime.intents import constraint_domains
from app.textutil import embedding_text
from app.write.ops import canonical_pred

BATCH = 256


async def main() -> None:
    warmup()
    done = 0
    last_id = None
    while True:
        async with admin_session() as conn:
            rows = (await conn.execute(
                text("SELECT id, kind, content, pred, metadata FROM memories "
                     "WHERE (CAST(:after AS uuid) IS NULL OR id > CAST(:after AS uuid)) ORDER BY id LIMIT :n"),
                {"after": last_id, "n": BATCH},
            )).mappings().all()
            if not rows:
                break
            vecs = await embed_passages([embedding_text(r["content"]) for r in rows])
            for r, v in zip(rows, vecs):
                meta = dict(r["metadata"] or {})
                doms = constraint_domains(r["content"], r["kind"]) if r["kind"] != "episode" else None
                if doms:
                    meta["constraint_domains"] = doms
                await conn.execute(
                    text("UPDATE memories SET embedding = CAST(:v AS vector), pred = :p, "
                         "metadata = CAST(:m AS jsonb) WHERE id = :id"),
                    {"v": v, "p": canonical_pred(r["pred"]), "m": orjson.dumps(meta).decode(), "id": r["id"]},
                )
            done += len(rows)
            last_id = rows[-1]["id"]
        print(f"re-embedded {done}", flush=True)
    await dispose()
    print(f"done: {done} memories")


if __name__ == "__main__":
    asyncio.run(main())
