"""Stage C: chat consolidation -> episodic memory.

When a chat goes idle / is closed / every N turns, the new segment of the chat is summarised into
an *episode gist* (what happened, decisions, open threads). Episodes are what a NEW chat in the
same project recalls ("where we left off", "what did we discuss about X"). The project brief is
refreshed afterwards.
"""

import logging
import uuid
from datetime import datetime, timezone
from uuid import UUID

import orjson
from sqlalchemy import text

from app.cache import Keys, redis
from app.consolidation.brief import refresh_brief
from app.config import settings
from app.db.session import tenant_session
from app.embeddings import embed_passages
from app.llm import json_complete, llm_enabled
from app.textutil import embedding_text, first_sentence
from app.write.ops import ApplyReport, Op, TurnCtx, apply, insert_memory, validate

log = logging.getLogger(__name__)

SYSTEM = """Summarise this chat segment as an episodic memory for an AI assistant. Reply JSON only.
Keep only what matters later: the user's situation, plans, decisions, progress and open threads. Leave out
greetings, small talk, jokes, weather, incidental happenings (e.g. a pet knocking something over) and
general-knowledge Q&A. State current values (e.g. "renamed the protagonist to Mara"), not superseded ones.
If nothing in the segment matters later, reply {"skip": true}.
{"title": "<=8 words", "gist": "<=50 words, past tense, concrete: what was discussed/decided/built",
 "decisions": ["..."], "open_threads": ["..."], "entities": ["<=6 key named things"],
 "ops": [<optional durable facts that only become clear across several turns, same op schema:
          {"op":"ADD","kind":"fact|decision|goal|preference|commitment|open_question","content":"...",
           "subj":"...","pred":"...","obj":"...","importance":0..1,"confidence":0..1,"speaker":"user"}>]}"""


async def consolidate_chat(tenant_id: UUID, chat_id: UUID) -> dict | None:
    now = datetime.now(timezone.utc)
    async with tenant_session(tenant_id) as conn:
        chat = (await conn.execute(
            text(
                "SELECT c.id, c.project_id, c.title, p.user_id, p.isolated FROM chats c "
                "JOIN projects p ON p.id = c.project_id WHERE c.id = :c"
            ),
            {"c": chat_id},
        )).mappings().first()
        if not chat:
            return None
        last_to = (await conn.execute(
            text("SELECT COALESCE(max(turn_to), 0) FROM episodes WHERE chat_id = :c"), {"c": chat_id}
        )).scalar_one()
        turns = (await conn.execute(
            text(
                "SELECT seq, user_msg, assistant_msg, salience, "
                "(SELECT count(*) FROM provenance pv WHERE pv.turn_id = turns.id) AS ops "
                "FROM turns WHERE chat_id = :c AND seq > :s "
                "ORDER BY seq LIMIT 200"
            ),
            {"c": chat_id, "s": last_to},
        )).mappings().all()
        mems = (await conn.execute(
            text(
                "SELECT kind, content, thread_status FROM memories WHERE source_chat_id = :c "
                "AND valid_until IS NULL AND kind <> 'episode' AND NOT is_belief ORDER BY importance DESC LIMIT 20"
            ),
            {"c": chat_id},
        )).mappings().all()
        # Like people, don't form an episode for a chat where nothing memorable happened.
        trivial = bool(turns) and not mems and all((t["salience"] or 0) < settings.salience_extract for t in turns)
        if not turns or trivial:
            await conn.execute(text("UPDATE chats SET consolidated_at = now() WHERE id = :c"), {"c": chat_id})
            return None

    title, gist, decisions, open_threads, entities, extra_ops = "", "", [], [], [], []
    if llm_enabled():
        # Encode what mattered: turns that formed memories (or were clearly salient). Small talk stays out of
        # the episode even when the summariser would have mentioned it.
        kept = [t for t in turns if t["ops"] or (t["salience"] or 0) >= 0.5] or list(turns)
        transcript = "\n".join(f"[{t['seq']}] user: {t['user_msg'][:1500]}\n[{t['seq']}] assistant: "
                               f"{t['assistant_msg'][:1500]}" for t in kept)[-24000:]
        data = await json_complete(SYSTEM, orjson.dumps({"chat_title": chat["title"], "transcript": transcript,
                                                         "already_stored": [m["content"] for m in mems]}).decode())
        if data and data.get("skip"):
            async with tenant_session(tenant_id) as conn:
                await conn.execute(text("UPDATE chats SET consolidated_at = now() WHERE id = :c"), {"c": chat_id})
            return None
        if data:
            title = str(data.get("title") or "")[:80]
            gist = str(data.get("gist") or "")[:600]
            decisions = [str(x)[:200] for x in data.get("decisions") or []][:6]
            open_threads = [str(x)[:200] for x in data.get("open_threads") or []][:6]
            entities = [str(x)[:80] for x in data.get("entities") or []][:6]
            extra_ops = [o for o in (Op.from_dict(d) for d in data.get("ops") or [] if isinstance(d, dict)) if o]
    if not gist:  # heuristic episode
        def _s(text_: str, n: int) -> str:
            return first_sentence(text_, n).rstrip(".!?;, ")

        title = _s(chat["title"] or turns[0]["user_msg"], 60)
        topics = [_s(t["user_msg"], 90) for t in turns[1:5]]
        decisions = [m["content"] for m in mems if m["kind"] in ("decision", "goal", "fact")][:5]
        open_threads = [m["content"] for m in mems if m["thread_status"] == "open"][:4]
        parts = []
        if topics:
            parts.append("Also discussed: " + "; ".join(topics) + ".")
        if decisions:
            parts.append("Outcomes: " + "; ".join(d.rstrip(".") for d in decisions[:3]) + ".")
        if open_threads:
            parts.append("Open: " + "; ".join(o.rstrip(".") for o in open_threads[:2]) + ".")
        gist = (" ".join(parts) or "Short exchange.")[:600]

    content = f"{title}: {gist}" if title else gist
    vec = (await embed_passages([embedding_text(content)]))[0]
    ctx = TurnCtx(tenant_id, chat["user_id"], chat["project_id"], chat_id, None, chat["isolated"],
                  "consolidation", now)
    report = ApplyReport()
    async with tenant_session(tenant_id) as conn:
        mid = await insert_memory(conn, ctx, Op(op="ADD", kind="episode", content=content, importance=0.5,
                                                confidence=0.9, entities=entities, evidence=title), vec, report)
        await conn.execute(
            text(
                "INSERT INTO episodes (id, tenant_id, project_id, chat_id, memory_id, title, gist, decisions, "
                "open_threads, turn_from, turn_to) VALUES (:id, :t, :p, :c, :m, :title, :gist, "
                "CAST(:d AS jsonb), CAST(:o AS jsonb), :f, :to)"
            ),
            {"id": uuid.uuid4(), "t": tenant_id, "p": chat["project_id"], "c": chat_id, "m": UUID(mid),
             "title": title or content[:80], "gist": gist, "d": orjson.dumps(decisions).decode(),
             "o": orjson.dumps(open_threads).decode(), "f": turns[0]["seq"], "to": turns[-1]["seq"]},
        )
        if extra_ops:
            await apply(conn, validate(extra_ops, set()), ctx)
        await conn.execute(
            text(
                "UPDATE chats SET consolidated_at = now(), "
                "status = CASE WHEN status = 'closed' THEN 'closed' ELSE 'idle' END WHERE id = :c"
            ),
            {"c": chat_id},
        )
    await redis.incr(Keys.memory_version(chat["project_id"]))
    await refresh_brief(tenant_id, chat["project_id"])
    return {"episode_memory_id": mid, "title": title, "turns": [turns[0]["seq"], turns[-1]["seq"]]}
