"""Stage B: per-turn memory formation (runs in the worker, seconds after the response).

persist verbatim turn (idempotent) -> signals -> salience (novelty vs what we already know)
-> extraction (LLM ops, heuristic fallback) + adoption/feedback ops -> validate -> apply in one
transaction (dedupe / contradiction / reconsolidation / provenance / versions)
-> spaced reinforcement of the memories that were retrieved -> invalidate caches.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

import numpy as np
from sqlalchemy import text

from app.cache import Keys, redis
from app.config import settings
from app.db.session import tenant_session
from app.embeddings import as_array, embed_passages
from app.memory.salience import salience
from app.memory.strength import reinforce, retrievability
from app.runtime import working_memory
from app.runtime.brief import get_brief, mark_dirty
from app.textutil import content_tokens, detect_tech, embedding_text
from app.write import extractor
from app.write.ops import FUNCTIONAL, TurnCtx, apply, canonical_pred, grounded, validate
from app.write.signals import detect, informative

log = logging.getLogger(__name__)

_MEM_COLS = (
    "id, kind, content, subj, pred, obj, scope, is_belief, thread_status, embedding, created_at, last_recalled_at, "
    "stability"
)
NEIGHBOURS_K = 8
NEIGHBOURS_MIN_COS = 0.55  # bge-small: unrelated pairs sit at ~0.45-0.5


async def _neighbours(job: "TurnJob", vec) -> list[dict]:  # noqa: ANN001
    """Active memories most related to this turn (project + user level). The extractor needs these to
    retire or refine what the turn changes; the strictly selected context alone is too narrow."""
    async with tenant_session(job.tenant_id) as conn:
        rows = (await conn.execute(
            text(
                f"SELECT {_MEM_COLS}, 1 - (embedding <=> CAST(:v AS vector)) AS sim FROM memories "
                "WHERE tenant_id = :t AND (project_id = :p OR (NOT CAST(:iso AS boolean) AND scope = 'user' "
                "AND user_id = :u AND project_id IS NULL)) AND valid_until IS NULL AND status <> 'archived' "
                "AND kind <> 'episode' ORDER BY embedding <=> CAST(:v AS vector) LIMIT :k"
            ),
            {"v": vec, "t": job.tenant_id, "p": job.project_id, "u": job.user_id, "iso": job.isolated,
             "k": NEIGHBOURS_K},
        )).mappings().all()
    return [_row({k: v for k, v in r.items() if k != "sim"}) for r in rows if r["sim"] >= NEIGHBOURS_MIN_COS]


@dataclass
class TurnJob:
    turn_id: UUID
    tenant_id: UUID
    user_id: UUID
    project_id: UUID
    chat_id: UUID
    isolated: bool
    seq: int
    user_msg: str
    assistant_msg: str
    retrieved_ids: list[str]
    model: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "TurnJob":
        return cls(
            turn_id=UUID(d["turn_id"]), tenant_id=UUID(d["tenant_id"]), user_id=UUID(d["user_id"]),
            project_id=UUID(d["project_id"]), chat_id=UUID(d["chat_id"]), isolated=bool(d.get("isolated")),
            seq=int(d["seq"]), user_msg=d["user_msg"], assistant_msg=d["assistant_msg"],
            retrieved_ids=[str(x) for x in d.get("retrieved_ids") or []], model=d.get("model"),
        )


def _row(r) -> dict:  # noqa: ANN001
    d = dict(r)
    d["id"] = str(d["id"])
    return d


def was_used(content: str, answer: str) -> bool:
    """Did the answer actually draw on this memory? (cheap lexical overlap)"""
    mem = set(content_tokens(content))
    if not mem:
        return False
    ans = set(content_tokens(answer))
    return len(mem & ans) / len(mem) >= 0.5


async def process_turn(job: TurnJob) -> dict:
    now = datetime.now(timezone.utc)
    # 1) persist verbatim turn (idempotent on turn_id)
    async with tenant_session(job.tenant_id) as conn:
        inserted = (await conn.execute(
            text(
                "INSERT INTO turns (id, tenant_id, project_id, chat_id, seq, user_msg, assistant_msg, model, "
                "retrieved_ids) VALUES (:id, :t, :p, :c, :seq, :u, :a, :m, CAST(:r AS uuid[])) "
                "ON CONFLICT (id) DO NOTHING RETURNING id"
            ),
            {"id": job.turn_id, "t": job.tenant_id, "p": job.project_id, "c": job.chat_id, "seq": job.seq,
             "u": job.user_msg, "a": job.assistant_msg, "m": job.model,
             "r": [UUID(x) for x in job.retrieved_ids]},
        )).first()
        if inserted is None:
            done = (await conn.execute(text("SELECT processed_at FROM turns WHERE id = :id"),
                                       {"id": job.turn_id})).scalar_one_or_none()
            if done is not None:
                return {"skipped": "already-processed"}
        else:
            await conn.execute(
                text(
                    "UPDATE chats SET turn_count = turn_count + 1, last_active_at = now(), status = 'open', "
                    "title = CASE WHEN title = '' THEN :title ELSE title END WHERE id = :c"
                ),
                {"c": job.chat_id, "title": job.user_msg.strip().split("\n")[0][:80]},
            )
        retrieved = [_row(r) for r in (await conn.execute(
            text(f"SELECT {_MEM_COLS} FROM memories WHERE id = ANY(CAST(:ids AS uuid[]))"),
            {"ids": [UUID(x) for x in job.retrieved_ids]},
        )).mappings().all()] if job.retrieved_ids else []
        # assistant-sourced memories from the previous turn (targets for adoption / outcome feedback)
        prev = [_row(r) for r in (await conn.execute(
            text(
                f"SELECT {_MEM_COLS} FROM memories WHERE source_turn_id = "
                "(SELECT id FROM turns WHERE chat_id = :c AND seq < :seq ORDER BY seq DESC LIMIT 1) "
                "AND (is_belief OR kind = 'decision') ORDER BY created_at"
            ),
            {"c": job.chat_id, "seq": job.seq},
        )).mappings().all()]

    # 2) signals + 3) salience (novelty = distance from what was recalled for this turn)
    signals = detect(job.user_msg, job.assistant_msg)
    uvec = (await embed_passages([job.user_msg[:2000]]))[0]
    related = await _neighbours(job, uvec)
    seen_ids = {m["id"] for m in retrieved}
    related = [m for m in related if m["id"] not in seen_ids]
    known = [as_array(m["embedding"]) for m in retrieved + related]
    novelty = 1.0 - max((float(uvec @ v) for v in known), default=0.0)
    sal = salience(signals, novelty)

    # 4) extraction
    method = "heuristic"
    ops = []
    if sal >= settings.salience_extract or informative(job.user_msg):
        wm = await working_memory.load(job.chat_id)
        brief = await get_brief(job.tenant_id, job.project_id)
        existing = retrieved + related
        existing += [m for m in prev if m["id"] not in {e["id"] for e in existing}]
        llm_ops = await extractor.extract_llm(job.user_msg, job.assistant_msg, existing, brief,
                                              wm.recent_text(3, 1500), now)
        if llm_ops is not None:
            ops, method = llm_ops, "llm"
            # Deterministic backstop: a clear assistant recommendation is always kept as an (unconfirmed)
            # belief, so a later "let's do that" has something to confirm. Dedupe absorbs overlaps.
            if "recommendation" in signals.assistant and not any(o.speaker == "assistant" for o in ops):
                ops += [o for o in extractor.extract_heuristic("", job.assistant_msg, signals, now)
                        if o.speaker == "assistant"]
            # Same for stated stack decisions ("we use X"): never lose them to LLM variance.
            if signals.tech:
                # only fill gaps: a stack slot the LLM already handled (e.g. a switch) is left alone
                covered = {o.pred for o in ops if o.pred} | {
                    f"uses_{cat}" for o in ops for cat, _ in detect_tech(o.content or "")}
                ops += [o for o in extractor.extract_heuristic(job.user_msg, "", signals, now)
                        if o.pred and o.pred.startswith("uses_") and o.pred not in covered]
        else:
            ops = extractor.extract_heuristic(job.user_msg, job.assistant_msg, signals, now)
    if method == "heuristic" or not any(o.op == "OUTCOME" for o in ops):
        ops += extractor.outcome_ops(signals, prev, job.user_msg)
    allowed = {m["id"] for m in retrieved} | {m["id"] for m in prev} | {m["id"] for m in related}
    ops = validate(ops, allowed, source=job.user_msg)
    # Grounding: a new value for a single-valued attribute must actually appear in this turn. Stops the model
    # from "restoring" a stale value it saw elsewhere (which would supersede the current one).
    turn_text = f"{job.user_msg} {job.assistant_msg}"
    ops = [o for o in ops if not (o.op in ("ADD", "UPDATE") and o.pred in FUNCTIONAL and not grounded(o.obj, turn_text))]
    # Retiring a memory needs evidence of change: the user signalled one ("moved", "renamed", "actually"...)
    # or the same turn states a new value for that attribute. Stops e.g. a new peanut allergy from
    # "replacing" a lactose intolerance.
    changed = bool({"correction", "feedback_neg"} & signals.user)
    new_ops = [o for o in ops if o.op in ("ADD", "UPDATE") and o.content]
    new_preds = {o.pred for o in new_ops if o.pred}
    known = {m["id"]: m for m in retrieved + related + prev}
    new_vecs = await embed_passages([embedding_text(o.content) for o in new_ops]) if new_ops else None

    def may_retire(o) -> bool:  # noqa: ANN001
        m = known.get(o.memory_id)
        if m is None:
            return False
        if canonical_pred(m.get("pred")) in new_preds:
            return True  # same attribute, new value
        if not changed:
            return False
        if new_vecs is None:
            return True  # pure retraction ("I no longer ...")
        # a change was signalled: only retire what the new statement is actually about
        return float((new_vecs @ as_array(m["embedding"])).max()) >= 0.75

    ops = [o for o in ops if o.op != "INVALIDATE" or may_retire(o)]

    # 5) apply + 6) spaced reinforcement of retrieved memories, one transaction
    ctx = TurnCtx(job.tenant_id, job.user_id, job.project_id, job.chat_id, job.turn_id, job.isolated, method, now)
    async with tenant_session(job.tenant_id) as conn:
        report = await apply(conn, ops, ctx)
        touched = set(report.reinforced) | set(report.updated) | set(report.invalidated)
        for m in retrieved:
            if m["id"] in touched:
                continue
            if was_used(m["content"], job.assistant_msg):
                r = retrievability(now, m["created_at"], m["last_recalled_at"], m["stability"])
                await conn.execute(
                    text(
                        "UPDATE memories SET recall_count = recall_count + 1, last_recalled_at = :now, "
                        "stability = :s, utility = utility + 1, status = 'active' WHERE id = :id"
                    ),
                    {"now": now, "s": reinforce(m["stability"], r), "id": UUID(m["id"])},
                )
            else:
                await conn.execute(
                    text("UPDATE memories SET utility = GREATEST(utility - 0.2, -5) WHERE id = :id"),
                    {"id": UUID(m["id"])},
                )
        await conn.execute(
            text("UPDATE turns SET processed_at = now(), salience = :s WHERE id = :id"),
            {"s": sal, "id": job.turn_id},
        )
        turn_count = (await conn.execute(text("SELECT turn_count FROM chats WHERE id = :c"),
                                         {"c": job.chat_id})).scalar_one_or_none() or 0

    # 7) caches: new memory version (invalidates prefetch pools + graph cache), focus, brief, consolidation
    pipe = redis.pipeline(transaction=False)
    if report.changed or retrieved:
        pipe.incr(Keys.memory_version(job.project_id))
    if report.entity_ids:
        working_memory.touch_entities_cmds(pipe, job.chat_id, report.entity_ids)
    if turn_count and turn_count % settings.consolidate_every_turns == 0:
        pipe.sadd(Keys.CONSOLIDATE_PENDING, f"{job.tenant_id}:{job.chat_id}")
    pipe.set(Keys.chat_applied(job.chat_id), job.seq)
    await pipe.execute()
    if report.important:
        await mark_dirty(job.tenant_id, job.project_id)
    return {
        "salience": sal, "method": method, "ops": len(ops), "added": report.added, "reinforced": report.reinforced,
        "invalidated": report.invalidated, "updated": report.updated, "closed": report.closed,
        "outcomes": report.outcomes,
    }
