"""Memory operations: validate, resolve (dedupe / contradiction / reconsolidation) and apply.

ADD        new memory (auto-resolved to REINFORCE for duplicates, INVALIDATE+ADD for changed facts)
UPDATE     refine an existing memory in place (new version row) - reconsolidation
REINFORCE  restated / confirmed: stronger, more confident; a user restating an assistant belief confirms it
INVALIDATE the fact stopped being true: set valid_until, keep it for history
CLOSE      an open commitment / question got resolved
OUTCOME    action -> result: strengthen or weaken the memory that led to it, and record the lesson
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

import numpy as np
import orjson
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.config import settings
from app.embeddings import embed_passages
from app.memory.polarity import contradicts
from app.memory.strength import initial_stability
from app.runtime.intents import constraint_domains
from app.textutil import canonical_entity, detect_tech, embedding_text, normalize

log = logging.getLogger(__name__)

KINDS = {"fact", "preference", "decision", "goal", "commitment", "open_question", "outcome", "entity_relation",
         "episode"}
OPS = {"ADD", "UPDATE", "REINFORCE", "INVALIDATE", "CLOSE", "OUTCOME", "NOOP"}
THREAD_KINDS = {"commitment", "open_question"}
# Single-valued attributes: a different value means the old one stopped being true.
FUNCTIONAL = {
    "uses_database", "uses_framework", "uses_language", "uses_cloud", "hosted_on", "uses_cache", "uses_queue",
    "deadline", "lives_in", "works_at", "name", "job_title", "timezone", "primary_goal", "status",
    "response_style", "budget", "launch_date", "exam_date", "monthly_investment", "weekly_distance", "goal_time",
    "protagonist_name", "current_focus",
}
# Free-form predicates from the extractor -> one canonical name, so "moved_to Bangalore" supersedes
# "location Chennai". Unknown predicates pass through unchanged.
_PRED_SYNONYMS = {
    "lives_in": {"location", "lives", "lives_at", "moved_to", "relocated_to", "based_in", "city", "home_city",
                 "current_city", "resides_in", "home", "current_location", "residence"},
    "works_at": {"employer", "company", "works_for", "job_at", "workplace", "employed_at", "joined", "current_employer"},
    "job_title": {"role", "title", "position", "occupation", "job", "works_as", "profession", "joined_as"},
    "name": {"is_named", "called", "first_name", "full_name", "user_name"},
    "exam_date": {"exam", "exam_on", "test_date"},
    "monthly_investment": {"sip", "sip_amount", "monthly_sip", "invests_monthly", "monthly_sip_amount"},
    "weekly_distance": {"weekly_mileage", "mileage", "weekly_km", "runs_per_week"},
    "protagonist_name": {"protagonist", "main_character", "protagonist_is", "hero"},
    "current_focus": {"focus", "working_on", "current_weakness", "focus_on", "now_working_on"},
}
_PRED_CANON = {syn: canon for canon, syns in _PRED_SYNONYMS.items() for syn in syns}
IDENTITY_PREDS = {"lives_in", "works_at", "job_title", "name", "timezone"}
# Assistant-sourced memories may only be proposals/promises; "facts" must come from the user.
ASSISTANT_KINDS = {"decision", "commitment"}
# Never user-level: these belong to the project they were said in.
PROJECT_ONLY_KINDS = {"commitment", "open_question", "decision", "goal", "outcome", "episode"}


def synthesize_content(subj: str | None, pred: str, obj: str) -> str:
    """Readable statement from a triple, e.g. ("user", "monthly_budget", "60000 INR") ->
    "The user's monthly budget: 60000 INR." """
    who = {"user": "The user", "project": "The project", None: "The user", "": "The user"}.get(
        (subj or "").strip().lower(), (subj or "").strip())
    words = pred.replace("_", " ").strip()
    return f"{who}'s {words}: {obj}." if not who.endswith("s") else f"{who}' {words}: {obj}."


# 4) User-level memory is for things true about the person in every project (identity, health, diet, family,
#    pets) and preferences they say apply everywhere. Decided from the predicate, not loose content words
#    ("protagonist_name" is about a novel, not the user).
_USER_LEVEL_PRED = re.compile(
    r"^(?:name|lives_in|works_at|job_title|timezone)$|(^|_)(pronouns?|allerg\w*|intoleran\w*|diet\w*|vegan|vegetarian|"
    r"halal|kosher|health|injur\w*|medical|condition|wife|husband|partner|son|daughter|kids?|child(ren)?|sister|"
    r"brother|mother|mom|father|dad|sibling|family|pet|dog|cat|birthday)($|_)",
    re.I,
)
# "Apply everywhere" instructions ("always keep answers short", "in general use bullets") - not habits
# ("I always forget birthdays").
_GLOBAL_MARKER = re.compile(
    r"\b(in general|generally|every project|all projects|all my projects|everywhere|for everything|"
    r"always (?:keep|use|answer|respond|reply|give|write|format|be|include|explain|call me|address))\b",
    re.I,
)
# A change narrated as an event ("switched from X to Y") duplicates the new state it comes with.
_CHANGE_EVENT = re.compile(
    r"\b(switch\w*|chang\w*|mov\w*|migrat\w*|renam\w*|increas\w*|decreas\w*|rais\w*|reduc\w*|updat\w*)\b.*\bfrom\b.*\bto\b",
    re.I,
)
_NAME_CLAIM = re.compile(r"\b(my name is|i'?m|i am|call me|this is|name'?s)\s+", re.I)


def user_level_ok(o: "Op", source: str = "") -> bool:
    if o.kind == "preference":
        return bool(_GLOBAL_MARKER.search(f"{o.content} {o.evidence or ''} {source}"))
    return o.subj in (None, "user") and bool(o.pred and _USER_LEVEL_PRED.search(o.pred))


_ISO_DATE = re.compile(r"^\d{4}-\d{2}(-\d{2})?")


def grounded(obj: str | None, turn_text: str) -> bool:
    """Does the value actually appear in what was said? Numbers compare without separators ("20,000" ==
    "20000"); normalised dates are accepted (the extractor resolves "March 3rd" to 2027-03-03)."""
    if not obj or _ISO_DATE.match(obj.strip()):
        return True
    said = turn_text.lower().replace(",", "")
    toks = [t for t in re.findall(r"[a-z0-9.]+", obj.lower().replace(",", "")) if t.isdigit() or len(t) >= 3]
    return not toks or any(t in said or canonical_entity(t) in said for t in toks)


def canonical_pred(pred: str | None) -> str | None:
    if not pred:
        return pred
    p = pred.strip().lower().replace(" ", "_").replace("-", "_")
    return _PRED_CANON.get(p, p)
_SUBJ_ALIASES = {
    "we": "project", "us": "project", "our project": "project", "the project": "project", "this project": "project",
    "project": "project", "team": "project", "our team": "project",
    "i": "user", "me": "user", "my": "user", "myself": "user", "user": "user", "the user": "user",
}
MIN_CONFIDENCE = 0.3


@dataclass
class Op:
    op: str
    kind: str = "fact"
    content: str = ""
    subj: str | None = None
    pred: str | None = None
    obj: str | None = None
    memory_id: str | None = None
    importance: float = 0.5
    confidence: float = 0.8
    scope: str = "project"
    speaker: str = "user"
    evidence: str = ""
    due_at: datetime | None = None
    reason: str = ""
    result: str | None = None
    lesson: str = ""
    entities: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Op | None":
        def num(v, default: float) -> float:  # noqa: ANN001
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        def when(v) -> datetime | None:  # noqa: ANN001
            if not isinstance(v, str) or not v.strip():
                return None
            try:
                return datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
            except ValueError:
                return None  # a malformed date must not drop the whole memory

        try:
            subj = str(d["subj"]).strip() if d.get("subj") else None
            pred = str(d["pred"]).strip().lower().replace(" ", "_") if d.get("pred") else None
            obj = str(d["obj"]).strip() if d.get("obj") not in (None, "") else None
            content = str(d.get("content") or d.get("new_content") or "").strip()
            if not content and pred and obj:  # small models sometimes emit only the triple
                content = synthesize_content(subj, pred, obj)
            return cls(
                op=str(d.get("op", "")).upper(),
                kind=str(d.get("kind") or "fact").lower(),
                content=content,
                subj=subj,
                pred=pred,
                obj=obj,
                memory_id=(str(d.get("memory_id") or d.get("target_memory_id") or "") or None),
                importance=num(d.get("importance"), 0.5),
                confidence=num(d.get("confidence"), 0.8),
                scope=str(d.get("scope") or "project").lower(),
                speaker=str(d.get("speaker") or "user").lower(),
                evidence=str(d.get("evidence_span") or d.get("evidence") or "")[:500],
                due_at=when(d.get("due_at")),
                reason=str(d.get("reason") or "")[:300],
                result=(str(d["result"]).lower() if d.get("result") else None),
                lesson=str(d.get("lesson") or "")[:300],
                entities=[str(e)[:80] for e in (d.get("entities") or []) if e][:6],
            )
        except (TypeError, ValueError):
            return None


@dataclass
class TurnCtx:
    tenant_id: UUID
    user_id: UUID
    project_id: UUID
    chat_id: UUID | None
    turn_id: UUID | None
    isolated: bool
    method: str  # llm | heuristic | consolidation | explicit | sleep
    now: datetime


@dataclass
class ApplyReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    reinforced: list[str] = field(default_factory=list)
    invalidated: list[str] = field(default_factory=list)
    closed: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    important: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.reinforced or self.invalidated or self.closed or self.outcomes)


def norm_subj(subj: str | None) -> str | None:
    if not subj:
        return None
    s = normalize(subj)
    return _SUBJ_ALIASES.get(s, s)


def validate(ops: list[Op], allowed_ids: set[str], source: str = "") -> list[Op]:
    """Guards: known ops/kinds, ids must be ones we showed the extractor, assistant claims -> beliefs.

    ``source`` is the user's message (when known): scope and identity guards check what was actually said.
    """
    out: list[Op] = []
    for o in ops:
        if o.op not in OPS or o.op == "NOOP":
            continue
        if o.kind not in KINDS:
            o.kind = "fact"
        o.importance = max(0.0, min(1.0, o.importance))
        o.confidence = max(0.0, min(1.0, o.confidence))
        o.content = o.content[:400]
        o.subj = norm_subj(o.subj)
        o.pred = canonical_pred(o.pred)
        o.obj = o.obj[:120] if o.obj else None
        if o.scope not in ("project", "user"):
            o.scope = "project"
        if o.speaker not in ("user", "assistant"):
            o.speaker = "user"
        if o.op == "ADD" and o.speaker == "assistant" and o.kind not in ASSISTANT_KINDS:
            continue  # the assistant can propose, not assert facts about the user's world
        if o.op == "ADD" and o.pred == "name" and o.subj in (None, "user") and source and not (
                _NAME_CLAIM.search(source) and o.obj and o.obj.lower() in source.lower()):
            continue  # "Alice owns payments" must never become "the user is Alice"
        if o.scope == "user" and (o.kind in PROJECT_ONLY_KINDS or not user_level_ok(o, source)):
            o.scope = "project"
        elif o.scope == "project" and o.op == "ADD" and user_level_ok(o, source) and (
                o.kind == "preference" or (o.subj in (None, "user") and o.pred in IDENTITY_PREDS)):
            o.scope = "user"  # identity facts and "always / every project" preferences follow the user
        if o.op in ("UPDATE", "REINFORCE", "INVALIDATE", "CLOSE") and o.memory_id not in allowed_ids:
            if o.op == "UPDATE" and o.content:
                o.op, o.memory_id = "ADD", None      # unknown target: treat as new info (dedupe will catch it)
            else:
                continue
        if o.op == "OUTCOME" and o.memory_id and o.memory_id not in allowed_ids:
            o.memory_id = None
        if o.op in ("ADD", "UPDATE") and not o.content:
            continue
        if o.op == "ADD" and o.confidence < MIN_CONFIDENCE:
            continue
        if o.speaker == "assistant":
            o.confidence = min(o.confidence, 0.6)
        out.append(o)
    # Keep the new state, not the narrated change, when both are present in the same turn.
    adds = [o for o in out if o.op == "ADD"]
    if len(adds) > 1:
        out = [o for o in out if not (o.op == "ADD" and _CHANGE_EVENT.search(o.content))] or out
    return out


# --------------------------------------------------------------------------- apply

async def _provenance(conn: AsyncConnection, ctx: TurnCtx, mid: str, op: str, speaker: str, evidence: str) -> None:
    await conn.execute(
        text(
            "INSERT INTO provenance (tenant_id, memory_id, op, turn_id, chat_id, speaker, evidence, method) "
            "VALUES (:t, :m, :op, :turn, :chat, :sp, :ev, :me)"
        ),
        {"t": ctx.tenant_id, "m": UUID(mid), "op": op, "turn": ctx.turn_id, "chat": ctx.chat_id, "sp": speaker,
         "ev": evidence[:500], "me": ctx.method},
    )


async def _version(conn: AsyncConnection, ctx: TurnCtx, mid: str, op: str, reason: str = "") -> None:
    await conn.execute(
        text(
            "INSERT INTO memory_versions (tenant_id, memory_id, version, content, op, turn_id, reason) "
            "SELECT tenant_id, id, version, content, :op, :turn, :r FROM memories WHERE id = :m"
        ),
        {"m": UUID(mid), "op": op, "turn": ctx.turn_id, "r": reason[:300]},
    )


def _scope_filter(ctx: TurnCtx, scope: str) -> tuple[str, dict]:
    if scope == "user":
        return "user_id = :uid AND project_id IS NULL", {"uid": ctx.user_id}
    return "project_id = :pid", {"pid": ctx.project_id}


async def _reinforce(conn: AsyncConnection, ctx: TurnCtx, mid: str, speaker: str, evidence: str) -> None:
    await conn.execute(
        text(
            "UPDATE memories SET observed_count = observed_count + 1, "
            "confidence = LEAST(0.99, 1 - (1 - confidence) * 0.7), stability = LEAST(3650, stability * 1.2), "
            "is_belief = is_belief AND CAST(:asst AS boolean), "
            "source_reliability = CASE WHEN CAST(:asst AS boolean) THEN source_reliability ELSE 0.9 END, "
            "status = 'active', updated_at = now() WHERE id = :m"
        ),
        {"m": UUID(mid), "asst": speaker == "assistant"},
    )
    await _provenance(conn, ctx, mid, "REINFORCE", speaker, evidence)


async def _invalidate(conn: AsyncConnection, ctx: TurnCtx, mid: str, by: str | None, reason: str,
                      speaker: str, evidence: str) -> None:
    await conn.execute(
        text(
            "UPDATE memories SET valid_until = :now, superseded_by = :by, version = version + 1, updated_at = now() "
            "WHERE id = :m AND valid_until IS NULL"
        ),
        {"m": UUID(mid), "by": UUID(by) if by else None, "now": ctx.now},
    )
    await conn.execute(
        text("UPDATE edges SET valid_until = :now WHERE memory_id = :m AND valid_until IS NULL"),
        {"m": UUID(mid), "now": ctx.now},
    )
    await _version(conn, ctx, mid, "INVALIDATE", reason)
    await _provenance(conn, ctx, mid, "INVALIDATE", speaker, evidence)


async def _link_entities(conn: AsyncConnection, ctx: TurnCtx, mid: str, o: Op) -> list[str]:
    # Extractor-provided names first: they keep their original capitalisation for display.
    names: list[str] = list(o.entities)
    if o.obj and len(o.obj.split()) <= 4:
        names.append(o.obj)
    if o.subj and o.subj not in ("user", "project") and len(o.subj.split()) <= 4:
        names.append(o.subj)
    names += [n for _, n in detect_tech(o.content)]
    seen: dict[str, str] = {}
    for name in names:
        norm = canonical_entity(name)  # entity resolution: "PostgreSQL" and "postgres" are one entity
        if not norm or norm in seen or norm in ("user", "project") or len(norm) > 80:
            continue
        alias = normalize(name) if normalize(name) != norm else ""
        eid = (
            await conn.execute(
                text(
                    "INSERT INTO entities (id, tenant_id, project_id, name, norm_name, aliases) "
                    "VALUES (:id, :t, :p, :n, :nn, CASE WHEN :al = '' THEN CAST('{}' AS text[]) ELSE ARRAY[CAST(:al AS text)] END) "
                    "ON CONFLICT (project_id, norm_name) DO UPDATE SET mention_count = entities.mention_count + 1, "
                    "aliases = CASE WHEN :al = '' OR :al = ANY(entities.aliases) THEN entities.aliases "
                    "ELSE array_append(entities.aliases, CAST(:al AS text)) END "
                    "RETURNING id"
                ),
                {"id": uuid.uuid4(), "t": ctx.tenant_id, "p": ctx.project_id, "n": name.strip(), "nn": norm,
                 "al": alias},
            )
        ).scalar_one()
        seen[norm] = str(eid)
        await conn.execute(
            text(
                "INSERT INTO memory_entities (memory_id, entity_id, tenant_id) VALUES (:m, :e, :t) "
                "ON CONFLICT DO NOTHING"
            ),
            {"m": UUID(mid), "e": eid, "t": ctx.tenant_id},
        )
    ids = list(seen.values())
    subj_e = seen.get(canonical_entity(o.subj)) if o.subj else None
    obj_e = seen.get(canonical_entity(o.obj)) if o.obj else None
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            rel = o.pred if o.pred and {a, b} == {subj_e, obj_e} else "co_mentioned"
            await conn.execute(
                text(
                    "INSERT INTO edges (id, tenant_id, project_id, src_entity, dst_entity, relation, memory_id, weight) "
                    "VALUES (:id, :t, :p, :a, :b, :r, :m, 0.5) ON CONFLICT (project_id, src_entity, dst_entity, relation) "
                    "DO UPDATE SET weight = LEAST(1.0, edges.weight + 0.1), valid_until = NULL, memory_id = :m"
                ),
                {"id": uuid.uuid4(), "t": ctx.tenant_id, "p": ctx.project_id, "a": UUID(a), "b": UUID(b),
                 "r": rel, "m": UUID(mid)},
            )
    return ids


async def insert_memory(conn: AsyncConnection, ctx: TurnCtx, o: Op, vec: np.ndarray, report: ApplyReport,
                  extra_meta: dict | None = None) -> str:
    mid = str(uuid.uuid4())
    scope = "user" if (o.scope == "user" and not ctx.isolated) else "project"
    is_belief = o.speaker == "assistant"
    meta = dict(extra_meta or {})
    doms = constraint_domains(o.content, o.kind) if o.kind != "episode" else None
    if doms:
        meta["constraint_domains"] = doms  # standing constraint: recalled for any request in these domains
    await conn.execute(
        text(
            "INSERT INTO memories (id, tenant_id, user_id, project_id, scope, source_chat_id, source_turn_id, kind, "
            "is_belief, content, subj, pred, obj, embedding, importance, confidence, source_reliability, salience, "
            "stability, thread_status, due_at, valid_from, metadata, encoding_context) VALUES (:id, :t, :u, :p, :scope, "
            ":chat, :turn, :kind, :belief, :content, :subj, :pred, :obj, CAST(:emb AS vector), :imp, :conf, :rel, :sal, "
            ":stab, :thread, :due, :vf, CAST(:meta AS jsonb), CAST(:ctx AS jsonb))"
        ),
        {
            "id": UUID(mid), "t": ctx.tenant_id, "u": ctx.user_id,
            "p": None if scope == "user" else ctx.project_id, "scope": scope, "chat": ctx.chat_id,
            "turn": ctx.turn_id, "kind": o.kind, "belief": is_belief, "content": o.content, "subj": o.subj,
            "pred": o.pred, "obj": o.obj, "emb": vec, "imp": o.importance, "conf": o.confidence,
            "rel": 0.5 if is_belief else 0.9, "sal": o.importance, "stab": initial_stability(o.importance),
            "thread": "open" if o.kind in THREAD_KINDS else None, "due": o.due_at, "vf": ctx.now,
            "meta": orjson.dumps(meta).decode(),
            "ctx": orjson.dumps({"project_id": str(ctx.project_id), "chat_id": str(ctx.chat_id) if ctx.chat_id else None,
                                 "method": ctx.method}).decode(),
        },
    )
    await _version(conn, ctx, mid, "ADD")
    await _provenance(conn, ctx, mid, "ADD", o.speaker, o.evidence or o.content)
    report.entity_ids += await _link_entities(conn, ctx, mid, o)
    report.added.append(mid)
    if o.importance >= 0.7 or o.kind in ("decision", "goal"):
        report.important = True
    return mid


ATTITUDE_KINDS = ("preference", "fact", "goal")


async def _retire_opposites(conn: AsyncConnection, ctx: TurnCtx, new_id: str, o: Op, vec: np.ndarray,
                            report: ApplyReport) -> None:
    """Preference reversal: "hates chess" retires "enjoys chess" / "interested in chess" (newest attitude wins)."""
    if o.speaker != "user" or o.kind not in ATTITUDE_KINDS or ctx.method == "consolidation":
        return
    rows = (await conn.execute(
        text(
            "SELECT id, content FROM memories WHERE (project_id = :pid OR (user_id = :uid AND project_id IS NULL)) "
            "AND id <> :new AND valid_until IS NULL AND status <> 'archived' AND kind = ANY(CAST(:kinds AS text[])) "
            "AND 1 - (embedding <=> CAST(:v AS vector)) >= 0.45 ORDER BY embedding <=> CAST(:v AS vector) LIMIT 12"
        ),
        {"pid": ctx.project_id, "uid": ctx.user_id, "new": UUID(new_id), "kinds": list(ATTITUDE_KINDS), "v": vec},
    )).all()
    for r in rows:
        topic = contradicts(o.content, r.content)
        if topic:
            await _invalidate(conn, ctx, str(r.id), new_id, f"reversed attitude toward '{topic}'", o.speaker,
                              o.evidence or o.content)
            report.invalidated.append(str(r.id))
            report.important = True


async def _resolve_add(conn: AsyncConnection, ctx: TurnCtx, o: Op, vec: np.ndarray, report: ApplyReport) -> None:
    scope = "user" if (o.scope == "user" and not ctx.isolated) else "project"
    where, params = _scope_filter(ctx, scope)
    # 1) Functional predicate: same (subj, pred) -> same value reinforces, different value supersedes.
    #    Facts about the user are checked at BOTH levels: a project-level "lives in Bangalore" must retire a
    #    user-level "lives in Chennai" (and vice versa).
    if o.subj and o.pred and o.pred in FUNCTIONAL:
        fwhere, fparams = (
            ("(project_id = :pid OR (user_id = :uid AND project_id IS NULL))",
             {"pid": ctx.project_id, "uid": ctx.user_id})
            if o.subj == "user" and not ctx.isolated else (where, params)
        )
        rows = (
            await conn.execute(
                text(f"SELECT id, obj FROM memories WHERE {fwhere} AND subj = :s AND pred = :pr AND valid_until IS NULL"),
                {**fparams, "s": o.subj, "pr": o.pred},
            )
        ).all()
        same = [r for r in rows if r.obj and o.obj and canonical_entity(r.obj) == canonical_entity(o.obj)]
        if same:
            await _reinforce(conn, ctx, str(same[0].id), o.speaker, o.evidence or o.content)
            report.reinforced.append(str(same[0].id))
            return
        if rows and ctx.method == "consolidation":
            return  # re-reading old transcripts must never overwrite a current value
        if rows and o.speaker == "user":
            new_id = await insert_memory(conn, ctx, o, vec, report, {"supersedes": [str(r.id) for r in rows]})
            for r in rows:
                await _invalidate(conn, ctx, str(r.id), new_id, f"superseded: {o.pred} -> {o.obj}", o.speaker,
                                  o.evidence or o.content)
                report.invalidated.append(str(r.id))
            report.important = True
            return
    # 2) Near-duplicate of something still true -> reinforce instead of storing twice. Facts about the user
    #    are deduplicated across project + user level (the same fact must not live in both).
    dwhere, dparams = (
        ("(project_id = :pid OR (user_id = :uid AND project_id IS NULL))", {"pid": ctx.project_id, "uid": ctx.user_id})
        if o.subj in (None, "user") and not ctx.isolated else (where, params)
    )
    dup = (
        await conn.execute(
            text(
                f"SELECT id, kind, 1 - (embedding <=> CAST(:v AS vector)) AS sim FROM memories WHERE {dwhere} "
                "AND valid_until IS NULL AND status <> 'archived' ORDER BY embedding <=> CAST(:v AS vector) LIMIT 1"
            ),
            {**dparams, "v": vec},
        )
    ).first()
    same_family = dup is not None and (
        dup.kind == o.kind or {dup.kind, o.kind} <= {"fact", "preference", "entity_relation"} or o.kind == "fact")
    if dup and dup.sim >= settings.dedupe_write_sim and same_family:
        await _reinforce(conn, ctx, str(dup.id), o.speaker, o.evidence or o.content)
        report.reinforced.append(str(dup.id))
        return
    new_id = await insert_memory(conn, ctx, o, vec, report)
    await _retire_opposites(conn, ctx, new_id, o, vec, report)


async def apply(conn: AsyncConnection, ops: list[Op], ctx: TurnCtx) -> ApplyReport:
    report = ApplyReport()
    texts = [o.content for o in ops if o.op in ("ADD", "UPDATE") and o.content]
    texts += [o.lesson for o in ops if o.op == "OUTCOME" and o.lesson]
    # embedded in first person (how users ask), displayed as written
    vecs = dict(zip(texts, await embed_passages([embedding_text(t) for t in texts]))) if texts else {}

    for o in ops:
        if o.op == "ADD":
            await _resolve_add(conn, ctx, o, vecs[o.content], report)
        elif o.op == "UPDATE":
            mid = UUID(o.memory_id)
            await conn.execute(
                text(
                    "UPDATE memories SET content = :c, embedding = CAST(:v AS vector), version = version + 1, "
                    "subj = COALESCE(:s, subj), pred = COALESCE(:pr, pred), obj = COALESCE(:o, obj), "
                    "is_belief = is_belief AND CAST(:asst AS boolean), updated_at = now() WHERE id = :m"
                ),
                {"c": o.content, "v": vecs[o.content], "s": o.subj, "pr": o.pred, "o": o.obj, "m": mid,
                 "asst": o.speaker == "assistant"},
            )
            await _version(conn, ctx, o.memory_id, "UPDATE", o.reason)
            await _provenance(conn, ctx, o.memory_id, "UPDATE", o.speaker, o.evidence or o.content)
            report.entity_ids += await _link_entities(conn, ctx, o.memory_id, o)
            report.updated.append(o.memory_id)
        elif o.op == "REINFORCE":
            await _reinforce(conn, ctx, o.memory_id, o.speaker, o.evidence)
            report.reinforced.append(o.memory_id)
        elif o.op == "INVALIDATE":
            await _invalidate(conn, ctx, o.memory_id, None, o.reason, o.speaker, o.evidence)
            report.invalidated.append(o.memory_id)
            report.important = True
        elif o.op == "CLOSE":
            await conn.execute(
                text(
                    "UPDATE memories SET thread_status = 'closed', updated_at = now(), "
                    "metadata = metadata || CAST(:meta AS jsonb) WHERE id = :m"
                ),
                {"m": UUID(o.memory_id), "meta": orjson.dumps({"outcome": o.result or o.reason or "resolved",
                                                               "closed_at": ctx.now.isoformat()}).decode()},
            )
            await _provenance(conn, ctx, o.memory_id, "CLOSE", o.speaker, o.evidence)
            report.closed.append(o.memory_id)
        elif o.op == "OUTCOME":
            success = o.result in ("success", "worked", "positive")
            if o.memory_id:
                await conn.execute(
                    text(
                        "UPDATE memories SET "
                        "confidence = CASE WHEN CAST(:ok AS boolean) THEN LEAST(0.99, confidence + 0.15) ELSE confidence * 0.6 END, "
                        "stability = CASE WHEN CAST(:ok AS boolean) THEN LEAST(3650, stability * 1.5) ELSE stability END, "
                        "is_belief = CASE WHEN CAST(:ok AS boolean) THEN false ELSE is_belief END, "
                        "metadata = metadata || CAST(:meta AS jsonb), updated_at = now() WHERE id = :m"
                    ),
                    {"ok": success, "m": UUID(o.memory_id),
                     "meta": orjson.dumps({"last_outcome": "success" if success else "failure"}).decode()},
                )
                await _provenance(conn, ctx, o.memory_id, "OUTCOME", o.speaker, o.evidence)
            if o.lesson:
                lesson = Op(op="ADD", kind="outcome", content=o.lesson, importance=0.6 if success else 0.7,
                            confidence=0.8, speaker="user", evidence=o.evidence, entities=o.entities)
                mid = await insert_memory(conn, ctx, lesson, vecs[o.lesson], report,
                                    {"result": "success" if success else "failure", "about": o.memory_id})
                report.outcomes.append(mid)
    # Explicit INVALIDATE + ADD for the same (subj, pred): link old -> new so history reads as a chain.
    if report.invalidated and report.added:
        await conn.execute(
            text(
                "UPDATE memories o SET superseded_by = n.id FROM memories n "
                "WHERE o.id = ANY(CAST(:old AS uuid[])) AND n.id = ANY(CAST(:new AS uuid[])) "
                "AND o.superseded_by IS NULL AND o.pred IS NOT NULL AND o.pred = n.pred "
                "AND COALESCE(o.subj, '') = COALESCE(n.subj, '')"
            ),
            {"old": [UUID(x) for x in report.invalidated], "new": [UUID(x) for x in report.added]},
        )
    return report
