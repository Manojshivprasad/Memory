"""Rebuild the project brief (~150 tokens) from the strongest memories that are still true."""

from uuid import UUID

import orjson
from sqlalchemy import text

from app.db.session import tenant_session
from app.llm import json_complete, llm_enabled, parse_json
from app.runtime.brief import set_cached_brief
from app.tokens import count_tokens

MAX_TOKENS = 150

SYSTEM = """Write a compact project brief for an AI assistant (max 110 words) using ONLY the given
memories, which are all currently true. State current values only; never mention earlier values or
changes. Plain text lines, omit empty sections: "Goal: ...", "Stack/decisions: ...", "State: ...",
"Open: ...", "Preferences: ...". Skip unconfirmed items unless essential.
Reply as JSON with ONE string (lines separated by \\n): {"brief": "Goal: ...\\nState: ..."}"""


def _trim(lines: list[str]) -> str:
    out: list[str] = []
    for line in lines:
        if count_tokens("\n".join(out + [line])) > MAX_TOKENS:
            break
        out.append(line)
    return "\n".join(out)


def _join(items: list[str], n: int) -> str:
    return "; ".join(i.rstrip(".") for i in items[:n])


async def refresh_brief(tenant_id: UUID, project_id: UUID) -> str:
    async with tenant_session(tenant_id) as conn:
        proj = (await conn.execute(text("SELECT name, description FROM projects WHERE id = :p"),
                                   {"p": project_id})).mappings().first()
        if not proj:
            return ""
        mems = (await conn.execute(
            text(
                "SELECT kind, content, is_belief, thread_status, importance, created_at FROM memories "
                "WHERE project_id = :p AND valid_until IS NULL AND status <> 'archived' AND kind <> 'episode' "
                "AND (thread_status IS NULL OR thread_status = 'open') "
                "ORDER BY importance DESC, observed_count DESC, created_at DESC LIMIT 40"
            ),
            {"p": project_id},
        )).mappings().all()

    by_kind: dict[str, list[str]] = {}
    for m in mems:
        if m["is_belief"]:
            continue
        key = "open" if m["thread_status"] == "open" else m["kind"]
        by_kind.setdefault(key, []).append(m["content"])

    brief = ""
    if llm_enabled() and mems:
        data = await json_complete(SYSTEM, orjson.dumps({
            "project": {"name": proj["name"], "description": proj["description"]},
            "memories": [{"kind": m["kind"], "content": m["content"], "unconfirmed": m["is_belief"],
                          "open": m["thread_status"] == "open", "date": m["created_at"].strftime("%Y-%m-%d")}
                         for m in mems],
        }).decode(), max_tokens=1500)
        raw = data.get("brief") if data else None
        if isinstance(raw, str) and raw.strip().startswith("{"):
            raw = parse_json(raw) or raw  # some models double-encode the sections
        if isinstance(raw, dict):  # {"Goal:": "...", "State:": "..."} -> lines
            raw = "\n".join(f"{k.rstrip(':')}: {v}" for k, v in raw.items() if isinstance(v, str) and v.strip())
        if isinstance(raw, str) and raw.strip():
            brief = _trim([ln for ln in raw.strip().splitlines() if ln.strip()])
    if not brief:
        lines = [f"Project: {proj['name']}" + (f" - {proj['description']}" if proj["description"] else "")]
        if by_kind.get("goal"):
            lines.append("Goal: " + _join(by_kind["goal"], 2))
        if by_kind.get("decision"):
            lines.append("Decisions: " + _join(by_kind["decision"], 5))
        if by_kind.get("fact"):
            lines.append("Facts: " + _join(by_kind["fact"], 4))
        if by_kind.get("open"):
            lines.append("Open: " + _join(by_kind["open"], 3))
        if by_kind.get("preference"):
            lines.append("Preferences: " + _join(by_kind["preference"], 3))
        if by_kind.get("episode"):
            lines.append("Last session: " + by_kind["episode"][0][:160])
        brief = _trim(lines) if len(lines) > 1 else ""

    async with tenant_session(tenant_id) as conn:
        await conn.execute(
            text("UPDATE projects SET brief = :b, brief_updated_at = now() WHERE id = :p"),
            {"b": brief, "p": project_id},
        )
    await set_cached_brief(project_id, brief)
    return brief
