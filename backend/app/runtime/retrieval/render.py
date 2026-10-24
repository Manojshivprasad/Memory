"""Render memories as compact, self-describing lines (facts, not raw chunks)."""

from datetime import datetime

CONTEXT_HEADER = "Relevant memory (use only if helpful; newer facts override older ones):"
BRIEF_HEADER = "Project brief:"


def _d(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d") if dt else ""


def render_line(s, now: datetime) -> str:  # noqa: ANN001 - Scored (avoid import cycle)
    c = s.cand
    tags: list[str] = []
    if c.kind == "episode":
        return f"- [past chat {_d(c.created_at)}] {c.content}"
    if c.thread_status == "open":
        tag = f"open {c.kind}"
        if c.due_at:
            tag += f", due {_d(c.due_at)}"
        tags.append(tag)
    else:
        tags.append(("user " if c.scope == "user" else "") + c.kind)
    if c.is_belief:
        tags.append(f"unconfirmed ~{int(round(c.confidence * 100))}%")
    suffix = f" ({_d(c.valid_from)}"
    if c.valid_until and c.valid_until <= now:
        suffix += f" → no longer true since {_d(c.valid_until)}"
    suffix += ")"
    return f"- [{', '.join(tags)}] {c.content}{suffix}"


def render_context(lines: list[str]) -> str:
    return CONTEXT_HEADER + "\n" + "\n".join(lines) if lines else ""
