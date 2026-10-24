"""Optional memory agent for hard queries.

It gets the deterministic candidates up front, so the common case is ONE LLM call (a smart
filter / re-ranker). It may call up to ``agent_max_steps`` tools to dig further (timeline,
entity graph, verbatim turns). It can only select real memory ids it has seen, never invent
facts; its optional note is short and must cite ids. Hard timeout, and the caller falls back to
the deterministic selection on any failure.
"""

import asyncio
import logging
import re
from dataclasses import dataclass

import orjson

from app.config import settings
from app.llm import acompletion, llm_enabled, parse_json
from app.runtime.agent.tools import TOOL_SPECS, ToolContext
from app.runtime.retrieval.scoring import Scored
from app.runtime.temporal import TemporalCue

log = logging.getLogger(__name__)

SYSTEM = """You are the memory retrieval agent of an AI assistant. Decide which stored memories the
assistant needs to answer the user's next message well. Be STRICT: every selected memory costs
tokens. Select only memories that are clearly needed; selecting none is fine.
Prefer memories that are still true; include an outdated one only if the user asks about the past.
Use tools only if the candidates are insufficient. Today is {today}.
Finish by replying with JSON only:
{{"selected_ids": ["<memory id>", ...], "note": "<optional, <=40 words, only facts from cited memories or verbatim turns>"}}"""

_REFERENCE = re.compile(
    r"\b(that|this|those|these|the) (approach|idea|issue|bug|thing|one|plan|option|decision|problem|error|fix|"
    r"solution|design|suggestion)\b|\b(it|they) (again|still)\b",
    re.I,
)
_RELATIONAL = re.compile(r"\b(between|compare|compared|relationship|related|versus|vs\.?|difference)\b", re.I)


@dataclass
class AgentResult:
    ids: list[str]
    note: str
    steps: int


def needs_agent(query: str, scored: list[Scored], cue: TemporalCue, q_entities: list[str]) -> str | None:
    """Metamemory: return a reason when deterministic recall is likely insufficient."""
    relevant = [s for s in scored if s.relevant and s.score >= settings.min_score]
    if cue.has_window or cue.last_chat or cue.historical:
        return "temporal"
    if _REFERENCE.search(query) and not q_entities:
        return "unresolved-reference"
    if len(q_entities) >= 2 and _RELATIONAL.search(query):
        return "multi-hop"
    near_miss = [s for s in scored if not s.relevant and s.features["cos"] >= settings.semantic_floor - 0.08]
    if not relevant and near_miss:
        return "weak-recall"
    if relevant and relevant[0].score < settings.agent_hard_score:
        return "low-confidence"
    if len(relevant) >= 3 and relevant[0].score - relevant[2].score < 0.02:
        return "ambiguous"
    return None


def _candidate_payload(scored: list[Scored], limit: int = 15) -> list[dict]:
    return [
        {"id": s.id, "kind": s.cand.kind, "content": s.cand.content,
         "date": s.cand.created_at.strftime("%Y-%m-%d"), "still_true": s.cand.valid_until is None,
         "belief": s.cand.is_belief, "score": s.score}
        for s in scored[:limit]
    ]


async def _loop(query: str, recent: str, scored: list[Scored], tools: ToolContext) -> AgentResult:
    for s in scored[:15]:
        tools.seen[s.id] = s
    today = tools.score_ctx.now.strftime("%Y-%m-%d")
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM.format(today=today)},
        {"role": "user", "content": orjson.dumps({
            "next_user_message": query,
            "recent_conversation": recent,
            "candidates": _candidate_payload(scored),
        }).decode()},
    ]
    for step in range(settings.agent_max_steps + 1):
        final_step = step == settings.agent_max_steps
        try:
            resp = await acompletion(
                messages,
                model=settings.agent_llm_model or settings.llm_model,
                tools=None if final_step else TOOL_SPECS,
                max_tokens=400,
                temperature=0,
            )
        except Exception as e:  # noqa: BLE001 - e.g. provider rejected a malformed tool call
            if final_step:
                raise
            log.info("agent tool step failed (%s); answering from candidates", str(e)[:120])
            resp = await acompletion(messages, model=settings.agent_llm_model or settings.llm_model,
                                     max_tokens=400, temperature=0, response_format={"type": "json_object"})
            final_step = True
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        if calls and not final_step:
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name,
                                                              "arguments": c.function.arguments}}
                for c in calls
            ]})
            results = await asyncio.gather(*[
                tools.call(c.function.name, parse_json(c.function.arguments or "{}") or {}) for c in calls
            ])
            for c, r in zip(calls, results):
                messages.append({"role": "tool", "tool_call_id": c.id,
                                 "content": orjson.dumps(r, default=str).decode()[:6000]})
            continue
        data = parse_json(msg.content or "") or {}
        ids = [i for i in data.get("selected_ids", []) if isinstance(i, str) and i in tools.seen]
        note = str(data.get("note") or "")[:300]
        return AgentResult(ids=list(dict.fromkeys(ids)), note=note, steps=step + 1)
    return AgentResult(ids=[], note="", steps=settings.agent_max_steps)


async def run(query: str, recent: str, scored: list[Scored], tools: ToolContext, timeout_ms: int) -> AgentResult | None:
    if not llm_enabled():
        return None
    # Hard budget: abandon the task on timeout instead of awaiting its cancellation (provider calls can take
    # seconds to unwind, which would leak into the hot path).
    task = asyncio.create_task(_loop(query, recent, scored, tools))
    done, _ = await asyncio.wait({task}, timeout=timeout_ms / 1000)
    if not done:
        task.cancel()
        task.add_done_callback(lambda t: t.cancelled() or t.exception())  # silence "never retrieved"
        log.info("memory agent exceeded %sms; using deterministic selection", timeout_ms)
        return None
    try:
        return task.result()
    except Exception:  # noqa: BLE001
        log.exception("memory agent failed; using deterministic selection")
        return None
