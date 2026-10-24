"""Read-path orchestrator: prompt -> strict, budgeted memory context (optionally agent-assisted).

Hot path budget (runtime overhead): warm ~5-20 ms, cold ~30-60 ms.
    gate -> [WM, brief, memory-version, prefetch pool, query embedding]  (concurrent)
         -> entity match + spreading activation (in-process graph cache)
         -> candidates (prefetch pool rescore, or ONE SQL round trip)
         -> score -> strict select (floor, MMR, token budget)
         -> [optional] memory agent (hard timeout, falls back)
         -> render; trace written in the background
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

import numpy as np
import orjson

from app.auth.deps import ProjectMeta
from app.cache import Keys, redis
from app.config import settings
from app.embeddings import embed_query
from app.llm import llm_enabled
from app.observability import trace
from app.observability.trace import Timer
from app.runtime import prefetch_cache, working_memory
from app.runtime.agent import memory_agent
from app.runtime.agent.tools import ToolContext
from app.runtime.brief import get_brief
from app.runtime.gate import is_personal, should_retrieve
from app.runtime.intents import query_attributes, query_domains
from app.runtime.graph import ProjectGraph, get_graph, memory_version
from app.runtime.retrieval.candidates import CandidateQuery, fetch_candidates
from app.runtime.retrieval.render import render_context, render_line
from app.runtime.retrieval.scoring import ScoreContext, Scored, score_candidates
from app.runtime.retrieval.selector import Selection, select
from app.runtime.temporal import TemporalCue, parse as parse_temporal
from app.textutil import content_tokens, normalize, tsquery_or
from app.tokens import count_tokens

AgentMode = Literal["off", "auto", "always"]
BriefMode = Literal["auto", "always", "never"]
WM_PRIMING = 0.4  # activation seeded by entities already in focus in this chat


@dataclass
class RetrieveRequest:
    tenant_id: UUID
    user_id: UUID
    project: ProjectMeta
    chat_id: UUID | None
    query: str
    budget_tokens: int = settings.budget_tokens
    agent: AgentMode = "auto"
    brief: BriefMode = "auto"


@dataclass
class RetrieveResult:
    context: str = ""
    brief: str = ""
    memories: list[dict] = field(default_factory=list)
    selected_ids: list[str] = field(default_factory=list)
    query_entities: list[str] = field(default_factory=list)
    trace_id: str | None = None
    gate: str = "retrieve"
    cache_hit: bool = False
    agent_used: bool = False
    agent_reason: str | None = None
    tokens: int = 0
    latency_ms: dict = field(default_factory=dict)
    server_timing: str = ""


@dataclass
class _Prep:
    now: datetime
    wm: working_memory.WorkingMemory
    mver: str
    brief: str
    pool: prefetch_cache.Pool | None
    qvec: np.ndarray
    graph: ProjectGraph
    q_entities: list[str]
    activation: dict[str, float]
    cue: TemporalCue

    query: str = ""

    @property
    def cue_key(self) -> str:
        # anything that changes WHICH candidates are fetched must invalidate the prefetched pool
        c = self.cue
        return (f"{c.start.date() if c.start else ''}|{c.end.date() if c.end else ''}|{c.historical}|{c.last_chat}"
                f"|{','.join(query_attributes(self.query))}|{','.join(query_domains(self.query))}")


async def _prepare(req: RetrieveRequest) -> _Prep:
    pid = req.project.id

    async def _none():  # noqa: ANN202
        return None

    wm_t = working_memory.load(req.chat_id) if req.chat_id else _none()
    pool_t = prefetch_cache.load(req.chat_id) if req.chat_id else _none()
    wm, mver, brief, pool, qvec = await asyncio.gather(
        wm_t, memory_version(pid), get_brief(req.tenant_id, pid), pool_t, embed_query(req.query)
    )
    wm = wm or working_memory.WorkingMemory()
    graph = await get_graph(req.tenant_id, pid, mver)
    q_ents = graph.match(req.query)
    seeds = {e: WM_PRIMING for e in wm.entity_ids}
    seeds.update({e: 1.0 for e in q_ents})
    now = datetime.now(timezone.utc)
    return _Prep(
        now=now, wm=wm, mver=(mver or b"0").decode(), brief=brief, pool=pool, qvec=qvec, graph=graph,
        q_entities=q_ents, activation=graph.spread(seeds) if seeds else {}, cue=parse_temporal(req.query, now),
        query=req.query,
    )


def _pool_usable(prep: _Prep) -> bool:
    p = prep.pool
    return bool(
        p is not None
        and p.mver == prep.mver
        and p.entity_key == prefetch_cache.entity_key(prep.q_entities)
        and p.cue_key == prep.cue_key
        and float(p.qvec @ prep.qvec) >= settings.prefetch_reuse_sim
    )


async def _candidates(req: RetrieveRequest, prep: _Prep, statuses=("active", "weakening")):  # noqa: ANN001, ANN202
    if statuses == ("active", "weakening") and _pool_usable(prep):
        return prep.pool.cands, prep.pool.vecs, True
    cands, vecs = await fetch_candidates(CandidateQuery(
        tenant_id=req.tenant_id, project_id=req.project.id, user_id=req.user_id, chat_id=req.chat_id,
        include_user_scope=not req.project.isolated, qvec=prep.qvec, tsquery=tsquery_or(req.query, limit=6, salient_only=True),
        entity_activation=prep.activation, cue=prep.cue, statuses=statuses,
        attr_preds=query_attributes(req.query), domains=query_domains(req.query),
    ))
    return cands, vecs, False


def _score_ctx(req: RetrieveRequest, prep: _Prep) -> ScoreContext:
    c = prep.cue
    impersonal = not (prep.q_entities or c.has_window or c.last_chat or c.historical or is_personal(req.query))
    return ScoreContext(
        qvec=prep.qvec, query=req.query, activation=prep.activation, cue=prep.cue,
        wm_entities=set(prep.wm.entity_ids), chat_id=str(req.chat_id) if req.chat_id else None, now=prep.now,
        impersonal=impersonal,
    )


def _is_specific(query: str, q_entities: list[str]) -> bool:
    return bool(q_entities) or len(content_tokens(query)) >= 4


def _selection_from_ids(ids: list[str], seen: dict[str, Scored], note: str, now: datetime, budget: int) -> Selection:
    sel = Selection()
    for i in dict.fromkeys(ids):
        s = seen.get(i)
        if s is None:
            continue
        line = render_line(s, now)
        t = count_tokens(line) + 1
        if sel.tokens + t > budget:
            break
        sel.selected.append(s)
        sel.lines.append(line)
        sel.tokens += t
    if note:
        line = f"- [recalled] {note}"
        t = count_tokens(line) + 1
        if sel.tokens + t <= budget:
            sel.lines.append(line)
            sel.tokens += t
    return sel


def _memory_view(s: Scored) -> dict:
    c = s.cand
    return {
        "id": s.id, "kind": c.kind, "content": c.content, "scope": c.scope, "is_belief": c.is_belief,
        "score": s.score, "evidence": s.evidence, "features": s.features,
        "valid_from": c.valid_from.isoformat(), "valid_until": c.valid_until.isoformat() if c.valid_until else None,
        "thread_status": c.thread_status,
    }


async def _remember_last(chat_id: UUID | None, ids: list[str], ents: list[str]) -> None:
    if chat_id:
        await redis.set(Keys.last_retrieval(chat_id), orjson.dumps({"ids": ids, "ents": ents}), ex=3600)


def _include_brief(req: RetrieveRequest, prep: _Prep, selected: bool) -> bool:
    if req.brief == "never" or not prep.brief:
        return False
    if req.brief == "always":
        return True
    # auto: new chat (cross-chat continuity), something recalled, or the prompt names project entities
    return not prep.wm.turns or selected or bool(prep.q_entities)


async def retrieve(req: RetrieveRequest) -> RetrieveResult:
    timer = Timer()
    ok, gate = should_retrieve(req.query)
    if not ok:
        await _remember_last(req.chat_id, [], [])
        timer.mark("gate")
        return RetrieveResult(gate=gate, latency_ms={**timer.marks, "total": timer.total()},
                              server_timing=timer.server_timing())

    prep = await _prepare(req)
    timer.mark("prepare")
    cands, vecs, cache_hit = await _candidates(req, prep)
    timer.mark("candidates")
    ctx = _score_ctx(req, prep)
    scored = score_candidates(cands, vecs, ctx)
    sel = select(scored, prep.now, req.budget_tokens)
    # Semantic over episodic for current-state questions: past-chat gists describe what was true *then*
    # (and can mention since-changed facts). Use them only for "last time / when / before" questions or when
    # no fact answers the prompt.
    cue = prep.cue
    if not (cue.has_window or cue.last_chat or cue.historical) and any(s.cand.kind != "episode" for s in sel.selected):
        if any(s.cand.kind == "episode" for s in sel.selected):
            sel = select([s for s in scored if s.cand.kind != "episode"], prep.now, req.budget_tokens)
    timer.mark("select")

    # "Tip of the tongue": nothing active matched but the cue is specific -> try faded memories.
    if not sel.selected and _is_specific(req.query, prep.q_entities):
        a_cands, a_vecs, _ = await _candidates(req, prep, statuses=("archived",))
        if a_cands:
            a_scored = score_candidates(a_cands, a_vecs, ctx)
            a_sel = select(a_scored, prep.now, req.budget_tokens, min_score=settings.min_score + 0.1)
            if a_sel.selected:
                sel, scored = a_sel, a_scored + scored
        timer.mark("archive")

    # Optional memory agent (metamemory decides; speculative result from typing pause is reused).
    agent_used, agent_reason = False, None
    if req.agent != "off" and llm_enabled():
        agent_reason = "forced" if req.agent == "always" else memory_agent.needs_agent(
            req.query, scored, prep.cue, prep.q_entities)
        if agent_reason:
            spec = prep.pool.agent if prep.pool and prep.pool.agent else None
            seen = {s.id: s for s in scored}
            if spec and normalize(spec.get("query", "")) == normalize(req.query) and all(
                    i in seen for i in spec.get("ids", [])):
                keep = [s.id for s in sel.selected if s.score >= settings.agent_keep_score and s.id not in spec["ids"]]
                sel = _selection_from_ids(spec["ids"] + keep, seen, spec.get("note", ""), prep.now, req.budget_tokens)
                agent_used = True
                agent_reason += "+speculative"
            else:
                tools = ToolContext(req.tenant_id, req.project.id, req.user_id, req.chat_id,
                                    not req.project.isolated, prep.graph, ctx)
                budget_ms = settings.agent_forced_timeout_ms if req.agent == "always" else settings.agent_timeout_ms
                res = await memory_agent.run(req.query, prep.wm.recent_text(), scored, tools, budget_ms)
                if res is not None:
                    # Deliberate recall ADDS to automatic recall: strong automatic hits are kept.
                    keep = [s.id for s in sel.selected if s.score >= settings.agent_keep_score and s.id not in res.ids]
                    sel = _selection_from_ids(res.ids + keep, tools.seen, res.note, prep.now, req.budget_tokens)
                    scored = scored + [s for i, s in tools.seen.items() if i not in seen]
                    agent_used = True
            timer.mark("agent")

    selected_ids = [s.id for s in sel.selected]
    brief = prep.brief if _include_brief(req, prep, bool(sel.selected)) else ""
    await _remember_last(req.chat_id, selected_ids, prep.q_entities)
    timer.mark("finish")
    latency = {**timer.marks, "total": timer.total()}
    trace_id = trace.record(
        req.tenant_id, req.project.id, req.chat_id, req.query, gate, cache_hit, agent_used,
        [{"id": s.id, "score": s.score, "evidence": s.evidence, "features": s.features,
          "outcome": "selected" if s.id in selected_ids else sel.rejected.get(s.id, "not-chosen")}
         for s in scored[:40]],
        selected_ids, latency,
    )
    return RetrieveResult(
        context=render_context(sel.lines), brief=brief, memories=[_memory_view(s) for s in sel.selected],
        selected_ids=selected_ids, query_entities=prep.q_entities, trace_id=trace_id, gate=gate,
        cache_hit=cache_hit, agent_used=agent_used, agent_reason=agent_reason, tokens=sel.tokens,
        latency_ms=latency, server_timing=timer.server_timing(),
    )


async def prefetch(req: RetrieveRequest, speculate: bool = False) -> dict:
    """Warm the candidate pool for a partial prompt. With ``speculate`` (typing pause), also run the
    memory agent in the background budget so its answer is ready on submit."""
    timer = Timer()
    ok, gate = should_retrieve(req.query)
    if not ok or req.chat_id is None:
        return {"gate": gate, "candidates": 0, "latency_ms": timer.total()}
    prep = await _prepare(req)
    cands, vecs, reused = await _candidates(req, prep)
    pool = prefetch_cache.Pool(
        partial=req.query, qvec=prep.qvec, cands=cands, vecs=vecs, mver=prep.mver,
        entity_key=prefetch_cache.entity_key(prep.q_entities), cue_key=prep.cue_key,
        agent=prep.pool.agent if reused and prep.pool else None,
    )
    ctx = _score_ctx(req, prep)
    scored = score_candidates(cands, vecs, ctx)
    sel = select(scored, prep.now, req.budget_tokens)
    agent_ran = False
    if speculate and req.agent != "off" and llm_enabled():
        reason = "forced" if req.agent == "always" else memory_agent.needs_agent(
            req.query, scored, prep.cue, prep.q_entities)
        if reason:
            tools = ToolContext(req.tenant_id, req.project.id, req.user_id, req.chat_id,
                                not req.project.isolated, prep.graph, ctx)
            res = await memory_agent.run(req.query, prep.wm.recent_text(), scored, tools,
                                         settings.agent_speculative_timeout_ms)
            if res is not None:
                extra = [s for i, s in tools.seen.items() if i not in {c.id for c in cands}]
                if extra:  # keep agent-found memories in the pool so they can be reused on submit
                    pool.cands = pool.cands + [s.cand for s in extra]
                    pool.vecs = np.vstack([pool.vecs, np.stack([s.vec for s in extra])]) if len(cands) else \
                        np.stack([s.vec for s in extra])
                pool.agent = {"query": req.query, "ids": res.ids, "note": res.note}
                agent_ran = True
    await prefetch_cache.save(req.chat_id, pool)
    return {
        "gate": gate,
        "candidates": len(pool.cands),
        "reused": reused,
        "agent": agent_ran,
        "preview": [{"id": s.id, "content": s.cand.content, "score": s.score} for s in sel.selected[:5]],
        "latency_ms": timer.total(),
    }
