"""Proxy mode: OpenAI-compatible chat completions with memory.

    POST /v1/chat/completions   { "model": "...", "messages": [...], "stream": true,
                                  "memory": {"project_id": "...", "chat_id": "...", "agent": "auto"} }

Retrieve (strict, budgeted) -> inject [customer system prompt][project brief][memory] as ONE system
message (stable prefix first, so provider prompt caching works) -> stream the LLM answer through ->
after the last chunk, Stage A commit (Redis only) -> Stage B forms memories in the worker.
"""

import uuid
from typing import Any, Literal
from uuid import UUID

import orjson
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.chats import create_chat
from app.auth.deps import Principal, chat_for, project_for, require_app, resolve_user
from app.config import settings
from app.gateway import providers
from app.runtime.orchestrator import RetrieveRequest, RetrieveResult, retrieve
from app.runtime.retrieval.render import BRIEF_HEADER
from app.write.stage_a import commit_turn

router = APIRouter(prefix="/v1", tags=["proxy"])


class MemoryOptions(BaseModel):
    project_id: UUID
    chat_id: UUID | None = Field(default=None, description="Omit to start a new chat in the project")
    user_id: str | None = Field(default=None, description="End-user id (service tokens); defaults to `user`")
    budget_tokens: int = Field(default=settings.budget_tokens, ge=0, le=4000)
    agent: Literal["off", "auto", "always"] = "auto"
    brief: Literal["auto", "always", "never"] = "auto"
    commit: bool = True


class ChatCompletionIn(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)
    stream: bool = False
    user: str | None = None
    memory: MemoryOptions


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content parts
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def inject(messages: list[dict], res: RetrieveResult) -> list[dict]:
    blocks = []
    if res.brief:
        blocks.append(f"<project_memory>\n{BRIEF_HEADER}\n{res.brief}\n</project_memory>")
    if res.context:
        blocks.append(f"<memory>\n{res.context}\n</memory>")
    if not blocks:
        return messages
    mem = "\n\n".join(blocks)
    msgs = [dict(m) for m in messages]
    if msgs[0].get("role") == "system" and isinstance(msgs[0].get("content"), str):
        msgs[0]["content"] = f"{msgs[0]['content']}\n\n{mem}"
    else:
        msgs.insert(0, {"role": "system", "content": mem})
    return msgs


@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionIn, p: Principal = Depends(require_app)):  # noqa: ANN201
    p.require_scope("chat")
    opts = body.memory
    uid = await resolve_user(p, opts.user_id or body.user)
    project = await project_for(p, uid, opts.project_id)
    if opts.chat_id is None:
        chat_id = (await create_chat(p.tenant_id, project.id))["id"]
    else:
        chat_id = (await chat_for(p, project, opts.chat_id)).id

    last_user = next((m for m in reversed(body.messages) if m.get("role") == "user"), None)
    if last_user is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "messages must contain a user message")
    query = _text(last_user.get("content"))

    res = await retrieve(RetrieveRequest(p.tenant_id, uid, project, chat_id, query, opts.budget_tokens,
                                         opts.agent, opts.brief))
    messages = inject(body.messages, res)
    params = {k: v for k, v in (body.model_extra or {}).items() if k not in ("memory",)}
    if body.user:
        params["user"] = body.user
    turn_id = uuid.uuid4()  # known up front so clients can look up what memory did with this turn
    headers = {
        "X-Memory-Chat-Id": str(chat_id),
        "X-Memory-Turn-Id": str(turn_id),
        "X-Memory-Trace-Id": res.trace_id or "",
        "X-Memory-Count": str(len(res.selected_ids)),
        "Server-Timing": res.server_timing,
    }

    async def _commit(answer: str) -> None:
        if opts.commit and answer.strip():
            await commit_turn(
                tenant_id=p.tenant_id, user_id=uid, project_id=project.id, chat_id=chat_id,
                isolated=project.isolated, user_msg=query, assistant_msg=answer,
                retrieved_ids=res.selected_ids, query_entities=res.query_entities, model=body.model,
                turn_id=turn_id,
            )

    memory_meta = {"chat_id": str(chat_id), "turn_id": str(turn_id), "trace_id": res.trace_id, "memories": res.selected_ids,
                   "brief_used": bool(res.brief), "agent_used": res.agent_used, "latency_ms": res.latency_ms}

    if not body.stream:
        resp = await providers.complete(body.model, messages, params)
        data = resp.model_dump(exclude_none=True)
        await _commit(_text(data["choices"][0]["message"].get("content")))
        data["memory"] = memory_meta
        return data

    chunks = providers.stream(body.model, messages, params)
    first = await anext(chunks)  # surface provider errors as HTTP errors before streaming starts

    async def sse():  # noqa: ANN202
        parts: list[str] = []
        try:
            chunk = first
            while True:
                data = chunk.model_dump(exclude_none=True)
                for ch in data.get("choices", []):
                    delta = (ch.get("delta") or {}).get("content")
                    if delta:
                        parts.append(delta)
                yield b"data: " + orjson.dumps(data) + b"\n\n"
                try:
                    chunk = await anext(chunks)
                except StopAsyncIteration:
                    break
        except Exception as e:  # noqa: BLE001
            yield b"data: " + orjson.dumps({"error": {"message": str(e)[:500]}}) + b"\n\n"
            return
        yield b"data: [DONE]\n\n"
        await _commit("".join(parts))  # Stage A: ~1 ms of Redis after the client already has [DONE]

    return StreamingResponse(sse(), media_type="text/event-stream", headers=headers)


@router.get("/models")
async def list_models(_: Principal = Depends(require_app)) -> dict:
    """Chat models offered to end users (configure with MEMORY_CHAT_MODELS)."""
    return {"default": settings.chat_models[0] if settings.chat_models else None,
            "models": [{"id": m, "label": m.split("/")[-1]} for m in settings.chat_models]}
