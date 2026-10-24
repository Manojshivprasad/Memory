"""Type-ahead prefetch over WebSocket.

    ws://host/v1/prefetch?token=<subject or service JWT>&project_id=...&chat_id=...[&user_id=...][&agent=auto]

Client sends {"partial": "<current input text>"} on a ~150 ms debounce. The server warms the
candidate pool for that chat and replies with a small preview. If the user pauses typing for
``PAUSE_S``, the memory agent runs speculatively (when needed) so its answer is ready on submit.
"""

import asyncio
import logging
from uuid import UUID

import orjson
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from app.auth.deps import chat_for, principal_from_token, project_for, resolve_user
from app.runtime.orchestrator import RetrieveRequest, prefetch

router = APIRouter(tags=["prefetch"])
log = logging.getLogger(__name__)
PAUSE_S = 0.6


@router.websocket("/v1/prefetch")
async def prefetch_ws(
    ws: WebSocket,
    token: str = Query(...),
    project_id: UUID = Query(...),
    chat_id: UUID = Query(...),
    user_id: str | None = Query(default=None),
    agent: str = Query(default="auto", pattern="^(off|auto|always)$"),
) -> None:
    try:
        p = principal_from_token(token)
        if p.kind == "admin":
            raise HTTPException(403, "service or subject token required")
        p.require_scope("prefetch")
        uid = await resolve_user(p, user_id)
        project = await project_for(p, uid, project_id)
        await chat_for(p, project, chat_id)
    except HTTPException as e:
        await ws.close(code=4403 if e.status_code == 403 else 4401 if e.status_code == 401 else 4404,
                       reason=str(e.detail)[:120])
        return
    await ws.accept()

    speculation: asyncio.Task | None = None

    async def speculate(partial: str) -> None:
        await asyncio.sleep(PAUSE_S)
        res = await prefetch(RetrieveRequest(p.tenant_id, uid, project, chat_id, partial, agent=agent),
                             speculate=True)
        if res.get("agent"):
            await ws.send_bytes(orjson.dumps({"type": "speculated", **res}))

    try:
        while True:
            msg = await ws.receive_text()
            try:
                partial = str(orjson.loads(msg).get("partial", ""))
            except (orjson.JSONDecodeError, AttributeError):
                partial = msg
            if speculation and not speculation.done():
                speculation.cancel()
            res = await prefetch(RetrieveRequest(p.tenant_id, uid, project, chat_id, partial, agent=agent))
            await ws.send_bytes(orjson.dumps({"type": "prefetch", **res}))
            if agent != "off":
                speculation = asyncio.create_task(speculate(partial))
    except WebSocketDisconnect:
        pass
    finally:
        if speculation and not speculation.done():
            speculation.cancel()
