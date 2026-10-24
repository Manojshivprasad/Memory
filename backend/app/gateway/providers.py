"""LLM provider abstraction (LiteLLM): OpenAI, Anthropic, Google, Azure, Bedrock, Ollama, custom endpoints.

Customers switch models by changing the ``model`` string; memory behaviour is unchanged.
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from fastapi import HTTPException, status

# Use LiteLLM's bundled model/cost map instead of fetching it from the internet on first import.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
log = logging.getLogger(__name__)


def _litellm():  # noqa: ANN202
    import litellm

    litellm.drop_params = True  # ignore params a provider doesn't support instead of failing
    return litellm


def warm() -> None:
    """Import LiteLLM at startup (slow import) instead of on the first chat request."""
    _litellm()


def _http_error(e: Exception) -> HTTPException:
    code = getattr(e, "status_code", None) or status.HTTP_502_BAD_GATEWAY
    return HTTPException(code if 400 <= int(code) < 600 else 502, f"LLM provider error: {str(e)[:500]}")


# Network blips (DNS "getaddrinfo failed", connection resets, timeouts) are retried; real API errors are not.
_TRANSIENT_MARKERS = ("getaddrinfo", "cannot connect", "connection reset", "connection aborted", "timed out",
                      "timeout", "temporarily unavailable", "server disconnected", "502", "503", "504")
RETRY_DELAYS = (0.5, 1.5, 3.0)


def is_transient(e: Exception) -> bool:
    name = type(e).__name__
    if name in ("APIConnectionError", "Timeout", "ServiceUnavailableError", "InternalServerError"):
        return True
    return any(m in str(e).lower() for m in _TRANSIENT_MARKERS)


async def _with_retries(call):  # noqa: ANN001, ANN202
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            return await call()
        except Exception as e:  # noqa: BLE001
            if attempt < len(RETRY_DELAYS) and is_transient(e):
                log.warning("LLM provider unreachable (%s); retrying in %.1fs", str(e)[:120], RETRY_DELAYS[attempt])
                await asyncio.sleep(RETRY_DELAYS[attempt])
                continue
            raise _http_error(e) from e


async def complete(model: str, messages: list[dict], params: dict[str, Any]) -> Any:
    return await _with_retries(lambda: _litellm().acompletion(model=model, messages=messages, **params))


async def stream(model: str, messages: list[dict], params: dict[str, Any]) -> AsyncIterator[Any]:
    # retry only the connection phase: once tokens have been sent to the client we can't restart
    resp = await _with_retries(lambda: _litellm().acompletion(model=model, messages=messages, stream=True, **params))
    async for chunk in resp:
        yield chunk
