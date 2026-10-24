"""Internal LLM access (extraction, consolidation, memory agent) via LiteLLM.

Every caller must handle ``None`` / exceptions: the runtime always has a deterministic fallback.
"""

import asyncio
import logging
import re
from typing import Any

import orjson

from app.config import settings
from app.gateway.providers import is_transient

log = logging.getLogger(__name__)
_JSON_BLOCK = re.compile(r"\{.*\}", re.S)
_RETRY_AFTER = re.compile(r"try again in ([0-9.]+)s", re.I)


def llm_enabled() -> bool:
    return bool(settings.llm_model)


async def acompletion(messages: list[dict], **kwargs: Any):  # noqa: ANN201
    from app.gateway.providers import _litellm

    litellm = _litellm()
    kwargs.setdefault("timeout", settings.llm_timeout_s)
    return await litellm.acompletion(model=kwargs.pop("model", settings.llm_model), messages=messages, **kwargs)


def parse_json(text: str) -> dict | None:
    if not text:
        return None
    try:
        return orjson.loads(text)
    except orjson.JSONDecodeError:
        m = _JSON_BLOCK.search(text)
        if not m:
            return None
        try:
            return orjson.loads(m.group(0))
        except orjson.JSONDecodeError:
            return None


async def json_complete(system: str, user: str, max_tokens: int = 2500, retries: int = 2) -> dict | None:
    """One-shot JSON call. Returns None when disabled or on any failure (callers fall back to heuristics).

    ``max_tokens`` is generous because reasoning models spend completion tokens on hidden reasoning.
    """
    if not llm_enabled():
        return None
    resp = None
    for attempt in range(retries + 1):
        try:
            resp = await acompletion(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=0,
                response_format={"type": "json_object"},
                reasoning_effort="low",  # dropped automatically for models without reasoning controls
            )
            break
        except Exception as e:  # noqa: BLE001
            # Background work is not latency-bound: on a provider rate limit, wait as asked and retry.
            if type(e).__name__ == "RateLimitError" and attempt < retries:
                m = _RETRY_AFTER.search(str(e))
                wait = min(60.0, float(m.group(1)) + 0.5) if m else 5.0 * (attempt + 1)
                log.info("internal LLM rate-limited; retrying in %.1fs", wait)
                await asyncio.sleep(wait)
                continue
            if is_transient(e) and attempt < retries:  # DNS / connection blips
                log.info("internal LLM unreachable; retrying in %.1fs", 1.0 + attempt * 2)
                await asyncio.sleep(1.0 + attempt * 2)
                continue
            log.warning("internal LLM call failed (%s: %s); falling back", type(e).__name__, str(e)[:200])
            return None
    choice = resp.choices[0]
    data = parse_json(choice.message.content or "")
    if data is None:
        log.warning("internal LLM returned no parseable JSON (finish_reason=%s); falling back",
                    getattr(choice, "finish_reason", "?"))
    return data
