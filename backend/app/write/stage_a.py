"""Stage A: synchronous commit, Redis only (~1 ms, two round trips). Never blocks on the database.

1. assign an ordering seq for the chat
2. update chat working memory (turn + entities in focus) so the NEXT prompt already sees it
3. enqueue the turn for Stage B (Redis Stream); ``turn_id`` is the idempotency key
"""

import time
import uuid
from uuid import UUID

import orjson

from app.cache import Keys, redis
from app.config import settings
from app.runtime import working_memory


async def commit_turn(
    *,
    tenant_id: UUID,
    user_id: UUID,
    project_id: UUID,
    chat_id: UUID,
    isolated: bool,
    user_msg: str,
    assistant_msg: str,
    retrieved_ids: list[str] | None = None,
    query_entities: list[str] | None = None,
    turn_id: UUID | None = None,
    model: str | None = None,
) -> dict:
    turn_id = turn_id or uuid.uuid4()
    pipe = redis.pipeline(transaction=False)
    pipe.incr(Keys.chat_seq(chat_id))
    pipe.get(Keys.last_retrieval(chat_id))
    seq, last_raw = await pipe.execute()
    last = orjson.loads(last_raw) if last_raw else {}
    if retrieved_ids is None:
        retrieved_ids = last.get("ids", [])
    if query_entities is None:
        query_entities = last.get("ents", [])

    job = {
        "turn_id": str(turn_id), "tenant_id": str(tenant_id), "user_id": str(user_id),
        "project_id": str(project_id), "chat_id": str(chat_id), "isolated": isolated, "seq": seq,
        "user_msg": user_msg, "assistant_msg": assistant_msg, "retrieved_ids": retrieved_ids,
        "model": model, "ts": time.time(),
    }
    pipe = redis.pipeline(transaction=False)
    working_memory.append_turn_cmds(pipe, chat_id, seq, user_msg, assistant_msg)
    working_memory.touch_entities_cmds(pipe, chat_id, query_entities)
    pipe.delete(Keys.last_retrieval(chat_id))
    pipe.xadd(settings.stream_turns, {"chat": str(chat_id), "data": orjson.dumps(job)}, maxlen=200_000,
              approximate=True)
    await pipe.execute()
    return {"turn_id": str(turn_id), "seq": seq}
