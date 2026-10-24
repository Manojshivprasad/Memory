"""Chat working memory (Redis): the last few turns + the entities currently "in focus".

Written synchronously at commit time (Stage A) so the very next prompt in the chat sees it,
before the background workers have processed anything.
"""

import time
from dataclasses import dataclass, field
from uuid import UUID

import orjson

from app.cache import Keys, redis
from app.config import settings


@dataclass
class WorkingMemory:
    turns: list[dict] = field(default_factory=list)       # [{seq, u, a, ts}] oldest -> newest
    entity_ids: list[str] = field(default_factory=list)   # most recent first

    def recent_text(self, n: int = 2, max_chars: int = 600) -> str:
        parts = [f"user: {t['u']}\nassistant: {t['a']}" for t in self.turns[-n:]]
        return "\n".join(parts)[-max_chars:]


async def load(chat_id: UUID) -> WorkingMemory:
    pipe = redis.pipeline(transaction=False)
    pipe.lrange(Keys.wm_turns(chat_id), 0, -1)
    pipe.zrevrange(Keys.wm_entities(chat_id), 0, settings.wm_entities - 1)
    turns_raw, ents = await pipe.execute()
    return WorkingMemory(
        turns=[orjson.loads(t) for t in turns_raw],
        entity_ids=[e.decode() for e in ents],
    )


def append_turn_cmds(pipe, chat_id: UUID, seq: int, user_msg: str, assistant_msg: str) -> None:  # noqa: ANN001
    key = Keys.wm_turns(chat_id)
    pipe.rpush(key, orjson.dumps({"seq": seq, "u": user_msg[:4000], "a": assistant_msg[:4000], "ts": time.time()}))
    pipe.ltrim(key, -settings.wm_turns, -1)
    pipe.expire(key, settings.wm_ttl_s)


def touch_entities_cmds(pipe, chat_id: UUID, entity_ids: list[str]) -> None:  # noqa: ANN001
    if not entity_ids:
        return
    key = Keys.wm_entities(chat_id)
    now = time.time()
    pipe.zadd(key, {e: now for e in entity_ids})
    pipe.zremrangebyrank(key, 0, -(settings.wm_entities + 1))
    pipe.expire(key, settings.wm_ttl_s)
