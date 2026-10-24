from redis.asyncio import Redis

from app.config import settings

# Binary-safe client: values are orjson bytes or raw numpy buffers.
redis: Redis = Redis.from_url(settings.redis_url, decode_responses=False)


class Keys:
    """All Redis key names in one place."""

    @staticmethod
    def wm_turns(chat_id) -> str:  # noqa: ANN001
        return f"wm:{chat_id}:turns"

    @staticmethod
    def wm_entities(chat_id) -> str:  # noqa: ANN001
        return f"wm:{chat_id}:ents"

    @staticmethod
    def chat_seq(chat_id) -> str:  # noqa: ANN001
        return f"chat:{chat_id}:seq"

    @staticmethod
    def chat_applied(chat_id) -> str:  # noqa: ANN001
        return f"chat:{chat_id}:applied"

    @staticmethod
    def chat_lock(chat_id) -> str:  # noqa: ANN001
        return f"lock:chat:{chat_id}"

    @staticmethod
    def last_retrieval(chat_id) -> str:  # noqa: ANN001
        return f"lastret:{chat_id}"

    @staticmethod
    def prefetch(chat_id) -> str:  # noqa: ANN001
        return f"prefetch:{chat_id}"

    @staticmethod
    def prefetch_vecs(chat_id) -> str:  # noqa: ANN001
        return f"prefetch:{chat_id}:vecs"

    @staticmethod
    def brief(project_id) -> str:  # noqa: ANN001
        return f"brief:{project_id}"

    @staticmethod
    def memory_version(project_id) -> str:  # noqa: ANN001
        """Bumped on every memory change in a project; invalidates prefetch + graph caches."""
        return f"proj:{project_id}:mver"

    BRIEFS_DIRTY = "briefs:dirty"            # set of "tenant:project"
    CONSOLIDATE_PENDING = "consolidate:pending"  # set of "tenant:chat"
    DEAD_LETTER = "mem:turns:dead"
