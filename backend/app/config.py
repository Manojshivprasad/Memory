from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ScoringWeights(BaseModel):
    """Retrieval score = relevance x strength modulation. Pluggable per deployment (later per tenant).

    relevance = 1 - prod(1 - w_c * f_c) over recall channels (any strong channel can carry a memory)
    strength  = weighted mean of retrievability, importance, confidence*reliability, scope
    score     = relevance * ((1 - strength_mix) + strength_mix * strength)
    """

    # recall channels
    semantic: float = 1.0
    lexical: float = 0.8
    activation: float = 0.8
    temporal: float = 0.7
    context: float = 0.3
    attribute: float = 0.85   # the prompt asks about this memory's attribute (structured recall)
    constraint: float = 0.9   # standing safety/health constraint in the prompt's domain
    # strength (human-like: how alive / trusted / close the memory is)
    retrievability: float = 0.35
    importance: float = 0.25
    confidence: float = 0.25
    scope: float = 0.15
    strength_mix: float = 0.4


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="MEMORY_", env_nested_delimiter="__", extra="ignore"
    )

    env: str = "dev"
    cors_origins: list[str] = ["http://localhost:3000"]

    # --- infrastructure ---
    database_url: str = "postgresql+asyncpg://memory_app:memory_app@localhost:5433/memory"
    database_admin_url: str = "postgresql+asyncpg://memory:memory@localhost:5433/memory"
    db_pool_size: int = 20
    redis_url: str = "redis://localhost:6380/0"

    # --- auth ---
    jwt_secret: str = "dev-only-secret-change-me-0123456789abcdef"
    jwt_kid: str = "v1"
    jwt_issuer: str = "memory-runtime"
    access_ttl_s: int = 15 * 60
    refresh_ttl_s: int = 30 * 24 * 3600
    service_ttl_s: int = 60 * 60
    subject_ttl_s: int = 10 * 60

    # --- models ---
    embedder: str = "fastembed"  # fastembed | hash
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    embed_threads: int = 4  # ONNX intra-op threads (0 = library default); 4 measured best for 1-query latency
    embed_query_prefix: str = "Represent this sentence for searching relevant passages: "
    # Internal LLM (LiteLLM model string) for extraction, consolidation and the memory agent.
    # Empty => heuristic mode: the runtime never calls an LLM itself.
    llm_model: str = ""
    # Optional faster model for the latency-bound memory agent (defaults to llm_model).
    agent_llm_model: str = ""
    llm_timeout_s: float = 25.0
    # Models end users can pick in the chat UI (LiteLLM model strings; provider keys come from the env).
    chat_models: list[str] = ["groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b", "groq/qwen/qwen3.8-27b"]

    # --- retrieval (read path) ---
    budget_tokens: int = 300
    max_memories: int = 12
    candidate_limit: int = 40
    min_score: float = 0.35
    # raw cosine window used to normalise semantic similarity to 0..1
    sem_lo: float = 0.45
    sem_hi: float = 0.75
    # a candidate needs at least one real relevance signal; raw cosine needed for "semantic evidence"
    semantic_floor: float = 0.55  # calibrated on bge-small v1.5: unrelated p95 ~0.53, relevant p10 ~0.55
    impersonal_floor_boost: float = 0.08  # general-knowledge prompts need cos >= 0.63
    r_min: float = 0.05
    mmr_lambda: float = 0.75
    dedupe_sim: float = 0.95
    prefetch_reuse_sim: float = 0.90
    prefetch_ttl_s: int = 120
    graph_cache_ttl_s: float = 15.0
    weights: ScoringWeights = ScoringWeights()

    # --- memory agent ---
    agent_timeout_ms: int = 800              # agent="auto" on the submit path
    agent_forced_timeout_ms: int = 2500      # agent="always": caller opted into the extra latency
    agent_speculative_timeout_ms: int = 3000  # typing pause (hidden latency)
    agent_keep_score: float = 0.55           # automatic-recall hits this strong are kept even if the agent omits them
    agent_max_steps: int = 3
    agent_hard_score: float = 0.5

    # --- working memory ---
    wm_turns: int = 6
    wm_entities: int = 24
    wm_ttl_s: int = 24 * 3600

    # --- write path / consolidation ---
    salience_extract: float = 0.25
    dedupe_write_sim: float = 0.92
    chat_idle_s: int = 600
    consolidate_every_turns: int = 20
    sleep_interval_s: int = 24 * 3600
    scheduler_tick_s: float = 15.0
    stream_turns: str = "mem:turns"
    stream_group: str = "writers"


# Provider keys (GROQ_API_KEY, ANTHROPIC_API_KEY, ...) are read by LiteLLM from the process environment.
load_dotenv(override=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
