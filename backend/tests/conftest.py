import os

# Unit tests use the deterministic hash embedder; integration tests may override (MEMORY_EMBEDDER=fastembed).
os.environ.setdefault("MEMORY_EMBEDDER", "hash")
os.environ.setdefault("MEMORY_LLM_MODEL", "")
