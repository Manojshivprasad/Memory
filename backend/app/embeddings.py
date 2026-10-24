"""Local embeddings (no network on the hot path).

``fastembed`` runs bge-small-en-v1.5 on ONNX (~5-15 ms per short query on CPU). ``hash`` is a
deterministic feature-hashing embedder used in unit tests and offline development.
All vectors are L2-normalised float32, so dot product == cosine similarity.
"""

import asyncio
import hashlib
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np

from app.config import settings

_TOKEN = re.compile(r"[a-z0-9_]+")


class HashEmbedder:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = _TOKEN.findall(text.lower())
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


class FastEmbedder:
    def __init__(self, model: str, dim: int, threads: int | None) -> None:
        from fastembed import TextEmbedding

        self.dim = dim
        # Few intra-op threads: waking a large ONNX pool for one short query costs more than it saves.
        self._model = TextEmbedding(model_name=model, threads=threads)
        self._lock = Lock()  # ONNX session is thread-safe, but keep batches from interleaving

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        with self._lock:
            arr = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


_embedder: HashEmbedder | FastEmbedder | None = None
_query_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
_QUERY_CACHE_MAX = 4096
# One dedicated thread keeps ONNX on a stable OS thread and serialises tiny inference calls.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed")


def get_embedder() -> HashEmbedder | FastEmbedder:
    global _embedder
    if _embedder is None:
        if settings.embedder == "hash":
            _embedder = HashEmbedder(settings.embed_dim)
        else:
            _embedder = FastEmbedder(settings.embed_model, settings.embed_dim, settings.embed_threads or None)
    return _embedder


def embed_passages_sync(texts: list[str]) -> np.ndarray:
    return get_embedder().embed(texts)


def embed_query_sync(text: str) -> np.ndarray:
    key = text.strip()
    hit = _query_cache.get(key)
    if hit is not None:
        _query_cache.move_to_end(key)
        return hit
    prefix = settings.embed_query_prefix if settings.embedder == "fastembed" else ""
    vec = get_embedder().embed([prefix + key])[0]
    _query_cache[key] = vec
    if len(_query_cache) > _QUERY_CACHE_MAX:
        _query_cache.popitem(last=False)
    return vec


async def embed_query(text: str) -> np.ndarray:
    hit = _query_cache.get(text.strip())
    if hit is not None:
        return hit
    return await asyncio.get_running_loop().run_in_executor(_executor, embed_query_sync, text)


async def embed_passages(texts: list[str]) -> np.ndarray:
    return await asyncio.get_running_loop().run_in_executor(_executor, embed_passages_sync, texts)


def as_array(value) -> np.ndarray:  # noqa: ANN001
    """pgvector column value (Vector / ndarray / list) -> float32 ndarray."""
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value, dtype=np.float32)


def warmup() -> None:
    """Load the model and run a few input lengths (ONNX is slow the first time it sees a new shape)."""
    for n in (1, 4, 8, 16, 32, 64):
        text = " ".join(["memory"] * n)
        get_embedder().embed([text])
        get_embedder().embed([text, text])
    _query_cache.clear()
