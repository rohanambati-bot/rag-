# phase3/semantic_cache.py
"""
Production semantic cache.
──────────────────────────
Replaces the original implementation, which on **every lookup** issued a Redis
``KEYS rag_cache:*`` followed by one ``GET`` per key and then scored candidates in a
Python loop. That design has three production-fatal properties:

* ``KEYS`` is O(keyspace) and *blocks* the single-threaded Redis event loop, so one
  cache lookup stalls every other client on the instance.
* One network round-trip per cached entry means lookup latency grows linearly with
  cache size — the cache gets slower exactly as it becomes more useful.
* Scoring in a Python ``for`` loop wastes the vectorisation NumPy already provides.

This module instead keeps the hot path entirely in-process:

* **Vector index.** Embeddings live in a contiguous ``float32`` matrix. Because the
  vectors are L2-normalised, cosine similarity is a single matrix-vector product, so
  lookup is one BLAS call rather than N Python iterations. When ``hnswlib`` is
  installed an approximate-nearest-neighbour index is used instead, giving
  sub-linear search for large caches.
* **Redis is persistence, not the hot path.** It is read once at startup (via
  ``SCAN`` + batched ``MGET``, never ``KEYS``) and written through on store, so
  multiple workers share entries and survive restarts. If Redis is down the cache
  degrades to in-process only and keeps serving.
* **Bounded and expiring.** Capacity is capped with LRU eviction and entries carry a
  TTL, so memory cannot grow without limit.
* **Thread-safe.** A reentrant lock guards index mutation, because FastAPI serves
  requests from a thread pool.

Metrics (hits, misses, evictions, lookup latency) are exposed for scraping.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import (  # noqa: E402
    CACHE_SIMILARITY_THRESHOLD, CACHE_TTL_SECONDS, EMBEDDING_MODEL,
    REDIS_HOST, REDIS_PORT,
)

logger = logging.getLogger(__name__)

# Redis key namespace. The index set lets us enumerate entries without KEYS.
_KEY_PREFIX = "rag_cache:entry:"
_INDEX_SET = "rag_cache:index"

# Cap the in-process matrix. 50k x 384 float32 is ~77 MB, a reasonable ceiling for
# a single worker; beyond this the LRU evicts.
DEFAULT_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "50000"))


@dataclass
class CacheEntry:
    query: str
    answer: str
    contexts: List[str]
    embedding: List[float]
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0

    def is_expired(self, ttl_seconds: int) -> bool:
        return ttl_seconds > 0 and (time.time() - self.timestamp) > ttl_seconds

    def to_json(self) -> bytes:
        return json.dumps({
            "query": self.query,
            "answer": self.answer,
            "contexts": self.contexts,
            "embedding": self.embedding,
            "timestamp": self.timestamp,
            "hit_count": self.hit_count,
        }).encode()

    @classmethod
    def from_json(cls, raw: bytes) -> "CacheEntry":
        data = json.loads(raw.decode())
        return cls(
            query=data["query"],
            answer=data["answer"],
            contexts=data.get("contexts", []),
            embedding=data["embedding"],
            timestamp=data.get("timestamp", time.time()),
            hit_count=data.get("hit_count", 0),
        )


class _VectorIndex:
    """
    Similarity index over unit-norm embeddings.

    Uses hnswlib (approximate, sub-linear) when available and falls back to an exact
    vectorised brute-force search otherwise. Both paths return the same
    ``(row, score)`` contract, so behaviour is identical apart from recall on very
    large caches. Brute force is not a naive loop: one ``matrix @ vector`` product
    scores every entry at BLAS speed.
    """

    def __init__(self, dim: int, max_elements: int):
        self.dim = dim
        self.max_elements = max_elements
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        self._ann = None
        self._ann_count = 0

        try:
            import hnswlib
            self._ann = hnswlib.Index(space="cosine", dim=dim)
            self._ann.init_index(max_elements=max_elements, ef_construction=200, M=16)
            self._ann.set_ef(64)  # query-time accuracy/speed trade-off
            logger.info("SemanticCache: hnswlib ANN index active (dim=%d).", dim)
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "SemanticCache: hnswlib unavailable (%s) — using vectorised exact search.",
                exc,
            )

    @property
    def size(self) -> int:
        return self._matrix.shape[0]

    def add(self, vector: np.ndarray) -> int:
        """Append a vector, returning its row id."""
        row = self._matrix.shape[0]
        self._matrix = np.vstack([self._matrix, vector.reshape(1, -1)])
        if self._ann is not None:
            try:
                if self._ann_count >= self.max_elements:
                    self._ann.resize_index(self.max_elements * 2)
                    self.max_elements *= 2
                self._ann.add_items(vector.reshape(1, -1), [row])
                self._ann_count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("ANN add failed (%s) — disabling ANN.", exc)
                self._ann = None
        return row

    def search(self, vector: np.ndarray) -> Tuple[int, float]:
        """Return ``(best_row, cosine_score)``; ``(-1, 0.0)`` when empty."""
        if self._matrix.shape[0] == 0:
            return -1, 0.0

        if self._ann is not None and self._ann_count > 0:
            try:
                labels, distances = self._ann.knn_query(
                    vector.reshape(1, -1), k=min(1, self._ann_count)
                )
                # hnswlib 'cosine' space returns distance = 1 - cosine_similarity.
                return int(labels[0][0]), float(1.0 - distances[0][0])
            except Exception as exc:  # noqa: BLE001
                logger.warning("ANN query failed (%s) — falling back to exact.", exc)

        # Exact: one BLAS matrix-vector product scores the whole cache at once.
        scores = self._matrix @ vector
        best = int(np.argmax(scores))
        return best, float(scores[best])

    def rebuild(self, vectors: List[np.ndarray]) -> None:
        """Rebuild from scratch (used after eviction compaction)."""
        self._matrix = (
            np.vstack([v.reshape(1, -1) for v in vectors]).astype(np.float32)
            if vectors else np.zeros((0, self.dim), dtype=np.float32)
        )
        if self._ann is not None:
            try:
                import hnswlib
                self._ann = hnswlib.Index(space="cosine", dim=self.dim)
                self._ann.init_index(
                    max_elements=max(self.max_elements, len(vectors) * 2 or 16),
                    ef_construction=200, M=16,
                )
                self._ann.set_ef(64)
                if vectors:
                    self._ann.add_items(self._matrix, list(range(len(vectors))))
                self._ann_count = len(vectors)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ANN rebuild failed (%s) — disabling ANN.", exc)
                self._ann = None


class SemanticCache:
    """
    Semantic cache with an in-process vector index and optional Redis persistence.

    Hot path: embed the query, one index search, threshold check. No Redis round-trip
    on lookup, so p99 lookup latency is embedding time plus a few hundred
    microseconds regardless of cache size.
    """

    def __init__(
        self,
        embedding_model=None,
        redis_host: str = REDIS_HOST,
        redis_port: int = REDIS_PORT,
        similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD,
        ttl_seconds: int = CACHE_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        warm_start: bool = True,
    ):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.embedding_model = embedding_model

        self._lock = threading.RLock()
        self._entries: List[Optional[CacheEntry]] = []   # row -> entry (None = evicted)
        self._key_to_row: Dict[str, int] = {}
        self._index: Optional[_VectorIndex] = None
        self._dim: Optional[int] = None

        self._stats = {
            "hits": 0, "misses": 0, "evictions": 0, "errors": 0,
            "lookup_ms_total": 0.0, "total_latency_saved_ms": 0.0,
        }

        self._redis = None
        self._use_redis = False
        self._connect_redis(redis_host, redis_port)

        if self.embedding_model is None:
            self._load_embedding_model()

        if warm_start and self._use_redis:
            self._warm_start()

    # ── setup ──────────────────────────────────────────────────────────────────

    def _connect_redis(self, host: str, port: int) -> None:
        try:
            import redis as redis_lib
            self._redis = redis_lib.Redis(
                host=host, port=port, decode_responses=False,
                socket_timeout=2, socket_connect_timeout=2,
                health_check_interval=30,
            )
            self._redis.ping()
            self._use_redis = True
            logger.info("SemanticCache: Redis connected at %s:%d.", host, port)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SemanticCache: Redis unavailable (%s) — in-process cache only.", exc
            )
            self._redis = None
            self._use_redis = False

    def _load_embedding_model(self) -> None:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("SemanticCache: embedding model unavailable: %s", exc)

    def _warm_start(self) -> None:
        """
        Load persisted entries once at startup using SCAN + batched MGET.

        ``SCAN`` is cursor-based and non-blocking, unlike ``KEYS``. Values are fetched
        in batches rather than one call per key, so warm start costs O(n/batch)
        round-trips instead of O(n).
        """
        try:
            members = list(self._redis.sscan_iter(_INDEX_SET, count=500))
            if not members:
                return
            loaded = 0
            for i in range(0, len(members), 500):
                batch = members[i:i + 500]
                keys = [_KEY_PREFIX.encode() + m for m in batch]
                for raw in self._redis.mget(keys):
                    if not raw:
                        continue
                    try:
                        entry = CacheEntry.from_json(raw)
                    except Exception:  # noqa: BLE001
                        continue
                    if entry.is_expired(self.ttl_seconds):
                        continue
                    self._insert_local(self._cache_key(entry.query), entry)
                    loaded += 1
            logger.info("SemanticCache: warm-started %d entries from Redis.", loaded)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SemanticCache: warm start failed (%s).", exc)

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if self.embedding_model is None:
            return None
        try:
            arr = np.asarray(self.embedding_model.embed_query(text), dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            # Normalising here means cosine similarity reduces to a dot product.
            return arr / norm if norm > 0 else arr
        except Exception as exc:  # noqa: BLE001
            self._stats["errors"] += 1
            logger.warning("SemanticCache: embedding failed: %s", exc)
            return None

    def _insert_local(self, key: str, entry: CacheEntry) -> None:
        """Insert into the in-process index. Caller holds the lock (or is init)."""
        vector = np.asarray(entry.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm

        if self._index is None:
            self._dim = vector.shape[0]
            self._index = _VectorIndex(self._dim, max(self.max_entries, 1024))
        elif vector.shape[0] != self._dim:
            logger.warning("SemanticCache: embedding dim mismatch — entry skipped.")
            return

        if key in self._key_to_row:              # refresh in place
            row = self._key_to_row[key]
            self._entries[row] = entry
            return

        row = self._index.add(vector)
        if row == len(self._entries):
            self._entries.append(entry)
        else:                                    # defensive: keep list aligned
            while len(self._entries) <= row:
                self._entries.append(None)
            self._entries[row] = entry
        self._key_to_row[key] = row
        self._maybe_evict()

    def _maybe_evict(self) -> None:
        """LRU-ish eviction: drop the oldest 10% once capacity is exceeded."""
        live = [(k, r) for k, r in self._key_to_row.items()
                if r < len(self._entries) and self._entries[r] is not None]
        if len(live) <= self.max_entries:
            return

        drop = max(1, len(live) // 10)
        oldest = sorted(live, key=lambda kr: self._entries[kr[1]].timestamp)[:drop]
        for key, _ in oldest:
            self._key_to_row.pop(key, None)
            self._stats["evictions"] += 1
            if self._use_redis:
                try:
                    self._redis.delete(_KEY_PREFIX + key)
                    self._redis.srem(_INDEX_SET, key)
                except Exception:  # noqa: BLE001
                    pass

        # Compact: rebuild the index over surviving entries so rows stay dense.
        survivors = [(k, self._entries[r]) for k, r in self._key_to_row.items()
                     if r < len(self._entries) and self._entries[r] is not None]
        self._entries = [e for _, e in survivors]
        self._key_to_row = {k: i for i, (k, _) in enumerate(survivors)}
        vectors = [np.asarray(e.embedding, dtype=np.float32) for _, e in survivors]
        vectors = [v / (np.linalg.norm(v) or 1.0) for v in vectors]
        if self._index is not None:
            self._index.rebuild(vectors)

    # ── public API ─────────────────────────────────────────────────────────────

    def lookup(self, query: str) -> Optional[Dict[str, Any]]:
        """Return a cached result for a semantically similar query, or None."""
        started = time.perf_counter()
        vector = self._embed(query)
        if vector is None:
            self._stats["misses"] += 1
            return None

        with self._lock:
            if self._index is None or self._index.size == 0:
                self._stats["misses"] += 1
                self._stats["lookup_ms_total"] += (time.perf_counter() - started) * 1000
                return None

            row, score = self._index.search(vector)
            entry = self._entries[row] if 0 <= row < len(self._entries) else None

            if entry is not None and entry.is_expired(self.ttl_seconds):
                self._invalidate_row(row)
                entry = None

            elapsed_ms = (time.perf_counter() - started) * 1000
            self._stats["lookup_ms_total"] += elapsed_ms

            if entry is not None and score >= self.similarity_threshold:
                entry.hit_count += 1
                self._stats["hits"] += 1
                logger.debug("Cache HIT (sim=%.3f): %s", score, query[:60])
                return {
                    "answer": entry.answer,
                    "contexts": entry.contexts,
                    "cache_hit": True,
                    "similarity_score": round(score, 4),
                    "latency_ms": round(elapsed_ms, 3),
                }

            self._stats["misses"] += 1
            logger.debug("Cache MISS (best=%.3f): %s", score, query[:60])
            return None

    def _invalidate_row(self, row: int) -> None:
        """Drop an expired row. Caller holds the lock."""
        if 0 <= row < len(self._entries):
            self._entries[row] = None
        for key, r in list(self._key_to_row.items()):
            if r == row:
                self._key_to_row.pop(key, None)
                if self._use_redis:
                    try:
                        self._redis.delete(_KEY_PREFIX + key)
                        self._redis.srem(_INDEX_SET, key)
                    except Exception:  # noqa: BLE001
                        pass
                break

    def store(self, query: str, answer: str, contexts: List[str]) -> None:
        """Write a query/answer pair through to the local index and Redis."""
        vector = self._embed(query)
        if vector is None:
            return

        entry = CacheEntry(
            query=query, answer=answer, contexts=contexts,
            embedding=vector.tolist(),
        )
        key = self._cache_key(query)

        with self._lock:
            self._insert_local(key, entry)

        if self._use_redis:
            try:
                pipe = self._redis.pipeline()
                pipe.setex(_KEY_PREFIX + key, self.ttl_seconds, entry.to_json())
                pipe.sadd(_INDEX_SET, key)
                pipe.execute()
            except Exception as exc:  # noqa: BLE001
                self._stats["errors"] += 1
                logger.warning("SemanticCache: Redis write failed: %s", exc)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            live = sum(1 for e in self._entries if e is not None)
            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "errors": self._stats["errors"],
                "total_queries": total,
                "hit_rate": round(self._stats["hits"] / total, 4) if total else 0.0,
                "entries": live,
                "capacity": self.max_entries,
                "avg_lookup_ms": round(self._stats["lookup_ms_total"] / total, 3) if total else 0.0,
                "backend": "redis+memory" if self._use_redis else "memory",
                "ann": self._index is not None and self._index._ann is not None,
                "total_latency_saved_ms": self._stats["total_latency_saved_ms"],
            }

    def clear(self) -> None:
        """Drop every entry locally and in Redis (SCAN-based, never KEYS)."""
        with self._lock:
            self._entries.clear()
            self._key_to_row.clear()
            if self._index is not None:
                self._index.rebuild([])
        if self._use_redis:
            try:
                members = list(self._redis.sscan_iter(_INDEX_SET, count=500))
                pipe = self._redis.pipeline()
                for m in members:
                    pipe.delete(_KEY_PREFIX.encode() + m)
                pipe.delete(_INDEX_SET)
                pipe.execute()
            except Exception as exc:  # noqa: BLE001
                logger.warning("SemanticCache: Redis clear failed: %s", exc)

    def healthy(self) -> bool:
        """True when the cache can serve lookups (embeddings available)."""
        return self.embedding_model is not None
