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
* **Redis persists entries and owns shared generation authority.** It is read at startup
  (via ``SCAN`` + batched ``MGET``, never ``KEYS``) and written through on store,
  so multiple workers share entries and survive restarts. A small generation token
  is polled to invalidate a warm local index after another worker clears or updates
  the namespace; a changed generation triggers a fresh ``SCAN``/``MGET``. If Redis
   is down, ``redis_failure_policy='local'`` retains availability but may return
   revoked answers. ServingContext uses ``'miss'`` to reject unverified snapshots.
* **Bounded and expiring.** Capacity is capped with LRU eviction and entries carry a
  TTL, so memory cannot grow without limit.
* **Thread-safe.** A reentrant lock guards index mutation, because FastAPI serves
  requests from a thread pool.

Legacy availability lookups can disable snapshot polling. Strict ``'miss'``
lookups and the request-token API always verify shared revocation authority;
polling opt-outs do not weaken the supported API/UI/wrapper boundary. Ordinary
stores rotate the snapshot version only; atomic clear rotates a separate random
revocation epoch. Request tokens are captured before generation and Redis Lua
conditionally publishes against that epoch. Direct ``store`` is a compatibility
operation and cannot fence work that happened before its call.

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
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import (  # noqa: E402
    CACHE_SIMILARITY_THRESHOLD, CACHE_TTL_SECONDS, EMBEDDING_MODEL,
    REDIS_HOST, REDIS_PORT,
)

logger = logging.getLogger(__name__)

# Redis key namespace. The index set lets us enumerate entries without KEYS.
_KEY_PREFIX = "rag_cache:v4:default:entry:"
_INDEX_SET = "rag_cache:v4:default:index"
_GENERATION_SUFFIX = "generation"

# Cap the in-process matrix. 50k x 384 float32 is ~77 MB, a reasonable ceiling for
# a single worker; beyond this the LRU evicts.
DEFAULT_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "50000"))


class GenerationFencedError(RuntimeError):
    """Work was revoked, changed scope, or could not be verified. Do not serve it."""


def require_generation_cache(cache):
    """Reject old cache adapters at serving boundaries, rather than unsafe fallback."""
    if not all(callable(getattr(cache, name, None)) for name in (
        "begin_generation", "lookup_generation", "publish_generation", "validate_generation",
    )):
        raise GenerationFencedError("Cache adapter does not support generation fencing.")
    return cache


@dataclass(frozen=True)
class GenerationToken:
    """Request-start authority; snapshot versions deliberately are NOT authority.

    Tokens are instance-bound and include the namespace and index/config identity.
    A successful ordinary publication changes only the snapshot version, so other
    outstanding generations remain valid. Clear rotates random revocation epochs.
    """

    instance: str
    namespace: str
    index_identity: str
    local_epoch: str
    shared_epoch: Optional[str]


_BEGIN_GENERATION_LUA = """-- fencing:begin
local epoch = redis.call('GET', KEYS[1])
if not epoch then
    epoch = ARGV[1]
    redis.call('SET', KEYS[1], epoch)
    redis.call('SET', KEYS[2], ARGV[2])
end
return epoch
"""

_PUBLISH_GENERATION_LUA = """-- fencing:publish
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
if tonumber(ARGV[4]) > 0 then
    redis.call('SET', KEYS[3], ARGV[3], 'EX', ARGV[4])
else
    redis.call('SET', KEYS[3], ARGV[3])
end
redis.call('SADD', KEYS[4], ARGV[2])
redis.call('SET', KEYS[2], ARGV[5])
return 1
"""

_CLEAR_GENERATION_LUA = """-- fencing:clear
for _, member in ipairs(redis.call('SMEMBERS', KEYS[1])) do
    redis.call('DEL', ARGV[1] .. member)
end
redis.call('DEL', KEYS[1])
redis.call('SET', KEYS[2], ARGV[2])
redis.call('SET', KEYS[3], ARGV[3])
return 1
"""


@dataclass
class CacheEntry:
    query: str
    answer: str
    contexts: List[str]
    embedding: List[float]
    citations: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    hit_count: int = 0
    revocation_epoch: Optional[str] = None

    def is_expired(self, ttl_seconds: int) -> bool:
        return ttl_seconds > 0 and (time.time() - self.timestamp) > ttl_seconds

    def to_json(self) -> bytes:
        return json.dumps({
            "query": self.query,
            "answer": self.answer,
            "contexts": self.contexts,
            "citations": self.citations,
            "embedding": self.embedding,
            "timestamp": self.timestamp,
            "last_access": self.last_access,
            "hit_count": self.hit_count,
            "revocation_epoch": self.revocation_epoch,
        }).encode()

    @classmethod
    def from_json(cls, raw: bytes) -> "CacheEntry":
        data = json.loads(raw.decode())
        return cls(
            query=data["query"],
            answer=data["answer"],
            contexts=data.get("contexts", []),
            citations=data.get("citations", []),
            embedding=data["embedding"],
            timestamp=data.get("timestamp", time.time()),
            last_access=data.get("last_access", data.get("timestamp", time.time())),
            hit_count=data.get("hit_count", 0),
            revocation_epoch=data.get("revocation_epoch"),
        )


class _VectorIndex:
    """
    Similarity index over unit-norm embeddings.

    Uses hnswlib (approximate, sub-linear) when available and falls back to an exact
    vectorised brute-force search otherwise. Both paths return the same
    ``(row, score)`` contract, so behaviour is identical apart from recall on very
    large caches. Brute force is not a naive loop: one ``matrix @ vector`` product
    scores every entry at BLAS speed.

    The backing matrix is a capacity-doubling buffer: appending amortises to O(1)
    instead of copying the whole matrix on every insert (np.vstack each add is
    O(n) per insert, O(n²) overall — ~76 MB of memcpy per insert at 50k entries).
    """

    def __init__(self, dim: int, max_elements: int):
        self.dim = dim
        self.max_elements = max_elements
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        self._count = 0
        # Capacity for the dense buffer; allocated lazily on first add because
        # __init__ does not know the embedding width until then.
        self._capacity = 0
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
        return self._count

    def _grow(self, needed: int) -> None:
        """Ensure the dense buffer can hold ``needed`` rows (amortised doubling)."""
        if needed <= self._capacity:
            return
        new_cap = max(needed, self._capacity * 2 or 1024)
        grown = np.zeros((new_cap, self.dim), dtype=np.float32)
        grown[: self._count] = self._matrix[: self._count]
        self._matrix = grown
        self._capacity = new_cap

    def add(self, vector: np.ndarray) -> int:
        """Append a vector, returning its row id."""
        row = self._count
        self._grow(row + 1)
        self._matrix[row] = vector.reshape(-1)
        self._count = row + 1
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

    def replace(self, row: int, vector: np.ndarray) -> None:
        """Replace an existing vector, rebuilding ANN state when necessary."""
        if not 0 <= row < self._count:
            raise IndexError(f"vector row {row} is outside the index")
        self._matrix[row] = vector.reshape(-1)
        if self._ann is not None:
            # hnswlib does not allow replacing an item unless the index was
            # created with allow_replace_deleted=True. Rebuilding is rare (only
            # an exact-key refresh) and keeps ANN and dense fallback in sync.
            self.rebuild([self._matrix[i].copy() for i in range(self._count)])

    def search(self, vector: np.ndarray) -> Tuple[int, float]:
        """Return ``(best_row, cosine_score)``; ``(-1, 0.0)`` when empty."""
        if self._count == 0:
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
        scores = self._matrix[: self._count] @ vector
        best = int(np.argmax(scores))
        return best, float(scores[best])

    def rebuild(self, vectors: List[np.ndarray]) -> None:
        """Rebuild from scratch (used after eviction compaction)."""
        self._count = len(vectors)
        if vectors:
            self._grow(len(vectors))
            if vectors:
                self._matrix[: self._count] = np.vstack(
                    [v.reshape(1, -1) for v in vectors]
                ).astype(np.float32)
        else:
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)
            self._capacity = 0
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
                    self._ann.add_items(self._matrix[: self._count], list(range(len(vectors))))
                self._ann_count = len(vectors)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ANN rebuild failed (%s) — disabling ANN.", exc)
                self._ann = None


class SemanticCache:
    """
    Semantic cache with an in-process vector index and optional Redis persistence.

    Hot path: embed the query, check the namespace generation, then perform one
    local index search. A generation change is the exceptional path that reloads
    Redis entries; ``generation_poll_interval=None`` removes even that check and
    explicitly accepts restart-only cross-worker invalidation.
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
        namespace: str = "",
        generation_poll_interval: Optional[float] = 0.0,
        redis_failure_policy: str = "local",
        index_identity: Optional[str] = None,
        identity_provider: Optional[Callable[[], str]] = None,
    ):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")
        if generation_poll_interval is not None and generation_poll_interval < 0:
            raise ValueError("generation_poll_interval must be non-negative or None")
        if redis_failure_policy not in {"local", "miss"}:
            raise ValueError("redis_failure_policy must be 'local' or 'miss'")
        self._redis_failure_policy = redis_failure_policy
        self._generation_verified = False
        self._reload_required = False
        self._pending_generation = None
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.embedding_model = embedding_model
        # Never mix answers from different architectures, corpora, or generators.
        # Hash the public configuration namespace, not credentials or query text.
        scope = hashlib.sha256(namespace.encode()).hexdigest() if namespace else ""
        self._namespace_identity = scope or "default"
        self._instance_identity = uuid.uuid4().hex
        self._local_epoch = uuid.uuid4().hex
        self._index_identity = index_identity if index_identity is not None else namespace
        self._identity_provider = identity_provider
        self._key_prefix = f"rag_cache:v4:{scope}:entry:" if scope else _KEY_PREFIX
        self._index_set = f"rag_cache:v4:{scope}:index" if scope else _INDEX_SET
        self._generation_key = (
            f"rag_cache:v4:{scope}:{_GENERATION_SUFFIX}" if scope
            else f"rag_cache:v4:default:{_GENERATION_SUFFIX}"
        )
        self._revocation_key = f"rag_cache:v4:{self._namespace_identity}:revocation"
        self._generation_poll_interval = generation_poll_interval
        self._generation: Optional[Any] = None
        self._generation_last_checked = 0.0
        self._generation_supported = True

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

        self._initialize_generation()
        if warm_start and self._use_redis:
            self._warm_start()

    # ── setup ──────────────────────────────────────────────────────────────────

    def _connect_redis(self, host: str, port: int) -> None:
        try:
            import redis as redis_lib
            self._redis = redis_lib.Redis(
                host=host, port=port, decode_responses=False,
                socket_timeout=0.5, socket_connect_timeout=0.5,
                health_check_interval=30,
            )
            # Keep the reconnecting client even when the initial PING fails.
            self._use_redis = True
            self._generation_supported = self._supports_generation_commands()
            self._redis.ping()
            logger.info("SemanticCache: Redis connected at %s:%d.", host, port)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SemanticCache: Redis unavailable (%s) — in-process cache only.", exc
            )
            self._use_redis = False
            self._generation_supported = False

    def _load_embedding_model(self) -> None:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("SemanticCache: embedding model unavailable: %s", exc)

    def _supports_generation_commands(self) -> bool:
        """Return whether this Redis client exposes the generation contract."""
        return bool(
            self._redis is not None
            and callable(getattr(self._redis, "get", None))
            and callable(getattr(self._redis, "set", None))
        )

    @staticmethod
    def _generation_value(raw: Any) -> Optional[Any]:
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode(errors="replace")
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
        return str(raw)

    def _read_generation(self) -> Tuple[bool, Optional[Any]]:
        """Read the namespace generation, distinguishing outage from a missing key."""
        if not self._use_redis or not self._generation_supported:
            return False, None
        try:
            value = self._generation_value(self._redis.get(self._generation_key))
            self._generation_verified = True
            return True, value
        except AttributeError:
            # A lightweight Redis test double or older adapter may not implement
            # the contract. Keep the legacy local-only behavior explicitly.
            self._generation_supported = False
            self._generation_verified = False
            return False, None
        except Exception as exc:  # noqa: BLE001
            self._generation_verified = False
            self._reload_required = True
            self._stats["errors"] += 1
            logger.warning("SemanticCache: generation read failed: %s", exc)
            return False, None

    def _initialize_generation(self) -> None:
        if not self._use_redis or not self._generation_supported:
            return
        ok, generation = self._read_generation()
        if ok:
            self._generation = generation
            self._generation_last_checked = time.monotonic()

    def _generation_poll_due(self, force: bool = False) -> bool:
        if force:
            return True
        if self._generation_poll_interval is None:
            return False
        if self._generation_poll_interval == 0:
            return True
        return (
            time.monotonic() - self._generation_last_checked
            >= self._generation_poll_interval
        )

    def _clear_local(self) -> None:
        """Drop the local index without issuing any Redis deletion commands."""
        self._entries.clear()
        self._key_to_row.clear()
        if self._index is not None:
            self._index.rebuild([])

    def _sync_generation(self, force: bool = False) -> bool:
        """Invalidate and reload local state when the shared generation changes.

        The check is intentionally performed while the cache lock is held by the
        caller. A Redis outage leaves the last known generation and local answers
        intact; treating an unavailable read as generation ``None`` would silently
        discard valid cache state during a transient outage.
        """
        if not self._use_redis or not self._generation_supported:
            return False
        if not self._generation_poll_due(force=force):
            return False
        ok, current = self._read_generation()
        if not ok:
            return False
        self._generation_last_checked = time.monotonic()
        if current == self._generation and not self._reload_required:
            return False

        self._clear_local()
        if not self._warm_start():
            # Leave the old token in place so a later lookup retries the reload
            # rather than permanently accepting an empty local index.
            return True
        return True

    def _record_generation_from_pipeline(self, results: Any) -> None:
        """Update the local generation after a successful write-through."""
        generation = self._pending_generation
        if generation is None:
            ok, generation = self._read_generation()
            if not ok:
                return
        self._generation = generation
        self._generation_last_checked = time.monotonic()
        # Another writer may have committed between our pre-write sync and our
        # transaction. The returned token alone does not prove our snapshot is complete.
        self._reload_required = True

    def _queue_generation_increment(self, pipe: Any) -> bool:
        """Append an atomic generation increment when the client supports it."""
        if not self._generation_supported:
            return False
        if not callable(getattr(pipe, "set", None)):
            self._generation_supported = False
            return False
        # Random epochs avoid ABA after a nonpersistent Redis restart (INCR can
        # return the same counter value for entirely different contents).
        self._pending_generation = uuid.uuid4().hex
        pipe.set(self._generation_key, self._pending_generation)
        return True

    def _warm_start(self) -> bool:
        """Accept only a snapshot bracketed by the same generation token.

        SCAN/MGET is not a transaction. A clear/update midway through loading
        must never leave a partial or revoked snapshot marked current.
        """
        if not self._generation_supported:
            return self._load_persisted_entries()
        for _ in range(3):
            ok, before = self._read_generation()
            if not ok:
                break
            self._clear_local()
            if not self._load_persisted_entries():
                break
            ok, after = self._read_generation()
            if ok and before == after:
                self._generation = after
                self._reload_required = False
                return True
        self._clear_local()
        self._generation_verified = False
        self._reload_required = True
        return False

    def _load_persisted_entries(self) -> bool:
        """
        Load persisted entries once at startup using SCAN + batched MGET.

        ``SCAN`` is cursor-based and non-blocking, unlike ``KEYS``. Values are fetched
        in batches rather than one call per key, so warm start costs O(n/batch)
        round-trips instead of O(n).
        """
        try:
            epoch = None
            if callable(getattr(self._redis, "eval", None)):
                raw_epoch = self._redis.get(self._revocation_key)
                if raw_epoch is None:
                    return False
                epoch = raw_epoch.decode() if isinstance(raw_epoch, bytes) else str(raw_epoch)
            members = list(self._redis.sscan_iter(self._index_set, count=500))
            if not members:
                return True
            loaded = 0
            for i in range(0, len(members), 500):
                batch = members[i:i + 500]
                keys = [
                    self._key_prefix.encode()
                    + (m.encode() if isinstance(m, str) else m)
                    for m in batch
                ]
                for raw in self._redis.mget(keys):
                    if not raw:
                        continue
                    try:
                        entry = CacheEntry.from_json(raw)
                    except Exception:  # noqa: BLE001
                        continue
                    if entry.is_expired(self.ttl_seconds):
                        continue
                    if epoch is not None and entry.revocation_epoch != epoch:
                        continue
                    self._insert_local(self._cache_key(entry.query), entry)
                    loaded += 1
            logger.info("SemanticCache: warm-started %d entries from Redis.", loaded)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("SemanticCache: warm start failed (%s).", exc)
            return False

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
            self._index.replace(row, vector)
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

    def _compact_local(self) -> None:
        """Remove tombstones and rebuild the vector index over live entries."""
        survivors = [(k, self._entries[r]) for k, r in self._key_to_row.items()
                     if r < len(self._entries) and self._entries[r] is not None]
        self._entries = [e for _, e in survivors]
        self._key_to_row = {k: i for i, (k, _) in enumerate(survivors)}
        vectors = [np.asarray(e.embedding, dtype=np.float32) for _, e in survivors]
        vectors = [v / (np.linalg.norm(v) or 1.0) for v in vectors]
        if self._index is not None:
            self._index.rebuild(vectors)

    def _maybe_evict(self) -> None:
        """LRU-ish eviction: drop the oldest 10% once capacity is exceeded."""
        live = [(k, r) for k, r in self._key_to_row.items()
                if r < len(self._entries) and self._entries[r] is not None]
        if len(live) <= self.max_entries:
            return

        drop = max(1, len(live) // 10)
        oldest = sorted(live, key=lambda kr: self._entries[kr[1]].last_access)[:drop]
        for key, _ in oldest:
            self._key_to_row.pop(key, None)
            self._stats["evictions"] += 1
            # Capacity eviction is local. A stale reader must not delete a newer
            # persisted answer owned by another worker; Redis TTL owns expiry.

        self._compact_local()

    # ── public API ─────────────────────────────────────────────────────────────

    def _check_identity(self) -> None:
        if self._identity_provider is not None:
            try:
                current = self._identity_provider()
            except Exception as exc:
                raise GenerationFencedError("Index/config identity is unavailable.") from exc
            if current != self._index_identity:
                self._clear_local()
                raise GenerationFencedError("Index/config identity changed; restart serving resources.")

    def begin_generation(self) -> GenerationToken:
        """Capture BEFORE retrieval, lazy construction, embedding or generation.

        Strict shared fencing requires Redis EVAL. Local-only fencing is available
        only with the explicit availability policy and no configured Redis client;
        legacy adapters never silently become a supported serving boundary.
        """
        with self._lock:
            self._check_identity()
            shared_epoch = None
            if self._use_redis:
                try:
                    raw = self._redis.eval(
                        _BEGIN_GENERATION_LUA, 2, self._revocation_key,
                        self._generation_key, uuid.uuid4().hex, uuid.uuid4().hex,
                    )
                    if not raw:
                        raise ValueError("missing revocation epoch")
                    shared_epoch = raw.decode() if isinstance(raw, bytes) else str(raw)
                except Exception as exc:
                    self._generation_verified = False
                    self._reload_required = True
                    self._stats["errors"] += 1
                    raise GenerationFencedError("Shared generation authority unavailable.") from exc
            elif self._redis_failure_policy != "local":
                raise GenerationFencedError("Shared generation authority unavailable.")
            return GenerationToken(f"{self._instance_identity}:{os.getpid()}", self._namespace_identity,
                                   self._index_identity, self._local_epoch, shared_epoch)

    def validate_generation(self, token: GenerationToken) -> None:
        """Fail closed; a read is the response-validation linearization point.

        This cannot retract bytes already returned before a concurrent clear. A
        clear ordered before this check must fence even a non-cached response.
        """
        with self._lock:
            self._check_identity()
            if not isinstance(token, GenerationToken) or (
                token.instance != f"{self._instance_identity}:{os.getpid()}"
                or token.namespace != self._namespace_identity
                or token.index_identity != self._index_identity
                or token.local_epoch != self._local_epoch
            ):
                raise GenerationFencedError("Generation scope was revoked or does not match.")
            if token.shared_epoch is not None:
                try:
                    raw = self._redis.get(self._revocation_key)
                    current = raw.decode() if isinstance(raw, bytes) else raw
                except Exception as exc:
                    self._generation_verified = False
                    self._reload_required = True
                    raise GenerationFencedError("Shared generation verification unavailable.") from exc
                if current != token.shared_epoch:
                    self._clear_local()
                    self._reload_required = True
                    raise GenerationFencedError("Generation was revoked.")
            elif self._use_redis or self._redis_failure_policy != "local":
                raise GenerationFencedError("Generation authority changed.")

    def lookup_generation(self, token: GenerationToken, query: str) -> Optional[Dict[str, Any]]:
        """Request-bound lookup; polling opt-outs cannot bypass revocation checks."""
        with self._lock:
            self.validate_generation(token)
            self._sync_generation(force=True)
            if token.shared_epoch is not None and not self._generation_verified:
                raise GenerationFencedError("Shared cache snapshot is unavailable.")
            result = self._lookup(query)
            self.validate_generation(token)
            return result

    def publish_generation(
        self, token: GenerationToken, query: str, answer: str,
        contexts: List[str], citations: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Atomically compare revocation and publish, or raise; never false success.

        In shared mode local rows are loaded only from a verified snapshot, not
        optimistically inserted before the conditional Redis commit. An ambiguous
        write may have committed; it is never retried or acknowledged as success.
        """
        self.validate_generation(token)
        vector = self._embed(query)
        if vector is None:
            raise GenerationFencedError("Generation cache embedding unavailable.")
        entry = CacheEntry(query=query, answer=answer, contexts=contexts,
                           citations=citations or [], embedding=vector.tolist(),
                           revocation_epoch=token.shared_epoch)
        key = self._cache_key(query)
        with self._lock:
            self.validate_generation(token)
            if token.shared_epoch is None:
                self._insert_local(key, entry)
            else:
                try:
                    accepted = self._redis.eval(
                        _PUBLISH_GENERATION_LUA, 4, self._revocation_key,
                        self._generation_key, self._key_prefix + key, self._index_set,
                        token.shared_epoch, key, entry.to_json(), self.ttl_seconds,
                        uuid.uuid4().hex,
                    )
                except Exception as exc:
                    self._clear_local()
                    self._generation_verified = False
                    self._reload_required = True
                    self._stats["errors"] += 1
                    raise GenerationFencedError("Shared publication could not be confirmed.") from exc
                self._reload_required = True
                if accepted != 1:
                    self._clear_local()
                    raise GenerationFencedError("Generation was revoked before publication.")
            self.validate_generation(token)
            return True

    def lookup(self, query: str) -> Optional[Dict[str, Any]]:
        """Strict policies bracket even standalone lookups with request authority."""
        if self._redis_failure_policy == "miss":
            try:
                return self.lookup_generation(self.begin_generation(), query)
            except GenerationFencedError:
                self._stats["misses"] += 1
                return None
        return self._lookup(query)

    def _lookup(self, query: str) -> Optional[Dict[str, Any]]:
        """Return a cached result for a semantically similar query, or None."""
        started = time.perf_counter()
        vector = self._embed(query)
        if vector is None:
            self._stats["misses"] += 1
            return None

        with self._lock:
            try:
                self._check_identity()
            except GenerationFencedError:
                self._stats["misses"] += 1
                return None
            self._sync_generation(force=self._redis_failure_policy == "miss")
            if self._redis_failure_policy == "miss" and not self._generation_verified:
                self._stats["misses"] += 1
                return None
            if self._index is None or self._index.size == 0:
                self._stats["misses"] += 1
                self._stats["lookup_ms_total"] += (time.perf_counter() - started) * 1000
                return None

            # Expired/tombstoned rows must not hide the next-best live entry.
            # Search again after compaction rather than returning a false miss.
            row, score, entry = -1, 0.0, None
            while self._index is not None and self._index.size:
                row, score = self._index.search(vector)
                entry = self._entries[row] if 0 <= row < len(self._entries) else None
                if entry is not None and not entry.is_expired(self.ttl_seconds):
                    break
                if entry is not None:
                    self._invalidate_row(row)
                entry = None
                self._compact_local()

            elapsed_ms = (time.perf_counter() - started) * 1000
            self._stats["lookup_ms_total"] += elapsed_ms

            if entry is not None and score >= self.similarity_threshold:
                entry.hit_count += 1
                entry.last_access = time.time()
                self._stats["hits"] += 1
                logger.debug("Cache HIT (sim=%.3f): %s", score, query[:60])
                return {
                    "answer": entry.answer,
                    "contexts": entry.contexts,
                    "citations": entry.citations,
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
                break

    def store(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Legacy immediate store, NOT a request-start generation boundary.

        Kept for the CLI/demo and old adapters. Supported serving callers MUST use
        begin_generation/lookup_generation/publish_generation instead. Capturing a
        token here cannot revoke generation that happened before this call.
        """
        if self._use_redis and callable(getattr(self._redis, "eval", None)):
            try:
                token = self.begin_generation()
                self.publish_generation(token, query, answer, contexts, citations)
                with self._lock:
                    self._warm_start()
            except GenerationFencedError:
                return
            return
        vector = self._embed(query)
        if vector is None:
            return

        entry = CacheEntry(
            query=query, answer=answer, contexts=contexts,
            citations=citations or [],
            embedding=vector.tolist(),
        )
        key = self._cache_key(query)

        with self._lock:
            self._sync_generation(force=True)
            self._insert_local(key, entry)
            # Serialize write-through with local writes/clear. Otherwise a delayed
            # old store can overwrite a newer refresh in Redis after local publish.
            if self._use_redis:
                try:
                    pipe = self._redis.pipeline()
                    if self.ttl_seconds:
                        pipe.setex(self._key_prefix + key, self.ttl_seconds, entry.to_json())
                    else:
                        pipe.set(self._key_prefix + key, entry.to_json())
                    pipe.sadd(self._index_set, key)
                    generation_queued = self._queue_generation_increment(pipe)
                    results = pipe.execute()
                    if generation_queued:
                        self._record_generation_from_pipeline(results)
                except Exception as exc:  # noqa: BLE001
                    self._generation_verified = False
                    self._reload_required = True
                    self._stats["errors"] += 1
                    logger.warning("SemanticCache: Redis write failed: %s", exc)

    def store_with_citations(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        citations: List[Dict[str, Any]],
    ) -> None:
        """Explicit citation-aware alias for integrations with old cache doubles."""
        self.store(query, answer, contexts, citations=citations)

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
                "backend": "redis+memory" if self._use_redis and self._generation_verified else "memory",
                "generation": self._generation,
                "fencing": "atomic-redis" if self._use_redis and callable(getattr(self._redis, "eval", None)) else "legacy-local",
                "generation_verified": self._generation_verified,
                "redis_failure_policy": self._redis_failure_policy,
                "generation_polling": bool(
                    self._generation_supported
                    and self._generation_poll_interval is not None
                ),
                "ann": self._index is not None and self._index._ann is not None,
                "total_latency_saved_ms": self._stats["total_latency_saved_ms"],
            }

    def clear(self) -> bool:
        """Clear this namespace; False means shared revocation was NOT confirmed."""
        with self._lock:
            self._local_epoch = uuid.uuid4().hex
            self._clear_local()
            self._generation_verified = False
            if self._use_redis:
                try:
                    if callable(getattr(self._redis, "eval", None)):
                        token = uuid.uuid4().hex
                        # Atomic with concurrent stores: scanning outside the
                        # transaction can strand a newly added index member.
                        receipt = self._redis.eval(
                            _CLEAR_GENERATION_LUA, 3, self._index_set,
                            self._generation_key, self._revocation_key,
                            self._key_prefix, token, uuid.uuid4().hex,
                        )
                        if receipt != 1:
                            raise RuntimeError("Shared clear was not acknowledged")
                        # Do not adopt our token: another clear/write may already
                        # have committed. The next read reloads shared state.
                        self._reload_required = True
                        return True
                    if self._redis_failure_policy == "miss":
                        raise RuntimeError("Shared clear requires atomic Redis support")
                    members = list(self._redis.sscan_iter(self._index_set, count=500))
                    pipe = self._redis.pipeline()
                    for m in members:
                        member = m.encode() if isinstance(m, str) else m
                        pipe.delete(self._key_prefix.encode() + member)
                    pipe.delete(self._index_set)
                    generation_queued = self._queue_generation_increment(pipe)
                    results = pipe.execute()
                    if generation_queued:
                        self._record_generation_from_pipeline(results)
                except Exception as exc:  # noqa: BLE001
                    self._generation_verified = False
                    self._reload_required = True
                    logger.warning("SemanticCache: Redis clear failed: %s", exc)
                    return False
            return self._redis_failure_policy == "local" or self._use_redis

    def refresh_generation(self) -> bool:
        """Force a shared-generation check and reload local state if it changed."""
        with self._lock:
            return self._sync_generation(force=True)

    def healthy(self) -> bool:
        """True when the cache can serve lookups (embeddings available)."""
        return self.embedding_model is not None
