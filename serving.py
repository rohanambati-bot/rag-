# serving.py
"""
Single source of truth for serving-side resources.
──────────────────────────────────────────────────
Previously ``api.py`` and ``app.py`` each built their own retrieval stack, and both
diverged from the CLI in two ways that made the served system a *different system*
from the evaluated one:

* They bootstrapped documents from **SQuAD** while the CLI and the persisted Chroma
  index were built from **MS MARCO**. The BM25 index therefore covered a different
  corpus than the dense index it was fused with.
* They registered **4** architectures out of the 12 the framework implements, so most
  of the project was unreachable over HTTP.

Everything now goes through this module, which owns:

* the embedding model, vector store and chunk corpus (MS MARCO, matching the CLI),
* the full architecture registry from ``build_all_architectures``,
* the semantic cache,
* readiness state, so ``/readyz`` reflects whether the stack is genuinely usable.

Resources are built once per process and reused, guarded by a lock so concurrent
requests cannot trigger duplicate model loads.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import (  # noqa: E402
    DATASET_SUBSET_SIZE, VECTOR_STORE_DIR,
)
from observability import METRICS, get_logger  # noqa: E402

logger = get_logger(__name__)


class ServingContext:
    """
    Lazily-built, process-wide container for the retrieval stack.

    ``warmup()`` is called from the application lifespan hook so the first user
    request does not pay model-load latency, and so readiness only flips true once the
    stack actually works.
    """

    def __init__(self) -> None:
        # Guards resource construction. Held for the full duration of an expensive
        # build (model load, corpus chunking, BM25 index), which can be tens of
        # seconds.
        self._lock = threading.RLock()
        # Guards only the readiness flags. Kept separate on purpose: health probes
        # must never block behind a warmup that is holding the build lock, otherwise
        # /readyz hangs during startup and an orchestrator reads a timeout as a dead
        # instance rather than a warming one.
        self._state_lock = threading.Lock()

        self._embedding = None
        self._vectorstore = None
        self._chunks: Optional[List[Any]] = None
        self._architectures: Optional[Dict[str, Any]] = None
        self._available_keys: Optional[List[str]] = None
        self._cache = None
        self._ready = False
        self._degraded_reason: Optional[str] = None
        self._arch_count = 0
        self._cache_ready = False
        self.started_at = time.time()

    # ── resources ──────────────────────────────────────────────────────────────

    # Each accessor uses double-checked locking: an already-built resource is
    # returned WITHOUT taking the lock. This matters because one slow construction
    # (GRACE-Ω's RAPTOR tree takes minutes) would otherwise hold the build lock and
    # block every unrelated request that merely wants an existing resource.
    # Reading an attribute is atomic under the GIL, so the unlocked fast path is safe.

    def embedding(self):
        if self._embedding is not None:
            return self._embedding
        with self._lock:
            if self._embedding is None:
                from module1.data_preprocessing import get_embedding_model
                logger.info("Loading embedding model.")
                self._embedding = get_embedding_model()
            return self._embedding

    def vectorstore(self):
        """
        Load the persisted Chroma index.

        Refuses to silently bootstrap a different corpus: if the store is missing we
        say so, because serving a SQuAD-derived index while claiming MS MARCO results
        is precisely the drift this module exists to prevent.
        """
        if self._vectorstore is not None:
            return self._vectorstore
        with self._lock:
            if self._vectorstore is None:
                from module1.data_preprocessing import load_vector_store
                if not os.path.isdir(VECTOR_STORE_DIR) or not os.listdir(VECTOR_STORE_DIR):
                    raise RuntimeError(
                        f"Vector store not found at {VECTOR_STORE_DIR}. Build it with "
                        "`python preprocess_full.py` (or `python main.py --mode full`) "
                        "before serving."
                    )
                logger.info("Loading vector store.", extra={"path": VECTOR_STORE_DIR})
                self._vectorstore = load_vector_store(self.embedding())
            return self._vectorstore

    def chunks(self) -> List[Any]:
        """
        The chunk corpus used for sparse (BM25) retrieval.

        Built from **MS MARCO**, the same source as the dense index, so hybrid
        architectures fuse two views of one corpus rather than two corpora.
        """
        if self._chunks is not None:
            return self._chunks
        with self._lock:
            if self._chunks is None:
                from module1.data_preprocessing import (
                    chunk_documents, load_msmarco_documents,
                )
                logger.info("Building chunk corpus (MS MARCO).",
                            extra={"max_samples": DATASET_SUBSET_SIZE})
                self._chunks = chunk_documents(load_msmarco_documents())
            return self._chunks

    def architectures(self) -> Dict[str, Any]:
        """
        Cheap-to-construct architectures, built eagerly.

        Deliberately excludes anything whose *constructor* calls an LLM. GRACE-Ω
        builds a RAPTOR summarisation tree in ``__init__`` — roughly 15 LLM calls per
        instance — so eagerly constructing every variant made startup take minutes
        and burn API quota before serving a single request. Those are created on first
        use by :meth:`architecture` instead.
        """
        if self._architectures is not None:
            return self._architectures
        with self._lock:
            if self._architectures is None:
                from module2.rag_architectures import build_all_architectures
                self._architectures = build_all_architectures(
                    vectorstore=self.vectorstore(),
                    chunks=self.chunks(),
                    llm_free_only=True,
                )
                logger.info("Eager architectures registered.",
                            extra={"count": len(self._architectures),
                                   "names": sorted(self._architectures)})
            return self._architectures

    def architecture(self, key: str):
        """
        Resolve one architecture by key, constructing lazily when needed.

        Expensive architectures are built on first request and then cached, so the
        cost is paid once by one caller instead of by every deploy.
        """
        eager = self.architectures()
        if key in eager:
            return eager[key]

        with self._lock:
            if key in self._architectures:      # another thread may have built it
                return self._architectures[key]
            if key not in self.available_keys():
                raise KeyError(key)

            logger.info("Lazily constructing architecture.", extra={"architecture": key})
            from module2.rag_architectures import build_all_architectures
            full = build_all_architectures(
                vectorstore=self.vectorstore(), chunks=self.chunks(),
            )
            if key not in full:
                raise KeyError(key)
            self._architectures.update(full)
            return self._architectures[key]

    def available_keys(self) -> List[str]:
        """
        Every architecture key the framework can serve, without constructing them.

        Reads declared registry metadata, so ``GET /architectures`` advertises the
        full set at zero cost. Never takes the build lock — this is called on the
        request path and must stay responsive during a slow construction.
        """
        if self._available_keys is None:
            from module2.rag_architectures import ARCHITECTURE_KEYS
            self._available_keys = sorted(ARCHITECTURE_KEYS)
        return self._available_keys

    def architecture_names(self) -> List[str]:
        return self.available_keys()

    def cache(self):
        if self._cache is not None:
            return self._cache
        with self._lock:
            if self._cache is None:
                from phase3.semantic_cache import SemanticCache
                self._cache = SemanticCache(embedding_model=self.embedding())
            return self._cache

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """
        Build everything up front and record readiness.

        A failure here leaves the process running but *not ready*, so an orchestrator
        will keep it out of the load-balancer rotation instead of routing traffic to a
        broken instance.
        """
        try:
            archs = self.architectures()
            cache = self.cache()
            with self._state_lock:
                self._ready = True
                self._degraded_reason = None
                self._arch_count = len(archs)
                self._cache_ready = bool(cache and cache.healthy())
            logger.info("Warmup complete — service ready.",
                        extra={"architectures": self._arch_count})
        except Exception as exc:  # noqa: BLE001
            with self._state_lock:
                self._ready = False
                self._degraded_reason = str(exc)
            logger.error("Warmup failed — service NOT ready.", extra={"error": str(exc)})

    def shutdown(self) -> None:
        """Release references so Chroma/torch can free resources deterministically."""
        # Flip readiness first (cheap lock) so no new work is admitted while we tear
        # down, then release resources under the build lock.
        with self._state_lock:
            self._ready = False
            self._cache_ready = False
        with self._lock:
            self._architectures = None
            self._vectorstore = None
            self._chunks = None
            self._cache = None
        logger.info("Serving context shut down.")

    # ── health ─────────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """Non-blocking readiness check (uses the state lock, never the build lock)."""
        with self._state_lock:
            return self._ready

    def readiness(self) -> Dict[str, Any]:
        """
        Detailed readiness payload for /readyz.

        Reads only cached state under the short-lived state lock, so this stays
        responsive while a warmup holds the build lock.
        """
        with self._state_lock:
            ready = self._ready
            reason = self._degraded_reason
            n_archs = self._arch_count
            cache_ok = self._cache_ready
        return {
            "ready": ready,
            "reason": reason if reason else (None if ready else "warming up"),
            "architectures": n_archs,
            "vector_store": os.path.isdir(VECTOR_STORE_DIR),
            "cache": cache_ok,
            "uptime_seconds": round(time.time() - self.started_at, 1),
        }

    def refresh_cache_gauge(self) -> None:
        """Publish live cache size to Prometheus."""
        try:
            if self._cache is not None:
                METRICS.cache_entries.set(self._cache.stats().get("entries", 0))
        except Exception:  # noqa: BLE001
            pass


# Process-wide singleton.
CONTEXT = ServingContext()
