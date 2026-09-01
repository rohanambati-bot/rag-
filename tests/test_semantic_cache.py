"""
Tests for the production semantic cache.

The original implementation issued ``KEYS rag_cache:*`` plus one ``GET`` per key on
every lookup and scored candidates in a Python loop. These tests pin the properties
of the replacement:

* correctness — near-duplicate queries hit, unrelated queries miss;
* the hot path performs **no** Redis command (Redis is warm-start + write-through);
* ``KEYS`` is never used anywhere;
* capacity is bounded with eviction, and TTL expiry is honoured;
* concurrent access is safe;
* lookup cost stays roughly flat as the cache grows.

All tests use a deterministic fake embedder, so no model download or network access.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase3.semantic_cache import CacheEntry, SemanticCache  # noqa: E402

DIM = 32


class FakeEmbedder:
    """Deterministic unit-norm embeddings derived from the text."""

    def __init__(self):
        self.calls = 0

    def embed_query(self, text: str):
        self.calls += 1
        rng = np.random.default_rng(abs(hash(text.strip().lower())) % (2**32))
        v = rng.normal(size=DIM).astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()


class NearEmbedder:
    """Maps any text to one of two clusters, so 'similar' queries really are similar."""

    def embed_query(self, text: str):
        base = np.zeros(DIM, dtype=np.float32)
        if "weather" in text.lower():
            base[0], base[1] = 1.0, 0.05
        else:
            base[2], base[3] = 1.0, 0.05
        # Tiny text-dependent jitter keeps vectors distinct but highly similar.
        base[4] = (len(text) % 7) * 0.001
        return (base / np.linalg.norm(base)).tolist()


def _cache(embedder=None, **kwargs) -> SemanticCache:
    """Cache with Redis disabled, so tests exercise the in-process index only."""
    return SemanticCache(
        embedding_model=embedder or FakeEmbedder(),
        warm_start=False,
        redis_host="127.0.0.1",
        redis_port=1,          # nothing listens here → in-process only
        **kwargs,
    )


# ── Correctness ────────────────────────────────────────────────────────────────

class TestCorrectness:
    def test_exact_repeat_is_a_hit(self):
        c = _cache()
        c.store("what is a corporation", "a legal entity", ["ctx"])
        hit = c.lookup("what is a corporation")
        assert hit is not None
        assert hit["answer"] == "a legal entity"
        assert hit["cache_hit"] is True
        assert hit["similarity_score"] >= c.similarity_threshold

    def test_unrelated_query_is_a_miss(self):
        c = _cache()
        c.store("what is a corporation", "a legal entity", ["ctx"])
        assert c.lookup("how tall is mount everest") is None

    def test_semantically_similar_query_hits(self):
        c = _cache(NearEmbedder())
        c.store("what is the weather today", "sunny", ["ctx"])
        hit = c.lookup("tell me the weather right now")
        assert hit is not None and hit["answer"] == "sunny"

    def test_different_cluster_misses(self):
        c = _cache(NearEmbedder())
        c.store("what is the weather today", "sunny", ["ctx"])
        assert c.lookup("explain deep learning briefly") is None

    def test_empty_cache_misses_without_error(self):
        assert _cache().lookup("anything") is None

    def test_lookup_is_case_and_whitespace_insensitive_for_keys(self):
        c = _cache()
        c.store("Hello World", "hi", ["ctx"])
        # Same key → refresh rather than a duplicate row.
        c.store("  hello world  ", "hi again", ["ctx"])
        assert c.stats()["entries"] == 1

    def test_contexts_round_trip(self):
        c = _cache()
        c.store("q", "a", ["ctx-1", "ctx-2"])
        assert c.lookup("q")["contexts"] == ["ctx-1", "ctx-2"]

    def test_missing_embedder_degrades_to_miss(self):
        c = _cache()
        c.embedding_model = None
        assert c.lookup("q") is None
        c.store("q", "a", [])          # must not raise
        assert c.stats()["entries"] == 0


# ── Redis is not on the hot path ───────────────────────────────────────────────

class _SpyRedis:
    """Records every Redis command so the hot path can be asserted clean."""

    def __init__(self):
        self.commands: list = []

    def _record(self, name, *a):
        self.commands.append(name)

    def ping(self):
        self._record("ping")
        return True

    def keys(self, *a):
        self._record("KEYS")
        return []

    def get(self, *a):
        self._record("get")
        return None

    def mget(self, *a):
        self._record("mget")
        return []

    def setex(self, *a):
        self._record("setex")
        return True

    def sadd(self, *a):
        self._record("sadd")
        return 1

    def srem(self, *a):
        self._record("srem")
        return 1

    def delete(self, *a):
        self._record("delete")
        return 1

    def sscan_iter(self, *a, **k):
        self._record("sscan_iter")
        return iter([])

    def pipeline(self):
        return self

    def execute(self):
        self._record("execute")
        return []


class TestRedisUsage:
    def _with_spy(self):
        c = _cache()
        spy = _SpyRedis()
        c._redis = spy
        c._use_redis = True
        return c, spy

    def test_lookup_issues_no_redis_commands(self):
        """The whole point: lookup must not touch the network."""
        c, spy = self._with_spy()
        c.store("q", "a", [])
        spy.commands.clear()
        c.lookup("q")
        assert spy.commands == [], f"lookup hit Redis: {spy.commands}"

    def test_keys_command_is_never_used(self):
        c, spy = self._with_spy()
        c.store("q1", "a", [])
        c.lookup("q1")
        c.clear()
        assert "KEYS" not in spy.commands

    def test_store_writes_through(self):
        c, spy = self._with_spy()
        spy.commands.clear()
        c.store("q", "a", [])
        assert "setex" in spy.commands and "sadd" in spy.commands

    def test_clear_uses_scan_not_keys(self):
        c, spy = self._with_spy()
        spy.commands.clear()
        c.clear()
        assert "sscan_iter" in spy.commands
        assert "KEYS" not in spy.commands

    def test_redis_failure_does_not_break_store(self):
        class Broken(_SpyRedis):
            def pipeline(self):
                raise RuntimeError("connection reset")

        c = _cache()
        c._redis, c._use_redis = Broken(), True
        c.store("q", "a", [])              # must not raise
        assert c.lookup("q") is not None   # local index still serves
        assert c.stats()["errors"] >= 1


# ── Capacity, eviction, TTL ────────────────────────────────────────────────────

class TestCapacityAndExpiry:
    def test_capacity_is_enforced(self):
        c = _cache(max_entries=20)
        for i in range(60):
            c.store(f"question {i}", f"answer {i}", [])
        assert c.stats()["entries"] <= 20
        assert c.stats()["evictions"] > 0

    def test_eviction_keeps_newest(self):
        c = _cache(max_entries=10)
        for i in range(30):
            c.store(f"question {i}", f"answer {i}", [])
            time.sleep(0.001)          # distinct timestamps
        assert c.lookup("question 29") is not None

    def test_expired_entry_is_not_returned(self):
        c = _cache(ttl_seconds=1)
        c.store("q", "a", [])
        assert c.lookup("q") is not None
        time.sleep(1.1)
        assert c.lookup("q") is None

    def test_zero_ttl_means_no_expiry(self):
        c = _cache(ttl_seconds=0)
        c.store("q", "a", [])
        time.sleep(0.05)
        assert c.lookup("q") is not None

    def test_clear_empties_the_cache(self):
        c = _cache()
        for i in range(5):
            c.store(f"q{i}", "a", [])
        c.clear()
        assert c.stats()["entries"] == 0
        assert c.lookup("q0") is None


# ── Stats and health ───────────────────────────────────────────────────────────

class TestStats:
    def test_hit_and_miss_counts(self):
        c = _cache()
        c.store("q", "a", [])
        c.lookup("q")                  # hit
        c.lookup("totally other text") # miss
        s = c.stats()
        assert s["hits"] == 1 and s["misses"] == 1
        assert s["total_queries"] == 2
        assert s["hit_rate"] == 0.5

    def test_stats_expose_operational_fields(self):
        s = _cache().stats()
        for field in ("entries", "capacity", "avg_lookup_ms", "backend",
                      "ann", "evictions", "errors"):
            assert field in s

    def test_backend_reports_memory_when_redis_absent(self):
        assert _cache().stats()["backend"] == "memory"

    def test_healthy_reflects_embedder_availability(self):
        c = _cache()
        assert c.healthy() is True
        c.embedding_model = None
        assert c.healthy() is False


# ── Concurrency ────────────────────────────────────────────────────────────────

class TestConcurrency:
    def test_parallel_store_and_lookup_is_safe(self):
        """FastAPI serves from a thread pool, so the index must be lock-guarded."""
        c = _cache(max_entries=500)
        errors: list = []

        def worker(base: int):
            try:
                for i in range(40):
                    c.store(f"q-{base}-{i}", f"a-{base}-{i}", [])
                    c.lookup(f"q-{base}-{i}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(b,)) for b in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"concurrent access raised: {errors}"
        assert c.stats()["entries"] > 0


# ── Scaling ────────────────────────────────────────────────────────────────────

class TestScaling:
    @pytest.mark.parametrize("n_entries", [200, 2000])
    def test_lookup_stays_fast_as_cache_grows(self, n_entries):
        """
        Lookup must not degrade linearly. The old design paid one Redis round-trip
        per entry plus a Python loop; this one is a single vector search.
        """
        c = _cache(max_entries=n_entries * 2)
        for i in range(n_entries):
            c.store(f"question number {i}", f"answer {i}", [])

        started = time.perf_counter()
        for i in range(0, n_entries, max(1, n_entries // 50)):
            c.lookup(f"question number {i}")
        per_lookup_ms = (time.perf_counter() - started) / 50 * 1000

        # Generous bound: the point is sub-linear behaviour, not a micro-benchmark.
        assert per_lookup_ms < 25, f"{per_lookup_ms:.2f} ms/lookup at n={n_entries}"


# ── Entry serialisation ────────────────────────────────────────────────────────

class TestCacheEntry:
    def test_json_round_trip(self):
        entry = CacheEntry(query="q", answer="a", contexts=["c"], embedding=[0.1, 0.2])
        restored = CacheEntry.from_json(entry.to_json())
        assert restored.query == "q"
        assert restored.answer == "a"
        assert restored.contexts == ["c"]
        assert restored.embedding == [0.1, 0.2]

    def test_expiry_predicate(self):
        entry = CacheEntry(query="q", answer="a", contexts=[], embedding=[1.0],
                           timestamp=time.time() - 100)
        assert entry.is_expired(50) is True
        assert entry.is_expired(500) is False
        assert entry.is_expired(0) is False   # 0 disables TTL
