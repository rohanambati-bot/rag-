"""Real cache behavior with isolated fake Redis persistence, never a live service."""
import asyncio

from fastapi import Request
import pytest

from phase3.semantic_cache import SemanticCache
from serving import ServingContext


class Embedder:
    def embed_query(self, text):
        return [1.0, 0.0]


class RedisDouble:
    def __init__(self):
        self.values, self.sets = {}, {}
        self.commands = []

    def pipeline(self):
        return self

    def setex(self, key, ttl, value):
        assert ttl > 0, "Real Redis rejects SETEX with a zero expiry"
        self.values[key] = value
        self.commands.append("setex")

    def set(self, key, value):
        self.values[key] = value
        self.commands.append("set")

    def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value.encode())

    def srem(self, key, value):
        self.sets.get(key, set()).discard(value.encode())

    def sscan_iter(self, key, **kwargs):
        return iter(list(self.sets.get(key, [])))

    def mget(self, keys):
        return [self.values.get(k.decode() if isinstance(k, bytes) else k) for k in keys]

    def delete(self, key):
        if isinstance(key, bytes):
            key = key.decode()
        self.values.pop(key, None)
        self.sets.pop(key, None)

    def execute(self):
        return []


@pytest.fixture
def cache_factory(monkeypatch):
    redis = RedisDouble()

    def connect(cache, *args):
        cache._redis, cache._use_redis = redis, True

    monkeypatch.setattr(SemanticCache, "_connect_redis", connect)
    return lambda scope="", **kw: SemanticCache(Embedder(), namespace=scope, **kw), redis


def test_namespaces_isolate_semantically_identical_questions_and_clear(cache_factory):
    make_cache, _ = cache_factory
    a, b = make_cache("naive-v1"), make_cache("rerank-v1")
    a.store("question", "naive", ["naive context"])
    b.store("question", "rerank", ["rerank context"])
    assert make_cache("naive-v1").lookup("question")["answer"] == "naive"
    assert make_cache("rerank-v1").lookup("question")["answer"] == "rerank"
    assert make_cache("rerank-v2").lookup("question") is None
    a.clear()
    assert make_cache("naive-v1").lookup("question") is None
    assert make_cache("rerank-v1").lookup("question")["answer"] == "rerank"


def test_zero_ttl_persists_without_setex(cache_factory):
    make_cache, redis = cache_factory
    cache = make_cache(ttl_seconds=0)
    cache.store("question", "answer", [])
    assert "set" in redis.commands and "setex" not in redis.commands
    assert cache.stats()["errors"] == 0
    assert make_cache(ttl_seconds=0).lookup("question")["answer"] == "answer"


def test_live_worker_invalidation_is_not_redis_write_through(cache_factory):
    """Pin the documented limitation rather than falsely claiming invalidation."""
    make_cache, _ = cache_factory
    first = make_cache("same-worker-contract")
    first.store("question", "old answer", [])
    second = make_cache("same-worker-contract")
    first.clear()
    assert second.lookup("question")["answer"] == "old answer"
    assert make_cache("same-worker-contract").lookup("question") is None


def test_api_uses_architecture_specific_answer_cache(monkeypatch, cache_factory):
    import api
    make_cache, _ = cache_factory
    ctx = ServingContext()
    ctx._caches = {name: make_cache(name) for name in ("naive", "rerank")}

    class Arch:
        def __init__(self, name):
            self.name = name

        def run(self, question):
            return {"answer": self.name, "contexts": [self.name], "latency_ms": 1}

    ctx._architectures = {name: Arch(name) for name in ctx._caches}
    monkeypatch.setattr(api, "CONTEXT", ctx)
    monkeypatch.setattr(api, "_WORK_SEMAPHORE", asyncio.Semaphore(2))

    async def exercise():
        request = Request({"type": "http", "method": "POST", "path": "/query", "headers": []})
        for name in ("naive", "rerank", "naive", "rerank"):
            response = await api.query(request, api.QueryRequest(question="same question", architecture=name))
            assert response.architecture == response.answer == name
        assert response.cache_hit

    asyncio.run(exercise())


def test_serving_namespace_changes_with_architecture_source_and_generator(monkeypatch, tmp_path):
    import serving
    from configs import config
    source = tmp_path / "chroma.sqlite3"
    source.write_bytes(b"original fixture")
    monkeypatch.setattr(serving, "VECTOR_STORE_DIR", str(tmp_path))
    context = ServingContext()
    first = context._cache_namespace("naive")
    assert context._cache_namespace("rerank") != first
    monkeypatch.setattr(config, "GROQ_MODEL", "different-generator")
    assert context._cache_namespace("naive") != first
    source.write_bytes(b"replacement fixture")
    assert context._cache_namespace("naive") != first


@pytest.mark.parametrize("setting,value", [
    ("BM25_WEIGHT", 0.123), ("DENSE_WEIGHT", 0.456), ("CHUNK_SIZE", 123),
    ("PARENT_CHUNK_SIZE", 999), ("TOP_K", 17), ("LLM_MODEL", "replacement"),
    ("MAX_CONTEXT_CHARS", 100), ("DATASET_SUBSET_SIZE", 99),
])
def test_serving_cache_identity_covers_retrieval_and_generation_contract(
    monkeypatch, tmp_path, setting, value
):
    import serving
    from configs import config
    (tmp_path / "chroma.sqlite3").write_bytes(b"fixture")
    monkeypatch.setattr(serving, "VECTOR_STORE_DIR", str(tmp_path))
    context = ServingContext()
    original = context._cache_namespace("naive")
    monkeypatch.setattr(config, setting, value)
    assert context._cache_namespace("naive") != original


def test_cache_identity_does_not_confuse_same_size_same_mtime_corpora(monkeypatch, tmp_path):
    import os
    import serving
    source = tmp_path / "chroma.sqlite3"
    source.write_bytes(b"corpus-a")
    stat = source.stat()
    monkeypatch.setattr(serving, "VECTOR_STORE_DIR", str(tmp_path))
    original = ServingContext()._cache_namespace("naive")
    source.write_bytes(b"corpus-b")
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert ServingContext()._cache_namespace("naive") != original
