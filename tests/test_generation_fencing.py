"""Request-start fencing regressions; barriers/events, no timing-based ordering."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import ast
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from phase3.cicd_pipeline import CachedRAG
from phase3.semantic_cache import GenerationFencedError, SemanticCache
from tests.test_cache_index_lifecycle import _Embedder, _SharedRedis


@pytest.fixture
def factory(monkeypatch):
    redis = _SharedRedis()

    def connect(cache, *_args):
        cache._redis, cache._use_redis, cache._generation_supported = redis, True, True

    monkeypatch.setattr(SemanticCache, "_connect_redis", connect)

    def make(scope="fencing", **kwargs):
        return SemanticCache(_Embedder(), namespace=scope, redis_failure_policy="miss", **kwargs)

    return make, redis


def test_clear_during_generation_rejects_response_and_both_caches(factory):
    make, redis = factory
    cache, clearer = make(), make()
    started, release = threading.Event(), threading.Event()

    class Rag:
        def run(self, query):
            started.set()
            assert release.wait(5)
            return {"answer": "revoked", "contexts": ["private"]}

    with ThreadPoolExecutor() as pool:
        future = pool.submit(CachedRAG(Rag(), cache).run, "question")
        assert started.wait(5)
        assert clearer.clear()
        release.set()
        with pytest.raises(GenerationFencedError):
            future.result(5)
    assert cache.lookup("question") is None
    assert make().lookup("question") is None
    assert not any(":entry:" in key for key in redis.values)


def test_atomic_compare_publish_fences_clear_after_last_python_read(factory, monkeypatch):
    make, redis = factory
    cache, clearer = make(), make()
    token = cache.begin_generation()
    entered, release = threading.Event(), threading.Event()
    original = redis.eval

    def pause(script, *args):
        if "fencing:publish" in script:
            entered.set()
            assert release.wait(5)
        return original(script, *args)

    monkeypatch.setattr(redis, "eval", pause)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(cache.publish_generation, token, "q", "revoked", [])
        assert entered.wait(5)
        assert clearer.clear()
        release.set()
        with pytest.raises(GenerationFencedError, match="revoked before publication"):
            future.result(5)
    assert not cache._key_to_row
    assert make().lookup("q") is None


def test_normal_stores_do_not_revoke_other_outstanding_work(factory):
    make, _ = factory
    first, second = make(), make()
    a, b = first.begin_generation(), second.begin_generation()
    assert first.publish_generation(a, "first", "one", [])
    snapshot = first._redis.get(first._generation_key)
    assert second.publish_generation(b, "second", "two", [])
    assert snapshot != first._redis.get(first._generation_key)
    first.validate_generation(a)
    second.validate_generation(b)
    assert len(make()._key_to_row) == 2


def test_overlapping_index_scopes_only_fence_relevant_work(factory):
    make, _ = factory
    epochs = {key: "build-1" for key in "abc"}

    def identity(keys):
        return ":".join(epochs[key] for key in keys)

    caches = [make(keys, index_identity=identity(keys),
                   identity_provider=lambda keys=keys: identity(keys)) for keys in ("ab", "bc", "c")]
    tokens = [cache.begin_generation() for cache in caches]
    epochs["b"] = "build-2"
    for cache, token in zip(caches[:2], tokens[:2]):
        with pytest.raises(GenerationFencedError, match="identity changed"):
            cache.publish_generation(token, "q", "obsolete", [])
        assert cache.lookup("q") is None
    assert caches[2].publish_generation(tokens[2], "q", "unrelated", [])


def test_concurrent_clears_fence_all_older_tokens(factory):
    make, _ = factory
    stale = make()
    token = stale.begin_generation()
    clearers = [make() for _ in range(8)]
    barrier = threading.Barrier(len(clearers))

    def clear(cache):
        barrier.wait(5)
        return cache.clear()

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(clear, clearers))
    with pytest.raises(GenerationFencedError):
        stale.publish_generation(token, "q", "obsolete", [])
    assert make().lookup("q") is None


def test_outage_and_initial_outage_never_acknowledge_generation(factory):
    make, redis = factory
    cache = make()
    token = cache.begin_generation()
    redis.fail = True
    with pytest.raises(GenerationFencedError):
        cache.publish_generation(token, "q", "answer", [])
    with pytest.raises(GenerationFencedError):
        make().begin_generation()
    assert cache.clear() is False
    redis.fail = False
    with pytest.raises(GenerationFencedError):
        cache.validate_generation(token)
    assert not cache._key_to_row


def test_ambiguous_publication_is_not_success_or_retried(factory, monkeypatch):
    make, redis = factory
    cache = make()
    token = cache.begin_generation()
    original = redis.eval
    calls = []

    def uncertain(script, *args):
        result = original(script, *args)
        if "fencing:publish" in script:
            calls.append(True)
            raise ConnectionError("reply lost after commit")
        return result

    monkeypatch.setattr(redis, "eval", uncertain)
    with pytest.raises(GenerationFencedError, match="could not be confirmed"):
        cache.publish_generation(token, "q", "answer", [])
    assert calls == [True]
    assert not cache._key_to_row
    assert make().lookup("q")["answer"] == "answer"  # may have committed, never claimed otherwise
    assert cache.clear()
    assert make().lookup("q") is None


def test_clear_server_success_with_delayed_local_bookkeeping_still_fences(factory, monkeypatch):
    make, redis = factory
    cache, stale = make(), make()
    token = stale.begin_generation()
    original = redis.eval

    def lost_clear_receipt(script, *args):
        result = original(script, *args)
        if "fencing:clear" in script:
            raise ConnectionError("server committed; caller has stale snapshot bookkeeping")
        return result

    monkeypatch.setattr(redis, "eval", lost_clear_receipt)
    assert cache.clear() is False
    with pytest.raises(GenerationFencedError):
        stale.publish_generation(token, "q", "obsolete", [])
    assert make().lookup("q") is None


def test_restart_random_epoch_fences_stale_worker_even_when_snapshot_is_empty(factory):
    make, redis = factory
    old = make()
    token = old.begin_generation()
    redis.values.clear()
    redis.sets.clear()
    new = make()
    fresh = new.begin_generation()
    assert fresh.shared_epoch != token.shared_epoch
    with pytest.raises(GenerationFencedError):
        old.publish_generation(token, "q", "before restart", [])
    assert new.publish_generation(fresh, "q", "after restart", [])
    assert old.lookup("q")["answer"] == "after restart"


def test_missing_revocation_authority_cannot_recertify_old_persisted_entries(factory):
    make, redis = factory
    cache = make()
    old = cache.begin_generation()
    cache.publish_generation(old, "q", "old epoch", [])
    assert cache.lookup_generation(old, "q")
    redis.values.pop(cache._revocation_key)
    fresh = cache.begin_generation()
    assert fresh.shared_epoch != old.shared_epoch
    assert cache.lookup_generation(fresh, "q") is None
    assert make().lookup("q") is None


def test_random_epoch_preserves_leading_zeroes_even_if_all_characters_are_digits(factory):
    make, redis = factory
    cache = make()
    epoch = "00012345678901234567890123456789"
    redis._set(cache._revocation_key, epoch)
    token = cache.begin_generation()
    assert token.shared_epoch == epoch
    assert cache.publish_generation(token, "q", "valid", [])


@pytest.mark.parametrize("field,value", [("instance", "other"), ("namespace", "other"),
                                         ("index_identity", "other"), ("local_epoch", "other")])
def test_token_cannot_cross_instance_namespace_or_index(factory, field, value):
    make, _ = factory
    cache = make()
    token = cache.begin_generation()
    with pytest.raises(GenerationFencedError):
        cache.publish_generation(replace(token, **{field: value}), "q", "bad", [])
    with pytest.raises(GenerationFencedError):
        make().validate_generation(token)
    assert not cache._key_to_row


def test_polling_optout_does_not_bypass_request_bound_lookup(factory):
    make, _ = factory
    cache, clearer = make(generation_poll_interval=None), make()
    token = cache.begin_generation()
    cache.publish_generation(token, "q", "old", [])
    assert cache.lookup_generation(token, "q")["answer"] == "old"
    clearer.clear()
    with pytest.raises(GenerationFencedError):
        cache.lookup_generation(token, "q")


def test_clear_during_cached_hit_scoring_cannot_return_old_current_hit(factory, monkeypatch):
    make, _ = factory
    cache, clearer = make(), make()
    token = cache.begin_generation()
    cache.publish_generation(token, "q", "revoked", [])
    assert cache.lookup_generation(token, "q")
    entered, release = threading.Event(), threading.Event()
    original = cache._index.search

    def search(vector):
        entered.set()
        assert release.wait(5)
        return original(vector)

    monkeypatch.setattr(cache._index, "search", search)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(cache.lookup, "q")
        assert entered.wait(5)
        assert clearer.clear()
        release.set()
        assert future.result(5) is None
    assert not cache._key_to_row


def test_explicit_local_token_fences_same_process_clear(monkeypatch):
    monkeypatch.setattr(SemanticCache, "_connect_redis", lambda *_: None)
    cache = SemanticCache(_Embedder(), redis_failure_policy="local")
    token = cache.begin_generation()
    assert token.shared_epoch is None
    assert cache.clear()
    with pytest.raises(GenerationFencedError):
        cache.publish_generation(token, "q", "revoked", [])


def test_no_redis_client_cannot_claim_a_shared_clear(monkeypatch):
    monkeypatch.setattr(SemanticCache, "_connect_redis", lambda *_: None)
    cache = SemanticCache(_Embedder(), redis_failure_policy="miss")
    assert cache.clear() is False
    with pytest.raises(GenerationFencedError):
        cache.begin_generation()


def test_legacy_adapter_requires_explicit_wrapper_optin():
    cache = SimpleNamespace(lookup=lambda _: None, store=lambda **_: None)
    rag = SimpleNamespace(run=lambda _: {"answer": "ok", "contexts": []})
    with pytest.raises(GenerationFencedError, match="adapter"):
        CachedRAG(rag, cache).run("q")
    assert CachedRAG(rag, cache, allow_legacy_cache=True).run("q")["answer"] == "ok"


@pytest.mark.parametrize("use_cache", [True, False])
def test_http_clear_during_generation_rejects_even_uncached_response(factory, monkeypatch, use_cache):
    from tests.test_api_production import _make_client
    make, _ = factory
    cache = make()
    entered, release = threading.Event(), threading.Event()
    with _make_client(monkeypatch, allow_unauth=True) as client:
        import api
        monkeypatch.setattr(api.CONTEXT, "cache", lambda *_: cache)

        def run(_):
            entered.set()
            assert release.wait(5)
            return {"answer": "never serve revoked", "contexts": [], "latency_ms": 1}

        monkeypatch.setattr(api.CONTEXT.architectures()["naive"], "run", run)
        with ThreadPoolExecutor() as pool:
            future = pool.submit(client.post, "/query", json={"question": "q", "use_cache": use_cache})
            assert entered.wait(5)
            assert client.delete("/cache").status_code == 200
            release.set()
            response = future.result(5)
        assert response.status_code == 503
        assert "never serve revoked" not in response.text
        assert cache.lookup("q") is None


def test_http_captures_before_lazy_architecture_construction(factory, monkeypatch):
    from tests.test_api_production import _make_client
    make, _ = factory
    cache = make()
    with _make_client(monkeypatch, allow_unauth=True) as client:
        import api
        monkeypatch.setattr(api.CONTEXT, "cache", lambda *_: cache)

        async def resolve(_):
            cache.clear()
            return SimpleNamespace(run=lambda _: {"answer": "obsolete", "contexts": []})

        monkeypatch.setattr(api, "resolve_architecture", resolve)
        response = client.post("/query", json={"question": "q"})
        assert response.status_code == 503


def test_http_rejects_legacy_cache_adapter_by_default(monkeypatch):
    from tests.test_api_production import _make_client
    with _make_client(monkeypatch, allow_unauth=True) as client:
        import api
        legacy = SimpleNamespace(lookup=lambda _: None, store=lambda *args: None)
        monkeypatch.setattr(api.CONTEXT, "cache", lambda *_: legacy)
        assert client.post("/query", json={"question": "q"}).status_code == 503


def test_http_uncached_ab_response_is_fenced(factory, monkeypatch):
    from tests.test_api_production import _make_client
    make, _ = factory
    cache = make()
    with _make_client(monkeypatch, allow_unauth=True) as client:
        import api
        monkeypatch.setattr(api.CONTEXT, "cache", lambda *_: cache)

        def run(_):
            cache.clear()
            return {"answer": "obsolete AB", "contexts": [], "latency_ms": 1}

        monkeypatch.setattr(api.CONTEXT.architectures()["naive"], "run", run)
        response = client.post("/ab-test", json={"question": "q"})
        assert response.status_code == 503
        assert "obsolete AB" not in response.text


def test_actual_serving_identity_fences_changed_manifest_and_config(factory, tmp_path, monkeypatch):
    import serving
    from configs import config
    (tmp_path / "chroma.sqlite3").write_bytes(b"unchanged sqlite")
    manifest = tmp_path / "_index_manifest.json"
    manifest.write_text('{"build_id":"one"}')
    monkeypatch.setattr(serving, "VECTOR_STORE_DIR", str(tmp_path))
    context = serving.ServingContext()
    monkeypatch.setattr(context, "embedding", _Embedder)
    cache = context.cache("naive")
    token = cache.begin_generation()
    manifest.write_text('{"build_id":"two"}')
    with pytest.raises(GenerationFencedError, match="identity changed"):
        cache.publish_generation(token, "q", "obsolete", [])
    replacement = serving.ServingContext()
    monkeypatch.setattr(replacement, "embedding", _Embedder)
    current = replacement.cache("naive")
    token = current.begin_generation()
    monkeypatch.setattr(config, "TOP_K", config.TOP_K + 1)
    with pytest.raises(GenerationFencedError, match="identity changed"):
        current.validate_generation(token)


def test_stale_worker_cannot_create_a_current_cache_for_old_loaded_resources(factory, tmp_path, monkeypatch):
    import serving
    (tmp_path / "chroma.sqlite3").write_bytes(b"unchanged sqlite")
    manifest = tmp_path / "_index_manifest.json"
    manifest.write_text('{"build_id":"old"}')
    monkeypatch.setattr(serving, "VECTOR_STORE_DIR", str(tmp_path))
    context = serving.ServingContext()
    monkeypatch.setattr(context, "embedding", _Embedder)
    # Loading the corpus/architecture stack captures ALL architecture identities,
    # including architectures whose answer caches are still cold.
    old = context.generation_identity("naive")
    manifest.write_text('{"build_id":"replacement"}')
    cache = context.cache("rerank")
    assert cache._index_identity != context._cache_namespace("rerank")
    assert context.generation_identity("naive") == old
    with pytest.raises(GenerationFencedError, match="identity changed"):
        cache.begin_generation()


@pytest.mark.parametrize("change", ["clear", "replace", "digest"])
def test_actual_ui_upload_fence_covers_its_private_index(factory, change):
    from security_boundary import session_cache
    tree = ast.parse(Path("app.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_get_upload_generation_cache")
    state = {"_upload_digest": "original"}
    uploader = SimpleNamespace(vectorstore=object())
    original_store = uploader.vectorstore  # retain old resource, as a running RAG does
    env = {"st": SimpleNamespace(session_state=state), "session_cache": session_cache,
           "_get_embedding_model": _Embedder}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "app.py", "exec"), env)
    cache = env["_get_upload_generation_cache"](uploader, "Naive RAG")
    token = cache.begin_generation()
    if change == "digest":
        state["_upload_digest"] = "replacement"
    else:
        uploader.vectorstore = None if change == "clear" else object()
    assert original_store is not None
    with pytest.raises(GenerationFencedError, match="identity changed"):
        cache.validate_generation(token)
    fresh = env["_get_upload_generation_cache"](uploader, "Naive RAG")
    assert fresh is not cache
    fresh.validate_generation(fresh.begin_generation())


@pytest.mark.parametrize("use_cache", [True, False])
def test_ui_chat_rejects_revoked_response_before_render(factory, use_cache):
    make, _ = factory
    cache, clearer = make(), make()
    # Execute the actual UI function without importing/running Streamlit itself.
    tree = ast.parse(Path("app.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "page_chat")
    rendered, failures = [], []

    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    class UI:
        session_state = State()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def markdown(self, value, **kwargs):
            rendered.append(value)
        def error(self, value):
            failures.append(value)
        def columns(self, _):
            return [self, self]
        def selectbox(self, *args, **kwargs):
            return "naive"
        def toggle(self, *args, **kwargs):
            return use_cache
        def button(self, *args, **kwargs):
            return False
        def chat_input(self, *args):
            return "q"
        def chat_message(self, *args):
            return self
        def empty(self):
            return self

    def run(_):
        assert clearer.clear()
        return {"answer": "never render revoked", "contexts": []}

    env = {"st": UI(), "_get_cache": lambda _: cache,
           "_render_untrusted_markdown": lambda value, target=None: (
               target.markdown(value) if target is not None else rendered.append(value)
           ),
           "_get_architectures": lambda: {"naive": SimpleNamespace(run=run)},
           "logging": __import__("logging")}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "app.py", "exec"), env)
    env["page_chat"]()
    assert failures
    assert "never render revoked" not in rendered
