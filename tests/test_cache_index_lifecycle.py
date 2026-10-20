"""Focused lifecycle regressions for cache generations and index publication."""

from __future__ import annotations

import json
from pathlib import Path
import threading

import pytest

from module1.data_preprocessing import (
    BuildLockError,
    atomic_write_json,
    build_lock_for,
    create_staging_dir,
)
from phase3.semantic_cache import SemanticCache


class _Embedder:
    def embed_query(self, _text):
        return [1.0, 0.0]


class _Pipeline:
    def __init__(self, redis):
        self.redis = redis
        self.operations = []

    def setex(self, key, ttl, value):
        self.operations.append(("setex", key, ttl, value))
        return self

    def set(self, key, value):
        self.operations.append(("set", key, value))
        return self

    def sadd(self, key, value):
        self.operations.append(("sadd", key, value))
        return self

    def delete(self, key):
        self.operations.append(("delete", key))
        return self

    def incr(self, key):
        self.operations.append(("incr", key))
        return self

    def execute(self):
        results = []
        for operation in self.operations:
            command, *args = operation
            if command == "setex":
                _, key, _ttl, value = operation
                results.append(self.redis._set(key, value))
            elif command == "set":
                _, key, value = operation
                results.append(self.redis._set(key, value))
            elif command == "sadd":
                _, key, value = operation
                results.append(self.redis._sadd(key, value))
            elif command == "delete":
                results.append(self.redis._delete(args[0]))
            elif command == "incr":
                results.append(self.redis._incr(args[0]))
        return results


class _SharedRedis:
    """Small Redis contract double with bytes-compatible set members."""

    def __init__(self):
        self.values = {}
        self.sets = {}
        self.fail = False
        self.commands = []
        self.lock = threading.RLock()

    def eval(self, script, numkeys, *args):
        """Atomic double; the separate-process runner also executes the real Lua."""
        with self.lock:
            self._check()
            keys, argv = args[:numkeys], args[numkeys:]
            if "fencing:begin" in script:
                if self.get(keys[0]) is None:
                    self._set(keys[0], argv[0])
                    self._set(keys[1], argv[1])
                return self.get(keys[0])
            if "fencing:publish" in script:
                expected, member, raw, _ttl, snapshot = argv
                if self.get(keys[0]) != expected.encode():
                    return 0
                self._set(keys[2], raw)
                self._sadd(keys[3], member)
                self._set(keys[1], snapshot)
                return 1
            assert "fencing:clear" in script
            prefix, snapshot, epoch = argv
            for member in list(self.sets.get(keys[0], set())):
                self._delete(prefix + member.decode())
            self._delete(keys[0])
            self._set(keys[1], snapshot)
            self._set(keys[2], epoch)
            return 1

    @staticmethod
    def _key(value):
        return value.decode() if isinstance(value, bytes) else value

    def _check(self):
        if self.fail:
            raise ConnectionError("Redis unavailable")

    def ping(self):
        self._check()
        return True

    def get(self, key):
        self._check()
        self.commands.append("get")
        return self.values.get(self._key(key))

    def incr(self, key):
        self._check()
        self.commands.append("incr")
        return self._incr(key)

    def _incr(self, key):
        key = self._key(key)
        value = int(self.values.get(key, b"0")) + 1
        self.values[key] = str(value).encode()
        return value

    def pipeline(self):
        self._check()
        return _Pipeline(self)

    def _set(self, key, value):
        self._check()
        if isinstance(value, str):
            value = value.encode()
        self.values[self._key(key)] = value
        return True

    def set(self, key, value):
        return self._set(key, value)

    def _sadd(self, key, value):
        self._check()
        key = self._key(key)
        value = value.encode() if isinstance(value, str) else value
        self.sets.setdefault(key, set()).add(value)
        return 1

    def _delete(self, key):
        self._check()
        key = self._key(key)
        existed = key in self.values or key in self.sets
        self.values.pop(key, None)
        self.sets.pop(key, None)
        return int(existed)

    def sscan_iter(self, key, **_kwargs):
        self._check()
        self.commands.append("sscan_iter")
        return iter(list(self.sets.get(self._key(key), set())))

    def mget(self, keys):
        self._check()
        self.commands.append("mget")
        return [self.values.get(self._key(key)) for key in keys]


@pytest.fixture
def shared_cache_factory(monkeypatch):
    redis = _SharedRedis()

    def connect(cache, *_args):
        cache._redis = redis
        cache._use_redis = True
        cache._generation_supported = True

    monkeypatch.setattr(SemanticCache, "_connect_redis", connect)

    def make(namespace="lifecycle", **kwargs):
        return SemanticCache(
            embedding_model=_Embedder(),
            namespace=namespace,
            warm_start=kwargs.pop("warm_start", True),
            **kwargs,
        )

    return make, redis


def test_shared_generation_reloads_updates_and_clear(shared_cache_factory):
    make, _redis = shared_cache_factory
    writer = make(warm_start=False)
    reader = make()

    writer.store("question", "old answer", ["old context"])
    # The reader was started before the first write, so its generation changes
    # from missing to 1 on the first lookup and it loads the shared entry.
    assert reader.lookup("question")["answer"] == "old answer"

    writer.store("question", "new answer", ["new context"])
    assert reader.lookup("question")["answer"] == "new answer"

    writer.clear()
    assert reader.lookup("question") is None


def test_generation_polling_keeps_local_answer_during_redis_outage(shared_cache_factory):
    make, redis = shared_cache_factory
    cache = make(warm_start=False)
    cache.store("question", "answer", [])
    redis.fail = True

    # A failed generation read is not interpreted as a new empty generation.
    assert cache.lookup("question")["answer"] == "answer"
    assert cache.stats()["errors"] >= 1


def test_fail_closed_policy_never_returns_unverified_outage_snapshot(shared_cache_factory):
    make, redis = shared_cache_factory
    cache = make(redis_failure_policy="miss")
    cache.store("question", "private answer", [])
    assert cache.lookup("question")["answer"] == "private answer"
    redis.fail = True
    assert cache.lookup("question") is None


def test_serving_context_selects_fail_closed_cache(shared_cache_factory, monkeypatch):
    from serving import ServingContext
    _make, redis = shared_cache_factory
    context = ServingContext()
    monkeypatch.setattr(context, "embedding", lambda: _Embedder())
    monkeypatch.setattr(context, "available_keys", lambda: ["naive"])
    monkeypatch.setattr(context, "_cache_namespace", lambda _key: "serving-contract")
    cache = context.cache()
    cache.store("question", "answer", [])
    assert cache.lookup("question")["answer"] == "answer"
    redis.fail = True
    assert cache.lookup("question") is None
    assert cache.clear() is False
    redis.fail = False
    # A failed clear is not silently replayed later. Its caller must retry and
    # verify a successful receipt before claiming shared revocation.
    assert cache.clear() is True
    assert cache.lookup("question") is None


def test_snapshot_retries_when_clear_occurs_during_scan(shared_cache_factory, monkeypatch):
    make, _redis = shared_cache_factory
    writer = make()
    writer.store("question", "revoked", [])
    reader = make()
    original = reader._load_persisted_entries
    calls = []

    def racing_load():
        result = original()
        if not calls:
            calls.append(True)
            writer.clear()
        return result

    monkeypatch.setattr(reader, "_load_persisted_entries", racing_load)
    reader._reload_required = True
    assert reader.lookup("question") is None
    assert reader.stats()["entries"] == 0


def test_own_write_token_does_not_certify_an_incomplete_snapshot(shared_cache_factory):
    make, _redis = shared_cache_factory
    reader, writer = make(), make()
    # Emulate another process committing between reader's pre-write sync and
    # reader's transaction. Recording the latest token cannot certify its rows.
    writer.store("question", "remote", [])
    reader._pending_generation = writer._generation
    reader._record_generation_from_pipeline([])
    assert reader.lookup("question")["answer"] == "remote"


def test_local_eviction_cannot_delete_another_workers_persisted_answer(shared_cache_factory):
    make, redis = shared_cache_factory
    reader = make(max_entries=1)
    reader.store("old", "answer", [])
    key = reader._key_prefix + reader._cache_key("old")
    reader.store("new", "answer", [])
    assert key in redis.values


def test_unstable_snapshot_is_a_miss_not_partial_state(shared_cache_factory, monkeypatch):
    make, _redis = shared_cache_factory
    writer = make()
    writer.store("question", "answer", [])
    reader = make(redis_failure_policy="miss")
    original = reader._load_persisted_entries

    def racing_load():
        result = original()
        writer.store("question", "changed", [])
        return result

    monkeypatch.setattr(reader, "_load_persisted_entries", racing_load)
    reader._reload_required = True
    assert reader.lookup("question") is None
    assert reader.stats()["entries"] == 0


def test_zero_ttl_uses_persistent_set_and_generation(shared_cache_factory):
    make, redis = shared_cache_factory
    cache = make(ttl_seconds=0, warm_start=False)
    cache.store("question", "answer", [])

    entry_key = next(key for key in redis.values if ":entry:" in key)
    assert redis.values[entry_key]
    assert cache.lookup("question")["answer"] == "answer"
    assert len(cache.stats()["generation"]) == 32


def test_no_polling_documents_the_cross_worker_limitation(shared_cache_factory):
    make, _redis = shared_cache_factory
    writer = make(warm_start=False)
    writer.store("question", "old answer", [])
    reader = make(generation_poll_interval=None)
    assert reader.lookup("question")["answer"] == "old answer"

    writer.clear()
    # Without polling, the reader cannot observe a remote generation change;
    # this is the documented limitation rather than silent contract breakage.
    assert reader.lookup("question")["answer"] == "old answer"
    assert reader.stats()["generation_polling"] is False


def test_builder_lock_wait_is_bounded_and_file_is_cleaned(tmp_path):
    target = tmp_path / "chroma_db_full"
    entered = threading.Event()
    release = threading.Event()

    def holder():
        with build_lock_for(str(target), timeout=0.2):
            entered.set()
            release.wait(timeout=2)

    thread = threading.Thread(target=holder)
    thread.start()
    assert entered.wait(timeout=2)
    with pytest.raises(BuildLockError, match="Index build lock is held"):
        with build_lock_for(str(target), timeout=0.03):
            pass
    assert Path(f"{target}.build.lock").exists()
    release.set()
    thread.join(timeout=2)
    assert not Path(f"{target}.build.lock").exists()


def test_builder_lock_releases_after_exception(tmp_path):
    target = tmp_path / "child-index"
    with pytest.raises(RuntimeError, match="failed build"):
        with build_lock_for(str(target), timeout=0):
            raise RuntimeError("failed build")
    assert not Path(f"{target}.build.lock").exists()


def test_staging_cleanup_removes_only_dead_marked_stages(tmp_path):
    import preprocess_full

    target = tmp_path / "chroma_db_full"
    with build_lock_for(str(target), timeout=0) as lock:
        active = Path(create_staging_dir(str(tmp_path), ".chroma_db_full.", str(target), lock))
        stale = Path(create_staging_dir(str(tmp_path), ".chroma_db_full.", str(target), lock))
        stale_metadata = stale / ".build-staging.json"
        metadata = json.loads(stale_metadata.read_text())
        metadata["pid"] = 2**31
        atomic_write_json(str(stale_metadata), metadata)
        unknown = tmp_path / ".chroma_db_full.unknown"
        unknown.mkdir()

        preprocess_full._cleanup_staged_vector_stores(
            str(tmp_path), target_dir=str(target), lock=lock
        )

        assert active.exists()
        assert not stale.exists()
        assert unknown.exists()
        # The active stage is cleaned explicitly by the test after the lock
        # check; production paths remove it on publish or failure.
        import shutil
        shutil.rmtree(active)


def test_atomic_manifest_replace_preserves_previous_file_on_failure(tmp_path, monkeypatch):
    import module1.data_preprocessing as preprocessing

    destination = tmp_path / "manifest.json"
    destination.write_text('{"version": 1}')
    real_replace = preprocessing.os.replace

    def fail_replace(source, destination_path):
        if Path(destination_path) == destination:
            raise OSError("simulated atomic replace failure")
        return real_replace(source, destination_path)

    monkeypatch.setattr(preprocessing.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated atomic replace failure"):
        atomic_write_json(str(destination), {"version": 2})
    assert json.loads(destination.read_text())["version"] == 1
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


def test_failed_publication_restores_previous_store_and_releases_lock(tmp_path, monkeypatch):
    import preprocess_full

    staged = tmp_path / "staged"
    final = tmp_path / "chroma_db_full"
    staged.mkdir()
    final.mkdir()
    (staged / "chroma.sqlite3").write_text("new")
    (final / "chroma.sqlite3").write_text("old")
    real_replace = preprocess_full.os.replace

    def fail_promotion(source, destination):
        if Path(source) == staged and Path(destination) == final:
            raise OSError("promotion failed")
        return real_replace(source, destination)

    monkeypatch.setattr(preprocess_full.os, "replace", fail_promotion)
    with pytest.raises(OSError, match="promotion failed"):
        preprocess_full._publish_vector_store(str(staged), str(final))

    assert (final / "chroma.sqlite3").read_text() == "old"
    assert staged.exists()
    assert not Path(f"{final}.build.lock").exists()
