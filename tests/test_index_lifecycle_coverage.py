"""Inventory plus real temporary-store lifecycle boundaries; no models/providers."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

from scripts.index_lifecycle_inventory import Inventory, build_inventory
from scripts.validate_live_lifecycle import CTX, DeterministicEmbeddings, Worker


def test_supported_persistent_constructors_are_only_shared_guarded_implementations():
    report = build_inventory()
    assert not report["unguarded_raw_opens"]
    persistent = [row for row in report["entries"] if row["operation"] == "raw-persistent-open"
                  and not row["path"].startswith("tests/")]
    assert persistent
    assert all(row["boundary"] == "guarded-shared-implementation" for row in persistent)
    assert not [row for row in report["entries"] if row["path"].startswith("module2/")
                and row["operation"] in {"raw-persistent-open", "raw-dynamic-open"}]


def test_inventory_detects_aliased_persistent_and_dynamic_bypasses():
    visitor = Inventory("query.py")
    visitor.visit(ast.parse("""
from langchain_chroma import Chroma as Raw
from chromadb import PersistentClient as Persistent
Raw.from_documents([], persist_directory='live')
Persistent(path='live')
Raw(**options)
"""))
    assert [row["operation"] for row in visitor.rows] == [
        "raw-persistent-open", "raw-persistent-open", "raw-dynamic-open",
    ]
    assert all(row["boundary"] == "UNGUARDED" for row in visitor.rows)


def _paused_first_builder(connection, path, fail):
    from langchain_core.documents import Document
    from module1.data_preprocessing import build_vector_store, load_vector_store

    class PausedEmbeddings(DeterministicEmbeddings):
        def embed_documents(self, texts):
            connection.send("writer-held-before-embedding")
            assert connection.recv() == "release"
            if fail:
                raise RuntimeError("injected embedding failure")
            return super().embed_documents(texts)

    try:
        store = build_vector_store([Document(page_content="first corpus")], PausedEmbeddings(), path)
        connection.send(store.get()["documents"])
    except RuntimeError as exc:
        connection.send(str(exc))
        try:
            load_vector_store(DeterministicEmbeddings(), path)
        except Exception as denied:
            connection.send(type(denied).__name__)
    assert connection.recv() == "exit"
    connection.close()


@pytest.mark.parametrize("fail", [False, True])
def test_first_build_barrier_lifetime_failure_and_restart(tmp_path, fail):
    path = str(tmp_path / "live")
    parent, child = CTX.Pipe()
    process = CTX.Process(target=_paused_first_builder, args=(child, path, fail))
    process.start()
    child.close()
    try:
        assert parent.poll(40) and parent.recv() == "writer-held-before-embedding"
        reader = Worker("index")
        writer = Worker("index")
        try:
            reader.send("read", path)
            assert reader.receive()["error"] == "BuildLockError"
            # First-build writers use the same barrier even before any documents
            # exist. No raw native open can sneak in while embedding is paused.
            writer.send("build", path, "racing corpus")
            assert writer.receive()["error"] == "BuildLockError"
        finally:
            reader.close()
            writer.close()
        parent.send("release")
        assert parent.poll(40)
        result = parent.recv()
        if fail:
            assert result == "injected embedding failure"
            assert parent.poll(40) and parent.recv() == "BuildLockError"
        else:
            assert result == ["first corpus"]
        # Failed writers retain EX; successful writers retain SH. Both refuse
        # publication/recovery until the native-owning process actually exits.
        recovery = Worker("index")
        try:
            recovery.send("recover", path)
            assert recovery.receive()["error"] == "BuildLockError"
        finally:
            recovery.close()
        parent.send("exit")
        process.join(15)
        assert process.exitcode == 0
        fresh = Worker("index")
        try:
            if fail:
                assert fresh.call("rebuild", path, "recovered corpus") == ["recovered corpus"]
                assert Path(path + ".previous", "chroma.sqlite3").is_file()
            else:
                assert fresh.call("read", path) == ["first corpus"]
        finally:
            fresh.close()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(15)
        parent.close()


def test_real_snapshot_active_reader_recovery_and_child_reopening(tmp_path):
    path, snapshot = str(tmp_path / "children"), str(tmp_path / "snapshot")
    builder = Worker("index")
    try:
        assert builder.call("rebuild", path, "child corpus") == ["child corpus"]
        # Rebuilding a native-cached path in the same process is refused before
        # writing anything, rather than reopening an old path-keyed Chroma client.
        builder.send("rebuild", path, "stale native replacement")
        assert builder.receive()["error"] == "BuildLockError"
        check = Worker("index")
        try:
            check.send("snapshot", path, snapshot)
            assert check.receive()["error"] == "BuildLockError"
            check.send("recover", path)
            assert check.receive()["error"] == "BuildLockError"
            assert not Path(snapshot).exists()
        finally:
            check.close()
    finally:
        builder.close()
    copier = Worker("index")
    try:
        assert copier.call("snapshot", path, snapshot) == os.path.realpath(snapshot)
        assert copier.call("read", snapshot) == ["child corpus"]
    finally:
        copier.close()
    # Simulate a process interrupted after moving live -> .previous. Recovery
    # runs with both builder serialization and the lifetime reader barrier.
    os.replace(path, path + ".previous")
    restart = Worker("index")
    try:
        assert restart.call("recover", path)
        assert restart.call("read", path) == ["child corpus"]
    finally:
        restart.close()


def test_internal_publish_and_recovery_fail_closed_without_guard(tmp_path):
    from module1.data_preprocessing import BuildLockError, index_publication_guard
    from preprocess_full import (
        _publish_vector_store_without_readers, _recover_previous_vector_store_without_readers,
    )
    path = str(tmp_path / "live")
    with pytest.raises(BuildLockError, match="active publication guard"):
        _publish_vector_store_without_readers(str(tmp_path / "stage"), path)
    with pytest.raises(BuildLockError, match="active publication guard"):
        _recover_previous_vector_store_without_readers(path)
    with index_publication_guard(path):
        with pytest.raises(BuildLockError, match="Nested"):
            with index_publication_guard(path):
                pytest.fail("nested exclusive guards must never deadlock")


def test_publish_retains_each_last_good_corpus(tmp_path):
    from preprocess_full import _publish_vector_store
    final = tmp_path / "live"
    for i in range(3):
        stage = tmp_path / f"stage-{i}"
        stage.mkdir()
        (stage / "chroma.sqlite3").write_text(str(i))
        _publish_vector_store(str(stage), str(final))
    assert (final / "chroma.sqlite3").read_text() == "2"
    assert Path(str(final) + ".previous", "chroma.sqlite3").read_text() == "1"
    archived = [path for path in tmp_path.glob("live.previous.*") if path.is_dir()]
    assert len(archived) == 1
    assert (archived[0] / "chroma.sqlite3").read_text() == "0"


def _crash_during_publication(connection, path):
    from module1.data_preprocessing import build_lock_for, index_publication_guard
    with build_lock_for(path), index_publication_guard(path), index_publication_guard(path + ".previous"):
        os.replace(path, path + ".previous")
        connection.send("interrupted-after-backup-rename")
        connection.close()
        os._exit(0)  # intentionally skip finally/atexit, reproducing owner death


def test_crashed_publisher_restart_recovers_real_store_and_archives_receipt(tmp_path):
    path = str(tmp_path / "live")
    builder = Worker("index")
    try:
        assert builder.call("build", path, "last good corpus") == ["last good corpus"]
    finally:
        builder.close()
    parent, child = CTX.Pipe()
    process = CTX.Process(target=_crash_during_publication, args=(child, path))
    process.start()
    child.close()
    try:
        assert parent.poll(40) and parent.recv() == "interrupted-after-backup-rename"
        process.join(15)
        assert process.exitcode == 0
        assert not Path(path).exists()
        assert Path(path + ".build.lock").exists()
        # Opening the backup is a supported explicit-path read. Recovery must
        # refuse to rename even that alias while its native reader is alive.
        backup_reader = Worker("index")
        recovery = Worker("index")
        try:
            assert backup_reader.call("read", path + ".previous") == ["last good corpus"]
            recovery.send("recover", path)
            assert recovery.receive()["error"] == "BuildLockError"
            assert not Path(path).exists()
        finally:
            backup_reader.close()
            recovery.close()
        restart = Worker("index")
        try:
            assert restart.call("recover", path)
            assert restart.call("read", path) == ["last good corpus"]
        finally:
            restart.close()
        assert list(tmp_path.glob("live.build.lock.abandoned.*"))
        assert not Path(path + ".build.lock").exists()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(15)
        parent.close()


def test_train_replacement_preserves_real_previous_corpus_on_build_failure(tmp_path, monkeypatch):
    import preprocess_train
    import module1.data_preprocessing as preprocessing
    path = str(tmp_path / "train")
    docs_file = tmp_path / "documents.jsonl"
    docs_file.write_text(json.dumps({"content": "a train passage"}) + "\n")
    for name, value in {
        "DATA_DIR": str(tmp_path), "VECTOR_STORE_DIR": path,
        "DOCS_FILE": str(docs_file), "CHECKPOINT_FILE": str(tmp_path / "checkpoint.json"),
    }.items():
        monkeypatch.setattr(preprocess_train, name, value)
    old = Worker("index")
    try:
        old.call("build", path, "last good training corpus")
    finally:
        old.close()

    def completed_build(builder, source, stage, *options):
        writer = Worker("index")
        try:
            assert writer.call("build", stage, "replacement training corpus") == ["replacement training corpus"]
        finally:
            writer.close()
        return 1

    monkeypatch.setattr(preprocessing, "run_isolated_index_builder", completed_build)
    assert preprocess_train.phase_embed() == 1
    assert docs_file.read_text() == json.dumps({"content": "a train passage"}) + "\n"

    def failed_build(*args):
        raise RuntimeError("injected train build failure")

    monkeypatch.setattr(preprocessing, "run_isolated_index_builder", failed_build)
    with pytest.raises(RuntimeError, match="injected train build failure"):
        preprocess_train.phase_embed()
    check = Worker("index")
    try:
        assert check.call("read", path) == ["replacement training corpus"]
        assert check.call("read", path + ".previous") == ["last good training corpus"]
    finally:
        check.close()


def test_real_shard_merge_writer_and_source_reader_sessions(tmp_path, monkeypatch):
    import train_in_shards
    from langchain_core.documents import Document
    from module1.data_preprocessing import build_vector_store, persistent_index_writer
    shard_dir = tmp_path / "shards"
    checkpoint_dir = tmp_path / "checkpoints"
    monkeypatch.setattr(train_in_shards, "SHARDS_DIR", str(shard_dir))
    monkeypatch.setattr(train_in_shards, "CHECKPOINT_DIR", str(checkpoint_dir))
    monkeypatch.setattr(train_in_shards, "get_embedding_model", DeterministicEmbeddings)
    build_vector_store(
        [Document(page_content="first shard passage", metadata={"source": "fixture"})],
        DeterministicEmbeddings(), str(shard_dir / "shard_0"),
    )
    train_in_shards.save_shard_checkpoint(0, {"status": "done", "total_chunks": 1})
    stage = str(tmp_path / "merged-stage")
    with persistent_index_writer(stage, DeterministicEmbeddings(), create=True) as store:
        assert train_in_shards._merge_shards_into(store) == 1
        assert store.get()["documents"] == ["first shard passage"]
    train_in_shards.save_shard_checkpoint(0, {"status": "in_progress", "total_chunks": 2})
    with pytest.raises(RuntimeError, match="incomplete"):
        train_in_shards._merge_shards_into(store)


def test_full_batch_builder_with_real_store_and_deterministic_vectors(tmp_path):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from module1.data_preprocessing import persistent_index_writer
    from preprocess_full import _add_full_batches
    source = tmp_path / "documents.jsonl"
    source.write_text("\n".join(json.dumps({"content": text, "answer": "fixture"}) for text in (
        "first full-build passage", "second full-build passage",
    )))
    with persistent_index_writer(str(tmp_path / "stage"), DeterministicEmbeddings(), create=True) as store:
        assert _add_full_batches(str(source), store, RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=0,
        ), 1) == 2
        assert sorted(store.get()["documents"]) == ["first full-build passage", "second full-build passage"]
