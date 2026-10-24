"""Regression tests for full-validation index publication."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def test_published_store_and_manifest_can_be_loaded(tmp_path):
    """A completed staged Chroma store remains usable after publication."""
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document

    from module1.data_preprocessing import load_vector_store, write_index_manifest
    from preprocess_full import _publish_vector_store

    class FakeEmbeddings:
        def embed_documents(self, texts):
            return [[float(len(text)), 0.0, 0.0] for text in texts]

        def embed_query(self, text):
            return [float(len(text)), 0.0, 0.0]

    staged_dir = tmp_path / "staged"
    final_dir = tmp_path / "chroma_db_full"
    Chroma.from_documents(
        [Document(page_content="a persisted validation passage", metadata={"source": "test"})],
        embedding=FakeEmbeddings(),
        persist_directory=str(staged_dir),
    )
    write_index_manifest(str(staged_dir), num_chunks=1)

    _publish_vector_store(str(staged_dir), str(final_dir))

    manifest = json.loads((final_dir / "_index_manifest.json").read_text())
    loaded = load_vector_store(FakeEmbeddings(), persist_dir=str(final_dir))
    assert manifest["num_chunks"] == 1
    assert loaded._collection.count() == 1
    assert not staged_dir.exists()


def test_publish_failure_restores_previous_store(tmp_path, monkeypatch):
    """A failed directory promotion leaves the previous store and manifest intact."""
    import preprocess_full

    staged_dir = tmp_path / "staged"
    final_dir = tmp_path / "chroma_db_full"
    staged_dir.mkdir()
    final_dir.mkdir()
    (staged_dir / "chroma.sqlite3").write_text("new")
    (staged_dir / "_index_manifest.json").write_text('{"num_chunks": 2}')
    (final_dir / "chroma.sqlite3").write_text("old")
    (final_dir / "_index_manifest.json").write_text('{"num_chunks": 1}')

    real_replace = os.replace

    def fail_new_store(source, destination):
        if Path(source) == staged_dir and Path(destination) == final_dir:
            raise OSError("simulated promotion failure")
        return real_replace(source, destination)

    monkeypatch.setattr(preprocess_full.os, "replace", fail_new_store)
    with pytest.raises(OSError, match="simulated promotion failure"):
        preprocess_full._publish_vector_store(str(staged_dir), str(final_dir))

    assert (final_dir / "chroma.sqlite3").read_text() == "old"
    assert json.loads((final_dir / "_index_manifest.json").read_text())["num_chunks"] == 1
    assert staged_dir.exists()
    assert not (tmp_path / "chroma_db_full.previous").exists()


def test_interrupted_publish_backup_is_recovered(tmp_path):
    """A later run restores a store left behind by a process interruption."""
    from preprocess_full import _recover_previous_vector_store

    backup_dir = tmp_path / "chroma_db_full.previous"
    backup_dir.mkdir()
    (backup_dir / "chroma.sqlite3").write_text("previous")

    _recover_previous_vector_store(str(tmp_path / "chroma_db_full"))

    assert (tmp_path / "chroma_db_full" / "chroma.sqlite3").read_text() == "previous"
    assert not backup_dir.exists()


def test_full_build_keeps_completed_store_and_preserves_it_on_rebuild_error(
    tmp_path, monkeypatch
):
    """The script publishes the manifest and never removes the last good build."""
    import importlib

    import preprocess_full

    class FakeDocument:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class FakeSplitter:
        def __init__(self, **kwargs):
            pass

        def split_documents(self, documents):
            return documents

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            pass

    class FakeDataset:
        rows = [{
            "answers": ["A validation answer"],
            "query": "Which answer is validated?",
            "query_type": "entity",
            "passages": {
                "passage_text": [
                    "The first validation passage contains enough text for indexing.",
                    "The second validation passage also contains enough text for indexing.",
                ],
                "is_selected": [1, 0],
                "url": ["", ""],
            },
        }]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    class FakeChroma:
        fail = False

        def __init__(self, persist_directory, count):
            self.persist_directory = Path(persist_directory)
            self.count = count

        @classmethod
        def from_documents(cls, documents, embedding, persist_directory):
            if cls.fail:
                raise RuntimeError("simulated embedding failure")
            path = Path(persist_directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / "chroma.sqlite3").write_text("completed build")
            return cls(persist_directory, len(documents))

        def add_documents(self, documents):
            self.count += len(documents)

        def persist(self):
            return None

    datasets = importlib.import_module("datasets")
    langchain_documents = importlib.import_module("langchain_core.documents")
    splitters = importlib.import_module("langchain_text_splitters")
    embeddings = importlib.import_module("langchain_community.embeddings")
    vectorstores = importlib.import_module("langchain_community.vectorstores")
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: FakeDataset())
    monkeypatch.setattr(langchain_documents, "Document", FakeDocument)
    monkeypatch.setattr(splitters, "RecursiveCharacterTextSplitter", FakeSplitter)
    monkeypatch.setattr(embeddings, "HuggingFaceEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(vectorstores, "Chroma", FakeChroma)
    # Keep this orchestration test model-free. Real process/native writer
    # lifecycle is exercised separately in test_index_lifecycle_coverage.py.
    import module1.data_preprocessing as preprocessing

    def fake_isolated_build(builder, docs_file, stage, *options):
        if FakeChroma.fail:
            raise RuntimeError("simulated embedding failure")
        Path(stage, "chroma.sqlite3").write_text("completed build")
        preprocessing.write_index_manifest(stage, num_chunks=2)
        return 2

    monkeypatch.setattr(preprocessing, "run_isolated_index_builder", fake_isolated_build)

    final_dir = tmp_path / "chroma_db_full"
    monkeypatch.setattr(preprocess_full, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(preprocess_full, "CHECKPOINT_FILE", str(tmp_path / "checkpoint.json"))
    monkeypatch.setattr(preprocess_full, "SYNTHETIC_DATA_PATH", str(tmp_path / "synthetic.json"))
    monkeypatch.setattr(preprocess_full, "VECTOR_STORE_DIR", str(final_dir))
    monkeypatch.setattr(preprocess_full, "BATCH_SIZE", 1)
    monkeypatch.setattr(preprocess_full, "NUM_SYNTHETIC_SAMPLES", 1)

    preprocess_full.main()
    assert (final_dir / "chroma.sqlite3").read_text() == "completed build"
    assert (final_dir / "_index_manifest.json").exists()
    assert json.loads((final_dir / "_index_manifest.json").read_text())["num_chunks"] == 2

    FakeChroma.fail = True
    with pytest.raises(RuntimeError, match="simulated embedding failure"):
        preprocess_full.main()

    assert (final_dir / "chroma.sqlite3").read_text() == "completed build"
    assert (final_dir / "_index_manifest.json").exists()
