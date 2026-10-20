# tests/test_audit_regressions.py
"""
Regression tests for issues found and fixed during the project audit.

Each test maps to a confirmed defect:

* ``compute_noise_fingerprint`` returned ``{"topical": ..}`` while
  ``evaluate_sample`` extracts ``["topical_noise"]`` — a KeyError that silently
  dropped every sample whose retrieval returned nothing.
* ``main._store_dir`` mapped both ``--store full`` and ``--store train`` to the
  same directory whenever VECTOR_STORE_DIR defaulted to the train store, so
  `--store full` evaluated against the wrong corpus.
* ``query.py`` bootstrapped hybrid/parent-child from SQuAD while the dense store
  holds MS MARCO — a split-corpus serving stack.
* New stores carry an index manifest so a stale/mismatched index is detectable.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


# ── compute_noise_fingerprint (empty retrieval) ──────────────────────────────

def test_noise_fingerprint_empty_retrieval_uses_expected_keys():
    """Empty retrieved text must return the *_noise keys, not {'topical': ..}."""
    from module3.evaluation import compute_noise_fingerprint

    fp = compute_noise_fingerprint("", "Paris", "What is the capital of France?")
    assert set(fp) == {"topical_noise", "peripheral_noise", "random_noise"}
    assert fp == {"topical_noise": 0.0, "peripheral_noise": 0.0, "random_noise": 0.0}


def test_noise_fingerprint_keys_match_evaluate_sample_extraction():
    """The non-empty path must use the same key names (no cross-path drift)."""
    from module3.evaluation import compute_noise_fingerprint

    fp = compute_noise_fingerprint(
        "Paris is the capital city of France.",
        "Paris",
        "What is the capital of France?",
    )
    for key in ("topical_noise", "peripheral_noise", "random_noise"):
        assert key in fp
        assert isinstance(fp[key], float)


def test_evaluate_sample_tolerates_empty_retrieval():
    """A sample whose retrieval returns nothing must be scored, not dropped.

    Regression: the fingerprint KeyError propagated through evaluate_architecture's
    per-sample try/except, so affected samples vanished from the results silently.
    """
    from module3.evaluation import EvaluationPipeline

    class EmptyRAG:
        name = "EmptyRAG"

        def run(self, query: str):
            return {"answer": "", "context_docs": [], "contexts": [],
                    "latency_ms": 0.0}

    pipeline = EvaluationPipeline(output_path=os.devnull)
    df = pipeline.evaluate_architecture(
        architecture_name="EmptyRAG",
        rag_system=EmptyRAG(),
        synthetic_dataset=[{"question": "q?", "ground_truth": "gt"}],
    )
    assert len(df) == 1
    row = df.iloc[0]
    assert row["num_docs_retrieved"] == 0
    assert row["precision_omega"] == 0.0
    assert row["noise_fingerprint_topical"] == 0.0


# ── --store shorthand resolution ──────────────────────────────────────────────

@pytest.mark.parametrize("default_store", ["full", "train", "custom"])
def test_store_shorthand_full_and_train_are_distinct_paths(tmp_path, monkeypatch, default_store):
    """Resolve shorthands independently of local datasets and the configured default."""
    import configs.config as cfg
    import main as main_mod

    full_dir = tmp_path / "chroma_db_full"
    train_dir = tmp_path / "chroma_db_train"
    custom_dir = tmp_path / "custom_store"
    for directory in (full_dir, train_dir, custom_dir):
        directory.mkdir()
    monkeypatch.setattr(cfg, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "VECTOR_STORE_DIR", str({
        "full": full_dir, "train": train_dir, "custom": custom_dir,
    }[default_store]))

    full = main_mod._store_dir(SimpleNamespace(store="full"))
    train = main_mod._store_dir(SimpleNamespace(store="train"))
    assert full == str(full_dir)
    assert train == str(train_dir)
    assert full != train


def test_store_shorthand_rejects_missing_custom_path(tmp_path, caplog):
    """An explicit --store path that does not exist exits with a clear error."""
    import logging

    import main as main_mod

    missing = tmp_path / "missing_custom_store"
    with caplog.at_level(logging.ERROR, logger="main"), pytest.raises(SystemExit) as caught:
        main_mod._store_dir(SimpleNamespace(store=str(missing)))
    assert caught.value.code == 1
    assert any("Vector store not found" in rec.message for rec in caplog.records)


@pytest.mark.parametrize("store", ["full", "train"])
def test_store_shorthand_rejects_missing_named_store(tmp_path, monkeypatch, caplog, store):
    """Keep the real missing-corpus guard: only the test fixtures create stores."""
    import configs.config as cfg
    import main as main_mod

    monkeypatch.setattr(cfg, "DATA_DIR", str(tmp_path))
    with pytest.raises(SystemExit) as caught:
        main_mod._store_dir(SimpleNamespace(store=store))
    assert caught.value.code == 1
    assert any("Vector store not found" in rec.message for rec in caplog.records)
    assert list(tmp_path.iterdir()) == []


def test_config_default_vector_store_is_the_full_store():
    """configs default must match the documented eval/cicd contract."""
    from configs.config import DATA_DIR, VECTOR_STORE_DIR

    assert VECTOR_STORE_DIR == os.path.join(DATA_DIR, "chroma_db_full")


# ── query.py corpus consistency ──────────────────────────────────────────────

def test_query_cli_uses_msmarco_not_squad():
    """query.py must build its BM25 corpus from MS MARCO, same as the dense store."""
    root = Path(__file__).resolve().parent.parent
    src = (root / "query.py").read_text(encoding="utf-8")
    assert "load_msmarco_documents" in src
    # The SQuAD fallback must no longer feed retrieval corpora.
    assert "chunk_documents(load_squad_documents())" not in src


# ── index manifest (stale/mismatched index detection) ────────────────────────

def test_index_manifest_written_and_roundtrips(tmp_path):
    from module1.data_preprocessing import (
        _check_index_manifest,
        _index_manifest_path,
        write_index_manifest,
    )

    store_dir = tmp_path / "chroma_test_db"
    store_dir.mkdir()
    write_index_manifest(str(store_dir), num_chunks=42)

    manifest_path = Path(_index_manifest_path(str(store_dir)))
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["num_chunks"] == 42
    assert manifest["embedding_model"] == importlib.import_module(
        "configs.config"
    ).EMBEDDING_MODEL

    # Compatible manifest → no warnings; incompatible → warning about the model.
    _check_index_manifest(str(store_dir))  # must not raise


def test_index_manifest_warns_on_embedding_model_mismatch(tmp_path, caplog):
    import logging

    import module1.data_preprocessing as dp

    store_dir = tmp_path / "chroma_misc"
    store_dir.mkdir()
    dp.write_index_manifest(str(store_dir), num_chunks=1)

    original = dp.EMBEDDING_MODEL
    dp.EMBEDDING_MODEL = "sentence-transformers/some-other-model"
    try:
        with caplog.at_level(logging.WARNING, logger="module1.data_preprocessing"):
            dp._check_index_manifest(str(store_dir))
    finally:
        dp.EMBEDDING_MODEL = original

    assert any("embedding model" in rec.message.lower() for rec in caplog.records)


# ── generation-grounding hardening ───────────────────────────────────────────

def test_answer_prompt_treats_context_as_untrusted():
    """Retrieved documents may contain hostile instructions; the prompt must say so."""
    from module2.rag_architectures import ANSWER_PROMPT

    assert "untrusted" in ANSWER_PROMPT.lower()
    assert "ignore any instructions" in ANSWER_PROMPT.lower()
    # The formatting contract must stay intact.
    assert ANSWER_PROMPT.format(context="ctx", question="q?").startswith("You are a")


# ── preprocess_full synthetic context quality ────────────────────────────────

def test_selected_context_joins_only_annotated_passages():
    from preprocess_full import selected_context

    passage_texts = ["Alpha passage.", "Beta passage.", "Gamma passage."]
    is_selected = [0, 1, 1]
    ctx = selected_context(passage_texts, is_selected)
    assert ctx == "Beta passage. Gamma passage."


def test_selected_context_no_selected_passages_or_mismatched_lengths():
    from preprocess_full import selected_context

    assert selected_context(["A.", "B."], [0, 0]) == ""
    # is_selected shorter than passage list must not raise.
    assert selected_context(["A.", "B."], [1]) == "A."


def test_preprocessing_rejects_invalid_limits_and_chunk_overlap():
    """Invalid user/configuration bounds fail before a dataset or splitter call."""
    from module1.data_preprocessing import chunk_documents, load_msmarco_documents

    assert load_msmarco_documents(max_samples=0) == []
    with pytest.raises(ValueError, match="non-negative"):
        load_msmarco_documents(max_samples=-1)
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        chunk_documents([], chunk_size=10, chunk_overlap=10)


# ── judge pacing (shared provider scheduler) ─────────────────────────────────

def test_judge_call_is_paced_by_shared_provider_scheduler(monkeypatch):
    """RAGASEvaluator._call must reserve a provider slot before hitting the API.

    The 2026-10-18 judged run made 866 unpaced 429s against Groq's judge model
    (judge had its own un-paced client) and died with the whole eval — delayed
    back to the generator's LLM_RATE_LIMIT_SECONDS pacing, then real API call.
    """
    import module3.evaluation as ev

    calls = []

    def fake_wait(provider, *, max_wait=None):
        calls.append((provider, max_wait))
        return True

    class FakeClient:
        class chat:  # noqa: N801
            class completions:
                @staticmethod
                def create(model, messages, temperature, max_tokens):
                    return type("R", (), {"choices": [
                        type("C", (), {"message": type(
                            "M", (), {"content": "1.0"})()})()
                    ]})()

    monkeypatch.setattr(ev, "_judge_wait", fake_wait, raising=True)
    evaluator = ev.RAGASEvaluator(provider="groq", model="test/model")
    evaluator._client = FakeClient()

    score = evaluator._score("probe")
    assert score == 1.0
    # Pacing happened BEFORE the API call, and the judge waits (never abstains).
    assert calls and calls[0][0] == "groq" and calls[0][1] is None
