"""Fast offline tests for the controlled retrieval ablation harness.

The fakes below deliberately replace both retrieval stages and the reranker.  No
Chroma store, embedding weights, cross-encoder weights, dataset loader, or LLM is
used.  The tests protect the user's important experimental boundary: all five
conditions must share one contract while reranked stages see the complete explicit
candidate pool before top-k truncation.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from scripts.retrieval_ablation import (
    AblationError,
    AblationHarness,
    Candidate,
    CorpusDocument,
    DatasetContract,
    FilterPolicy,
    QueryRecord,
    RankedDocument,
    build_harness_from_manifest,
    LocalCrossEncoderReranker,
)
from scripts.run_retrieval_ablation_local import _retain_judged_pairs


class FakeRetriever:
    def __init__(self, rankings, revision):
        self.rankings = rankings
        self.model_revision = revision
        self.calls = []

    def retrieve(self, query_id, query_text, k):
        del query_text
        self.calls.append((query_id, k))
        return [RankedDocument(doc_id) for doc_id in self.rankings[query_id][:k]]


class FakeReranker:
    model_revision = "fake-cross-encoder@rev-1"

    def __init__(self):
        self.calls = []

    def rerank(self, query_id, query_text, candidates, top_k):
        del query_text
        self.calls.append((query_id, [candidate.doc_id for candidate in candidates], top_k))
        # The gold passage is intentionally buried in each first-stage pool.
        ordered = sorted(
            candidates,
            key=lambda candidate: ("gold" not in candidate.text, candidate.source_rank),
        )
        return [
            Candidate(
                doc_id=candidate.doc_id,
                text=candidate.text,
                source=candidate.source,
                source_rank=candidate.source_rank,
                retrieval_score=candidate.retrieval_score,
                fusion_score=candidate.fusion_score,
                sources=candidate.sources,
                rerank_score=1.0 if "gold" in candidate.text else 0.0,
            )
            for candidate in ordered[:top_k]
        ]


def _contract():
    corpus = (
        CorpusDocument("d-noise", "noise passage"),
        CorpusDocument("d-gold", "gold answer passage"),
        CorpusDocument("d-other", "other passage"),
        CorpusDocument("d-four", "fourth passage"),
    )
    queries = (
        QueryRecord("q-1", "what is the answer"),
        QueryRecord("q-2", "what else"),
    )
    qrels = {"q-1": {"d-gold": 1}, "q-2": {"d-other": 1}}
    return DatasetContract(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        corpus_revision="fixture-corpus@1",
        qrels_revision="fixture-qrels@1",
        filter_policy=FilterPolicy(name="fixture-filter", revision="1"),
    )


def _harness():
    dense = FakeRetriever(
        {
            "q-1": ["d-noise", "d-gold", "d-other", "d-four"],
            "q-2": ["d-noise", "d-other", "d-gold", "d-four"],
        },
        "fake-dense@rev-1",
    )
    bm25 = FakeRetriever(
        {
            "q-1": ["d-other", "d-gold", "d-noise", "d-four"],
            "q-2": ["d-noise", "d-other", "d-four", "d-gold"],
        },
        "fake-bm25@rev-1",
    )
    reranker = FakeReranker()
    return AblationHarness(
        _contract(),
        dense,
        bm25,
        reranker,
        top_k=1,
        candidate_pool_k=3,
        package_metadata={"python": "fixture", "packages": {"fake": "1"}},
    ), reranker


def test_five_conditions_share_contract_and_rerank_full_pool():
    harness, reranker = _harness()
    run = harness.run()

    assert [spec["name"] for spec in run.metadata["conditions"]] == [
        "dense_only",
        "dense_reranked",
        "bm25_only",
        "hybrid_no_rerank",
        "hybrid_reranked",
    ]
    assert len(run.per_query) == 10
    expected_query_ids = ["q-1", "q-2"]
    for condition in [spec["name"] for spec in run.metadata["conditions"]]:
        rows = [row for row in run.per_query if row["condition"] == condition]
        assert [row["query_id"] for row in rows] == expected_query_ids
        assert {row["top_k"] for row in rows} == {1}
        assert {row["contract_fingerprint"] for row in rows} == {
            run.metadata["contract_fingerprint"]
        }
        assert {row["filter_fingerprint"] for row in rows} == {
            run.per_query[0]["filter_fingerprint"]
        }
        assert {row["model_revisions_fingerprint"] for row in rows} == {
            run.metadata["model_revisions_fingerprint"]
        }
        assert {row["package_metadata_fingerprint"] for row in rows} == {
            run.metadata["package_metadata_fingerprint"]
        }

    dense_reranked = [row for row in run.per_query if row["condition"] == "dense_reranked"]
    hybrid_reranked = [row for row in run.per_query if row["condition"] == "hybrid_reranked"]
    assert all(row["candidate_pool_size"] == 3 for row in dense_reranked)
    assert all(row["candidate_pool_size"] >= 3 for row in hybrid_reranked)
    assert all(row["reranker_candidate_count"] == row["candidate_pool_size"] for row in dense_reranked)
    assert all(row["reranker_candidate_count"] == row["candidate_pool_size"] for row in hybrid_reranked)
    assert len(reranker.calls) == 4  # two queries x two reranked conditions
    assert all(len(candidate_ids) >= 3 for _, candidate_ids, _ in reranker.calls)

    # The fake cross-encoder promotes a buried answer; rank-aware project metrics
    # observe the intended second-stage effect.
    dense_only_q1 = next(
        row for row in run.per_query if row["condition"] == "dense_only" and row["query_id"] == "q-1"
    )
    dense_reranked_q1 = next(
        row
        for row in run.per_query
        if row["condition"] == "dense_reranked" and row["query_id"] == "q-1"
    )
    assert dense_only_q1["ranked_ids"] == ["d-noise"]
    assert dense_reranked_q1["ranked_ids"] == ["d-gold"]
    assert dense_reranked_q1["mrr_at_10"] > dense_only_q1["mrr_at_10"]


def test_shared_filter_removes_same_doc_for_every_condition():
    harness, _ = _harness()
    harness.contract = DatasetContract(
        corpus=harness.contract.corpus,
        queries=harness.contract.queries,
        qrels=harness.contract.qrels,
        corpus_revision=harness.contract.corpus_revision,
        qrels_revision=harness.contract.qrels_revision,
        filter_policy=FilterPolicy(
            name="exclude-fourth", revision="1", allowed_doc_ids=frozenset({"d-noise", "d-gold", "d-other"})
        ),
    )
    run = harness.run()
    for row in run.per_query:
        assert "d-four" not in row["candidate_pool_ids"]
        assert "d-four" not in row["ranked_ids"]


def test_create_only_outputs_refuse_existing_directory(tmp_path):
    harness, _ = _harness()
    run = harness.run()
    output_dir = tmp_path / "ablation-v1"
    run.write_outputs(output_dir)
    assert (output_dir / "run_metadata.json").is_file()
    assert (output_dir / "per_query.jsonl").is_file()
    assert (output_dir / "condition_summary.csv").is_file()
    with pytest.raises(FileExistsError):
        run.write_outputs(output_dir)


def test_invalid_reranker_cannot_invent_a_document():
    harness, _ = _harness()

    class InventingReranker(FakeReranker):
        def rerank(self, query_id, query_text, candidates, top_k):
            del query_id, query_text, candidates, top_k
            return [RankedDocument("not-in-pool")]

    harness.reranker = InventingReranker()
    with pytest.raises(AblationError, match="not-in-pool"):
        harness.run()


def test_manifest_replay_is_local_and_deterministic(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "corpus": [
                    {"doc_id": "d1", "text": "noise"},
                    {"doc_id": "d2", "text": "gold"},
                ],
                "queries": [{"query_id": "q1", "text": "question"}],
                "qrels": {"q1": {"d2": 1}},
                "rankings": {
                    "dense": {"q1": ["d1", "d2"]},
                    "bm25": {"q1": ["d2", "d1"]},
                },
                "top_k": 1,
                "candidate_pool_k": 2,
                "corpus_revision": "corpus@1",
                "qrels_revision": "qrels@1",
                "model_revisions": {
                    "dense": "dense@1",
                    "bm25": "bm25@1",
                    "reranker": "reranker@1",
                },
                "reranker_scores": {"q1": {"d1": 0.0, "d2": 1.0}},
                "package_metadata": {"python": "fixture", "packages": {"x": "1"}},
            }
        ),
        encoding="utf-8",
    )
    first = build_harness_from_manifest(manifest).run()
    second = build_harness_from_manifest(manifest).run()
    assert first.metadata["contract_fingerprint"] == second.metadata["contract_fingerprint"]
    assert first.per_query == second.per_query
    assert first.summary == second.summary


def test_local_cross_encoder_passes_local_files_only_without_duplicate_keyword(
    tmp_path, monkeypatch
):
    model_dir = tmp_path / "cross-encoder"
    model_dir.mkdir()
    calls = []

    class FakeCrossEncoder:
        def __init__(self, model_name, *, device, local_files_only):
            calls.append((model_name, device, local_files_only))

        def predict(self, pairs, show_progress_bar):
            del show_progress_bar
            return [float(len(passage)) for _query, passage in pairs]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    reranker = LocalCrossEncoderReranker(
        str(model_dir), model_revision="fixture-reranker@1", device="cpu"
    )
    output = reranker.rerank(
        "q-1",
        "question",
        [
            Candidate("d1", "short", "dense", 1),
            Candidate("d2", "longer passage", "dense", 2),
        ],
        top_k=1,
    )
    assert calls == [(str(model_dir), "cpu", True)]
    assert [candidate.doc_id for candidate in output] == ["d2"]


def test_local_protocol_excludes_unjudged_queries_from_metric_denominator():
    selected = [
        {"id": "q-judged", "question": "judged"},
        {"id": "q-unjudged", "question": "unjudged"},
    ]
    judged, excluded = _retain_judged_pairs(
        selected,
        {"judged": {"d1": 1}},
    )
    assert [row["id"] for row in judged] == ["q-judged"]
    assert excluded == ["q-unjudged"]
