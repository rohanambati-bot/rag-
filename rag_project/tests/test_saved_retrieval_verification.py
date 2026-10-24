import json
import hashlib
import math

import pytest

from scripts.verify_saved_retrieval import holm, paired, scores, verify


def test_independent_binary_metric_closed_forms():
    result = scores(["wrong", "yes"], {"yes", "missing"})
    assert result["ndcg_at_5"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert result["mrr_at_10"] == .5
    assert result["recall_at_k"] == .5
    assert result["precision_at_k"] == .2
    with pytest.raises(ValueError, match="duplicate"):
        scores(["yes", "yes"], {"yes"})
    with pytest.raises(ValueError, match="unjudged"):
        scores([], set())


def test_paired_zero_deltas_and_holm():
    result = paired([1, 2, 3], [1, 2, 3], repetitions=100)
    assert result["p_two_sided_monte_carlo"] == 1
    assert result["bootstrap_95_percentile"] == [0, 0]
    assert holm([.01, .04, .03]) == pytest.approx([.03, .06, .06])


def test_replay_is_create_only_and_checks_pools(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    names = ["dense_only", "dense_reranked", "bm25_only", "hybrid_no_rerank", "hybrid_reranked"]
    fingerprints = {key: "fixture" for key in ["contract_fingerprint", "corpus_fingerprint", "qrels_fingerprint",
                    "filter_fingerprint", "model_revisions_fingerprint", "package_metadata_fingerprint"]}
    fingerprints["filter_fingerprint"] = hashlib.sha256(b"{}").hexdigest()
    metadata = dict(fingerprints, filter={}, query_ids=["q"], conditions=[{"name": name} for name in names])
    contract = {"corpus": [{"doc_id": "yes"}, {"doc_id": "wrong"}], "qrels": {"q": {"yes": 1}}}
    rows = [dict(fingerprints, condition=name, query_id="q", top_k=5,
                 ranked_ids=["yes", "wrong"], candidate_pool_ids=["yes", "wrong"], relevant_ids=["yes"],
                 ndcg_at_10=1, **scores(["yes", "wrong"], {"yes"})) for name in names]
    (source / "contract.json").write_text(json.dumps(contract))
    (source / "run_metadata.json").write_text(json.dumps(metadata))
    (source / "condition_summary.csv").write_text("fixture\n")
    receipt = source / "per_query.jsonl"
    receipt.write_text("\n".join(map(json.dumps, rows)))
    destination = tmp_path / "verified"
    summary = verify(source, destination)
    assert summary["maximum_historical_metric_error"] == 0
    assert summary["identical_hybrid_pools"] == 1
    assert summary["derived_dense_differs_from_historical_query_ids"] == []
    with pytest.raises(FileExistsError):
        verify(source, destination)
    rows[-1]["candidate_pool_ids"] = ["wrong"]
    receipt.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="invalid candidate pool"):
        verify(source, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
