"""Regression cases demonstrated during independent validation."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from langchain_core.documents import Document

from module1.data_preprocessing import (
    _manifest_matches_current, build_vector_store,
    load_vector_store, rebuild_vector_store_safely,
)
from module3.evaluation import EvaluationPipeline


class Embedder:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def test_rebuild_rejects_stale_chroma_handles_and_retains_last_store(tmp_path):
    path = str(tmp_path / "store")
    old_docs = [Document(page_content="old corpus content")]
    old = build_vector_store(old_docs, Embedder(), path)
    with pytest.raises(RuntimeError, match="fresh process"):
        rebuild_vector_store_safely(
            [Document(page_content="new corpus content")], Embedder(), path
        )
    assert old.get(include=["documents"])["documents"] == ["old corpus content"]
    assert load_vector_store(Embedder(), path).get()["documents"] == ["old corpus content"]


def test_loading_an_empty_directory_does_not_create_a_store(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_vector_store(Embedder(), str(tmp_path))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("manifest", ["[]", "null", "false", "42", "\"value\""])
def test_non_object_manifest_is_incompatible_not_an_attribute_error(tmp_path, manifest):
    (tmp_path / "_index_manifest.json").write_text(manifest)
    assert _manifest_matches_current(str(tmp_path), 1) is False


def test_retrieval_only_never_calls_supplied_judge(tmp_path):
    class Judge:
        def evaluate(self, **kwargs):
            pytest.fail("retrieval-only must not call a judge")

    pipeline = EvaluationPipeline(Judge(), output_path=str(tmp_path / "out.csv"))
    frame = pipeline.evaluate_architecture(
        "fixture", SimpleNamespace(retrieve=lambda query: []),
        [{"question": "q", "ground_truth": "a"}], retrieval_only=True,
    )
    assert len(frame) == 1
    assert "faithfulness" not in frame


def test_bh_retains_small_nonzero_pvalues():
    from module3.evaluation import test_metric_significance as significance
    from scipy.stats import wilcoxon
    differences = np.linspace(.01, .2, 30)
    frame = pd.DataFrame([
        {"question": f"q{i}", "architecture": arch, "ndcg_at_10": value}
        for i, delta in enumerate(differences)
        for arch, value in (("Naive RAG", .1), ("Other", .1 + delta))
    ])
    row = significance(frame, metric="ndcg_at_10").iloc[0]
    expected = wilcoxon(differences).pvalue
    assert 0 < row["wilcoxon_p"] < 1e-6
    assert row["p_value_bh"] == pytest.approx(expected, rel=1e-10, abs=0)


def test_failed_judge_preflight_stops_before_generation(monkeypatch):
    import main
    import module3.evaluation as evaluation

    class BrokenJudge:
        def health_check(self):
            raise RuntimeError("fixture unavailable")

    monkeypatch.setattr(main, "OPENROUTER_API_KEY", "fixture-not-a-credential")
    monkeypatch.setattr(evaluation, "RAGASEvaluator", BrokenJudge)
    monkeypatch.setattr(evaluation, "EvaluationPipeline",
                        lambda **kwargs: pytest.fail("Failed preflight must stop before evaluation"))
    with pytest.raises(RuntimeError, match="refusing to spend"):
        main._run_evaluation({}, [], retrieval_only=False)


def test_non_ascii_wrong_api_key_is_unauthorized_not_type_error(monkeypatch):
    import api
    from fastapi import HTTPException
    monkeypatch.setattr(api, "API_KEY", "fixture-production-key")
    with pytest.raises(HTTPException) as error:
        api.require_api_key("é")
    assert error.value.status_code == 401
