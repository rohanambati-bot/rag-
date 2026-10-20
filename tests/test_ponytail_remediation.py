"""Small regressions for the second audit's correctness fixes."""

from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import pytest


def test_evaluation_records_failed_samples_and_invalidates_gate(tmp_path):
    from module3.evaluation import EvaluationPipeline

    class RAG:
        def run(self, question):
            if question == "bad":
                raise RuntimeError("provider failed")
            return {"answer": "answer", "context_docs": [], "contexts": [], "latency_ms": 1}

    pipeline = EvaluationPipeline(output_path=str(tmp_path / "summary.csv"))
    frame = pipeline.evaluate_architecture(
        "fixture", RAG(),
        [{"question": "good", "ground_truth": "answer"},
         {"question": "bad", "ground_truth": "answer"},
         {"question": "", "ground_truth": "answer"}],
    )
    assert len(frame) == 3
    assert frame["sample_status"].tolist() == ["success", "failed", "skipped"]
    summary = pipeline.compare_architectures({"fixture": frame})
    row = summary.iloc[0]
    assert row["expected_samples"] == 3
    assert row["successful_samples"] == 1
    assert row["failed_samples"] == 1
    assert row["skipped_samples"] == 1
    assert not bool(row["evaluation_valid"])
    assert pipeline.check_quality_gate(frame)[0] is False


def test_all_failed_evaluation_has_no_plausible_metric(tmp_path):
    from module3.evaluation import EvaluationPipeline

    class RAG:
        def run(self, _question):
            raise ValueError("bad model output")

    pipeline = EvaluationPipeline(output_path=str(tmp_path / "summary.csv"))
    frame = pipeline.evaluate_architecture(
        "fixture", RAG(), [{"question": "q", "ground_truth": "a"}]
    )
    summary = pipeline.compare_architectures({"fixture": frame})
    assert summary.loc[0, "successful_samples"] == 0
    assert summary.loc[0, "failure_rate"] == 1.0
    assert "faithfulness" not in summary.columns


def test_quality_gate_summary_marks_failed_metric_with_cross_mark(capsys):
    from phase3.cicd_pipeline import CICDReport

    report = CICDReport(
        run_id="fixture", timestamp="now", architecture="fixture", num_samples=1,
        metrics={"faithfulness": 0.1},
        thresholds={"faithfulness": 0.2}, passed=False,
        failure_reasons=["faithfulness = 0.1000 < 0.2"],
        metric_status={"faithfulness": False},
    )
    report.print_summary()
    output = capsys.readouterr().out
    assert "✗  faithfulness" in output
    assert "✓  faithfulness" not in output


def test_graph_cooccurrence_is_symmetric(monkeypatch):
    from module1.data_preprocessing import KnowledgeGraphBuilder

    builder = KnowledgeGraphBuilder()
    monkeypatch.setattr(builder, "_extract_entities", lambda _text: ["alpha", "beta"])
    builder.build([{"id": "c1", "text": "fixture"}])
    assert "beta" in builder.graph["alpha"]
    assert "alpha" in builder.graph["beta"]
    assert builder.graph["alpha"].count("beta") == 1
    assert builder.graph["beta"].count("alpha") == 1


def test_raptor_cache_key_changes_when_corpus_identity_changes():
    from langchain_core.documents import Document
    from module2.rag_architectures import _raptor_cache_key

    first = [Document(page_content="same prefix", metadata={"revision": "one"})]
    second = [Document(page_content="same prefix", metadata={"revision": "two"})]
    assert _raptor_cache_key(first) != _raptor_cache_key(second)


def test_raptor_cache_key_includes_chunks_after_old_prefix_bound():
    from langchain_core.documents import Document
    from module2.rag_architectures import _raptor_cache_key

    first = [Document(page_content=f"chunk-{index}") for index in range(301)]
    second = list(first)
    second[-1] = Document(page_content="changed-final-chunk")
    assert _raptor_cache_key(first) != _raptor_cache_key(second)


def test_provider_scheduler_reserves_distinct_concurrent_slots(monkeypatch):
    import module2.rag_architectures as rag

    clock = [100.0]
    monkeypatch.setattr(rag.time, "monotonic", lambda: clock[0])
    scheduler = rag._ProviderSlotScheduler()
    assert scheduler.reserve("groq", 3.0) == 0.0
    assert scheduler.reserve("groq", 3.0) == 3.0
    assert scheduler.reserve("mistral", 3.0) == 0.0


def test_stale_warmup_cannot_restore_readiness_after_shutdown():
    from serving import ServingContext

    context = ServingContext()
    generation = context.begin_warmup()
    context.shutdown()
    context.warmup(generation)
    assert context.is_ready() is False


def test_current_warmup_publishes_readiness():
    from serving import ServingContext

    context = ServingContext()
    context.architectures = lambda: {"naive": object()}
    context.cache = lambda: type("Cache", (), {"healthy": lambda self: True})()
    generation = context.begin_warmup()
    context.warmup(generation)
    assert context.is_ready() is True


def test_invalid_manifest_does_not_leave_a_reader_lease(tmp_path):
    from module1 import data_preprocessing as preprocessing

    store = tmp_path / "store"
    store.mkdir()
    (store / "chroma.sqlite3").write_bytes(b"fixture")
    with pytest.raises(preprocessing.IndexManifestError):
        preprocessing.load_vector_store(object(), str(store))
    assert str(store.resolve()) not in preprocessing._READER_LEASES


def test_concurrent_evaluation_outputs_are_atomic(tmp_path):
    from module3.evaluation import EvaluationPipeline

    target = tmp_path / "summary.csv"

    def write(name):
        pipeline = EvaluationPipeline(output_path=str(target))
        pipeline.compare_architectures({name: pd.DataFrame([{
            "architecture": name, "sample_status": "success", "expected_samples": 1,
            "faithfulness": 1.0,
        }])})

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(write, ["one", "two"]))
    loaded = pd.read_csv(target)
    assert len(loaded) == 1
    assert loaded.loc[0, "architecture"] in {"one", "two"}
