"""Provider-free checks of the real answer boundary and frozen evaluation contract."""

from copy import deepcopy
import json
import socket

import pytest

from scripts import evaluate_answer_boundaries as evaluation


DATASET, RAW_DATASET = evaluation.load_dataset()


@pytest.fixture(autouse=True)
def no_network_or_provider(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Answer-boundary checks must never initialize/call a provider or use a network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    rag = evaluation.load_production()
    monkeypatch.setattr(rag._LLM, "__init__", forbidden)
    for method in ("_openrouter_complete", "_groq_complete", "_mistral_complete"):
        monkeypatch.setattr(rag._LLM, method, forbidden)


def test_fixed_exact_dataset_contract():
    assert evaluation.digest(RAW_DATASET) == evaluation.DATASET_SHA256
    assert evaluation.digest(evaluation.json_bytes(evaluation.DATASET_SCHEMA)) == evaluation.SCHEMA_SHA256
    assert DATASET["dataset_version"] == "1.0.0"
    assert DATASET["schema_version"] == "answer-boundaries.dataset/v1"
    assert len(DATASET["cases"]) == 64
    assert len({case["id"] for case in DATASET["cases"]}) == 64
    assert all(sum(case["category"] == category for case in DATASET["cases"]) == 8
               for category in evaluation.CATEGORIES)


@pytest.mark.parametrize("case", DATASET["cases"], ids=lambda case: case["id"])
def test_frozen_cases_exercise_production_boundaries(case):
    result = evaluation.evaluate_case(case)
    assert result["all_boundary_checks_pass"], result["boundary_checks"]
    assert result["semantic_dimensions_unscored"] == dict.fromkeys(evaluation.SEMANTIC_DIMENSIONS)


@pytest.mark.parametrize(
    ("answer", "count", "scope", "expected"),
    [
        ("An unsupported assertion [1].", 1, [1], []),
        ("An assertion.", 1, [1], ["missing_inline_citation"]),
        ("An assertion [0].", 1, [1], ["citation_index_out_of_range"]),
        ("An assertion [2].", 1, [1], ["citation_index_out_of_range"]),
        ("An assertion [2].", 2, [1], ["citation_outside_generation_scope"]),
        ("An assertion [01].", 1, [1], ["malformed_inline_citation", "missing_inline_citation"]),
        ("An assertion [１].", 1, [1], ["malformed_inline_citation", "missing_inline_citation"]),
        ("An assertion [1,2].", 2, [1, 2], ["malformed_inline_citation", "missing_inline_citation"]),
        ("An assertion [[1]].", 1, [1], ["malformed_inline_citation"]),
        ("An assertion [0][x].", 1, [1], ["citation_index_out_of_range", "malformed_inline_citation"]),
        ("An assertion [1", 1, [1], ["malformed_inline_citation", "missing_inline_citation"]),
        ("An assertion [-1]", 1, [1], ["malformed_inline_citation", "missing_inline_citation"]),
        ("", 1, [1], ["empty_answer", "missing_inline_citation"]),
    ],
)
def test_citation_diagnostics_are_syntax_and_scope_only(answer, count, scope, expected):
    assert evaluation.citation_diagnostics(answer, count, scope) == expected


def test_even_false_answer_with_valid_reference_has_no_semantic_score():
    case = deepcopy(DATASET["cases"][24])
    assert case["id"] == "cit-01"
    case["scripted_answer"] = "The capital of France is on Jupiter [1]."
    result = evaluation.evaluate_case(case)
    assert result["inline_diagnostics"] == []
    assert result["all_boundary_checks_pass"]
    assert all(value is None for value in result["semantic_dimensions_unscored"].values())


def test_injection_echo_and_contradiction_are_visible_without_invented_labels():
    injection = next(case for case in DATASET["cases"] if case["id"] == "inj-01")
    result = evaluation.evaluate_case(injection)
    assert result["observations_not_scores"]["literal_presence_in_answer"]["INJECTED_ALPHA"] is True
    assert result["semantic_dimensions_unscored"]["instruction_resistance"] is None
    contradiction = evaluation.evaluate_case(next(case for case in DATASET["cases"] if case["id"] == "con-01"))
    assert contradiction["answer"] == "The capital of Luma is Arin."
    assert contradiction["semantic_dimensions_unscored"]["contradiction_handling"] is None


def test_scoping_preserves_retrieval_provenance_without_claiming_support():
    result = evaluation.evaluate_case(next(case for case in DATASET["cases"] if case["id"] == "scp-02"))
    assert result["generation_context_indices"] == [1]
    assert result["retrieval_citations"] == [{"context_index": 1}, {"context_index": 2}]
    assert result["inline_diagnostics"] == ["citation_outside_generation_scope"]
    assert result["answer"] == "Rome [2]."  # Production passes the fake output through.


def test_evaluator_detects_production_answer_outside_prompt_evidence(monkeypatch):
    rag = evaluation.load_production()
    monkeypatch.setattr(rag.BaseRAG, "generate", lambda self, query, docs: "Invented outside evidence")
    result = evaluation.evaluate_case(DATASET["cases"][0])
    assert result["all_boundary_checks_pass"] is False
    assert result["boundary_checks"]["extractive_answer_is_visible_span_or_abstention"] is False
    assert result["boundary_checks"]["actual_prompt_matches_scoped_evidence"] is False


def test_evaluator_detects_production_fabricated_citation_metadata(monkeypatch):
    rag = evaluation.load_production()
    monkeypatch.setattr(rag.BaseRAG, "_citations", staticmethod(lambda docs: [
        {"context_index": 1, "source": "invented-source.txt"},
    ]))
    result = evaluation.evaluate_case(DATASET["cases"][0])
    assert result["all_boundary_checks_pass"] is False
    assert result["boundary_checks"]["retrieval_citations_match_metadata"] is False


def test_evaluator_detects_production_evidence_budget_regression(monkeypatch):
    rag = evaluation.load_production()
    monkeypatch.setattr(rag.BaseRAG, "_generation_evidence", staticmethod(lambda docs: [
        {"context_index": index, "text": doc.page_content} for index, doc in enumerate(docs, start=1)
    ]))
    result = evaluation.evaluate_case(next(case for case in DATASET["cases"] if case["id"] == "scp-01"))
    assert result["all_boundary_checks_pass"] is False
    assert result["boundary_checks"]["generation_slices_match_budget_and_order"] is False
    assert result["boundary_checks"]["context_with_separators_within_cap"] is False


def test_invocation_seam_still_avoids_clients_with_nonblank_config_keys(monkeypatch):
    rag = evaluation.load_production()
    for name in ("OPENROUTER_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.setattr(rag, name, "synthetic-not-a-real-key")
    assert evaluation.evaluate_case(DATASET["cases"][0])["all_boundary_checks_pass"]


@pytest.mark.parametrize("mutation", ["duplicate_id", "wrong_version", "unknown_field", "missing_script", "bad_cap", "bad_document", "unbalanced"])
def test_invalid_schema_contract_is_rejected(mutation):
    dataset = deepcopy(DATASET)
    if mutation == "duplicate_id":
        dataset["cases"][1]["id"] = dataset["cases"][0]["id"]
    elif mutation == "wrong_version":
        dataset["dataset_version"] = "2.0.0"
    elif mutation == "unknown_field":
        dataset["cases"][0]["human_correctness"] = True
    elif mutation == "missing_script":
        del dataset["cases"][24]["scripted_answer"]
    elif mutation == "bad_cap":
        dataset["cases"][0]["context_cap"] = True
    elif mutation == "bad_document":
        dataset["cases"][0]["documents"][0]["metadata"] = []
    else:
        dataset["cases"][0]["category"] = "unsupported"
    with pytest.raises(ValueError):
        evaluation.validate_dataset(dataset)


def test_hash_pin_rejects_even_whitespace_changes(tmp_path):
    changed = tmp_path / "changed.json"
    changed.write_bytes(RAW_DATASET + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        evaluation.load_dataset(changed)


def test_schema_pin_rejects_silent_protocol_changes(monkeypatch):
    schema = deepcopy(evaluation.DATASET_SCHEMA)
    schema["properties"]["dataset_version"]["const"] = "unversioned-change"
    monkeypatch.setattr(evaluation, "DATASET_SCHEMA", schema)
    with pytest.raises(ValueError, match="Schema SHA-256"):
        evaluation.load_dataset()


def test_outputs_are_deterministic_create_only_and_review_labels_blank(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    summary = evaluation.write_outputs(first)
    assert evaluation.write_outputs(second) == summary
    first_bytes = {file.name: file.read_bytes() for file in first.iterdir()}
    assert first_bytes == {file.name: file.read_bytes() for file in second.iterdir()}
    assert first_bytes["dataset.json"] == RAW_DATASET
    assert json.loads(first_bytes["schema.json"]) == evaluation.DATASET_SCHEMA
    manifest = json.loads(first_bytes["run_manifest.json"])
    for name, expected in manifest["artifacts_sha256"].items():
        assert evaluation.digest(first_bytes[name]) == expected
    with pytest.raises(FileExistsError):
        evaluation.write_outputs(first)
    assert first_bytes == {file.name: file.read_bytes() for file in first.iterdir()}
    assert summary["case_count"] == summary["cases_passing_boundary_regressions"] == 64
    assert summary["boundary_assertion_failures"] == 0
    assert summary["human_review"]["completed_annotations"] == 0
    assert summary["human_review"]["cohens_kappa"] is None
    records = [json.loads(line) for line in first_bytes["human_review.jsonl"].splitlines()]
    assert len(records) == 64
    assert len({record["review_id"] for record in records}) == 64
    assert len(json.loads(first_bytes["human_review_key.json"])) == 64
    for record in records:
        assert not {"case_id", "category", "mode", "expected_answer", "boundary_checks"} & record.keys()
        assert len(record["annotations"]) == 2
        for annotation in record["annotations"]:
            assert annotation["reviewer_id"] == annotation["notes"] == ""
            assert annotation["labels"] == dict.fromkeys(evaluation.SEMANTIC_DIMENSIONS)


def test_cli_exit_status_refuses_existing_output_and_preserves_it(tmp_path, capsys):
    existing = tmp_path / "old-evidence"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("preserve this evidence")
    with pytest.raises(SystemExit) as exc:
        evaluation.main(["--output-dir", str(existing)])
    assert exc.value.code == 2
    assert "Refusing existing" in capsys.readouterr().err
    assert sentinel.read_text() == "preserve this evidence"


def test_production_import_disables_dotenv_reads(monkeypatch):
    # Import is performed inside this patch even when the interpreter has loaded
    # config already. Fresh CLI execution uses the same protected import path.
    import dotenv
    original_import = evaluation.importlib.import_module

    def checking_import(name):
        assert dotenv.load_dotenv("must-not-be-read") is False
        return original_import(name)

    monkeypatch.setattr(evaluation.importlib, "import_module", checking_import)
    assert evaluation.load_production().__name__ == "module2.rag_architectures"
