"""
Tests for the cross-encoder rerank stage and the architecture-fidelity repairs.

Covered here:
  * ``_CrossEncoderReranker`` re-orders by joint (query, passage) score, degrades
    gracefully when the model is unavailable, and never invents documents.
  * ``RerankRAG`` widens the first stage before reranking (reranking an already
    truncated top-k cannot promote a passage that was never fetched).
  * ``CorrectiveRAG`` really grades with the LLM and really performs the corrective
    rewrite-retry — the behaviour that makes it CRAG rather than dense+filter.
  * ``AdaptiveRAG``'s three complexity branches produce three *distinct* behaviours.

No network calls: the reranker model and the LLM are both replaced with stubs.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import module2.rag_architectures as rag  # noqa: E402


class _Doc:
    def __init__(self, page_content: str, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class _StubVectorStore:
    """Returns deterministic docs and records the k it was asked for."""

    def __init__(self, docs=None):
        self.docs = docs or [_Doc(f"doc{i}") for i in range(60)]
        self.requested_k = []

    def similarity_search(self, query, k=5):
        self.requested_k.append(k)
        return self.docs[:k]


class _StubCrossEncoder:
    """Scores by a supplied keyword: passages containing it rank highest."""

    def __init__(self, keyword="gold"):
        self.keyword = keyword
        self.pairs_seen = 0

    def predict(self, pairs, show_progress_bar=False):
        self.pairs_seen += len(pairs)
        return [1.0 if self.keyword in passage else 0.0 for _, passage in pairs]


@pytest.fixture
def reranker_with_stub(monkeypatch):
    """A reranker whose model is a stub cross-encoder (no weights downloaded)."""
    r = rag._CrossEncoderReranker()
    stub = _StubCrossEncoder()
    r._model = stub
    monkeypatch.setattr(rag, "_GLOBAL_RERANKER", r)
    return r, stub


# ── Reranker core ───────────────────────────────────────────────────────────────

class TestCrossEncoderReranker:
    def test_promotes_the_relevant_passage_to_rank_one(self, reranker_with_stub):
        """The gold passage starts last and must end first."""
        r, _ = reranker_with_stub
        docs = [_Doc("filler a"), _Doc("filler b"), _Doc("the gold passage")]
        out = r.rerank("q", docs, top_k=3)
        assert out[0].page_content == "the gold passage"

    def test_truncates_to_top_k(self, reranker_with_stub):
        r, _ = reranker_with_stub
        docs = [_Doc(f"d{i}") for i in range(10)]
        assert len(r.rerank("q", docs, top_k=4)) == 4

    def test_attaches_rerank_score_metadata(self, reranker_with_stub):
        r, _ = reranker_with_stub
        out = r.rerank("q", [_Doc("gold here")], top_k=1)
        assert "rerank_score" in out[0].metadata

    def test_scores_every_candidate_pair(self, reranker_with_stub):
        r, stub = reranker_with_stub
        r.rerank("q", [_Doc(f"d{i}") for i in range(7)], top_k=3)
        assert stub.pairs_seen == 7

    def test_empty_input_returns_empty(self, reranker_with_stub):
        r, _ = reranker_with_stub
        assert r.rerank("q", [], top_k=5) == []

    def test_unavailable_model_preserves_first_stage_order(self):
        """A missing model must degrade to a no-op, never crash retrieval."""
        r = rag._CrossEncoderReranker()
        r._unavailable = True
        docs = [_Doc("a"), _Doc("b"), _Doc("c")]
        out = r.rerank("q", docs, top_k=2)
        assert [d.page_content for d in out] == ["a", "b"]

    def test_predict_failure_falls_back_to_first_stage(self):
        class _Boom:
            def predict(self, *a, **k):
                raise RuntimeError("cuda oom")

        r = rag._CrossEncoderReranker()
        r._model = _Boom()
        docs = [_Doc("a"), _Doc("b")]
        assert [d.page_content for d in r.rerank("q", docs, top_k=2)] == ["a", "b"]

    def test_never_returns_documents_it_was_not_given(self, reranker_with_stub):
        r, _ = reranker_with_stub
        docs = [_Doc("a"), _Doc("gold")]
        out = r.rerank("q", docs, top_k=2)
        assert {d.page_content for d in out} <= {"a", "gold"}


# ── RerankRAG wiring ────────────────────────────────────────────────────────────

class TestRerankRAG:
    def test_first_stage_is_wider_than_top_k(self, reranker_with_stub):
        """
        The candidate pool must exceed top_k, otherwise there is nothing to
        re-order and the reranker cannot rescue a low-ranked gold passage.
        """
        vs = _StubVectorStore()
        arch = rag.RerankRAG(vectorstore=vs, chunks=[], top_k=5, candidates=40)
        arch.retrieve("q")
        assert vs.requested_k and max(vs.requested_k) == 40

    def test_returns_at_most_top_k(self, reranker_with_stub):
        arch = rag.RerankRAG(
            vectorstore=_StubVectorStore(), chunks=[], top_k=3, candidates=20
        )
        assert len(arch.retrieve("q")) <= 3

    def test_gold_buried_deep_is_promoted(self, reranker_with_stub):
        """End-to-end statement of the 'retrieval paradox' fix."""
        docs = [_Doc(f"filler {i}") for i in range(29)] + [_Doc("the gold passage")]
        arch = rag.RerankRAG(
            vectorstore=_StubVectorStore(docs), chunks=[], top_k=3, candidates=30
        )
        out = arch.retrieve("q")
        assert out[0].page_content == "the gold passage"

    def test_is_registered_by_the_builder(self):
        assert "RerankRAG" in dir(rag)

    def test_run_emits_metadata_backed_citations(self, monkeypatch):
        class InvalidAnswerInvocation:
            def complete_answer(self, payload):
                return "grounded answer"  # Free-form answers now fail closed.

        monkeypatch.setattr(rag, "_llm", lambda: InvalidAnswerInvocation())
        doc = _Doc("source passage", {"source": "fixture.txt", "page": 2})
        vs = _StubVectorStore([doc])
        result = rag.NaiveRAG(vectorstore=vs, top_k=1).run("What does the source say?")
        assert result["citations"] == [
            {"context_index": 1, "source": "fixture.txt", "page": 2}
        ]
        assert result["answer"] == rag._ABSTENTION_ANSWER
        assert result["answer_citations"] == []
        assert result["answer_validation"]["reason"] == "malformed_json"


# ── CorrectiveRAG fidelity ──────────────────────────────────────────────────────

class _RecordingLLM:
    """Stub LLM that records prompts and replays scripted replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompt, max_tokens=None):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "0.0"


class TestCorrectiveRAGFidelity:
    def test_llm_grader_is_actually_called(self, monkeypatch):
        llm = _RecordingLLM(["0.9"] * 10)
        monkeypatch.setattr(rag, "_llm", lambda: llm)
        crag = rag.CorrectiveRAG(vectorstore=_StubVectorStore(), top_k=3, grader="llm")
        crag.retrieve("what is x")
        # One grading prompt per retrieved document.
        assert any("relevant" in p.lower() for p in llm.prompts)
        assert len(llm.prompts) >= 3

    def test_grades_are_parsed_and_filter_documents(self, monkeypatch):
        # Two pass (>=0.5), one fails → only the passing docs survive.
        llm = _RecordingLLM(["0.9", "0.8", "0.1"])
        monkeypatch.setattr(rag, "_llm", lambda: llm)
        crag = rag.CorrectiveRAG(
            vectorstore=_StubVectorStore(), top_k=3, grader="llm",
            allow_corrective_rewrite=False,
        )
        assert len(crag.retrieve("q")) == 2

    def test_corrective_rewrite_triggers_when_retrieval_is_poor(self, monkeypatch):
        """The defining CRAG behaviour: bad retrieval → rewrite → retry."""
        # First pass: all three fail grading. Then a rewrite. Then retries pass.
        llm = _RecordingLLM(["0.0", "0.0", "0.0", "a better query", "0.9", "0.9", "0.9"])
        monkeypatch.setattr(rag, "_llm", lambda: llm)
        crag = rag.CorrectiveRAG(vectorstore=_StubVectorStore(), top_k=3, grader="llm")
        docs = crag.retrieve("vague q")
        assert any("rewrite" in p.lower() for p in llm.prompts), "no rewrite attempted"
        assert docs, "correction should recover documents"

    def test_no_rewrite_when_retrieval_is_already_good(self, monkeypatch):
        llm = _RecordingLLM(["0.9", "0.9", "0.9"])
        monkeypatch.setattr(rag, "_llm", lambda: llm)
        crag = rag.CorrectiveRAG(vectorstore=_StubVectorStore(), top_k=3, grader="llm")
        crag.retrieve("clear q")
        assert not any("rewrite" in p.lower() for p in llm.prompts)

    def test_lexical_mode_makes_no_llm_calls(self, monkeypatch):
        llm = _RecordingLLM([])
        monkeypatch.setattr(rag, "_llm", lambda: llm)
        crag = rag.CorrectiveRAG(
            vectorstore=_StubVectorStore(), top_k=3, grader="lexical",
            allow_corrective_rewrite=False,
        )
        crag.retrieve("q")
        assert llm.prompts == []

    def test_never_returns_empty_context(self, monkeypatch):
        """Even when everything fails grading, the generator must get something."""
        llm = _RecordingLLM(["0.0"] * 20)
        monkeypatch.setattr(rag, "_llm", lambda: llm)
        crag = rag.CorrectiveRAG(vectorstore=_StubVectorStore(), top_k=3, grader="llm")
        assert crag.retrieve("q")

    def test_grading_failure_falls_back_to_lexical(self, monkeypatch):
        class _Boom:
            def complete(self, *a, **k):
                raise RuntimeError("api down")

        monkeypatch.setattr(rag, "_llm", lambda: _Boom())
        crag = rag.CorrectiveRAG(vectorstore=_StubVectorStore(), top_k=3, grader="llm")
        assert crag.retrieve("q")  # no exception escapes


# ── AdaptiveRAG branch distinctness ─────────────────────────────────────────────

class TestAdaptiveRAGBranches:
    def _arch(self, monkeypatch, label):
        monkeypatch.setattr(rag, "_llm", lambda: _RecordingLLM([label]))
        chunks = [_Doc(f"chunk {i}") for i in range(10)]
        return rag.AdaptiveRAG(
            vectorstore=_StubVectorStore(), chunks=chunks, top_k=3
        )

    def test_simple_and_moderate_request_different_depths(self, monkeypatch):
        """
        Regression: SIMPLE and MODERATE used identical code, so a 3-way classifier
        had only 2 behaviours and Adaptive RAG was indistinguishable from Naive.
        """
        simple = self._arch(monkeypatch, "SIMPLE")
        simple.retrieve("q")
        simple_k = list(simple.vectorstore.requested_k)

        moderate = self._arch(monkeypatch, "MODERATE")
        moderate.retrieve("q")
        moderate_k = list(moderate.vectorstore.requested_k)

        assert simple_k != moderate_k
        assert max(moderate_k) > max(simple_k)

    def test_records_route_for_analysis(self, monkeypatch):
        arch = self._arch(monkeypatch, "MODERATE")
        arch.retrieve("q")
        assert arch._last_route == "MODERATE"

    def test_all_branches_respect_top_k(self, monkeypatch):
        for label in ("SIMPLE", "MODERATE", "COMPLEX"):
            arch = self._arch(monkeypatch, label)
            assert len(arch.retrieve("q")) <= 3
