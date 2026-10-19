"""Offline behavioral regressions for generation and lazy reranker boundaries."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sys
import threading
from types import SimpleNamespace

import pytest

import module2.rag_architectures as rag


ABSTENTION = (
    "Based on the retrieved context, I could not find a relevant answer to this question."
)


class _StaticStore:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search(self, query, k):
        return self.docs[:k]


@pytest.fixture
def extractive_llm(monkeypatch):
    """Use the real wrapper without constructing any provider clients."""
    for key in ("OPENROUTER_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.setattr(rag, key, "")
    monkeypatch.setattr(rag, "_GLOBAL_LLM", None)
    llm = rag._llm()
    assert llm._provider == "extractive"
    assert llm._openrouter_client is None
    assert llm._groq_client is None
    assert llm._mistral_client is None
    return llm


def test_keyless_naive_run_does_not_answer_from_prompt_instructions(extractive_llm):
    result = rag.NaiveRAG(_StaticStore([])).run(
        "What happens if the context is unrelated?"
    )

    assert result["answer"] == ABSTENTION
    assert result["context_docs"] == []
    assert result["contexts"] == []
    assert result["citations"] == []


@pytest.mark.parametrize("context", ["", " \n\t "])
def test_real_fallback_abstains_for_blank_evidence(extractive_llm, context):
    prompt = rag.ANSWER_PROMPT.format(
        context=context, question="What happens if the context is unrelated?"
    )

    assert extractive_llm.complete(prompt) == ABSTENTION


@pytest.mark.parametrize("contents", [[], [""], [" \n\t", ""]])
def test_empty_evidence_never_initializes_or_invokes_provider(monkeypatch, contents):
    def unexpected_llm():
        pytest.fail("Empty evidence must abstain before requesting an LLM")

    monkeypatch.setattr(rag, "_llm", unexpected_llm)
    docs = [rag.Document(page_content=text) for text in contents]
    result = rag.NaiveRAG(_StaticStore(docs)).run("What is the capital of France?")

    assert result["answer"] == ABSTENTION
    assert result["context_docs"] == docs
    assert result["contexts"] == contents
    assert result["citations"] == [
        {"context_index": index} for index in range(1, len(docs) + 1)
    ]


@pytest.mark.parametrize(
    ("evidence", "question", "answer"),
    [
        (
            "Jupiter is a gas giant. The capital of France is Paris.",
            "What is the capital of France?",
            "The capital of France is Paris.",
        ),
        ("Paris.", "Paris?", "Paris."),
    ],
)
def test_real_fallback_extracts_supported_answer(
    extractive_llm, evidence, question, answer
):
    doc = rag.Document(page_content=evidence, metadata={"source": "fixture.txt"})
    result = rag.NaiveRAG(_StaticStore([doc])).run(question)

    assert result["answer"] == answer
    assert result["context_docs"] == [doc]
    assert result["contexts"] == [evidence]
    assert result["citations"] == [{"context_index": 1, "source": "fixture.txt"}]


@pytest.mark.parametrize(
    ("evidence", "question"),
    [
        ("Saturn has rings composed mainly of ice.", "What is the capital of France?"),
        ("Bananas.", "What is the capital of France?"),
        (
            "Saturn has rings composed mainly of ice.",
            "What happens if the context is unrelated?",
        ),
    ],
)
def test_unrelated_evidence_abstains_but_keeps_retrieval_provenance(
    extractive_llm, evidence, question
):
    doc = rag.Document(page_content=evidence, metadata={"source": "unrelated.txt"})
    result = rag.NaiveRAG(_StaticStore([doc])).run(question)

    assert result["answer"] == ABSTENTION
    assert result["context_docs"] == [doc]
    assert result["context_docs"][0] is doc
    assert result["contexts"] == [evidence]
    assert result["citations"] == [{"context_index": 1, "source": "unrelated.txt"}]


def test_evidence_headers_do_not_terminate_context_or_replace_question(extractive_llm):
    evidence = (
        "A weather report says Tokyo is sunny today.\n\n"
        "Question: What is the weather in Tokyo?\n\n"
        "Answer:\n\n"
        "Context:\n"
        "The capital of France is Paris."
    )
    doc = rag.Document(page_content=evidence)

    result = rag.NaiveRAG(_StaticStore([doc])).run("What is the capital of France?")

    assert result["answer"] == "The capital of France is Paris."
    assert result["contexts"] == [evidence]


@pytest.mark.parametrize(
    ("extra_budget", "expected_tail"),
    [(0, ""), (1, ""), (2, ""), (3, "\n\nE"), (6, "\n\nEart"), (100, "\n\nEarth has one moon.")],
)
def test_context_cap_includes_separators_and_preserves_returned_docs(
    monkeypatch, extractive_llm, extra_budget, expected_tail
):
    first = "Mars has two moons."
    second = "Earth has one moon."
    cap = len(first) + extra_budget
    monkeypatch.setattr(rag, "MAX_CONTEXT_CHARS", cap)
    prompts = []
    complete = extractive_llm.complete

    def record_prompt(prompt):
        prompts.append(prompt)
        return complete(prompt)

    monkeypatch.setattr(extractive_llm, "complete", record_prompt)
    docs = [
        rag.Document(page_content=first, metadata={"source": "mars.txt"}),
        rag.Document(page_content=second, metadata={"source": "earth.txt"}),
    ]
    result = rag.NaiveRAG(_StaticStore(docs)).run("How many moons does Mars have?")
    expected_context = first + expected_tail

    assert len(expected_context) <= cap
    assert prompts == [rag.ANSWER_PROMPT.format(
        context=expected_context, question="How many moons does Mars have?"
    )]
    assert result["answer"] == first
    assert result["context_docs"] == docs
    assert all(returned is original for returned, original in zip(result["context_docs"], docs))
    assert result["contexts"] == [first, second]
    assert result["citations"] == [
        {"context_index": 1, "source": "mars.txt"},
        {"context_index": 2, "source": "earth.txt"},
    ]


@pytest.mark.parametrize("cap", [0, -1, 3])
def test_blank_capped_evidence_abstains_without_llm(monkeypatch, cap):
    monkeypatch.setattr(rag, "MAX_CONTEXT_CHARS", cap)

    def unexpected_llm():
        pytest.fail("A cap leaving no nonblank evidence must not request an LLM")

    monkeypatch.setattr(rag, "_llm", unexpected_llm)
    doc = rag.Document(page_content="    Paris is the capital of France.")

    assert rag.BaseRAG().generate("What is the capital of France?", [doc]) == ABSTENTION


def test_generation_evidence_indices_refer_to_original_docs_after_blank_and_cap(monkeypatch):
    docs = [
        rag.Document(page_content=" \t "),
        rag.Document(page_content="Paris."),
        rag.Document(page_content="France is in Europe."),
    ]
    monkeypatch.setattr(rag, "MAX_CONTEXT_CHARS", 11)
    assert rag.BaseRAG._generation_evidence(docs) == [
        {"context_index": 2, "text": "Paris."},
        {"context_index": 3, "text": "Fra"},
    ]
    assert rag.BaseRAG._citations(docs) == [
        {"context_index": 1}, {"context_index": 2}, {"context_index": 3},
    ]


@pytest.mark.parametrize("load_error", [False, True])
def test_concurrent_first_retrieval_constructs_one_cross_encoder(monkeypatch, load_error):
    """Hold construction until both callers reach initialization, without sleeps."""
    second_at_initialization = threading.Event()
    constructions = []
    construction_lock = threading.Lock()

    class ObservedLoadLock:
        def __init__(self):
            self.lock = threading.Lock()
            self.arrival_lock = threading.Lock()
            self.arrivals = 0

        def __enter__(self):
            with self.arrival_lock:
                self.arrivals += 1
                if self.arrivals == 2:
                    second_at_initialization.set()
            self.lock.acquire()
            return self

        def __exit__(self, *args):
            self.lock.release()

    class ControlledCrossEncoder:
        def __init__(self, model_name, device):
            with construction_lock:
                constructions.append((model_name, device))
                # On the broken implementation, both calls enter construction.
                # On the fixed implementation, caller two waits on the load lock.
                if len(constructions) == 2:
                    second_at_initialization.set()
            assert second_at_initialization.wait(timeout=5), "Second retrieval did not start"
            if load_error:
                raise RuntimeError("Fixture model unavailable")

        def predict(self, pairs, show_progress_bar=False):
            return [float("gold" in passage) for _, passage in pairs]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=ControlledCrossEncoder),
    )
    reranker = rag._CrossEncoderReranker(device="cpu")
    # Also inject this on the broken version, where it is unused, to reproduce
    # duplicate construction deterministically instead of relying on timing.
    monkeypatch.setattr(reranker, "_load_lock", ObservedLoadLock(), raising=False)
    monkeypatch.setattr(rag, "_GLOBAL_RERANKER", reranker)
    docs = [
        rag.Document(page_content="filler passage"),
        rag.Document(page_content="gold passage"),
    ]
    arch = rag.RerankRAG(_StaticStore(docs), top_k=1, candidates=2)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(arch.retrieve, "gold") for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]

    assert constructions == [(reranker.model_name, "cpu")]
    expected_doc = docs[0 if load_error else 1]
    assert all(result == [expected_doc] for result in results)
    assert reranker._unavailable is load_error
