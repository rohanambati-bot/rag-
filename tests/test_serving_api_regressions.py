"""Focused regressions for serving selection and executor ownership."""
from __future__ import annotations

import asyncio
import threading

import pytest
from fastapi import HTTPException

import api
import module2.rag_architectures as rag
import serving


def _stub_constructor(name: str, calls: list[str]):
    class StubArchitecture:
        def __init__(self, **_kwargs):
            calls.append(name)

    return StubArchitecture


def test_selected_architectures_construct_only_requested_keys(monkeypatch):
    """Selection must prevent unrequested GRACE constructors from running."""
    calls: list[str] = []
    class_names = (
        "NaiveRAG",
        "CorrectiveRAG",
        "RAGFusion",
        "HybridMultiQueryRAG",
        "AdaptiveRAG",
        "GRACEOmegaRAG",
        "ParentChildRAG",
        "RerankRAG",
        "GRACEOmegaNoGate",
        "GRACEOmegaNoKG",
        "GRACEOmegaNoHyDE",
    )
    for name in class_names:
        monkeypatch.setattr(rag, name, _stub_constructor(name, calls))

    result = rag.build_all_architectures(
        vectorstore=object(),
        chunks=[object()],
        kg_builder=object(),
        selected_architectures=["naive", "grace_omega"],
    )

    assert set(result) == {"naive", "grace_omega"}
    assert calls == ["NaiveRAG", "GRACEOmegaRAG"]


def test_serving_eager_set_uses_advertised_llm_free_architectures(monkeypatch):
    """Eager serving must not introduce the legacy ``crag_lexical`` key."""
    context = serving.ServingContext()
    vectorstore = object()
    chunks = [object()]
    monkeypatch.setattr(context, "vectorstore", lambda: vectorstore)
    monkeypatch.setattr(context, "chunks", lambda: chunks)

    observed = {}

    def build(**kwargs):
        observed.update(kwargs)
        return {key: object() for key in kwargs["selected_architectures"]}

    monkeypatch.setattr(rag, "build_all_architectures", build)

    architectures = context.architectures()
    expected = [
        key for key in rag.ARCHITECTURE_KEYS
        if rag.ARCHITECTURE_REGISTRY[key]["llm_free"]
    ]

    assert list(architectures) == expected
    assert "crag_lexical" not in architectures
    assert observed["selected_architectures"] == expected
    assert "llm_free_only" not in observed


def test_serving_lazy_resolution_builds_only_requested_architecture(monkeypatch):
    """Resolving one lazy key must not construct the other GRACE variants."""
    context = serving.ServingContext()
    context._architectures = {"naive": object()}
    context._available_keys = list(rag.ARCHITECTURE_KEYS)
    vectorstore = object()
    chunks = [object()]
    monkeypatch.setattr(context, "vectorstore", lambda: vectorstore)
    monkeypatch.setattr(context, "chunks", lambda: chunks)

    requested = []
    lazy_architecture = object()

    def build(**kwargs):
        requested.append(kwargs["selected_architectures"])
        return {"grace_omega": lazy_architecture}

    monkeypatch.setattr(rag, "build_all_architectures", build)

    assert context.architecture("grace_omega") is lazy_architecture
    assert requested == [["grace_omega"]]
    assert context.architecture("grace_omega") is lazy_architecture
    assert requested == [["grace_omega"]]


def test_run_blocking_holds_semaphore_until_thread_finishes(monkeypatch):
    """A timed-out thread keeps its concurrency slot until completion."""
    semaphore = asyncio.Semaphore(1)
    monkeypatch.setattr(api, "_WORK_SEMAPHORE", semaphore)

    first_started = threading.Event()
    first_finished = threading.Event()
    second_started = threading.Event()

    def first_work():
        first_started.set()
        if not first_finished.wait(2):
            raise AssertionError("first executor task was not released")
        return "first"

    def second_work():
        second_started.set()
        return "second"

    async def exercise():
        first_task = asyncio.create_task(api.run_blocking(first_work, timeout=0.02))
        for _ in range(1000):
            if first_started.is_set():
                break
            await asyncio.sleep(0.001)
        assert first_started.is_set()

        with pytest.raises(HTTPException) as exc_info:
            await first_task
        assert exc_info.value.status_code == 504

        second_task = asyncio.create_task(api.run_blocking(second_work, timeout=1))
        await asyncio.sleep(0.05)
        assert not second_started.is_set()

        first_finished.set()
        assert await asyncio.wait_for(second_task, timeout=1) == "second"
        assert second_started.is_set()

    asyncio.run(exercise())


def test_run_blocking_holds_semaphore_after_cancellation(monkeypatch):
    """Client disconnects must not release a slot while sync work is still running."""
    semaphore = asyncio.Semaphore(1)
    monkeypatch.setattr(api, "_WORK_SEMAPHORE", semaphore)

    started = threading.Event()
    finished = threading.Event()
    second_started = threading.Event()

    def first_work():
        started.set()
        if not finished.wait(2):
            raise AssertionError("cancelled executor task was not allowed to finish")
        return "first"

    def second_work():
        second_started.set()
        return "second"

    async def exercise():
        first_task = asyncio.create_task(api.run_blocking(first_work, timeout=1))
        for _ in range(1000):
            if started.is_set():
                break
            await asyncio.sleep(0.001)
        assert started.is_set()

        first_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_task

        second_task = asyncio.create_task(api.run_blocking(second_work, timeout=1))
        await asyncio.sleep(0.05)
        assert not second_started.is_set()

        finished.set()
        assert await asyncio.wait_for(second_task, timeout=1) == "second"
        assert second_started.is_set()

    asyncio.run(exercise())
