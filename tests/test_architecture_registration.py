"""Property-based test for BUG 2 — input-driven architecture registration.

Feature: eval-pipeline-fixes, Property 1: Architecture registration depends only
on declared inputs.
Validates: Requirements 2.2, 3.4

Property (from design.md):
    For any invocation of ``build_all_architectures`` the returned mapping
      * always contains ``naive``, ``crag``, ``rag_fusion``;
      * contains ``hybrid``, ``adaptive``, ``grace_omega``, ``parent_child``
        if and only if a non-empty ``chunks`` list is provided;
      * contains ``grace_no_gate``, ``grace_no_kg``, ``grace_no_hyde`` if and
        only if BOTH a non-empty ``chunks`` list AND a knowledge-graph builder
        are provided.

Why the import shim below is needed
-----------------------------------
``module2/rag_architectures.py`` imports ``langchain_core.documents.Document`` at
module load time and imports ``configs.config`` (which imports ``python-dotenv``).
Neither dependency is installed on the Python 3.9 analysis machine, and neither
is relevant to the registration logic under test. We therefore install *minimal*
stub modules for those two TOP-LEVEL import dependencies **only when they are not
already importable**, so that the REAL ``build_all_architectures`` function can be
imported and exercised as-is.

The heavy retrieval / LLM libraries (Chroma, scikit-learn, sentence-transformers,
rank-bm25, groq, mistralai) are imported *lazily inside* the RAG classes'
methods/constructors — which this test never triggers, because we monkeypatch the
ten RAG constructor classes with lightweight stubs (approach (a) from the task).
The registration control-flow itself (the ``if chunks:`` / nested ``if kg_builder:``
structure that decides which keys are added) is executed from the real, fixed
source — this test does NOT re-implement that logic.
"""
from __future__ import annotations

import os
import sys
import types

import pytest
from hypothesis import given, settings, strategies as st


# ─────────────────────────────────────────────────────────────────────────────
# Import shim — make the real module importable without the heavy stack.
# ─────────────────────────────────────────────────────────────────────────────
def _stub_dotenv_if_missing() -> None:
    """configs.config calls ``load_dotenv`` at import time; stub it if absent."""
    try:  # pragma: no cover - depends on the runtime
        import dotenv  # noqa: F401
    except Exception:
        mod = types.ModuleType("dotenv")
        mod.load_dotenv = lambda *a, **k: False  # no-op, mirrors "no .env" case
        sys.modules["dotenv"] = mod


def _stub_langchain_document_if_missing() -> None:
    """module2 does ``from langchain_core.documents import Document`` at import."""
    try:  # pragma: no cover - depends on the runtime
        from langchain_core.documents import Document  # noqa: F401
    except Exception:
        pkg = sys.modules.get("langchain_core") or types.ModuleType("langchain_core")
        docs = types.ModuleType("langchain_core.documents")

        class _StubDocument:  # minimal placeholder, only ever imported
            def __init__(self, page_content: str = "", metadata=None):
                self.page_content = page_content
                self.metadata = metadata or {}

        docs.Document = _StubDocument
        pkg.documents = docs
        sys.modules["langchain_core"] = pkg
        sys.modules["langchain_core.documents"] = docs


_stub_dotenv_if_missing()
_stub_langchain_document_if_missing()

# Ensure the project root is importable even when run standalone.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import module2.rag_architectures as rag  # noqa: E402  (import after shim)


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight test doubles.
# ─────────────────────────────────────────────────────────────────────────────
class _StubRAG:
    """Stand-in for a real RAG class: swallows any constructor kwargs cheaply."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyChunk:
    """Chunk-like object; its contents are irrelevant to registration."""

    def __init__(self, text: str = ""):
        self.page_content = text
        self.metadata = {}


class _DummyVectorStore:
    """Placeholder vectorstore — always supplied to the builder."""


class _DummyKGBuilder:
    """Truthy placeholder knowledge-graph builder (no __bool__/__len__)."""


# The RAG constructor classes referenced inside build_all_architectures.
_ARCH_CLASS_NAMES = [
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
]

# Expected registration groups.
ALWAYS = frozenset({"naive", "crag", "rag_fusion"})
CHUNK_DEPENDENT = frozenset({
    "hybrid", "adaptive", "grace_omega", "parent_child",
    # Two-stage retrieval variants: both need chunks (BM25 first stage /
    # GRACE-Ω base) and are registered under the same `if chunks:` branch.
    "rerank", "grace_rerank",
})
KG_DEPENDENT = frozenset({"grace_no_gate", "grace_no_kg", "grace_no_hyde"})


@pytest.fixture(scope="module", autouse=True)
def _stub_rag_classes():
    """Replace the heavy RAG classes with cheap stubs for the whole module.

    Module-scoped + autouse so the patch is applied once and every Hypothesis
    example runs against it (this avoids the function-scoped-fixture health
    check while still exercising the real build_all_architectures body).
    """
    originals = {}
    for name in _ARCH_CLASS_NAMES:
        assert hasattr(rag, name), f"fixed source must define {name}"
        originals[name] = getattr(rag, name)
        setattr(rag, name, _StubRAG)
    try:
        yield
    finally:
        for name, obj in originals.items():
            setattr(rag, name, obj)


def _expected_keys(chunks, kg_builder) -> set:
    """The exact key set the property demands for the given inputs."""
    keys = set(ALWAYS)
    if chunks:  # non-empty list (matches the source's ``if chunks:``)
        keys |= CHUNK_DEPENDENT
        if kg_builder:
            keys |= KG_DEPENDENT
    return keys


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis strategies.
# ─────────────────────────────────────────────────────────────────────────────
# chunks: None or [] (both falsy → "no chunks"), or a non-empty randomized list.
_chunks_strategy = st.one_of(
    st.none(),
    st.just([]),
    st.lists(
        st.builds(_DummyChunk, st.text(max_size=32)),
        min_size=1,
        max_size=6,
    ),
)
# kg_builder: absent (None) or a truthy dummy object.
_kg_strategy = st.one_of(st.none(), st.builds(_DummyKGBuilder))
# Optional pass-through args the builder forwards to ParentChildRAG.
_parent_child_strategy = st.one_of(st.none(), st.builds(dict))
_child_vs_strategy = st.one_of(st.none(), st.builds(_DummyVectorStore))


# ─────────────────────────────────────────────────────────────────────────────
# Property 1.
# ─────────────────────────────────────────────────────────────────────────────
@settings(max_examples=200, deadline=None)
@given(
    chunks=_chunks_strategy,
    kg_builder=_kg_strategy,
    parent_child=_parent_child_strategy,
    child_vectorstore=_child_vs_strategy,
)
def test_registration_depends_only_on_declared_inputs(
    chunks, kg_builder, parent_child, child_vectorstore
):
    """Feature: eval-pipeline-fixes, Property 1: Architecture registration
    depends only on declared inputs. Validates: Requirements 2.2, 3.4."""
    result = rag.build_all_architectures(
        vectorstore=_DummyVectorStore(),
        chunks=chunks,
        parent_child=parent_child,
        child_vectorstore=child_vectorstore,
        kg_builder=kg_builder,
    )
    keys = set(result.keys())

    chunks_present = bool(chunks)
    kg_present = bool(kg_builder)

    # Unconditional trio.
    assert ALWAYS <= keys

    # Chunk-dependent group present iff non-empty chunks.
    if chunks_present:
        assert CHUNK_DEPENDENT <= keys
    else:
        assert keys.isdisjoint(CHUNK_DEPENDENT)

    # KG-dependent group present iff BOTH non-empty chunks AND a kg_builder.
    if chunks_present and kg_present:
        assert KG_DEPENDENT <= keys
    else:
        assert keys.isdisjoint(KG_DEPENDENT)

    # Nothing extra, nothing missing.
    assert keys == _expected_keys(chunks, kg_builder)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic corner cases (the four input combinations spelled out).
# ─────────────────────────────────────────────────────────────────────────────
class TestRegistrationCorners:
    def _build(self, chunks, kg_builder):
        return rag.build_all_architectures(
            vectorstore=_DummyVectorStore(),
            chunks=chunks,
            kg_builder=kg_builder,
        )

    def test_no_chunks_no_kg_registers_only_unconditional(self):
        assert set(self._build(None, None)) == set(ALWAYS)

    def test_empty_chunks_behaves_like_no_chunks(self):
        assert set(self._build([], _DummyKGBuilder())) == set(ALWAYS)

    def test_eval_mode_chunks_without_kg_includes_parent_child(self):
        # BUG 2 fix: parent_child depends only on chunks (eval mode → 9 archs
        # once the two reranking variants are registered).
        result = self._build([_DummyChunk("a")], None)
        assert set(result) == set(ALWAYS | CHUNK_DEPENDENT)
        assert "parent_child" in result
        assert KG_DEPENDENT.isdisjoint(result)
        assert len(result) == len(ALWAYS | CHUNK_DEPENDENT) == 9

    def test_full_mode_registers_every_architecture(self):
        # Regression 3.4: full mode keeps every architecture (→ 12 total).
        result = self._build([_DummyChunk("a")], _DummyKGBuilder())
        assert set(result) == set(ALWAYS | CHUNK_DEPENDENT | KG_DEPENDENT)
        assert len(result) == 12

    def test_no_hyde_ablation_receives_the_kg(self):
        """
        Single-variable ablation guard: `grace_no_hyde` must be built WITH the KG.

        Passing kg_builder=None there would ablate HyDE *and* cross-structure
        verification at once, confounding the two and making the variant's results
        indistinguishable from `grace_no_kg`.
        """
        kg = _DummyKGBuilder()
        result = self._build([_DummyChunk("a")], kg)
        no_hyde = result["grace_no_hyde"]
        assert no_hyde.kwargs.get("kg_builder") is kg


def test_builder_is_the_real_module_function():
    """Guard: we exercise the actual source, not a re-implementation."""
    assert rag.build_all_architectures.__module__ == "module2.rag_architectures"
    assert callable(rag.build_all_architectures)
