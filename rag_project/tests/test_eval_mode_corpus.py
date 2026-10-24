"""Integration test for the eval-mode corpus wiring (BUG 4, Task 6.2).

Spec: eval-pipeline-fixes -- Requirements 2.4, 3.8.

What is under test
------------------
Task 6.1 fixed ``main.py.run_eval_mode`` so eval mode builds its chunk set from the
**same MS MARCO corpus** that backs the persisted vectorstore (previously it loaded
SQuAD for the sparse/graph indexes while the dense store held MS MARCO -- BUG 4). The
fixed body is::

    def run_eval_mode(args) -> None:
        _require_api_key()
        from module1.data_preprocessing import (
            load_vector_store, get_embedding_model, chunk_documents, load_msmarco_documents,
        )
        from module2.rag_architectures import build_all_architectures
        embedding_model = get_embedding_model()
        vectorstore = load_vector_store(embedding_model)
        docs = load_msmarco_documents()          # was load_squad_documents()
        chunks = chunk_documents(docs)
        architectures = build_all_architectures(vectorstore=vectorstore, chunks=chunks)
        synthetic_data = _load_synthetic_data()
        _run_evaluation(architectures, synthetic_data)

This test exercises the REAL ``run_eval_mode`` (its module is imported, not
re-implemented) and asserts the single-corpus wiring: MS MARCO is loaded (SQuAD is
not), the loaded docs flow through ``chunk_documents``, and the resulting chunks +
vectorstore are handed to ``build_all_architectures``.

Why the stubbing below is needed
---------------------------------
* ``import main`` pulls ``pandas`` (module top) and ``configs.config`` (which imports
  ``python-dotenv``). Neither is installed on the Python 3.9 analysis machine, so both
  are stubbed **only when genuinely absent** -- on the 3.12 runtime the real packages
  are used unchanged.
* ``run_eval_mode`` imports its loaders *locally* at call time via
  ``from module1.data_preprocessing import ...`` / ``from module2.rag_architectures
  import ...``. A ``from X import Y`` where ``X`` is already in ``sys.modules`` binds
  ``Y`` from that cached module without importing it, so we install instrumented stub
  modules for ``module1.data_preprocessing`` and ``module2.rag_architectures`` (and
  their parent packages, defensively) *before* calling the function. The function then
  binds our ``MagicMock`` loaders and the heavy LangChain/Chroma/LLM stack is never
  touched.
* ``_require_api_key`` / ``_load_synthetic_data`` / ``_run_evaluation`` are module-level
  collaborators of ``run_eval_mode`` that are out of scope for the corpus-wiring
  assertion (and would ``sys.exit`` without keys/a dataset, or run the whole heavy
  evaluation). They are patched to inert mocks so the call terminates at the wiring
  boundary we care about.

The live single-corpus retrieval run (dense + BM25 + graph over one MS MARCO corpus)
is covered separately by Task 6.3 and is **(deferred)** to the Python 3.12 runtime.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock, sentinel

import pytest

# Project root (tests/ parent): importable + the location of main.py.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_MAIN_PATH = os.path.join(_ROOT, "main.py")


# ─────────────────────────────────────────────────────────────────────────────
# Import shims — make ``import main`` succeed without the heavy stack.
# ─────────────────────────────────────────────────────────────────────────────
def _build_dotenv_stub() -> types.ModuleType:
    """``configs.config`` runs ``from dotenv import load_dotenv`` at import time."""
    mod = types.ModuleType("dotenv")
    mod.load_dotenv = lambda *a, **k: False  # no-op, mirrors the "no .env" case
    return mod


def _install_stub_if_missing(monkeypatch, name, builder) -> None:
    """Install a stub module under ``name`` only when the real dependency is not
    importable, so the test behaves identically on the 3.9 analysis machine (deps
    absent -> stub) and the 3.12 runtime (deps present -> real package)."""
    if name in sys.modules:
        return
    try:
        available = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        available = False
    if not available:
        monkeypatch.setitem(sys.modules, name, builder())


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: real ``main`` imported with instrumented corpus loaders in place.
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def eval_mode(monkeypatch):
    """Import the real ``main`` module with the corpus loaders / architecture
    builder replaced by instrumented ``MagicMock`` stubs, and the out-of-scope
    module-level collaborators neutralized. Yields a namespace of the mocks so
    tests can assert exactly how ``run_eval_mode`` wired them together."""
    # 1) Let ``import main`` succeed on the analysis machine.
    _install_stub_if_missing(monkeypatch, "dotenv", _build_dotenv_stub)
    _install_stub_if_missing(monkeypatch, "pandas", lambda: types.ModuleType("pandas"))

    # 2) Instrumented loaders for module1.data_preprocessing. Distinct ``sentinel``
    #    return values make the identity of every wired object observable.
    get_embedding_model = MagicMock(
        name="get_embedding_model", return_value=sentinel.embedding_model)
    load_vector_store = MagicMock(
        name="load_vector_store", return_value=sentinel.vectorstore)
    load_msmarco_documents = MagicMock(
        name="load_msmarco_documents", return_value=sentinel.msmarco_docs)
    load_squad_documents = MagicMock(
        name="load_squad_documents", return_value=sentinel.squad_docs)
    chunk_documents = MagicMock(
        name="chunk_documents", return_value=sentinel.chunks)

    dp = types.ModuleType("module1.data_preprocessing")
    dp.get_embedding_model = get_embedding_model
    dp.load_vector_store = load_vector_store
    dp.load_msmarco_documents = load_msmarco_documents
    dp.load_squad_documents = load_squad_documents  # present but must stay unused
    dp.chunk_documents = chunk_documents

    # 3) Instrumented architecture builder for module2.rag_architectures.
    build_all_architectures = MagicMock(
        name="build_all_architectures", return_value=sentinel.architectures)
    ra = types.ModuleType("module2.rag_architectures")
    ra.build_all_architectures = build_all_architectures

    # 4) Parent-package stubs. The submodule stubs above are what the function-local
    #    ``from module1.data_preprocessing import ...`` actually resolves against
    #    (short-circuited via sys.modules); the parent stubs are belt-and-suspenders
    #    so no real heavy package ``__init__`` can run if the import machinery ever
    #    consults the parent.
    m1 = types.ModuleType("module1")
    m1.__path__ = []
    m1.data_preprocessing = dp
    m2 = types.ModuleType("module2")
    m2.__path__ = []
    m2.rag_architectures = ra

    for name, mod in (
        ("module1", m1),
        ("module1.data_preprocessing", dp),
        ("module2", m2),
        ("module2.rag_architectures", ra),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    # 5) Import the REAL main (fresh, so it binds under the stubs in place).
    monkeypatch.delitem(sys.modules, "main", raising=False)
    main = importlib.import_module("main")

    # 6) Neutralize the out-of-scope module-level collaborators.
    monkeypatch.setattr(main, "_require_api_key", MagicMock(name="_require_api_key"))
    load_synthetic_data = MagicMock(
        name="_load_synthetic_data", return_value=sentinel.synthetic_data)
    monkeypatch.setattr(main, "_load_synthetic_data", load_synthetic_data)
    run_evaluation = MagicMock(name="_run_evaluation")
    monkeypatch.setattr(main, "_run_evaluation", run_evaluation)

    return types.SimpleNamespace(
        main=main,
        get_embedding_model=get_embedding_model,
        load_vector_store=load_vector_store,
        load_msmarco_documents=load_msmarco_documents,
        load_squad_documents=load_squad_documents,
        chunk_documents=chunk_documents,
        build_all_architectures=build_all_architectures,
        load_synthetic_data=load_synthetic_data,
        run_evaluation=run_evaluation,
    )


def _call_run_eval_mode(env):
    """Invoke the real ``run_eval_mode``. ``args`` is unused by the body, so a bare
    Namespace suffices."""
    env.main.run_eval_mode(argparse.Namespace())


# ─────────────────────────────────────────────────────────────────────────────
# Integration assertions — the single-corpus wiring (Req 2.4, 3.8).
# ─────────────────────────────────────────────────────────────────────────────
class TestEvalModeCorpusWiring:
    def test_loads_msmarco_and_never_squad(self, eval_mode):
        """(Req 2.4) Eval mode loads the MS MARCO corpus exactly once and never
        touches the SQuAD loader (the BUG 4 mismatch)."""
        _call_run_eval_mode(eval_mode)

        eval_mode.load_msmarco_documents.assert_called_once_with()
        eval_mode.load_squad_documents.assert_not_called()
        assert eval_mode.load_squad_documents.call_count == 0

    def test_msmarco_docs_are_routed_through_chunk_documents(self, eval_mode):
        """(Req 2.4) The exact object returned by ``load_msmarco_documents`` is the
        one passed into ``chunk_documents`` -- proving chunks derive from MS MARCO."""
        _call_run_eval_mode(eval_mode)

        eval_mode.chunk_documents.assert_called_once_with(
            eval_mode.load_msmarco_documents.return_value
        )

    def test_builder_receives_matching_vectorstore_and_chunks(self, eval_mode):
        """(Req 2.4, 3.8) ``build_all_architectures`` gets ``vectorstore`` from
        ``load_vector_store`` and ``chunks`` from ``chunk_documents`` -- i.e. dense
        and sparse/graph retrievers are wired to one shared MS MARCO corpus."""
        _call_run_eval_mode(eval_mode)

        # vectorstore := load_vector_store(get_embedding_model(), persist_dir=<store>)
        eval_mode.get_embedding_model.assert_called_once_with()
        assert eval_mode.load_vector_store.call_count == 1
        vs_args, vs_kwargs = eval_mode.load_vector_store.call_args
        assert vs_args[0] is eval_mode.get_embedding_model.return_value
        # A --store override resolves to an explicit persist_dir; the default still
        # resolves through configs.config.VECTOR_STORE_DIR.
        assert "persist_dir" in vs_kwargs

        # build_all_architectures(vectorstore=<store>, chunks=<msmarco chunks>)
        assert eval_mode.build_all_architectures.call_count == 1
        _, build_kwargs = eval_mode.build_all_architectures.call_args
        assert build_kwargs["vectorstore"] is eval_mode.load_vector_store.return_value
        assert build_kwargs["chunks"] is eval_mode.chunk_documents.return_value
        # Cost gating is decided at construction time (GRACE-Ω's RAPTOR build makes
        # LLM calls in __init__), so the flag must reach the builder itself.
        assert build_kwargs.get("llm_free_only") is False

    def test_evaluation_receives_built_architectures_and_dataset(self, eval_mode):
        """The built architectures and loaded synthetic dataset flow into the
        evaluation step, so the fixed wiring reaches evaluation intact."""
        _call_run_eval_mode(eval_mode)

        assert eval_mode.run_evaluation.call_count == 1
        args, kwargs = eval_mode.run_evaluation.call_args
        # Positional payload is unchanged: (architectures, synthetic_dataset).
        assert args[0] is eval_mode.build_all_architectures.return_value
        assert args[1] is eval_mode.load_synthetic_data.return_value
        # Newer keyword arguments carry the qrels / sampling / mode wiring.
        assert "qrels" in kwargs
        assert kwargs["retrieval_only"] is False
        assert kwargs["max_samples"] == 200

    def test_exercises_the_real_run_eval_mode(self, eval_mode):
        """Guard: we drove the actual ``main.run_eval_mode`` (not a stand-in)."""
        assert eval_mode.main.run_eval_mode.__module__ == "main"
        assert callable(eval_mode.main.run_eval_mode)


# ─────────────────────────────────────────────────────────────────────────────
# Static backstop — the source of run_eval_mode chooses MS MARCO, not SQuAD.
# ─────────────────────────────────────────────────────────────────────────────
def _run_eval_mode_ast():
    with open(_MAIN_PATH, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=_MAIN_PATH)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run_eval_mode":
            return node
    return None


class TestStaticCorpusChoice:
    """Cheap, runtime-independent backstop that catches a revert to SQuAD even if
    the dynamic test were somehow satisfied."""

    def test_references_msmarco_and_not_squad(self):
        func = _run_eval_mode_ast()
        assert func is not None, "run_eval_mode not found in main.py"

        imported = {
            alias.name
            for node in ast.walk(func)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        names = {n.id for n in ast.walk(func) if isinstance(n, ast.Name)}
        referenced = imported | names

        assert "load_msmarco_documents" in referenced, (
            "run_eval_mode must load the MS MARCO corpus"
        )
        assert "load_squad_documents" not in referenced, (
            "run_eval_mode must not reference the SQuAD loader (BUG 4 mismatch)"
        )

    def test_imports_msmarco_from_data_preprocessing(self):
        func = _run_eval_mode_ast()
        assert func is not None, "run_eval_mode not found in main.py"
        found = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "module1.data_preprocessing"
            and any(alias.name == "load_msmarco_documents" for alias in node.names)
            for node in ast.walk(func)
        )
        assert found, (
            "expected `from module1.data_preprocessing import ... load_msmarco_documents`"
        )
