"""Property + example tests for the RAGAS construction predicate (BUG 3).

Feature: eval-pipeline-fixes, Property 2: LLM-judged evaluation is enabled exactly
when a key is available.

Property statement (design.md -> Correctness Properties -> Property 2):
    For any combination of ``GROQ_API_KEY`` and ``MISTRAL_API_KEY`` environment values,
    the evaluation pipeline constructs a RAGAS evaluator if and only if at least one of
    the two keys is non-empty; when neither key is set the evaluator is absent and
    evaluation proceeds without raising.

Validates: Requirements 2.3, 3.2

How this test exercises the REAL predicate without the heavy stack
------------------------------------------------------------------
The guard under test lives in ``main.py._run_evaluation``:

    ragas_evaluator = RAGASEvaluator() if (GROQ_API_KEY or MISTRAL_API_KEY) else None

Calling the real ``_run_evaluation`` would import ``module3.evaluation`` (and with it
the full LangChain / Chroma / Groq / Mistral stack), which the design pins to the
Python 3.12 runtime and is unavailable on the 3.9 analysis machine; it would also run
the entire evaluation loop, visualization, and significance testing -- machinery far
beyond the one boolean we care about.

Instead we parse the REAL, fixed ``main.py`` with ``ast``, locate the exact assignment
to ``ragas_evaluator`` inside ``_run_evaluation``, and ``exec`` *only that statement* in
a controlled namespace where ``RAGASEvaluator`` is a lightweight sentinel factory and
``GROQ_API_KEY`` / ``MISTRAL_API_KEY`` are bound to the Hypothesis-generated values. This
means:

  * We execute the *actual* guard expression pulled from source -- the predicate is NOT
    hand-reimplemented here. If the guard ever regresses to ``if MISTRAL_API_KEY`` (the
    original BUG 3), this test fails for the Groq-only combination.
  * No project module is imported, so there is no LangChain/Chroma/LLM dependency and no
    network access. ``RAGASEvaluator`` is stubbed, so no real provider setup ever runs.

The oracle ``bool(groq) or bool(mistral)`` is the property itself (at least one key
non-empty), independent of the exact source expression -- so agreement between the
executed real statement and the oracle is a meaningful check, not a tautology.
"""
from __future__ import annotations

import ast
import os

from hypothesis import example, given, settings
from hypothesis import strategies as st

# ── Locate the real main.py (project root is the tests/ parent). ─────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MAIN_PATH = os.path.join(_ROOT, "main.py")


# ── Sentinel evaluator: stands in for RAGASEvaluator so construction is observable
#    and cheap (no provider setup, no network). ────────────────────────────────────
class _StubRAGASEvaluator:
    """Cheap stand-in for ``module3.evaluation.RAGASEvaluator``.

    The real guard calls ``RAGASEvaluator()`` with no arguments; we accept ``*args`` /
    ``**kwargs`` defensively so the test stays valid if a no-op argument is ever added.
    Its mere construction (an instance appearing in the namespace) is what the property
    observes -- it performs no work.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


# ── Extract the real guard statement from main.py via AST. ───────────────────────
def _read_main_source() -> str:
    with open(_MAIN_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


def _find_function(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _find_ragas_assignment(func: ast.AST):
    """
    Return the ``ragas_evaluator = <conditional>`` Assign node inside ``func``.

    ``_run_evaluation`` assigns ``ragas_evaluator`` more than once: retrieval-only
    runs short-circuit it to ``None`` (no generator, no judge), and the real
    key-presence guard lives in the other branch. We want the guard, so prefer the
    assignment whose value is a conditional expression and fall back to the first
    match only if no conditional is found.
    """
    fallback = None
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ragas_evaluator":
                    if isinstance(node.value, ast.IfExp):
                        return node
                    fallback = fallback or node
    return fallback


def _extract_ragas_guard_assignment() -> ast.Assign:
    """Pull the exact ``ragas_evaluator`` assignment out of the real fixed source."""
    assert os.path.exists(_MAIN_PATH), f"main.py not found at {_MAIN_PATH}"
    tree = ast.parse(_read_main_source(), filename=_MAIN_PATH)

    func = _find_function(tree, "_run_evaluation")
    assert func is not None, "could not find function _run_evaluation in main.py"

    assign = _find_ragas_assignment(func)
    assert assign is not None, (
        "could not find the `ragas_evaluator = ...` assignment inside _run_evaluation; "
        "the RAGAS construction guard may have moved or been renamed"
    )
    return assign


def _compile_single_statement(assign: ast.Assign):
    """Compile the lone assignment node into an executable code object."""
    module = ast.Module(body=[assign], type_ignores=[])
    ast.fix_missing_locations(module)
    return compile(module, filename=_MAIN_PATH, mode="exec")


# Extract + compile once; every example re-execs the same real guard cheaply.
_GUARD_ASSIGN = _extract_ragas_guard_assignment()
_GUARD_CODE = _compile_single_statement(_GUARD_ASSIGN)


def _build_evaluator(groq_key, mistral_key):
    """Exec the real guard with the given key values; return what it assigned.

    Returns a ``_StubRAGASEvaluator`` instance when the guard constructs one, else
    ``None``. Raises nothing for any input -- exec simply evaluates the ternary.
    """
    namespace = {
        "RAGASEvaluator": _StubRAGASEvaluator,
        "GROQ_API_KEY": groq_key,
        "MISTRAL_API_KEY": mistral_key,
        # OpenRouter joined the provider cascade; absent here so these examples keep
        # testing exactly the Groq/Mistral presence matrix they were written for.
        "OPENROUTER_API_KEY": "",
    }
    exec(_GUARD_CODE, namespace)
    return namespace["ragas_evaluator"]


def _build_evaluator_any(openrouter_key="", groq_key="", mistral_key=""):
    """Exec the real guard across all three providers."""
    namespace = {
        "RAGASEvaluator": _StubRAGASEvaluator,
        "OPENROUTER_API_KEY": openrouter_key,
        "GROQ_API_KEY": groq_key,
        "MISTRAL_API_KEY": mistral_key,
    }
    exec(_GUARD_CODE, namespace)
    return namespace["ragas_evaluator"]


# ── Key-value strategy: model os.getenv(...) results (str, default "") plus None. ──
# Covers absent ("" / None), whitespace-only (non-empty -> truthy), arbitrary
# non-empty text, and realistic key-shaped strings. Two independent draws give all
# four presence/absence combinations across a run.
_key_values = st.one_of(
    st.just(""),                                   # absent (os.getenv default)
    st.none(),                                     # absent (unset)
    st.text(alphabet=" \t", min_size=1, max_size=4),   # whitespace-only, non-empty
    st.text(min_size=1, max_size=24),              # arbitrary non-empty
    st.sampled_from(["gsk_live_abc123", "sk-mistral-xyz", "openai/gpt-oss-20b"]),
)


@settings(max_examples=200, deadline=None)
@given(groq_key=_key_values, mistral_key=_key_values)
# Pin the canonical corners so they are always exercised, independent of sampling.
@example(groq_key="", mistral_key="")        # neither set -> None
@example(groq_key=None, mistral_key=None)    # neither set (unset) -> None
@example(groq_key="gsk_x", mistral_key="")   # Groq only (free tier) -> built  [BUG 3]
@example(groq_key="", mistral_key="ms_x")    # Mistral only -> built
@example(groq_key="gsk_x", mistral_key="ms_x")  # both set -> built
@example(groq_key=" ", mistral_key="")       # whitespace is non-empty -> built
def test_ragas_evaluator_built_iff_a_key_is_present(groq_key, mistral_key):
    """Feature: eval-pipeline-fixes, Property 2: LLM-judged evaluation is enabled
    exactly when a key is available. Validates: Requirements 2.3, 3.2.

    The evaluator is constructed iff at least one key is non-empty; when neither is
    set the result is None and no exception is raised.
    """
    expected_built = bool(groq_key) or bool(mistral_key)

    result = _build_evaluator(groq_key, mistral_key)  # no exception for any input (3.2)

    assert (result is not None) == expected_built
    if expected_built:
        assert isinstance(result, _StubRAGASEvaluator)
    else:
        # Neither key set -> evaluator absent (pipeline would run with None -> 3.2).
        assert result is None


# ── Deterministic corner cases (the four combinations spelled out). ──────────────
class TestRagasGateCorners:
    def test_neither_key_yields_none_without_raising(self):
        assert _build_evaluator("", "") is None

    def test_neither_key_as_none_yields_none_without_raising(self):
        assert _build_evaluator(None, None) is None

    def test_groq_only_builds_evaluator(self):
        # The documented free-tier setup (Groq key, no Mistral) must build the
        # evaluator -- this is the exact regression BUG 3 fixed.
        result = _build_evaluator("gsk_live_key", "")
        assert isinstance(result, _StubRAGASEvaluator)

    def test_mistral_only_builds_evaluator(self):
        result = _build_evaluator("", "ms_live_key")
        assert isinstance(result, _StubRAGASEvaluator)

    def test_both_keys_build_evaluator(self):
        result = _build_evaluator("gsk_live_key", "ms_live_key")
        assert isinstance(result, _StubRAGASEvaluator)

    def test_whitespace_key_counts_as_present(self):
        # A non-empty string (even whitespace) is truthy, matching os.getenv semantics.
        assert isinstance(_build_evaluator(" ", None), _StubRAGASEvaluator)


# ── Guard: confirm we exercise the real source, not a re-implementation. ─────────
def test_guard_is_extracted_from_real_main_source():
    """The executed statement is the real ``ragas_evaluator`` guard and depends on
    BOTH keys (the BUG 3 fix), constructing ``RAGASEvaluator``."""
    assign = _extract_ragas_guard_assignment()

    # The guard is a ternary: build-or-None.
    assert isinstance(assign.value, ast.IfExp)

    # Its target is exactly `ragas_evaluator`.
    assert any(
        isinstance(t, ast.Name) and t.id == "ragas_evaluator" for t in assign.targets
    )

    # The predicate references every provider key and constructs RAGASEvaluator --
    # so a revert to `if MISTRAL_API_KEY` (dropping the others) fails this assertion.
    names = {n.id for n in ast.walk(assign) if isinstance(n, ast.Name)}
    assert {"GROQ_API_KEY", "MISTRAL_API_KEY", "RAGASEvaluator"} <= names
    assert "OPENROUTER_API_KEY" in names, (
        "the guard must also honour OPENROUTER_API_KEY, otherwise an OpenRouter-only "
        "setup silently loses all LLM-judged metrics"
    )


def test_openrouter_only_builds_evaluator():
    """An OpenRouter-only setup (no Groq, no Mistral) must still build the judge."""
    assert isinstance(_build_evaluator_any(openrouter_key="sk-or-v1-x"), _StubRAGASEvaluator)


def test_no_keys_at_all_yields_none():
    assert _build_evaluator_any() is None
