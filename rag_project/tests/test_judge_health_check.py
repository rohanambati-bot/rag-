"""Unit tests for RAGASEvaluator.health_check (Task 10.1 — optional hardening).

The health check issues ONE trivial judge prompt through the *real* judge call
path (``RAGASEvaluator._call``, which forwards ``self.model``) and must FAIL
LOUDLY — raising rather than silently returning ``None`` the way ``_score`` does
— when the judge errors or returns a non-numeric response.

These tests mock the judge call, so they never touch the network or any LLM SDK.
The evaluator is built with ``object.__new__`` to bypass ``__init__`` (which would
otherwise require a live API key and the provider SDK), keeping the tests
stack-free.

Runtime note: like the rest of this repo's suite, these run on the Python 3.12 CI
runtime (pytest + numpy/pandas installed). The 3.9 analysis box lacks those deps,
so execution is deferred to CI.
"""
from __future__ import annotations

import pytest

from module3.evaluation import RAGASEvaluator


def _make_evaluator(call_impl):
    """Build a RAGASEvaluator without running __init__ (no SDK / API key needed).

    ``_call`` is installed as an instance attribute, so ``self._call(probe)`` is
    invoked with just the prompt (an instance attribute is not bound as a method).
    """
    ev = object.__new__(RAGASEvaluator)
    ev.provider = "groq"
    ev.model = "openai/gpt-oss-20b"
    ev._client = None
    ev._mode = "groq"
    ev._call = call_impl  # type: ignore[attr-defined]
    return ev


def test_health_check_returns_score_on_numeric_response():
    ev = _make_evaluator(lambda prompt: "0.9")
    assert ev.health_check() == pytest.approx(0.9)


def test_health_check_clamps_score_into_unit_interval():
    ev = _make_evaluator(lambda prompt: "1.7")
    assert ev.health_check() == 1.0


def test_health_check_raises_on_non_numeric_response():
    ev = _make_evaluator(lambda prompt: "I cannot grade this.")
    with pytest.raises(RuntimeError):
        ev.health_check()


def test_health_check_raises_when_judge_call_errors():
    def boom(prompt):
        raise RuntimeError("model_decommissioned")

    ev = _make_evaluator(boom)
    with pytest.raises(RuntimeError):
        ev.health_check()


def test_health_check_exercises_model_via_real_call_path():
    """The probe must be issued through _call (the path that forwards self.model)."""
    seen = {}

    def record(prompt):
        seen["prompt"] = prompt
        return "0.5"

    ev = _make_evaluator(record)
    assert ev.health_check() == pytest.approx(0.5)
    assert seen.get("prompt")  # a real, non-empty judge prompt was issued
