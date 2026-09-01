"""Property-based tests for the token-overlap retrieval metrics.

Feature: eval-pipeline-fixes, Property 4: Token-overlap retrieval metrics are
LLM-free and bounded.

Property statement (design.md → Correctness Properties → Property 4):
    For any retrieved-context string and any ground-truth string,
    ``compute_answer_recall``, ``compute_token_iou``, ``compute_precision_omega``,
    and ``compute_noise_ratio`` each return a value in the closed interval
    ``[0.0, 1.0]`` and complete without invoking any LLM or network call.

**Validates: Requirements 3.3**

Runtime note
------------
The framework targets Python 3.12 but the analysis machine runs 3.9, where the
heavy stack (numpy / pandas / groq / mistralai) is unavailable. ``module3.evaluation``
imports ``numpy``/``pandas`` at module top and pulls in ``configs.config`` (which
imports ``python-dotenv``); the LLM SDKs are imported lazily inside ``RAGASEvaluator``
methods that the four token-overlap functions never touch.

To exercise the *real* metric functions without the heavy stack, this module injects
minimal ``sys.modules`` stubs for ONLY the dependencies that are genuinely missing,
*before* importing ``module3.evaluation``. The four functions under test are pure
(they use ``re`` + stdlib only), so stubbing ``numpy``/``pandas`` does not change the
code paths being tested. The LLM SDK stubs raise on any real use, and the metric
calls run inside a network-blocking guard, so any hidden LLM/network access would
surface as a test failure rather than passing silently.
"""

from __future__ import annotations

import importlib
import math
import os
import socket
import sys
import types

# ── Ensure the project root is importable (conftest also does this) ──────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── Stub ONLY the heavy top-level import deps that are actually missing ──────────
def _ensure_importable(name: str, build_stub) -> None:
    """If ``name`` cannot be imported, register a lightweight stub in its place.

    Only activates when the real module is absent, so a full runtime (3.12) tests
    the genuine libraries and the constrained runtime (3.9) still tests the real
    metric function bodies.
    """
    try:
        importlib.import_module(name)
    except Exception:  # noqa: BLE001 — any import failure means we must stub it
        sys.modules[name] = build_stub()


def _plain_stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


class _RaisingModule(types.ModuleType):
    """Module stub whose real (non-dunder) attribute access raises.

    Used for the LLM SDKs so that an accidental judge/LLM call inside a metric
    would blow up loudly instead of silently succeeding. Dunder lookups return
    normally so the import machinery stays happy.
    """

    def __getattr__(self, item: str):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        raise AssertionError(
            f"Unexpected LLM SDK access ({self.__name__}.{item}): token-overlap "
            f"metrics must be LLM-free (Property 4)."
        )


def _build_numpy_stub() -> types.ModuleType:
    """A ``numpy`` stub complete enough for both ``module3.evaluation`` (which only
    does ``import numpy as np``) and Hypothesis, which — whenever a module named
    ``numpy`` is importable — introspects ``numpy.ndarray`` (``sampled_from``) and
    manages ``numpy.random`` for deterministic seeding. The four functions under
    test never call numpy, so these no-op shims don't affect what's being tested.
    """
    numpy_mod = types.ModuleType("numpy")
    numpy_mod.ndarray = type("ndarray", (), {})  # isinstance() guard target
    random_mod = types.ModuleType("numpy.random")
    random_mod.seed = lambda *a, **k: None
    random_mod.get_state = lambda *a, **k: None
    random_mod.set_state = lambda *a, **k: None
    numpy_mod.random = random_mod
    sys.modules["numpy.random"] = random_mod  # so ``import numpy.random`` resolves
    return numpy_mod


# numpy / pandas: only used by helpers the four functions never call → stubs are
# enough to let ``module3.evaluation`` import (numpy needs a Hypothesis-friendly
# shape; see _build_numpy_stub).
_ensure_importable("numpy", _build_numpy_stub)
_ensure_importable("pandas", lambda: _plain_stub("pandas"))
# python-dotenv: configs.config does ``from dotenv import load_dotenv`` at import.
_ensure_importable(
    "dotenv", lambda: _plain_stub("dotenv", load_dotenv=lambda *a, **k: False)
)
# LLM SDKs: lazily imported inside RAGASEvaluator; stub them to raise on real use.
_ensure_importable("groq", lambda: _RaisingModule("groq"))
_ensure_importable("mistralai", lambda: _RaisingModule("mistralai"))


# ── Import the REAL metric functions under test ──────────────────────────────────
from module3.evaluation import (  # noqa: E402  (import after stub injection)
    compute_answer_recall,
    compute_noise_ratio,
    compute_precision_omega,
    compute_token_iou,
)

from hypothesis import example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


# The four functions share the signature (retrieved_text, ground_truth) -> float.
_METRICS = (
    ("compute_answer_recall", compute_answer_recall),
    ("compute_token_iou", compute_token_iou),
    ("compute_precision_omega", compute_precision_omega),
    ("compute_noise_ratio", compute_noise_ratio),
)


class _NoNetwork:
    """Context manager that makes any socket connection attempt raise.

    Scoped tightly around the metric calls so Hypothesis's own machinery is not
    affected (keeps the guard strong but not brittle).
    """

    _MSG = (
        "network access attempted inside a token-overlap metric — "
        "metrics must be LLM-free (Property 4)"
    )

    def __enter__(self) -> "_NoNetwork":
        def _blocked(*_args, **_kwargs):
            raise AssertionError(self._MSG)

        self._saved = {
            "socket": socket.socket,
            "create_connection": socket.create_connection,
            "getaddrinfo": socket.getaddrinfo,
        }
        socket.socket = _blocked  # type: ignore[assignment]
        socket.create_connection = _blocked  # type: ignore[assignment]
        socket.getaddrinfo = _blocked  # type: ignore[assignment]
        return self

    def __exit__(self, *_exc) -> bool:
        socket.socket = self._saved["socket"]  # type: ignore[assignment]
        socket.create_connection = self._saved["create_connection"]  # type: ignore[assignment]
        socket.getaddrinfo = self._saved["getaddrinfo"]  # type: ignore[assignment]
        return False


def _assert_bounded(name: str, value, retrieved: str, ground_truth: str) -> None:
    """Assert ``value`` is a finite number (float/int, not bool) within [0.0, 1.0]."""
    assert isinstance(value, (int, float)) and not isinstance(value, bool), (
        f"{name} returned non-numeric {type(value).__name__}={value!r}"
    )
    # NaN / inf are genuine Property-4 violations (e.g. a hidden division-by-zero).
    assert math.isfinite(value), (
        f"{name} returned non-finite value {value!r} for "
        f"retrieved={retrieved!r}, ground_truth={ground_truth!r}"
    )
    assert 0.0 <= value <= 1.0, (
        f"{name} returned {value!r} outside [0.0, 1.0] for "
        f"retrieved={retrieved!r}, ground_truth={ground_truth!r}"
    )


# ── Input strategies ─────────────────────────────────────────────────────────────
# A small shared vocabulary (including stopwords) so two independently-drawn
# "wordy" strings frequently share tokens — this exercises 0.0, intermediate, and
# 1.0 outcomes rather than only the disjoint (near-0/near-1) regime.
_WORDS = [
    "paris", "france", "capital", "einstein", "physics", "london", "england",
    "river", "quantum", "theory", "water", "boils", "everest", "mountain",
    "the", "and", "of", "is",  # stopwords → can collapse token sets to empty
]

_wordy_text = st.lists(st.sampled_from(_WORDS), min_size=0, max_size=40).map(" ".join)

# Arbitrary text: unicode, punctuation, whitespace, control chars, and empty.
_arbitrary_text = st.text(min_size=0, max_size=300)

# Occasionally very long strings to stress the token-frequency capping paths.
_long_text = st.integers(min_value=0, max_value=4000).map(lambda n: "paris france " * n)

_any_text = st.one_of(_wordy_text, _arbitrary_text, _long_text)


@settings(max_examples=250, deadline=None)
@given(retrieved=_any_text, ground_truth=_any_text)
# Explicit edge cases (also documents the degenerate branches each function has).
@example(retrieved="", ground_truth="")                       # both empty
@example(retrieved="anything here", ground_truth="")          # empty ground truth
@example(retrieved="", ground_truth="paris")                  # empty retrieved
@example(retrieved="   \t\n  ", ground_truth="  \n ")         # whitespace-only
@example(retrieved=".,;:!?()[]{}-—…", ground_truth="!!! ??? ")  # punctuation-only
@example(retrieved="the and of is", ground_truth="the of")    # stopwords-only → empty sets
@example(retrieved="café résumé naïve Zürich", ground_truth="café Zürich")  # unicode
@example(retrieved="Paris is the capital of France", ground_truth="Paris")  # overlap
@example(retrieved="london england thames", ground_truth="paris france")    # disjoint
@example(retrieved="paris", ground_truth="paris")             # exact → 1.0 cases
@example(retrieved="paris " * 5000, ground_truth="paris")     # very long, high repetition
def test_token_overlap_metrics_are_llm_free_and_bounded(retrieved, ground_truth):
    """Property 4: every token-overlap metric returns a value in [0.0, 1.0]
    and touches no LLM/network, for arbitrary retrieved/ground-truth strings.
    """
    with _NoNetwork():
        results = [(name, fn(retrieved, ground_truth)) for name, fn in _METRICS]

    for name, value in results:
        _assert_bounded(name, value, retrieved, ground_truth)
