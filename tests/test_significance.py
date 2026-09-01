"""
Tests for the corrected significance testing.

Three properties are asserted, matching the three defects in the original
``test_pomega_significance``:

1. **Pairing is by question**, not by position. The old code truncated both series
   to a common length and zipped them, so a single failed sample silently
   mis-aligned every pair afterwards and invalidated the "paired" test.
2. **Architectures that return identical context are flagged**, rather than
   producing NaN t-statistics (previously seen for RAG-Fusion and Adaptive RAG).
3. **Multiple comparisons are corrected**, because one baseline versus k
   architectures is k hypothesis tests.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import module3.evaluation as _ev  # noqa: E402

# Bound to underscore-prefixed names on purpose: the production helpers are called
# ``test_metric_significance`` / ``test_pomega_significance``, so any module-level
# name starting with "test_" would be collected by pytest as a test case (and fail
# asking for a ``df`` fixture).
_significance = _ev.test_metric_significance
_pomega_significance = _ev.test_pomega_significance


def _frame(rows):
    return pd.DataFrame(rows)


def _arch_rows(arch, values, metric="precision_omega", start=0):
    return [
        {"architecture": arch, "question": f"q{i + start}", metric: v}
        for i, v in enumerate(values)
    ]


class TestPairing:
    def test_pairs_are_joined_on_question_not_position(self):
        """
        The treatment is missing q0, so a positional zip would pair q1-with-q0,
        q2-with-q1 ... and smear the true +0.1 offset. A question join must report
        exactly 9 pairs and a mean difference of 0.1.
        """
        base_vals = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
        rows = _arch_rows("Naive RAG", base_vals)
        # Treatment: same questions except q0, each +0.1.
        rows += [
            {"architecture": "Better", "question": f"q{i}", "precision_omega": v + 0.1}
            for i, v in enumerate(base_vals)
            if i != 0
        ]
        out = _significance(_frame(rows))
        row = out.iloc[0]
        assert row["n_pairs"] == 9
        assert row["mean_diff"] == pytest.approx(0.1, abs=1e-9)

    def test_duplicate_questions_are_averaged(self):
        rows = _arch_rows("Naive RAG", [0.1] * 6)
        rows += _arch_rows("Other", [0.2] * 6)
        # A duplicate row for q0 must not create a second pair.
        rows.append({"architecture": "Other", "question": "q0", "precision_omega": 0.4})
        out = _significance(_frame(rows))
        assert out.iloc[0]["n_pairs"] == 6

    def test_disjoint_questions_are_skipped(self):
        rows = _arch_rows("Naive RAG", [0.1] * 8, start=0)
        rows += _arch_rows("Other", [0.2] * 8, start=100)  # no overlap
        assert _significance(_frame(rows)).empty

    def test_too_few_pairs_is_skipped(self):
        rows = _arch_rows("Naive RAG", [0.1, 0.2])
        rows += _arch_rows("Other", [0.3, 0.4])
        assert _significance(_frame(rows)).empty


class TestIdenticalArchitectures:
    def test_clone_is_flagged_not_nan(self):
        """
        A variant that returns the same context as the baseline is a real finding
        (the variant is a no-op). It must be labelled, not silently emitted as NaN.
        """
        vals = list(np.linspace(0.1, 0.9, 12))
        rows = _arch_rows("Naive RAG", vals) + _arch_rows("Clone", vals)
        out = _significance(_frame(rows))
        row = out.iloc[0]
        assert bool(row["identical_to_baseline"]) is True
        assert row["p_value"] == 1.0
        assert row["mean_diff"] == 0.0
        assert not np.isnan(row["p_value"])

    def test_genuine_difference_is_not_flagged_identical(self):
        rows = _arch_rows("Naive RAG", list(np.linspace(0.1, 0.5, 12)))
        rows += _arch_rows("Better", list(np.linspace(0.3, 0.7, 12)))
        out = _significance(_frame(rows))
        assert bool(out.iloc[0]["identical_to_baseline"]) is False


class TestMultipleComparisonCorrection:
    def test_bh_column_is_present_and_at_least_raw_p(self):
        rows = _arch_rows("Naive RAG", list(np.linspace(0.1, 0.6, 15)))
        for name, shift in (("A", 0.02), ("B", 0.05), ("C", 0.30)):
            rows += _arch_rows("x", [])  # no-op, keeps the helper honest
            rows += [
                {"architecture": name, "question": f"q{i}", "precision_omega": v + shift}
                for i, v in enumerate(np.linspace(0.1, 0.6, 15))
            ]
        out = _significance(_frame(rows))
        assert "p_value_bh" in out.columns
        assert len(out) == 3
        # Correction can only make p-values larger (or equal).
        primary = out["wilcoxon_p"].fillna(out["p_value"])
        assert (out["p_value_bh"] >= primary - 1e-12).all()

    def test_significance_flag_uses_corrected_p(self):
        rows = _arch_rows("Naive RAG", list(np.linspace(0.1, 0.6, 20)))
        rows += [
            {"architecture": "Better", "question": f"q{i}", "precision_omega": v + 0.25}
            for i, v in enumerate(np.linspace(0.1, 0.6, 20))
        ]
        out = _significance(_frame(rows))
        flag_col = next(c for c in out.columns if c.startswith("significant"))
        assert bool(out.iloc[0][flag_col]) is True

    def test_results_sorted_by_corrected_p(self):
        rows = _arch_rows("Naive RAG", list(np.linspace(0.1, 0.6, 15)))
        for name, shift in (("Weak", 0.005), ("Strong", 0.4)):
            rows += [
                {"architecture": name, "question": f"q{i}", "precision_omega": v + shift}
                for i, v in enumerate(np.linspace(0.1, 0.6, 15))
            ]
        out = _significance(_frame(rows))
        assert list(out["p_value_bh"]) == sorted(out["p_value_bh"])


class TestMetricSelectionAndSafety:
    def test_works_on_rank_aware_metrics(self):
        rows = [
            {"architecture": "Naive RAG", "question": f"q{i}", "ndcg_at_10": 0.2}
            for i in range(10)
        ] + [
            {"architecture": "Rerank RAG", "question": f"q{i}", "ndcg_at_10": 0.6}
            for i in range(10)
        ]
        out = _significance(_frame(rows), metric="ndcg_at_10")
        assert out.iloc[0]["metric"] == "ndcg_at_10"
        assert out.iloc[0]["mean_diff"] == pytest.approx(0.4)

    def test_missing_metric_returns_empty(self):
        rows = _arch_rows("Naive RAG", [0.1] * 8) + _arch_rows("Other", [0.2] * 8)
        assert _significance(_frame(rows), metric="nope").empty

    def test_missing_question_column_returns_empty(self):
        df = _frame([
            {"architecture": "Naive RAG", "precision_omega": 0.1},
            {"architecture": "Other", "precision_omega": 0.2},
        ])
        assert _significance(df).empty

    def test_absent_baseline_returns_empty(self):
        rows = _arch_rows("A", [0.1] * 8) + _arch_rows("B", [0.2] * 8)
        assert _significance(_frame(rows), baseline_arch="Naive RAG").empty

    def test_empty_dataframe_is_safe(self):
        assert _significance(pd.DataFrame()).empty

    def test_nan_values_are_dropped_not_propagated(self):
        rows = _arch_rows("Naive RAG", [0.1] * 10)
        rows += _arch_rows("Other", [0.3] * 10)
        rows.append({"architecture": "Other", "question": "qX", "precision_omega": np.nan})
        out = _significance(_frame(rows))
        assert out.iloc[0]["n_pairs"] == 10


class TestBackwardCompatibility:
    def test_pomega_wrapper_still_works(self):
        rows = _arch_rows("Naive RAG", list(np.linspace(0.1, 0.5, 12)))
        rows += _arch_rows("Better", list(np.linspace(0.3, 0.7, 12)))
        out = _pomega_significance(_frame(rows))
        assert not out.empty
        assert out.iloc[0]["metric"] == "precision_omega"
