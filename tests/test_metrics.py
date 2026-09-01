"""Unit tests for module3.evaluation metric functions."""
from __future__ import annotations

import math

import pytest

from module3.evaluation import (
    bootstrap_ci, compute_answer_recall, compute_hit_rate, compute_mrr,
    compute_noise_ratio, compute_precision_at_k, compute_precision_omega,
    compute_token_iou,
)


class TestAnswerRecall:
    def test_full_recall_when_answer_present(self):
        assert compute_answer_recall("Paris is the capital of France.", "Paris") == 1.0

    def test_zero_recall_when_answer_absent(self):
        assert compute_answer_recall("London is in England.", "Paris") == 0.0

    def test_partial_recall(self):
        # "albert einstein" → both tokens present
        assert compute_answer_recall("Albert was a physicist.", "Albert Einstein") == pytest.approx(0.5)

    def test_empty_ground_truth_returns_one(self):
        assert compute_answer_recall("anything", "") == 1.0


class TestTokenIoU:
    def test_iou_full_match(self):
        # Single GT token must be exactly captured.
        assert compute_token_iou("Paris", "Paris") == 1.0

    def test_iou_no_overlap(self):
        assert compute_token_iou("foo bar baz", "Paris") == 0.0

    def test_iou_bounded_for_long_context(self):
        # The cap on retrieved-token-set size means a long context with the
        # GT token still yields a meaningful, non-near-zero score.
        long_ctx = "Paris " + " ".join(f"word{i}" for i in range(500))
        score = compute_token_iou(long_ctx, "Paris")
        assert score > 0.0
        assert score >= 1.0 / 200  # not floor-clipped to ~0


class TestPrecisionOmega:
    def test_zero_when_nothing_retrieved(self):
        assert compute_precision_omega("", "Paris") == 0.0

    def test_full_concentration(self):
        # Every retrieved token is in the GT.
        assert compute_precision_omega("Paris France", "Paris France capital") == 1.0


class TestNoiseRatio:
    def test_no_noise_when_perfect_concentration(self):
        assert compute_noise_ratio("Paris", "Paris") == 0.0

    def test_full_noise_when_no_overlap(self):
        assert compute_noise_ratio("foo bar baz", "Paris") == 1.0


class TestRankingMetrics:
    def test_precision_at_k(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["b", "d"]
        assert compute_precision_at_k(retrieved, relevant, k=4) == 0.5

    def test_precision_at_k_zero(self):
        assert compute_precision_at_k([], ["a"], k=5) == 0.0

    def test_hit_rate_true(self):
        assert compute_hit_rate(["x", "b", "y"], ["b"]) == 1.0

    def test_hit_rate_false(self):
        assert compute_hit_rate(["x", "y"], ["z"]) == 0.0

    def test_mrr_first_position(self):
        assert compute_mrr(["a", "b", "c"], ["a"]) == 1.0

    def test_mrr_third_position(self):
        assert compute_mrr(["a", "b", "c"], ["c"]) == pytest.approx(1.0 / 3)

    def test_mrr_no_hit(self):
        assert compute_mrr(["a", "b"], ["z"]) == 0.0


class TestBootstrapCI:
    def test_constant_values_yield_zero_width(self):
        lo, hi = bootstrap_ci([0.5] * 50, iters=200, seed=0)
        assert math.isclose(lo, 0.5, abs_tol=1e-6)
        assert math.isclose(hi, 0.5, abs_tol=1e-6)

    def test_ci_contains_mean(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        lo, hi = bootstrap_ci(values, iters=500, seed=0)
        mean = sum(values) / len(values)
        assert lo <= mean <= hi

    def test_empty_input(self):
        assert bootstrap_ci([]) == (0.0, 0.0)
