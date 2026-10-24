"""
Tests for the qrels-based, rank-aware IR metrics.

These metrics matter because they are graded against MS MARCO's *human* relevance
judgements (``is_selected``) rather than bag-of-words overlap, so they are directly
comparable to published retrieval numbers. They also need no LLM, which is what
allows large-n evaluation at zero API cost.

Property tests use Hypothesis with >=100 examples and make no network calls.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from module3.evaluation import (  # noqa: E402
    benjamini_hochberg,
    build_qrels,
    compute_hit_rate,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    doc_id_for,
)


class _Doc:
    """Minimal Document stand-in (avoids importing the heavy LangChain stack)."""

    def __init__(self, page_content: str, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


# ── nDCG@k ──────────────────────────────────────────────────────────────────────

class TestNDCG:
    def test_perfect_ranking_is_one(self):
        assert compute_ndcg_at_k(["a", "b", "x", "y"], ["a", "b"], k=10) == 1.0

    def test_no_relevant_retrieved_is_zero(self):
        assert compute_ndcg_at_k(["x", "y"], ["a"], k=10) == 0.0

    def test_empty_qrels_is_zero(self):
        assert compute_ndcg_at_k(["a"], [], k=10) == 0.0

    def test_rank_one_beats_rank_five(self):
        """Rank awareness is the whole point: earlier hits must score higher."""
        early = compute_ndcg_at_k(["a", "b", "c", "d", "e"], ["a"], k=10)
        late = compute_ndcg_at_k(["b", "c", "d", "e", "a"], ["a"], k=10)
        assert early > late

    def test_matches_hand_computed_value(self):
        # One relevant doc at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2) = 1.
        expected = 1.0 / math.log2(3)
        assert compute_ndcg_at_k(["x", "a"], ["a"], k=10) == pytest.approx(expected)

    def test_beyond_cutoff_is_ignored(self):
        assert compute_ndcg_at_k(["x", "y", "a"], ["a"], k=2) == 0.0

    @settings(max_examples=150, deadline=None)
    @given(
        retrieved=st.lists(st.integers(0, 20), min_size=0, max_size=15, unique=True),
        relevant=st.lists(st.integers(0, 20), min_size=1, max_size=8, unique=True),
        k=st.integers(1, 15),
    )
    def test_always_bounded(self, retrieved, relevant, k):
        """Property: nDCG is always in [0, 1] for any inputs."""
        value = compute_ndcg_at_k(
            [str(d) for d in retrieved], [str(r) for r in relevant], k=k
        )
        assert 0.0 <= value <= 1.0


# ── Recall@k / Precision@k / MRR / hit rate ─────────────────────────────────────

class TestRecallPrecisionMRR:
    def test_recall_counts_all_known_relevant(self):
        assert compute_recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0
        assert compute_recall_at_k(["a", "b", "c"], ["a", "b"], k=1) == 0.5

    def test_recall_with_no_qrels_is_zero(self):
        assert compute_recall_at_k(["a"], [], k=5) == 0.0

    def test_precision_at_k_divides_by_k(self):
        assert compute_precision_at_k(["a", "x", "y", "z"], ["a"], k=4) == 0.25

    def test_mrr_is_reciprocal_of_first_hit(self):
        assert compute_mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)
        assert compute_mrr(["a", "y"], ["a"]) == 1.0

    def test_mrr_at_10_truncates(self):
        """MS MARCO's official metric is MRR@10, so hits past rank 10 score zero."""
        ranked = [f"x{i}" for i in range(10)] + ["a"]
        assert compute_mrr(ranked, ["a"], k=10) == 0.0
        assert compute_mrr(ranked, ["a"]) == pytest.approx(1 / 11)

    def test_hit_rate_is_binary(self):
        assert compute_hit_rate(["x", "a"], ["a"]) == 1.0
        assert compute_hit_rate(["x", "y"], ["a"]) == 0.0

    @settings(max_examples=150, deadline=None)
    @given(
        retrieved=st.lists(st.integers(0, 15), min_size=0, max_size=12, unique=True),
        relevant=st.lists(st.integers(0, 15), min_size=1, max_size=6, unique=True),
        k=st.integers(1, 12),
    )
    def test_all_metrics_bounded_and_consistent(self, retrieved, relevant, k):
        r = [str(x) for x in retrieved]
        g = [str(x) for x in relevant]
        recall = compute_recall_at_k(r, g, k)
        precision = compute_precision_at_k(r, g, k)
        mrr = compute_mrr(r, g, k=k)
        hit = compute_hit_rate(r[:k], g)
        for value in (recall, precision, mrr, hit):
            assert 0.0 <= value <= 1.0
        # A non-zero MRR within the cutoff implies at least one hit in the top-k.
        if mrr > 0:
            assert hit == 1.0


# ── doc ids and qrels construction ──────────────────────────────────────────────

class TestQrels:
    def test_doc_id_is_stable_and_content_derived(self):
        assert doc_id_for(_Doc("Same text")) == doc_id_for(_Doc("Same text"))
        assert doc_id_for(_Doc("A")) != doc_id_for(_Doc("B"))

    def test_doc_id_ignores_case_and_surrounding_whitespace(self):
        assert doc_id_for(_Doc("  Paris  ")) == doc_id_for(_Doc("paris"))

    def test_only_selected_passages_become_relevant(self):
        docs = [
            _Doc("gold", {"query": "q1", "is_selected": 1}),
            _Doc("noise", {"query": "q1", "is_selected": 0}),
            _Doc("gold2", {"query": "q1", "is_selected": 1}),
        ]
        qrels = build_qrels(docs)
        assert set(qrels) == {"q1"}
        assert qrels["q1"] == {doc_id_for(docs[0]), doc_id_for(docs[2])}

    def test_groups_by_question(self):
        docs = [
            _Doc("a", {"query": "q1", "is_selected": 1}),
            _Doc("b", {"query": "q2", "is_selected": 1}),
        ]
        qrels = build_qrels(docs)
        assert set(qrels) == {"q1", "q2"}
        assert all(len(v) == 1 for v in qrels.values())

    def test_missing_or_blank_query_is_skipped(self):
        docs = [
            _Doc("a", {"is_selected": 1}),
            _Doc("b", {"query": "   ", "is_selected": 1}),
        ]
        assert build_qrels(docs) == {}

    def test_non_iterable_degrades_to_empty(self):
        """Qrels are optional enrichment — a bad corpus must not abort evaluation."""
        assert build_qrels(object()) == {}  # type: ignore[arg-type]

    def test_no_documents_yields_no_qrels(self):
        assert build_qrels([]) == {}


# ── Benjamini-Hochberg correction ───────────────────────────────────────────────

class TestBenjaminiHochberg:
    def test_matches_hand_computed_values(self):
        adjusted, _ = benjamini_hochberg([0.001, 0.008])
        # n=2: 0.001*2/1 = 0.002 ; 0.008*2/2 = 0.008
        assert adjusted[0] == pytest.approx(0.002)
        assert adjusted[1] == pytest.approx(0.008)

    def test_adjusted_p_is_never_below_raw_p(self):
        raw = [0.01, 0.02, 0.03, 0.04]
        adjusted, _ = benjamini_hochberg(raw)
        assert all(a >= r - 1e-12 for a, r in zip(adjusted, raw))

    def test_empty_input(self):
        assert benjamini_hochberg([]) == ([], [])

    def test_preserves_caller_order(self):
        """Adjusted values must come back aligned with the input, not sorted."""
        adjusted, _ = benjamini_hochberg([0.9, 0.001])
        assert adjusted[0] > adjusted[1]

    @settings(max_examples=150, deadline=None)
    @given(p_values=st.lists(st.floats(0.0, 1.0), min_size=1, max_size=12))
    def test_bounded_and_monotone_in_rank(self, p_values):
        adjusted, reject = benjamini_hochberg(p_values)
        assert len(adjusted) == len(reject) == len(p_values)
        assert all(0.0 <= a <= 1.0 for a in adjusted)
        # Sorting by raw p must yield non-decreasing adjusted p (BH step-up property).
        by_raw = [a for _, a in sorted(zip(p_values, adjusted), key=lambda t: t[0])]
        assert all(by_raw[i] <= by_raw[i + 1] + 1e-12 for i in range(len(by_raw) - 1))
