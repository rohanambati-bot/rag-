# module5/confidence.py
"""
Confidence Scoring — Compute retrieval confidence for each answer.
Shows users how reliable the answer is based on chunk similarity.
"""

import os
import sys
import re
import logging
from typing import Dict, Any, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Computes a confidence score for RAG answers based on:
      1. Cosine similarity of retrieved chunks to the query
      2. Keyword overlap between query and retrieved context
      3. Answer coverage (how much of the answer is grounded in context)

    Returns a score 0-100% with a label:
      90-100%: Very High Confidence
      70-89%:  High Confidence
      50-69%:  Medium Confidence
      30-49%:  Low Confidence
      0-29%:   Very Low (likely hallucination)
    """

    LABELS = {
        90: ("Very High", "🟢"),
        70: ("High", "🟡"),
        50: ("Medium", "🟠"),
        30: ("Low", "🔴"),
        0:  ("Very Low", "⚫"),
    }

    def __init__(self, vectorstore=None, embedding_model=None):
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model

    def score(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        similarity_scores: List[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute confidence score for a RAG answer.

        Returns:
            {
                "score": float (0-100),
                "label": str,
                "emoji": str,
                "breakdown": {
                    "retrieval_relevance": float,
                    "answer_grounding": float,
                    "context_coverage": float,
                },
                "is_reliable": bool,
            }
        """
        # 1. Retrieval relevance (from similarity scores or keyword overlap)
        if similarity_scores:
            # Chroma returns distances (lower = better), convert to similarity
            retrieval_score = self._scores_to_confidence(similarity_scores)
        else:
            retrieval_score = self._keyword_relevance(query, contexts)

        # 2. Answer grounding (how much of the answer appears in context)
        grounding_score = self._answer_grounding(answer, contexts)

        # 3. Context coverage (how much useful content is in the retrieved chunks)
        coverage_score = self._context_coverage(query, contexts)

        # Weighted combination
        confidence = (
            retrieval_score * 0.4 +
            grounding_score * 0.35 +
            coverage_score * 0.25
        )
        confidence = max(0, min(100, confidence))

        label, emoji = self._get_label(confidence)

        return {
            "score": round(confidence, 1),
            "label": label,
            "emoji": emoji,
            "breakdown": {
                "retrieval_relevance": round(retrieval_score, 1),
                "answer_grounding": round(grounding_score, 1),
                "context_coverage": round(coverage_score, 1),
            },
            "is_reliable": confidence >= 50,
        }

    def _scores_to_confidence(self, scores: List[float]) -> float:
        """Convert Chroma distance scores to confidence (0-100)."""
        if not scores:
            return 0.0
        # Chroma L2 distances: 0 = identical, higher = worse
        # Typical range: 0.5 (very similar) to 2.0 (unrelated)
        avg_dist = np.mean(scores[:3])  # top 3 chunks
        # Map: 0.5 → 100, 1.0 → 70, 1.5 → 40, 2.0 → 10
        confidence = max(0, 100 - (avg_dist - 0.3) * 60)
        return min(100, confidence)

    def _keyword_relevance(self, query: str, contexts: List[str]) -> float:
        """Keyword overlap between query and contexts."""
        q_tokens = self._meaningful_tokens(query)
        if not q_tokens:
            return 50.0

        all_context = " ".join(contexts)
        c_tokens = set(re.findall(r"\w+", all_context.lower()))

        overlap = len(q_tokens & c_tokens)
        ratio = overlap / len(q_tokens)
        return min(100, ratio * 100)

    def _answer_grounding(self, answer: str, contexts: List[str]) -> float:
        """How much of the answer is grounded in the retrieved context."""
        if not answer or not contexts:
            return 0.0

        answer_tokens = self._meaningful_tokens(answer)
        if not answer_tokens:
            return 50.0

        all_context = " ".join(contexts).lower()
        context_tokens = set(re.findall(r"\w+", all_context))

        grounded = len(answer_tokens & context_tokens)
        ratio = grounded / len(answer_tokens)
        return min(100, ratio * 100)

    def _context_coverage(self, query: str, contexts: List[str]) -> float:
        """How much of the context is relevant (not noise)."""
        q_tokens = self._meaningful_tokens(query)
        if not q_tokens or not contexts:
            return 50.0

        relevant_chunks = 0
        for ctx in contexts:
            c_tokens = set(re.findall(r"\w+", ctx.lower()))
            overlap = len(q_tokens & c_tokens)
            if overlap >= 2:
                relevant_chunks += 1

        ratio = relevant_chunks / max(len(contexts), 1)
        return ratio * 100

    @staticmethod
    def _meaningful_tokens(text: str) -> set:
        """Extract meaningful tokens (no stopwords)."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "of", "to",
            "in", "on", "at", "and", "or", "what", "who", "where",
            "when", "why", "how", "which", "do", "does", "did", "can",
            "could", "would", "should", "will", "it", "its", "this",
            "that", "these", "those", "be", "been", "being", "have",
            "has", "had", "for", "with", "from", "by", "about", "not",
            "no", "but", "if", "then", "than", "so", "very", "just",
            "also", "more", "most", "some", "any", "all", "each",
        }
        tokens = set(re.findall(r"\w+", text.lower()))
        return tokens - stopwords

    def _get_label(self, score: float) -> Tuple[str, str]:
        for threshold in sorted(self.LABELS.keys(), reverse=True):
            if score >= threshold:
                return self.LABELS[threshold]
        return self.LABELS[0]
