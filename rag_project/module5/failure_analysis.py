# module5/failure_analysis.py
"""
Failure Analysis — Classify WHY a RAG answer failed.
Maps to the 7 failure points from Barnett et al. (IEEE/ACM CAIN 2024).

Failure Points:
  FP1: Missing content — query topic not in knowledge base
  FP2: Missed top-K — relevant doc exists but wasn't retrieved in top-K
  FP3: Not in context — relevant doc retrieved but answer span not in chunk
  FP4: Not extracted — context has the answer but LLM didn't extract it
  FP5: Wrong format — answer is correct but in wrong format
  FP6: Incorrect specificity — answer is too broad or too narrow
  FP7: Incomplete — answer is partially correct but missing key info
"""

import os
import sys
import re
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)


@dataclass
class FailurePoint:
    id: str
    name: str
    description: str
    solution: str
    detected: bool = False
    evidence: str = ""


FAILURE_POINTS = [
    FailurePoint(
        id="FP1", name="Missing Content",
        description="The query topic is not present in the knowledge base at all.",
        solution="Expand corpus or use web search fallback.",
    ),
    FailurePoint(
        id="FP2", name="Missed Top-K",
        description="Relevant document exists but wasn't retrieved in top-K results.",
        solution="Increase K, use hybrid retrieval (BM25 + Dense), or rewrite query.",
    ),
    FailurePoint(
        id="FP3", name="Not in Context Window",
        description="Relevant doc retrieved but the answer span was chunked out.",
        solution="Use Parent-Child chunking or increase chunk size.",
    ),
    FailurePoint(
        id="FP4", name="Not Extracted",
        description="Context contains the answer but LLM failed to extract it.",
        solution="Improve prompt engineering or use a stronger LLM.",
    ),
    FailurePoint(
        id="FP5", name="Wrong Format",
        description="Answer is factually correct but in the wrong format.",
        solution="Add format instructions to the prompt.",
    ),
    FailurePoint(
        id="FP6", name="Incorrect Specificity",
        description="Answer is too broad or too narrow for the question.",
        solution="Use RAGAS answer relevance scoring + CI gate.",
    ),
    FailurePoint(
        id="FP7", name="Incomplete",
        description="Answer is partially correct but missing key information.",
        solution="Retrieve more chunks or use multi-hop reasoning.",
    ),
]


class FailureAnalyzer:
    """
    Analyzes a RAG result to detect which failure points apply.
    Uses heuristics (no LLM needed) for fast, free analysis.
    """

    def analyze(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a single RAG result for failure points.

        Returns:
            {
                "has_failure": bool,
                "failure_points": List[FailurePoint],
                "primary_failure": Optional[str],
                "quality_score": float (0-100),
                "recommendations": List[str],
            }
        """
        fps = [FailurePoint(**{k: v for k, v in fp.__dict__.items()})
               for fp in FAILURE_POINTS]

        q_tokens = self._meaningful_tokens(query)
        context_text = " ".join(contexts).lower()
        c_tokens = set(re.findall(r"\w+", context_text))

        # FP1: Missing Content — query keywords not in any context
        if q_tokens:
            overlap = len(q_tokens & c_tokens)
            if overlap / len(q_tokens) < 0.2:
                fps[0].detected = True
                fps[0].evidence = (
                    f"Only {overlap}/{len(q_tokens)} query keywords found in context."
                )

        # FP2: Missed Top-K — context is generic/irrelevant
        if contexts:
            relevant_chunks = sum(
                1 for ctx in contexts
                if len(self._meaningful_tokens(ctx) & q_tokens) >= 2
            )
            if relevant_chunks < 2 and not fps[0].detected:
                fps[1].detected = True
                fps[1].evidence = (
                    f"Only {relevant_chunks}/{len(contexts)} chunks are relevant."
                )

        # FP3: Not in Context — if ground truth provided, check if it's in context
        if ground_truth:
            gt_tokens = self._meaningful_tokens(ground_truth)
            gt_in_context = len(gt_tokens & c_tokens) / max(len(gt_tokens), 1)
            if gt_in_context < 0.4 and not fps[0].detected:
                fps[2].detected = True
                fps[2].evidence = (
                    f"Ground truth has {gt_in_context*100:.0f}% overlap with context."
                )

        # FP4: Not Extracted — context has answer keywords but answer doesn't
        if ground_truth and answer:
            gt_tokens = self._meaningful_tokens(ground_truth)
            a_tokens = self._meaningful_tokens(answer)
            gt_in_answer = len(gt_tokens & a_tokens) / max(len(gt_tokens), 1)
            gt_in_ctx = len(gt_tokens & c_tokens) / max(len(gt_tokens), 1)
            if gt_in_ctx > 0.5 and gt_in_answer < 0.3:
                fps[3].detected = True
                fps[3].evidence = (
                    f"Context has {gt_in_ctx*100:.0f}% of ground truth but "
                    f"answer only has {gt_in_answer*100:.0f}%."
                )

        # FP5: Wrong Format — answer is very short or very long relative to expected
        if answer and ground_truth:
            len_ratio = len(answer) / max(len(ground_truth), 1)
            if len_ratio > 5 or len_ratio < 0.1:
                fps[4].detected = True
                fps[4].evidence = (
                    f"Answer length ratio: {len_ratio:.1f}x "
                    f"(answer={len(answer)} chars, expected≈{len(ground_truth)} chars)."
                )

        # FP6: Incorrect Specificity — answer is too generic
        if answer:
            generic_phrases = [
                "i don't know", "cannot answer", "not enough information",
                "based on the context", "the passage does not",
            ]
            if any(p in answer.lower() for p in generic_phrases):
                if ground_truth and len(ground_truth) > 10:
                    fps[5].detected = True
                    fps[5].evidence = "Answer is a refusal but ground truth exists."

        # FP7: Incomplete — answer partially matches ground truth
        if answer and ground_truth:
            gt_tokens = self._meaningful_tokens(ground_truth)
            a_tokens = self._meaningful_tokens(answer)
            if gt_tokens:
                coverage = len(gt_tokens & a_tokens) / len(gt_tokens)
                if 0.2 < coverage < 0.6:
                    fps[6].detected = True
                    fps[6].evidence = (
                        f"Answer covers {coverage*100:.0f}% of ground truth (partial)."
                    )

        # Compute quality score
        detected = [fp for fp in fps if fp.detected]
        quality = max(0, 100 - len(detected) * 15)

        # Primary failure
        primary = detected[0].id if detected else None

        # Recommendations
        recommendations = [fp.solution for fp in detected]

        return {
            "has_failure": len(detected) > 0,
            "failure_points": fps,
            "detected_failures": detected,
            "primary_failure": primary,
            "quality_score": quality,
            "recommendations": recommendations,
        }

    def batch_analyze(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze a batch of results and return aggregate statistics.
        Each result dict should have: query, answer, contexts, ground_truth.
        """
        fp_counts = {f"FP{i}": 0 for i in range(1, 8)}
        total = len(results)
        quality_scores = []

        for r in results:
            analysis = self.analyze(
                query=r.get("query", ""),
                answer=r.get("answer", ""),
                contexts=r.get("contexts", []),
                ground_truth=r.get("ground_truth"),
            )
            quality_scores.append(analysis["quality_score"])
            for fp in analysis["detected_failures"]:
                fp_counts[fp.id] = fp_counts.get(fp.id, 0) + 1

        return {
            "total_samples": total,
            "avg_quality": sum(quality_scores) / max(total, 1),
            "failure_distribution": fp_counts,
            "most_common_failure": max(fp_counts, key=fp_counts.get) if any(fp_counts.values()) else None,
            "failure_rate": sum(1 for q in quality_scores if q < 70) / max(total, 1),
        }

    @staticmethod
    def _meaningful_tokens(text: str) -> set:
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "of", "to",
            "in", "on", "at", "and", "or", "what", "who", "where",
            "when", "why", "how", "which", "do", "does", "did", "can",
            "could", "would", "should", "will", "it", "its", "this",
            "that", "these", "those", "be", "been", "being", "have",
            "has", "had", "for", "with", "from", "by", "about",
        }
        tokens = set(re.findall(r"\w+", text.lower()))
        return tokens - stopwords
