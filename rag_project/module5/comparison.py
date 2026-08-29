# module5/comparison.py
"""
Architecture Comparator — Run all 4 architectures on the same query
and present a side-by-side comparison table.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result from one architecture for a single query."""
    architecture: str
    answer: str
    latency_ms: float
    num_chunks: int
    contexts: List[str] = field(default_factory=list)
    confidence: float = 0.0
    confidence_label: str = ""

# In module5/comparison.py add:

def compute_ensemble_confidence(
    results: Dict[str, List[str]],  # arch → list of chunk texts
    query: str
) -> Dict[str, float]:
    """
    Novel: Cross-architecture agreement as a confidence signal.
    
    When multiple architectures retrieve the same chunk,
    that chunk has higher information reliability.
    
    Ensemble_confidence(chunk) = 
        (num_architectures_that_retrieved_chunk / total_architectures)
        × PΩ(chunk, query)
    
    This creates a voting mechanism across architectures,
    enabling an ensemble retrieval strategy.
    """
    # Count how many architectures retrieved each chunk
    chunk_votes: Dict[str, int] = {}
    chunk_texts: Dict[str, str] = {}
    
    for arch_name, chunks in results.items():
        for chunk in chunks:
            key = chunk[:100]
            chunk_votes[key] = chunk_votes.get(key, 0) + 1
            chunk_texts[key] = chunk
    
    total_archs = len(results)
    
    # Score each chunk
    scored = {}
    for key, votes in chunk_votes.items():
        agreement_score = votes / total_archs
        scored[key] = {
            "text": chunk_texts[key],
            "agreement": agreement_score,
            "voted_by": votes,
            "confidence": agreement_score  # extend with PΩ
        }
    
    return scored

class ArchitectureComparator:
    """
    Runs the same query through all architectures simultaneously
    and returns a structured comparison.
    """

    def __init__(self, architectures: Dict[str, Any]):
        self.architectures = architectures

    def compare(self, query: str) -> List[ComparisonResult]:
        """Run query through all architectures and return comparison."""
        results = []

        for name, rag in self.architectures.items():
            try:
                t0 = time.time()
                result = rag.run(query)
                latency = (time.time() - t0) * 1000

                results.append(ComparisonResult(
                    architecture=name,
                    answer=result.get("answer", "Error"),
                    latency_ms=round(latency, 1),
                    num_chunks=len(result.get("contexts", [])),
                    contexts=result.get("contexts", []),
                ))
            except Exception as exc:
                results.append(ComparisonResult(
                    architecture=name,
                    answer=f"Error: {exc}",
                    latency_ms=0,
                    num_chunks=0,
                ))

        return results

    def compare_with_confidence(
        self, query: str, confidence_scorer=None
    ) -> List[ComparisonResult]:
        """Run comparison with confidence scoring."""
        results = self.compare(query)

        if confidence_scorer:
            for r in results:
                score_data = confidence_scorer.score(
                    query=query,
                    answer=r.answer,
                    contexts=r.contexts,
                )
                r.confidence = score_data["score"]
                r.confidence_label = f"{score_data['emoji']} {score_data['label']}"

        return results

    def rank_by_confidence(self, results: List[ComparisonResult]) -> List[ComparisonResult]:
        """Rank results by confidence score (highest first)."""
        return sorted(results, key=lambda r: r.confidence, reverse=True)

    def rank_by_latency(self, results: List[ComparisonResult]) -> List[ComparisonResult]:
        """Rank results by latency (fastest first)."""
        return sorted(results, key=lambda r: r.latency_ms)
