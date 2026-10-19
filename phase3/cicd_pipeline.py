# phase3/cicd_pipeline.py
"""
Phase 3 — CI/CD Automated Evaluation Loop & Semantic Caching
─────────────────────────────────────────────────────────────
Pipeline stages:
  1. Code Commit / Config Change  (trigger)
  2. Synthetic Query Run          (load dataset, run RAG)
  3. RAGAS Scoring                (faithfulness, precision, relevance)
  4. Quality Gate Check           (threshold comparison)
  5. Deploy / Fail Build          (pass/fail output + JSON report)

Semantic Cache:
  • Redis-backed vector cache
  • Intercepts semantically similar queries before retrieval pipeline
  • ~0ms latency on cache hit
  • 40–60% cost reduction target
"""

import os
import json
import hashlib
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from datetime import datetime

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Cache-related settings now live with the cache itself in phase3/semantic_cache.py.
from configs.config import (
    FAITHFULNESS_THRESHOLD, CONTEXT_PRECISION_THRESHOLD, ANSWER_RELEVANCE_THRESHOLD,
    NOISE_RATIO_MAX, CICD_REPORT_PATH, OUTPUTS_DIR,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC CACHE (Redis-backed)
# ═══════════════════════════════════════════════════════════════════════════════

# The semantic cache lives in phase3/semantic_cache.py. It is re-exported here so
# existing imports (`from phase3.cicd_pipeline import SemanticCache`) keep working.
#
# The previous in-file implementation issued `KEYS rag_cache:*` plus one GET per key
# on EVERY lookup and then scored candidates in a Python loop — a blocking Redis
# command whose cost grew with cache size. The replacement keeps an in-process vector
# index (hnswlib ANN, or a single vectorised matrix product as fallback) and treats
# Redis purely as write-through persistence read once at startup via SCAN.
from phase3.semantic_cache import CacheEntry, SemanticCache  # noqa: E402,F401


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED RAG WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class CachedRAG:
    """
    Wraps any BaseRAG instance with the semantic cache layer.
    Cache hits skip the retrieval + generation pipeline entirely.
    """

    def __init__(self, rag_system: Any, cache: SemanticCache):
        self.rag = rag_system
        self.cache = cache
        self.name = f"{getattr(rag_system, 'name', 'RAG')} + SemanticCache"

    def run(self, query: str) -> Dict[str, Any]:
        # Check cache first
        cached = self.cache.lookup(query)
        if cached:
            return {
                "architecture": self.name,
                "query": query,
                **cached,
            }

        # Cache miss: run full pipeline
        result = self.rag.run(query)
        store_with_citations = getattr(self.cache, "store_with_citations", None)
        if store_with_citations is not None:
            store_with_citations(
                query=query,
                answer=result.get("answer", ""),
                contexts=result.get("contexts", []),
                citations=result.get("citations", []),
            )
        else:
            self.cache.store(
                query=query,
                answer=result.get("answer", ""),
                contexts=result.get("contexts", []),
            )
        result["cache_hit"] = False
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# CI/CD QUALITY GATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CICDReport:
    run_id: str
    timestamp: str
    architecture: str
    num_samples: int
    metrics: Dict[str, float]
    thresholds: Dict[str, float]
    passed: bool
    failure_reasons: List[str]
    commit_hash: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, path: str = CICD_REPORT_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("CI/CD report saved to %s", path)

    def print_summary(self) -> None:
        status = "✅  PASS — Deploy to Production" if self.passed else "❌  FAIL — Block & Notify"
        print("\n" + "═" * 60)
        print(f"  CI/CD Quality Gate: {status}")
        print("─" * 60)
        for k, v in self.metrics.items():
            threshold = self.thresholds.get(k)
            direction = "≥" if k != "noise_ratio" else "≤"
            ok = "✓" if self.passed or k not in self.failure_reasons else "✗"
            print(f"  {ok}  {k:<25} {v:.4f}  ({direction} {threshold})")
        if self.failure_reasons:
            print("\n  Failure reasons:")
            for r in self.failure_reasons:
                print(f"    • {r}")
        print("═" * 60 + "\n")


THRESHOLDS = {
    "faithfulness": FAITHFULNESS_THRESHOLD,
    "context_precision": CONTEXT_PRECISION_THRESHOLD,
    "answer_relevance": ANSWER_RELEVANCE_THRESHOLD,
    "noise_ratio": NOISE_RATIO_MAX,
}


def run_quality_gate(
    rag_system: Any,
    synthetic_dataset: List[Dict[str, Any]],
    architecture_name: str = "CRAG",
    max_samples: int = 50,
    commit_hash: str = "",
    report_path: str = CICD_REPORT_PATH,
) -> CICDReport:
    """
    Full CI/CD evaluation loop:
      1. Run RAG on synthetic queries.
      2. Compute metrics (uses module3 EvaluationPipeline).
      3. Check thresholds.
      4. Return and save CICDReport.
    """
    import importlib
    eval_mod = importlib.import_module("module3.evaluation")
    EvaluationPipeline = eval_mod.EvaluationPipeline
    RAGASEvaluator = eval_mod.RAGASEvaluator

    logger.info("CI/CD: Starting evaluation for %s (%d samples)…", architecture_name, max_samples)

    ragas_evaluator = None
    from configs.config import GROQ_API_KEY, MISTRAL_API_KEY, OPENROUTER_API_KEY
    if OPENROUTER_API_KEY or GROQ_API_KEY or MISTRAL_API_KEY:
        ragas_evaluator = RAGASEvaluator()
        # Refuse to spend a long run on a retired/misconfigured judge.  Without
        # this preflight, every per-sample call would fail independently and the
        # gate would only report missing metrics after wasting the whole budget.
        try:
            probe_score = ragas_evaluator.health_check()
            logger.info("CI/CD judge health check passed (probe score=%.3f).", probe_score)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "CI/CD judge health check failed; refusing to run a judged gate. "
                "Fix JUDGE_PROVIDER/JUDGE_MODEL before retrying."
            ) from exc

    pipeline = EvaluationPipeline(ragas_evaluator=ragas_evaluator)
    df = pipeline.evaluate_architecture(
        architecture_name=architecture_name,
        rag_system=rag_system,
        synthetic_dataset=synthetic_dataset,
        max_samples=max_samples,
    )

    metrics: Dict[str, float] = {}
    failure_reasons: List[str] = []

    for col in ["faithfulness", "context_precision", "answer_relevance", "noise_ratio"]:
        if col in df.columns:
            metrics[col] = round(float(df[col].mean()), 4)

    # Gate checks
    def _check(metric: str, threshold: float, higher_is_better: bool = True) -> None:
        val = metrics.get(metric, 0.0 if higher_is_better else 1.0)
        if higher_is_better and val < threshold:
            failure_reasons.append(
                f"{metric} = {val:.4f} < {threshold} (required ≥ {threshold})"
            )
        elif not higher_is_better and val > threshold:
            failure_reasons.append(
                f"{metric} = {val:.4f} > {threshold} (required ≤ {threshold})"
            )

    _check("faithfulness", FAITHFULNESS_THRESHOLD, higher_is_better=True)
    _check("context_precision", CONTEXT_PRECISION_THRESHOLD, higher_is_better=True)
    _check("answer_relevance", ANSWER_RELEVANCE_THRESHOLD, higher_is_better=True)
    _check("noise_ratio", NOISE_RATIO_MAX, higher_is_better=False)

    passed = len(failure_reasons) == 0

    report = CICDReport(
        run_id=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        timestamp=datetime.utcnow().isoformat(),
        architecture=architecture_name,
        num_samples=len(df),
        metrics=metrics,
        thresholds=THRESHOLDS,
        passed=passed,
        failure_reasons=failure_reasons,
        commit_hash=commit_hash or os.getenv("GITHUB_SHA", "local"),
    )

    report.save(report_path)
    report.print_summary()
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# TRULENS PRODUCTION DRIFT MONITOR (stub)
# ═══════════════════════════════════════════════════════════════════════════════

class TruLensMonitor:
    """
    Lightweight wrapper around TruLens for production drift monitoring.
    Logs faithfulness scores per request and alerts on rolling degradation.
    """

    def __init__(self, alert_threshold: float = FAITHFULNESS_THRESHOLD, window: int = 20):
        self.alert_threshold = alert_threshold
        self.window = window
        self._score_buffer: List[float] = []
        self._log_path = os.path.join(OUTPUTS_DIR, "trulens_monitor.jsonl")
        os.makedirs(OUTPUTS_DIR, exist_ok=True)

    def record(self, query: str, answer: str, faithfulness: float) -> None:
        self._score_buffer.append(faithfulness)
        if len(self._score_buffer) > self.window * 3:
            self._score_buffer = self._score_buffer[-self.window * 3:]

        entry = {
            "ts": datetime.utcnow().isoformat(),
            "query_hash": hashlib.md5(query.encode()).hexdigest()[:8],
            "faithfulness": faithfulness,
        }
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Rolling window alert
        if len(self._score_buffer) >= self.window:
            rolling_mean = np.mean(self._score_buffer[-self.window:])
            if rolling_mean < self.alert_threshold:
                logger.warning(
                    "TruLens ALERT: rolling faithfulness %.3f < %.3f over last %d requests.",
                    rolling_mean, self.alert_threshold, self.window,
                )

    def rolling_stats(self) -> Dict[str, float]:
        if not self._score_buffer:
            return {}
        arr = np.array(self._score_buffer[-self.window:])
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": len(arr),
        }


if __name__ == "__main__":
    print("Phase 3 module loaded. Use run_quality_gate() and SemanticCache.")
