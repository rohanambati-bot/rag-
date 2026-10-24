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
import math
from dataclasses import dataclass, asdict, field
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
from phase3.semantic_cache import (  # noqa: E402,F401
    CacheEntry, SemanticCache, require_generation_cache,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED RAG WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class CachedRAG:
    """
    Wraps any BaseRAG instance with the semantic cache layer.
    Cache hits skip the retrieval + generation pipeline entirely.
    """

    def __init__(self, rag_system: Any, cache: SemanticCache, *, allow_legacy_cache: bool = False):
        self.rag = rag_system
        self.cache = cache
        # Compatibility is opt-in and outside the revocation guarantee. Serving
        # never enables it; callers migrating old adapters can select it explicitly.
        self.allow_legacy_cache = allow_legacy_cache
        self.name = f"{getattr(rag_system, 'name', 'RAG')} + SemanticCache"

    def run(self, query: str) -> Dict[str, Any]:
        token = None
        if not self.allow_legacy_cache:
            require_generation_cache(self.cache)
            token = self.cache.begin_generation()
        cached = (self.cache.lookup_generation(token, query) if token is not None
                  else self.cache.lookup(query))
        if cached:
            return {
                "architecture": self.name,
                "query": query,
                **cached,
            }

        # Cache miss: run full pipeline
        result = self.rag.run(query)
        store_with_citations = getattr(self.cache, "store_with_citations", None)
        if token is not None:
            self.cache.publish_generation(token, query, result.get("answer", ""),
                                          result.get("contexts", []), result.get("citations", []))
            self.cache.validate_generation(token)
        elif store_with_citations is not None:
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
    metric_status: Dict[str, bool] = field(default_factory=dict)
    successful_samples: int = 0
    failed_samples: int = 0
    skipped_samples: int = 0
    evaluation_valid: bool = False
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
            if k not in self.thresholds:
                continue
            threshold = self.thresholds.get(k)
            direction = "≥" if k != "noise_ratio" else "≤"
            ok = "✓" if self.metric_status.get(k, False) else "✗"
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

    passed, gate_metrics = pipeline.check_quality_gate(df)
    metrics: Dict[str, float] = {
        name: round(float(gate_metrics[name]), 4)
        for name in THRESHOLDS
        if isinstance(gate_metrics.get(name), (int, float))
    }
    metrics.update({
        name: gate_metrics[name]
        for name in ("expected_samples", "successful_samples", "failed_samples", "skipped_samples")
        if name in gate_metrics
    })
    metric_status: Dict[str, bool] = {}
    failure_reasons: List[str] = []
    for metric, threshold in THRESHOLDS.items():
        value = gate_metrics.get(metric)
        higher = metric != "noise_ratio"
        ok = isinstance(value, (int, float)) and math.isfinite(float(value)) and (
            float(value) >= threshold if higher else float(value) <= threshold
        )
        metric_status[metric] = ok
        if not ok:
            direction = "<" if higher else ">"
            requirement = "≥" if higher else "≤"
            shown = "missing/non-finite" if not isinstance(value, (int, float)) or not math.isfinite(float(value)) else f"{float(value):.4f}"
            failure_reasons.append(f"{metric} = {shown} {direction} {threshold} (required {requirement} {threshold})")
    if not passed:
        coverage = ", ".join(
            f"{name}={gate_metrics.get(name, 0)}"
            for name in ("expected_samples", "successful_samples", "failed_samples", "skipped_samples")
        )
        if gate_metrics.get("failed_samples", 0) or gate_metrics.get("skipped_samples", 0):
            failure_reasons.append(f"incomplete sample coverage ({coverage})")

    report = CICDReport(
        run_id=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        timestamp=datetime.utcnow().isoformat(),
        architecture=architecture_name,
        num_samples=int(gate_metrics.get("expected_samples", len(df))),
        metrics=metrics,
        thresholds=THRESHOLDS,
        passed=passed,
        failure_reasons=failure_reasons,
        metric_status=metric_status,
        successful_samples=int(gate_metrics.get("successful_samples", 0)),
        failed_samples=int(gate_metrics.get("failed_samples", 0)),
        skipped_samples=int(gate_metrics.get("skipped_samples", 0)),
        evaluation_valid=(
            int(gate_metrics.get("successful_samples", 0))
            == int(gate_metrics.get("expected_samples", 0))
            and not gate_metrics.get("failed_samples", 0)
            and not gate_metrics.get("skipped_samples", 0)
        ),
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
