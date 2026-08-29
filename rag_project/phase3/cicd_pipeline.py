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
import time
import logging
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import (
    FAITHFULNESS_THRESHOLD, CONTEXT_PRECISION_THRESHOLD, ANSWER_RELEVANCE_THRESHOLD,
    NOISE_RATIO_MAX, CICD_REPORT_PATH, OUTPUTS_DIR,
    REDIS_HOST, REDIS_PORT, CACHE_SIMILARITY_THRESHOLD, CACHE_TTL_SECONDS,
    EMBEDDING_MODEL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC CACHE (Redis-backed)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CacheEntry:
    query: str
    answer: str
    contexts: List[str]
    embedding: List[float]
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0


class SemanticCache:
    """
    Redis-backed semantic cache with vector similarity lookup.

    On every query:
      1. Embed the query.
      2. Scan cached embeddings for cosine similarity ≥ threshold.
      3. Cache hit → return cached answer instantly.
      4. Cache miss → execute RAG pipeline → store result.

    Falls back to an in-memory dict if Redis is unavailable.
    """

    def __init__(
        self,
        embedding_model=None,
        redis_host: str = REDIS_HOST,
        redis_port: int = REDIS_PORT,
        similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD,
        ttl_seconds: int = CACHE_TTL_SECONDS,
    ):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self._memory_cache: Dict[str, CacheEntry] = {}  # fallback
        self._use_redis = False
        self._redis = None
        self.embedding_model = embedding_model
        self._stats = {"hits": 0, "misses": 0, "total_latency_saved_ms": 0.0}

        # Try Redis
        try:
            import redis as redis_lib
            self._redis = redis_lib.Redis(
                host=redis_host, port=redis_port, decode_responses=False, socket_timeout=2
            )
            self._redis.ping()
            self._use_redis = True
            logger.info("SemanticCache: connected to Redis at %s:%d", redis_host, redis_port)
        except Exception as exc:
            logger.warning(
                "SemanticCache: Redis unavailable (%s) — using in-memory fallback.", exc
            )

        # Load embedding model if not provided
        if self.embedding_model is None:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    encode_kwargs={"normalize_embeddings": True},
                )
            except Exception as exc:
                logger.warning("Embedding model unavailable: %s", exc)

    def _embed(self, text: str) -> Optional[np.ndarray]:
        """Embed text into a unit vector."""
        if self.embedding_model is None:
            return None
        try:
            vec = self.embedding_model.embed_query(text)
            arr = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(arr)
            return arr / norm if norm > 0 else arr
        except Exception as exc:
            logger.warning("Embedding failed: %s", exc)
            return None

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # assumes unit vectors

    def _cache_key(self, query: str) -> str:
        return "rag_cache:" + hashlib.sha256(query.encode()).hexdigest()

    # ─── Redis helpers ──────────────────────────────────────────────────────────

    def _redis_store(self, key: str, entry: CacheEntry) -> None:
        payload = json.dumps({
            "query": entry.query,
            "answer": entry.answer,
            "contexts": entry.contexts,
            "embedding": entry.embedding,
            "timestamp": entry.timestamp,
            "hit_count": entry.hit_count,
        }).encode()
        self._redis.setex(key, self.ttl_seconds, payload)

    def _redis_load_all(self) -> List[CacheEntry]:
        entries = []
        try:
            keys = self._redis.keys("rag_cache:*")
            for k in keys:
                raw = self._redis.get(k)
                if raw:
                    data = json.loads(raw.decode())
                    entries.append(CacheEntry(**data))
        except Exception as exc:
            logger.warning("Redis scan failed: %s", exc)
        return entries

    # ─── Public API ─────────────────────────────────────────────────────────────

    def lookup(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Check cache for a semantically similar previous query.
        Returns cached result dict or None on miss.
        """
        query_vec = self._embed(query)
        if query_vec is None:
            return None

        entries = (
            self._redis_load_all()
            if self._use_redis
            else list(self._memory_cache.values())
        )

        best_score = 0.0
        best_entry: Optional[CacheEntry] = None

        for entry in entries:
            cached_vec = np.array(entry.embedding, dtype=np.float32)
            score = self._cosine(query_vec, cached_vec)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self.similarity_threshold:
            self._stats["hits"] += 1
            logger.debug("Cache HIT (similarity=%.3f): %s", best_score, query[:60])
            return {
                "answer": best_entry.answer,
                "contexts": best_entry.contexts,
                "cache_hit": True,
                "similarity_score": round(best_score, 4),
                "latency_ms": 0,
            }

        self._stats["misses"] += 1
        logger.debug("Cache MISS (best_sim=%.3f): %s", best_score, query[:60])
        return None

    def store(self, query: str, answer: str, contexts: List[str]) -> None:
        """Store a query-answer pair in the cache."""
        query_vec = self._embed(query)
        if query_vec is None:
            return

        entry = CacheEntry(
            query=query,
            answer=answer,
            contexts=contexts,
            embedding=query_vec.tolist(),
        )

        if self._use_redis:
            self._redis_store(self._cache_key(query), entry)
        else:
            self._memory_cache[self._cache_key(query)] = entry

    def stats(self) -> Dict[str, Any]:
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0
        return {
            **self._stats,
            "total_queries": total,
            "hit_rate": round(hit_rate, 4),
        }

    def clear(self) -> None:
        if self._use_redis:
            keys = self._redis.keys("rag_cache:*")
            if keys:
                self._redis.delete(*keys)
        else:
            self._memory_cache.clear()


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
    from configs.config import MISTRAL_API_KEY
    if MISTRAL_API_KEY:
        ragas_evaluator = RAGASEvaluator()

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
