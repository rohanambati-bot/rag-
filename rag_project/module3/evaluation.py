# module3/evaluation.py
"""
Module 3 — Evaluation Framework
─────────────────────────────────
Metrics implemented:
  Retrieval (token-overlap, recall-oriented):
    • Answer-Recall  — does the retrieved context cover the gold-answer tokens?
    • Token-IoU      — bidirectional overlap between retrieved context and gold answer
    • Precision Omega (PΩ) — answer-token concentration in retrieved context
    • Noise Ratio    — fraction of retrieved tokens that are NOT in the gold answer
  Ranking:
    • Precision@K, Hit Rate, Mean Reciprocal Rank
  RAGAS-style (LLM-judged):
    • Faithfulness, Context Precision, Context Recall, Answer Relevance
    • The judge LLM is decoupled from the generator LLM to reduce self-preference
      bias (configurable via JUDGE_PROVIDER / JUDGE_MODEL).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import (  # noqa: E402
    ANSWER_RELEVANCE_THRESHOLD, CONTEXT_PRECISION_THRESHOLD, EVAL_BOOTSTRAP_ITERS,
    EVAL_RANDOM_SEED, EVAL_RESULTS_PATH, FAITHFULNESS_THRESHOLD, GROQ_API_KEY,
    GROQ_MODEL, JUDGE_MODEL, JUDGE_PROVIDER, LLM_MODEL, MISTRAL_API_KEY,
    NOISE_RATIO_MAX,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Common English stopwords for fairer token-overlap metrics.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "at", "by",
    "for", "to", "in", "on", "with", "without", "from", "into", "about",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
    "did", "have", "has", "had", "this", "that", "these", "those", "it",
    "its", "as", "than", "then", "so", "such", "what", "which", "who",
    "whom", "whose", "where", "when", "why", "how", "no", "not", "any",
})


# ═══════════════════════════════════════════════════════════════════════════════
# TOKENIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def _tokens(text: str, drop_stopwords: bool = True) -> List[str]:
    """Lowercase word tokens, optionally with stopwords removed."""
    toks = re.findall(r"\b\w+\b", text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in _STOPWORDS]
    return toks


def _token_set(text: str, drop_stopwords: bool = True) -> set:
    return set(_tokens(text, drop_stopwords=drop_stopwords))


# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL METRICS (fixed)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_answer_recall(retrieved_text: str, ground_truth: str) -> float:
    """
    Fraction of ground-truth (content) tokens present in the retrieved context.
    Recall-oriented: high when the answer is recoverable from what we retrieved.

    answer_recall = |GT_tokens ∩ Retrieved_tokens| / |GT_tokens|
    """
    gt = _token_set(ground_truth)
    if not gt:
        return 1.0
    rt = _token_set(retrieved_text)
    return len(gt & rt) / len(gt)


def compute_token_iou(retrieved_text: str, ground_truth: str) -> float:
    """
    Bidirectional Jaccard overlap between gold-answer tokens and the
    *salient* retrieved tokens (top-N most frequent content tokens).
    This bounds the denominator so a long context isn't unfairly penalised.

    iou = |GT ∩ R_top| / |GT ∪ R_top| where R_top = top-K retrieved tokens by frequency.
    """
    gt = _token_set(ground_truth)
    if not gt:
        return 1.0
    # Cap the retrieved-token set to ~5x the GT length so the metric reflects
    # signal rather than bag-of-words size.
    cap = max(20, 5 * len(gt))
    counts: Dict[str, int] = {}
    for t in _tokens(retrieved_text):
        counts[t] = counts.get(t, 0) + 1
    salient = {t for t, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:cap]}
    if not salient:
        return 0.0
    return len(gt & salient) / len(gt | salient)


def compute_precision_omega(retrieved_text: str, ground_truth: str) -> float:
    """
    Precision Omega — concentration of gold-answer tokens within the retrieved
    context. High when retrieval is tight; low when retrieval is bloated.

    PΩ = |GT_tokens ∩ Retrieved_tokens| / |Retrieved_tokens|
    """
    rt = _token_set(retrieved_text)
    if not rt:
        return 0.0
    gt = _token_set(ground_truth)
    return len(rt & gt) / len(rt)


def compute_noise_ratio(retrieved_text: str, ground_truth: str) -> float:
    """
    Fraction of retrieved tokens absent from the gold answer (lower is better).
    Note: bag-of-words noise will always be high when GT is short. Use this as
    a relative metric across architectures, not an absolute quality score.
    """
    rt = _token_set(retrieved_text)
    if not rt:
        return 0.0
    gt = _token_set(ground_truth)
    return 1.0 - len(rt & gt) / len(rt)


# Backward-compatible alias.
compute_iou = compute_token_iou


# ═══════════════════════════════════════════════════════════════════════════════
# RANKING METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_precision_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    relevant_set = set(relevant_doc_ids)
    return sum(1 for d in top_k if d in relevant_set) / k


def compute_hit_rate(retrieved_doc_ids: List[str], relevant_doc_ids: List[str]) -> float:
    relevant_set = set(relevant_doc_ids)
    return float(any(d in relevant_set for d in retrieved_doc_ids))


def compute_mrr(retrieved_doc_ids: List[str], relevant_doc_ids: List[str]) -> float:
    relevant_set = set(relevant_doc_ids)
    for rank, d in enumerate(retrieved_doc_ids, start=1):
        if d in relevant_set:
            return 1.0 / rank
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# RAGAS-STYLE LLM-JUDGED METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class RAGASEvaluator:
    """
    LLM-judged evaluation similar to RAGAS, but with a JUDGE LLM that is
    decoupled from the GENERATOR LLM to reduce self-preference bias.

    Provider priority for the judge:
      1. JUDGE_PROVIDER='groq' and GROQ_API_KEY set  → Groq with JUDGE_MODEL
      2. JUDGE_PROVIDER='mistral' and MISTRAL_API_KEY set → Mistral with JUDGE_MODEL
      3. Whichever provider has a key (with a warning that judge==generator)
    """

    FAITHFULNESS_PROMPT = (
        "You are a strict grader of RAG systems. Your task is to judge whether the "
        "ANSWER is grounded in the provided CONTEXT.\n\n"
        "Scoring rubric (return only a number between 0.0 and 1.0):\n"
        "  1.0  every claim in the answer is directly supported by the context\n"
        "  0.7  most claims are supported, with a minor unsupported detail\n"
        "  0.4  the answer is partially supported, partially fabricated\n"
        "  0.0  the answer contradicts the context or is fabricated\n\n"
        "Return ONLY a single floating-point number. No prose.\n\n"
        "Context:\n{context}\n\nAnswer:\n{answer}"
    )

    CONTEXT_PRECISION_PROMPT = (
        "You are evaluating RAG retrieval quality. Estimate the fraction of the "
        "retrieved CONTEXT that is RELEVANT to answering the QUESTION.\n\n"
        "Return ONLY a number between 0.0 (none relevant) and 1.0 (all relevant).\n\n"
        "Question: {question}\nContext:\n{context}"
    )

    CONTEXT_RECALL_PROMPT = (
        "You are evaluating RAG retrieval. Estimate how much of the GROUND TRUTH "
        "answer is recoverable from the retrieved CONTEXT.\n\n"
        "Return ONLY a number between 0.0 (not recoverable) and 1.0 (fully recoverable).\n\n"
        "Ground truth: {ground_truth}\nContext:\n{context}"
    )

    ANSWER_RELEVANCE_PROMPT = (
        "Evaluate how directly and completely the ANSWER addresses the QUESTION. "
        "Ignore factual correctness — judge only relevance and completeness.\n\n"
        "Return ONLY a number between 0.0 and 1.0.\n\n"
        "Question: {question}\nAnswer:\n{answer}"
    )

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = (provider or JUDGE_PROVIDER or "").lower()
        self.model = model or JUDGE_MODEL
        self._client = None
        self._mode = "extractive"  # "groq" | "mistral" | "extractive"

        # Prefer an explicit judge configuration to reduce self-eval bias.
        if self.provider == "groq" and GROQ_API_KEY:
            self._init_groq()
        elif self.provider == "mistral" and MISTRAL_API_KEY:
            self._init_mistral()
        elif GROQ_API_KEY and GROQ_MODEL != self.model:
            # Different model on the same provider is acceptable.
            self.provider = "groq"
            self._init_groq()
        elif MISTRAL_API_KEY:
            logger.warning(
                "Judge LLM falls back to the same provider as the generator. "
                "Set JUDGE_PROVIDER and JUDGE_MODEL to a different family for a fairer eval."
            )
            self.provider = "mistral"
            self._init_mistral()
        else:
            raise RuntimeError(
                "RAGASEvaluator requires GROQ_API_KEY or MISTRAL_API_KEY. "
                "Set one in your environment to enable LLM-judged metrics."
            )

    def _init_groq(self) -> None:
        from groq import Groq
        self._client = Groq(api_key=GROQ_API_KEY)
        self._mode = "groq"
        logger.info("Judge LLM: Groq / %s", self.model)

    def _init_mistral(self) -> None:
        from mistralai import Mistral
        self._client = Mistral(api_key=MISTRAL_API_KEY)
        self._mode = "mistral"
        logger.info("Judge LLM: Mistral / %s", self.model)

    def _call(self, prompt: str) -> str:
        if self._mode == "groq":
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8,
            )
            return resp.choices[0].message.content or ""
        if self._mode == "mistral":
            resp = self._client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8,
            )
            return resp.choices[0].message.content or ""
        return ""

    def _score(self, prompt: str) -> Optional[float]:
        """Return a score in [0,1] or None if the call failed (caller decides)."""
        try:
            text = self._call(prompt).strip()
            match = re.search(r"-?\d+(?:\.\d+)?", text)
            if match:
                return max(0.0, min(1.0, float(match.group())))
            logger.warning("Judge returned non-numeric: %r", text[:80])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Judge call failed: %s", exc)
        return None

    def evaluate(
        self,
        question: str,
        context: str,
        answer: str,
        ground_truth: str = "",
    ) -> Dict[str, Optional[float]]:
        return {
            "faithfulness": self._score(
                self.FAITHFULNESS_PROMPT.format(context=context, answer=answer)
            ),
            "context_precision": self._score(
                self.CONTEXT_PRECISION_PROMPT.format(question=question, context=context)
            ),
            "context_recall": self._score(
                self.CONTEXT_RECALL_PROMPT.format(
                    ground_truth=ground_truth or "N/A", context=context
                )
            ),
            "answer_relevance": self._score(
                self.ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer)
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    values: List[float],
    iters: int = EVAL_BOOTSTRAP_ITERS,
    confidence: float = 0.95,
    seed: int = EVAL_RANDOM_SEED,
) -> Tuple[float, float]:
    """Return (lower, upper) bootstrap CI bounds for the mean."""
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return (0.0, 0.0)
    means = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = sample.mean()
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(lo), float(hi)


# ═══════════════════════════════════════════════════════════════════════════════
# FULL EVALUATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluationPipeline:
    """
    Runs all metrics over a synthetic dataset and stores per-sample and summary
    results to CSV with bootstrapped confidence intervals.
    """

    def __init__(
        self,
        ragas_evaluator: Optional[RAGASEvaluator] = None,
        output_path: str = EVAL_RESULTS_PATH,
        seed: int = EVAL_RANDOM_SEED,
    ):
        self.ragas_evaluator = ragas_evaluator
        self.output_path = output_path
        self.seed = seed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def evaluate_sample(
        self,
        question: str,
        ground_truth: str,
        retrieved_docs: List[Any],
        answer: str,
        architecture: str = "unknown",
    ) -> Dict[str, Any]:
        retrieved_text = " ".join(getattr(d, "page_content", str(d)) for d in retrieved_docs)

        result: Dict[str, Any] = {
            "architecture": architecture,
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "num_docs_retrieved": len(retrieved_docs),
            "answer_recall": compute_answer_recall(retrieved_text, ground_truth),
            "iou": compute_token_iou(retrieved_text, ground_truth),
            "precision_omega": compute_precision_omega(retrieved_text, ground_truth),
            "noise_ratio": compute_noise_ratio(retrieved_text, ground_truth),
        }

        if self.ragas_evaluator:
            ragas = self.ragas_evaluator.evaluate(
                question=question, context=retrieved_text,
                answer=answer, ground_truth=ground_truth,
            )
            # Drop None (failed-call) values so they don't poison the mean.
            for k, v in ragas.items():
                if v is not None:
                    result[k] = v

        return result

    def evaluate_architecture(
        self,
        architecture_name: str,
        rag_system: Any,
        synthetic_dataset: List[Dict[str, Any]],
        max_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        # Deterministic sampling via a seeded RNG.
        rng = np.random.default_rng(self.seed)
        samples = list(synthetic_dataset)
        if max_samples and max_samples < len(samples):
            idx = rng.choice(len(samples), size=max_samples, replace=False)
            samples = [samples[i] for i in idx]

        rows: List[Dict[str, Any]] = []
        for i, sample in enumerate(samples):
            question = sample.get("question", "")
            ground_truth = sample.get("ground_truth", "")
            if not question:
                continue
            try:
                result = rag_system.run(question)
                metrics = self.evaluate_sample(
                    question=question,
                    ground_truth=ground_truth,
                    retrieved_docs=result.get("context_docs", []),
                    answer=result.get("answer", ""),
                    architecture=architecture_name,
                )
                metrics["latency_ms"] = result.get("latency_ms", 0)
                metrics["query_type"] = sample.get("query_type", "unknown")
                rows.append(metrics)
            except Exception as exc:  # noqa: BLE001 — keep evaluating other samples
                logger.warning("Eval failed for sample %d (%s): %s", i, architecture_name, exc)

        df = pd.DataFrame(rows)
        if not df.empty:
            mean_faith = df["faithfulness"].mean() if "faithfulness" in df.columns else float("nan")
            logger.info("[%s] n=%d | faithfulness=%.3f", architecture_name, len(df), mean_faith)
        return df

    def compare_architectures(
        self,
        results: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Aggregate per-architecture metrics with mean + 95% bootstrap CIs and
        sample sizes. Persists to CSV.
        """
        metric_cols = [
            "answer_recall", "iou", "precision_omega", "noise_ratio",
            "faithfulness", "context_precision", "context_recall",
            "answer_relevance", "latency_ms",
        ]
        rows = []
        for arch_name, df in results.items():
            row: Dict[str, Any] = {"architecture": arch_name, "n": int(len(df))}
            for col in metric_cols:
                if col in df.columns and not df[col].dropna().empty:
                    vals = df[col].dropna().tolist()
                    row[col] = round(float(np.mean(vals)), 4)
                    if col != "latency_ms":
                        lo, hi = bootstrap_ci(vals, seed=self.seed)
                        row[f"{col}_ci_lo"] = round(lo, 4)
                        row[f"{col}_ci_hi"] = round(hi, 4)
            rows.append(row)

        summary = pd.DataFrame(rows)
        summary.to_csv(self.output_path, index=False)
        logger.info("Evaluation summary saved to %s", self.output_path)
        return summary

    def check_quality_gate(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, float]]:
        metrics = {
            "faithfulness": float(df["faithfulness"].mean()) if "faithfulness" in df else 0.0,
            "context_precision": float(df["context_precision"].mean()) if "context_precision" in df else 0.0,
            "answer_relevance": float(df["answer_relevance"].mean()) if "answer_relevance" in df else 0.0,
            "noise_ratio": float(df["noise_ratio"].mean()) if "noise_ratio" in df else 1.0,
        }
        passed = (
            metrics["faithfulness"] >= FAITHFULNESS_THRESHOLD
            and metrics["context_precision"] >= CONTEXT_PRECISION_THRESHOLD
            and metrics["answer_relevance"] >= ANSWER_RELEVANCE_THRESHOLD
            and metrics["noise_ratio"] <= NOISE_RATIO_MAX
        )
        return passed, metrics


if __name__ == "__main__":
    print("Module 3 loaded. Use EvaluationPipeline and RAGASEvaluator.")
