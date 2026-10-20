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

import hashlib
import logging
import os
import re
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import (  # noqa: E402
    ANSWER_RELEVANCE_THRESHOLD, CONTEXT_PRECISION_THRESHOLD, EVAL_BOOTSTRAP_ITERS,
    EVAL_RANDOM_SEED, EVAL_RESULTS_PATH, FAITHFULNESS_THRESHOLD, GROQ_API_KEY,
    GROQ_MODEL, JUDGE_MODEL, JUDGE_PROVIDER, MISTRAL_API_KEY,
    NOISE_RATIO_MAX, OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _safe_failure_message(exc: Exception) -> str:
    """Keep per-sample diagnostics bounded and free of prompt/provider details."""
    message = " ".join(str(exc).split())
    message = re.sub(r"(?i)(bearer\s+|api[_ -]?key\s*[=:]\s*)\S+", r"\1[redacted]", message)
    message = re.sub(r"(?i)sk-[A-Za-z0-9_-]+", "[redacted]", message)
    return message[:240] or "operation failed"


def _successful_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows that completed the requested sample, including legacy frames."""
    if "sample_status" not in frame.columns:
        return frame
    return frame.loc[frame["sample_status"] == "success"]

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
    p = len(rt & gt) / len(rt)
    return p * p   # PΩ = p² — quadratic noise penalty


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


def compute_mrr(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: Optional[int] = None) -> float:
    """
    Mean Reciprocal Rank of the first relevant document.

    ``k`` truncates the ranked list (MS MARCO's official passage-ranking metric
    is MRR@10, so pass ``k=10`` to match published leaderboard numbers).
    """
    ranked = retrieved_doc_ids[:k] if k else retrieved_doc_ids
    relevant_set = set(relevant_doc_ids)
    for rank, d in enumerate(ranked, start=1):
        if d in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_recall_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int) -> float:
    """
    Fraction of ALL known-relevant documents that appear in the top-k.

    Unlike :func:`compute_answer_recall` (token overlap), this is graded against
    human relevance judgements (qrels), so it is directly comparable to published
    retrieval numbers.
    """
    relevant_set = set(relevant_doc_ids)
    if not relevant_set or k <= 0:
        return 0.0
    top_k = set(retrieved_doc_ids[:k])
    return len(top_k & relevant_set) / len(relevant_set)


def compute_ndcg_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int = 10) -> float:
    """
    Normalised Discounted Cumulative Gain at k with binary relevance gains.

    DCG@k  = Σ_{i=1..k} rel_i / log2(i + 1)
    IDCG@k = the DCG of the best possible ranking (all relevant docs first)
    nDCG@k = DCG@k / IDCG@k  ∈ [0, 1]

    Rank-aware, so unlike Recall@k it rewards putting the relevant passage at
    rank 1 rather than rank k — the property that makes it the standard headline
    metric on MS MARCO / TREC Deep Learning and BEIR.
    """
    relevant_set = set(relevant_doc_ids)
    if not relevant_set or k <= 0:
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant_set:
            dcg += 1.0 / np.log2(i + 1)

    # Ideal ranking: every relevant doc packed into the top positions.
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


# ═══════════════════════════════════════════════════════════════════════════════
# QRELS (ground-truth relevance judgements)
# ═══════════════════════════════════════════════════════════════════════════════

def doc_id_for(doc: Any) -> str:
    """
    Stable identity for a chunk/document, derived from its text.

    Chunks are re-created on every run (and Chroma does not hand back the ids we
    indexed with), so a content hash is the only identifier that survives the
    round-trip from corpus → vector store → retrieved document. The first 200
    characters are used because that is the same span the MS MARCO loader uses to
    de-duplicate passages, so ids agree with the corpus-level dedup.
    """
    text = getattr(doc, "page_content", str(doc))
    return hashlib.sha1(text[:200].strip().lower().encode("utf-8")).hexdigest()[:16]


def build_qrels(docs: List[Any]) -> Dict[str, set]:
    """
    Build ``{question: {relevant_doc_id, ...}}`` from MS MARCO document metadata.

    MS MARCO ships human relevance labels: each passage carries ``is_selected``
    (1 when the annotator judged it to contain the answer) alongside the ``query``
    it was retrieved for. Those two fields are preserved through
    ``load_msmarco_documents`` → ``chunk_documents``, so every chunk inherits its
    parent passage's judgement and we can recover real qrels for free — no LLM and
    no manual labelling.

    Caveat: ``load_msmarco_documents`` de-duplicates passages globally, so a
    passage that is relevant to query A but was first seen under query B appears
    only once. Recall is therefore very slightly conservative.
    """
    qrels: Dict[str, set] = {}
    try:
        iterator = iter(docs)
    except TypeError:
        # Qrels are optional enrichment: a corpus we cannot iterate should degrade
        # to "no relevance labels" (token-overlap + LLM metrics still run) rather
        # than abort the whole evaluation.
        logger.warning("build_qrels: documents are not iterable — no qrels built.")
        return qrels

    for doc in iterator:
        meta = getattr(doc, "metadata", None) or {}
        if not isinstance(meta, dict) or not meta.get("is_selected"):
            continue
        question = (meta.get("query") or "").strip()
        if not question:
            continue
        qrels.setdefault(question, set()).add(doc_id_for(doc))
    logger.info("Built qrels for %d questions with >=1 relevant chunk.", len(qrels))
    return qrels


# ═══════════════════════════════════════════════════════════════════════════════
# RAGAS-STYLE LLM-JUDGED METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class RAGASEvaluator:
    """
    LLM-judged evaluation similar to RAGAS, but with a JUDGE LLM that is
    decoupled from the GENERATOR LLM to reduce self-preference bias.

    Provider priority for the judge:
      0. JUDGE_PROVIDER='openrouter' and OPENROUTER_API_KEY set → OpenRouter (OpenAI-
         compatible) with JUDGE_MODEL as the slug
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
        self._mode = "extractive"  # "openrouter" | "groq" | "mistral" | "extractive"

        # Prefer an explicit judge configuration to reduce self-eval bias.
        if self.provider == "openrouter" and OPENROUTER_API_KEY:
            self._init_openrouter()
        elif self.provider == "groq" and GROQ_API_KEY:
            self._init_groq()
        elif self.provider == "mistral" and MISTRAL_API_KEY:
            self._init_mistral()
        elif OPENROUTER_API_KEY:
            # A key is present without an explicit matching provider → route the
            # judge through OpenRouter (OpenAI-compatible; JUDGE_MODEL is its slug).
            self.provider = "openrouter"
            self._init_openrouter()
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

    def _init_openrouter(self) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        self._mode = "openrouter"
        logger.info("Judge LLM: OpenRouter / %s", self.model)

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
        # OpenRouter and Groq both expose the OpenAI-style chat.completions API.
        if self._mode in ("groq", "openrouter"):
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,  # headroom for reasoning models (gpt-oss/qwen); the score is parsed out via regex
            )
            return resp.choices[0].message.content or ""
        if self._mode == "mistral":
            resp = self._client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,  # headroom for reasoning models (gpt-oss/qwen); the score is parsed out via regex
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

    def health_check(self) -> float:
        """
        Issue ONE trivial judge prompt and confirm a numeric score comes back.

        This exercises the *real* judge call path (``self._call`` → ``self._client``
        with ``self.model``), so a mis-set or retired judge model surfaces here,
        once, before a long run — instead of being silently swallowed sample by
        sample by :meth:`_score` (which returns ``None`` on any failure and made
        BUG 1 indistinguishable from having no API key).

        Returns
        -------
        float
            The judge's numeric score (clamped to ``[0.0, 1.0]``) for a toy,
            trivially grounded context/answer pair.

        Raises
        ------
        RuntimeError
            If the judge call errors (retired/mis-set model, auth or network
            failure) or returns a non-numeric response. Raising — rather than
            returning ``None`` like :meth:`_score` — is deliberate: this guard
            must fail LOUDLY so a broken judge cannot degrade silently.
        """
        # A trivially grounded pair: the answer is fully supported by the
        # context, so any working judge returns a numeric faithfulness score.
        probe = self.FAITHFULNESS_PROMPT.format(
            context="The Eiffel Tower is located in Paris, France.",
            answer="The Eiffel Tower is in Paris.",
        )
        try:
            text = self._call(probe).strip()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Judge health check failed: the judge call raised for "
                f"provider={self.provider!r} model={self.model!r}: {exc}"
            ) from exc

        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match is None:
            raise RuntimeError(
                f"Judge health check failed: judge (provider={self.provider!r}, "
                f"model={self.model!r}) returned a non-numeric response "
                f"{text[:80]!r}; the judge model may be mis-set or retired."
            )
        return max(0.0, min(1.0, float(match.group())))

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
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    def evaluate_sample(
        self,
        question: str,
        ground_truth: str,
        retrieved_docs: List[Any],
        answer: str,
        architecture: str = "unknown",
        relevant_doc_ids: Optional[set] = None,
        use_judge: bool = True,
    ) -> Dict[str, Any]:
        retrieved_text = " ".join(getattr(d, "page_content", str(d)) for d in retrieved_docs)

        # Computed once and reused — this used to be called three times per sample.
        pomega = compute_precision_omega(retrieved_text, ground_truth)
        fingerprint = compute_noise_fingerprint(retrieved_text, ground_truth, question)

        result: Dict[str, Any] = {
            "architecture": architecture,
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "num_docs_retrieved": len(retrieved_docs),
            "answer_recall": compute_answer_recall(retrieved_text, ground_truth),
            "iou": compute_token_iou(retrieved_text, ground_truth),
            "precision_omega": pomega,
            "noise_ratio": compute_noise_ratio(retrieved_text, ground_truth),
            "paradox_flag": pomega < 0.4,
            "noise_fingerprint_topical": fingerprint["topical_noise"],
            "noise_fingerprint_peripheral": fingerprint["peripheral_noise"],
            "noise_fingerprint_random": fingerprint["random_noise"],
        }

        # ── Rank-aware IR metrics graded against human relevance judgements ──
        # Only emitted when qrels are supplied, so callers without labels keep the
        # previous column set. These need no LLM, so they scale to thousands of
        # queries at zero API cost.
        if relevant_doc_ids:
            retrieved_ids = [doc_id_for(d) for d in retrieved_docs]
            result["recall_at_k"] = compute_recall_at_k(
                retrieved_ids, relevant_doc_ids, k=len(retrieved_ids) or 1
            )
            result["precision_at_k"] = compute_precision_at_k(
                retrieved_ids, relevant_doc_ids, k=len(retrieved_ids) or 1
            )
            result["hit_rate"] = compute_hit_rate(retrieved_ids, relevant_doc_ids)
            result["mrr_at_10"] = compute_mrr(retrieved_ids, relevant_doc_ids, k=10)
            result["ndcg_at_10"] = compute_ndcg_at_k(retrieved_ids, relevant_doc_ids, k=10)
            result["num_relevant_known"] = len(relevant_doc_ids)

        if self.ragas_evaluator and use_judge:
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
        qrels: Optional[Dict[str, set]] = None,
        retrieval_only: bool = False,
    ) -> pd.DataFrame:
        """
        Evaluate one architecture over the dataset.

        Parameters
        ----------
        qrels
            Optional ``{question: {relevant_doc_id}}`` map (see :func:`build_qrels`).
            When supplied, rank-aware IR metrics (Recall@k, MRR@10, nDCG@10) are
            added to every row.
        retrieval_only
            When True, skip answer generation and LLM judging and only exercise
            ``rag_system.retrieve``. This makes the whole run LLM-free, so it can
            scale to thousands of queries with no API cost or rate limiting —
            the intended mode for large-n retrieval experiments.
        """
        # Deterministic sampling via a seeded RNG.
        rng = np.random.default_rng(self.seed)
        samples = list(synthetic_dataset)
        if max_samples and max_samples < len(samples):
            idx = rng.choice(len(samples), size=max_samples, replace=False)
            samples = [samples[i] for i in idx]

        expected_samples = len(samples)
        rows: List[Dict[str, Any]] = []
        for i, sample in enumerate(samples):
            question = sample.get("question", "")
            ground_truth = sample.get("ground_truth", "")
            if not question:
                rows.append({
                    "architecture": architecture_name,
                    "sample_index": i,
                    "question": question,
                    "ground_truth": ground_truth,
                    "query_type": sample.get("query_type", "unknown"),
                    "sample_status": "skipped",
                    "skip_reason": "missing_question",
                    "failure_type": "",
                    "failure_message": "",
                    "expected_samples": expected_samples,
                })
                continue
            try:
                if retrieval_only:
                    # No generation, no judge — pure retrieval timing + metrics.
                    t0 = time.time()
                    docs = rag_system.retrieve(question)
                    result = {
                        "context_docs": docs,
                        "answer": "",
                        "latency_ms": round((time.time() - t0) * 1000, 2),
                    }
                else:
                    result = rag_system.run(question)

                metrics = self.evaluate_sample(
                    question=question,
                    ground_truth=ground_truth,
                    retrieved_docs=result.get("context_docs", []),
                    answer=result.get("answer", ""),
                    architecture=architecture_name,
                    relevant_doc_ids=(qrels or {}).get(question),
                    use_judge=not retrieval_only,
                )

                metrics["latency_ms"] = result.get("latency_ms", 0)
                metrics["query_type"] = sample.get("query_type", "unknown")
                metrics["retrieval_path"] = getattr(rag_system, "_last_route", "N/A")
                metrics["sample_index"] = i
                metrics["sample_status"] = "success"
                metrics["failure_type"] = ""
                metrics["failure_message"] = ""
                metrics["expected_samples"] = expected_samples
                rows.append(metrics)
            except Exception as exc:  # noqa: BLE001 — keep evaluating other samples
                message = _safe_failure_message(exc)
                logger.warning(
                    "Eval failed for sample %d (%s): %s", i, architecture_name, message
                )
                rows.append({
                    "architecture": architecture_name,
                    "sample_index": i,
                    "question": question,
                    "ground_truth": ground_truth,
                    "query_type": sample.get("query_type", "unknown"),
                    "sample_status": "failed",
                    "failure_type": type(exc).__name__,
                    "failure_message": message,
                    "expected_samples": expected_samples,
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            valid = _successful_rows(df)
            status = df["sample_status"] if "sample_status" in df else pd.Series(
                "success", index=df.index
            )
            failed = int((status == "failed").sum())
            skipped = int((status == "skipped").sum())
            mean_faith = valid["faithfulness"].mean() if "faithfulness" in valid else float("nan")
            logger.info(
                "[%s] expected=%d successful=%d failed=%d skipped=%d | faithfulness=%.3f",
                architecture_name, expected_samples, len(valid), failed, skipped, mean_faith,
            )
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
            # Rank-aware IR metrics (present only when qrels were supplied).
            "recall_at_k", "precision_at_k", "hit_rate", "mrr_at_10", "ndcg_at_10",
            # Token-overlap metrics.
            "answer_recall", "iou", "precision_omega", "noise_ratio",
            # LLM-judged metrics.
            "faithfulness", "context_precision", "context_recall",
            "answer_relevance", "latency_ms",
        ]
        rows = []
        for arch_name, df in results.items():
            valid = _successful_rows(df)
            status = df["sample_status"] if "sample_status" in df else None
            if status is None:
                expected = len(df)
                failed = skipped = 0
            else:
                expected = int(df["expected_samples"].max()) if "expected_samples" in df else len(df)
                failed = int((status == "failed").sum())
                skipped = int((status == "skipped").sum())
            row: Dict[str, Any] = {
                "architecture": arch_name,
                "n": int(len(valid)),
                "expected_samples": expected,
                "successful_samples": int(len(valid)),
                "failed_samples": failed,
                "skipped_samples": skipped,
                "failure_rate": round((failed + skipped) / expected, 4) if expected else 1.0,
                "evaluation_valid": bool(expected and not failed and not skipped
                                          and len(valid) == expected),
            }
            for col in metric_cols:
                if col in valid.columns and not valid[col].dropna().empty:
                    vals = valid[col].dropna().tolist()
                    row[col] = round(float(np.mean(vals)), 4)
                    if col != "latency_ms":
                        lo, hi = bootstrap_ci(vals, seed=self.seed)
                        row[f"{col}_ci_lo"] = round(lo, 4)
                        row[f"{col}_ci_hi"] = round(hi, 4)
            rows.append(row)

        summary = pd.DataFrame(rows)
        if self.output_path != os.devnull:
            parent = os.path.dirname(os.path.abspath(self.output_path)) or "."
            fd, temporary = tempfile.mkstemp(
                prefix=f".{os.path.basename(self.output_path)}.", suffix=".tmp", dir=parent
            )
            os.close(fd)
            try:
                summary.to_csv(temporary, index=False)
                os.replace(temporary, self.output_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        logger.info("Evaluation summary saved to %s", self.output_path)
        return summary


    def check_quality_gate(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        valid = _successful_rows(df)
        metrics = {
            "faithfulness": float(valid["faithfulness"].mean()) if "faithfulness" in valid else float("nan"),
            "context_precision": float(valid["context_precision"].mean()) if "context_precision" in valid else float("nan"),
            "answer_relevance": float(valid["answer_relevance"].mean()) if "answer_relevance" in valid else float("nan"),
            "noise_ratio": float(valid["noise_ratio"].mean()) if "noise_ratio" in valid else float("nan"),
        }
        expected = int(df["expected_samples"].max()) if "expected_samples" in df and not df.empty else len(df)
        status = df["sample_status"] if "sample_status" in df else pd.Series(
            "success", index=df.index
        )
        metrics.update({
            "expected_samples": expected,
            "successful_samples": len(valid),
            "failed_samples": int((status == "failed").sum()),
            "skipped_samples": int((status == "skipped").sum()),
        })
        passed = (
            len(valid) == expected
            and expected > 0
            and all(np.isfinite(value) for value in metrics.values() if isinstance(value, float))
            and
            metrics["faithfulness"] >= FAITHFULNESS_THRESHOLD
            and metrics["context_precision"] >= CONTEXT_PRECISION_THRESHOLD
            and metrics["answer_relevance"] >= ANSWER_RELEVANCE_THRESHOLD
            and metrics["noise_ratio"] <= NOISE_RATIO_MAX
        )
        return passed, metrics

# In module3/evaluation.py add:

def compute_adaptive_threshold(query_type: str,
                                architecture: str,
                                chunk_size: int) -> float:
    """
    Novel: PΩ threshold adapts based on query difficulty
    and retrieval context.

    Simple queries: higher threshold (easier to retrieve cleanly)
    Ambiguous queries: lower threshold (harder to get clean retrieval)
    Larger chunks: lower threshold (more noise expected)
    """
    base = 0.4

    # Query type adjustment
    type_adj = {
        "simple": +0.1,       # expect cleaner retrieval
        "paraphrased": 0.0,   # neutral
        "ambiguous": -0.1,    # accept more noise
    }.get(query_type, 0.0)

    # Chunk size adjustment
    size_adj = max(-0.1, min(0.1, (400 - chunk_size) / 4000))

    threshold = base + type_adj + size_adj
    return max(0.2, min(0.7, threshold))

# In module3/evaluation.py add:

def compute_noise_fingerprint(retrieved_text: str,
                               ground_truth: str,
                               query: str) -> Dict[str, float]:
    """
    Novel: Characterise the TYPE of noise, not just the amount.

    Three noise types identified in this paper:
    1. Topical noise: retrieved content is on-topic but not answer-relevant
    2. Peripheral noise: retrieved content is tangentially related
    3. Random noise: retrieved content has no relation to query

    This creates a noise fingerprint per architecture, enabling
    targeted noise reduction strategies.
    """
    rt = set(re.findall(r"\w+", retrieved_text.lower()))
    gt = set(re.findall(r"\w+", ground_truth.lower()))
    qt = set(re.findall(r"\w+", query.lower()))

    # Key names MUST match the extractors used in evaluate_sample
    # ("topical_noise" / "peripheral_noise" / "random_noise"). The earlier
    # {"topical": ..} spelling caused a KeyError there, which silently dropped
    # every sample whose retrieval returned nothing.
    if not rt:
        return {"topical_noise": 0.0, "peripheral_noise": 0.0, "random_noise": 0.0}

    # Tokens in retrieved but NOT in answer
    noise_tokens = rt - gt

    # Topical noise: noise tokens that ARE in query
    # (retrieved query-related content that isn't the answer)
    topical = len(noise_tokens & qt) / len(rt)

    # Peripheral noise: noise tokens related to query terms
    # (one-hop from query terms — approximated by partial match)
    query_stems = {t[:4] for t in qt if len(t) > 4}
    peripheral_tokens = {
        t for t in noise_tokens
        if any(t.startswith(s) for s in query_stems)
    } - qt
    peripheral = len(peripheral_tokens) / len(rt)

    # Random noise: everything else
    random_noise = max(0, (len(noise_tokens) / len(rt))
                       - topical - peripheral)

    return {
        "topical_noise": round(topical, 4),
        "peripheral_noise": round(peripheral, 4),
        "random_noise": round(random_noise, 4),
    }

def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """
    Benjamini-Hochberg FDR correction for multiple comparisons.

    Comparing k architectures against one baseline means k hypothesis tests, so at
    alpha=0.05 roughly one in twenty "findings" is noise. BH controls the false
    discovery rate while retaining more power than Bonferroni, and is the
    correction recommended for multi-system IR comparisons.

    Returns ``(adjusted_p_values, reject_flags)`` in the caller's original order.
    """
    n = len(p_values)
    if n == 0:
        return [], []
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [1.0] * n
    running_min = 1.0
    # Walk from the largest p-value down, enforcing monotonicity.
    for rank_idx in range(n - 1, -1, -1):
        i = order[rank_idx]
        rank = rank_idx + 1
        value = p_values[i] * n / rank
        running_min = min(running_min, value)
        adjusted[i] = min(1.0, running_min)
    return adjusted, [adj < alpha for adj in adjusted]


def test_metric_significance(
    df: pd.DataFrame,
    baseline_arch: str = "Naive RAG",
    metric: str = "precision_omega",
    alpha: float = 0.05,
    min_pairs: int = 5,
) -> pd.DataFrame:
    """
    Paired significance tests of each architecture against a baseline.

    Three corrections over the previous implementation:

    1. **Genuine pairing.** Samples are joined on ``question`` so each pair is the
       same query under two architectures. The old version truncated both series to
       a common length and zipped them positionally, which silently mismatched
       pairs whenever any sample errored out and made the "paired" test invalid.
    2. **Wilcoxon signed-rank** alongside the t-test. Metric distributions here are
       bounded and skewed, so a nonparametric test is the safer primary; the
       t-statistic is still reported for continuity.
    3. **Benjamini-Hochberg correction**, because one baseline versus k
       architectures is a multiple-comparison problem.

    Effect size is Cohen's d_z for paired samples: mean(diff) / sd(diff).
    """
    from scipy import stats

    if "question" not in df.columns or metric not in df.columns:
        logger.warning(
            "Significance test skipped: need 'question' and %r columns.", metric
        )
        return pd.DataFrame()

    base_df = df[df["architecture"] == baseline_arch][["question", metric]].dropna()
    if base_df.empty:
        logger.warning("Baseline %r not found — skipping significance tests.", baseline_arch)
        return pd.DataFrame()
    # Average duplicate questions so the join is one row per question.
    base_map = base_df.groupby("question")[metric].mean()

    rows = []
    for arch in df["architecture"].unique():
        if arch == baseline_arch:
            continue
        arch_df = df[df["architecture"] == arch][["question", metric]].dropna()
        if arch_df.empty:
            continue
        arch_map = arch_df.groupby("question")[metric].mean()

        common = base_map.index.intersection(arch_map.index)
        if len(common) < min_pairs:
            logger.info(
                "Skipping %s: only %d paired questions (min %d).",
                arch, len(common), min_pairs,
            )
            continue

        treat = arch_map.loc[common].to_numpy(dtype=float)
        base = base_map.loc[common].to_numpy(dtype=float)
        diff = treat - base

        # All-zero differences mean the architectures returned identical context —
        # a real finding (the variant is a no-op), not something to test.
        if np.allclose(diff, 0.0):
            rows.append({
                "architecture": arch, "metric": metric, "n_pairs": len(common),
                "mean": round(float(treat.mean()), 6),
                "baseline_mean": round(float(base.mean()), 6),
                "mean_diff": 0.0, "t_statistic": np.nan, "p_value": 1.0,
                "wilcoxon_p": 1.0, "effect_size_dz": 0.0,
                "identical_to_baseline": True,
            })
            continue

        t_stat, t_p = stats.ttest_rel(treat, base)
        try:
            _, w_p = stats.wilcoxon(treat, base)
        except Exception:  # noqa: BLE001 — e.g. too few non-zero differences
            w_p = float("nan")

        sd = float(np.std(diff, ddof=1))
        rows.append({
            "architecture": arch,
            "metric": metric,
            "n_pairs": int(len(common)),
            "mean": round(float(treat.mean()), 6),
            "baseline_mean": round(float(base.mean()), 6),
            "mean_diff": round(float(diff.mean()), 6),
            "t_statistic": round(float(t_stat), 4),
            # Keep full precision through FDR correction and serialization. Tiny
            # but nonzero p-values must not be rounded to zero before BH.
            "p_value": float(t_p),
            "wilcoxon_p": float(w_p) if w_p == w_p else np.nan,
            "effect_size_dz": round(float(diff.mean() / sd), 4) if sd > 0 else 0.0,
            "identical_to_baseline": False,
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    # Correct across the family of comparisons, preferring Wilcoxon where available.
    primary = [
        r["wilcoxon_p"] if r["wilcoxon_p"] == r["wilcoxon_p"] else r["p_value"]
        for r in rows
    ]
    adjusted, reject = benjamini_hochberg(primary, alpha=alpha)
    out["p_value_bh"] = adjusted
    out[f"significant_p{str(alpha).split('.')[-1]}_bh"] = reject
    return out.sort_values("p_value_bh").reset_index(drop=True)


def test_pomega_significance(
    df: pd.DataFrame,
    baseline_arch: str = "Naive RAG",
) -> pd.DataFrame:
    """
    Backward-compatible wrapper: paired significance tests on PΩ.

    Delegates to :func:`test_metric_significance`, which fixes the pairing and adds
    multiple-comparison correction.
    """
    return test_metric_significance(
        df, baseline_arch=baseline_arch, metric="precision_omega"
    )

if __name__ == "__main__":
    print("Module 3 loaded. Use EvaluationPipeline and RAGASEvaluator.")
