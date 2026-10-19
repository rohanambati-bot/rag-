#!/usr/bin/env python3
"""Controlled, offline retrieval ablation harness.

This module deliberately does not call an LLM, mutate the project's vector store,
or write ``outputs/eval_results.csv``.  It gives the five retrieval conditions a
small common interface so that the only intended differences are the first-stage
retriever, the candidate-pool width, and the optional cross-encoder second stage::

    dense-only
    dense candidate pool + cross-encoder reranking
    BM25-only
    hybrid without reranking
    hybrid with cross-encoder reranking

The harness is usable in two ways:

* Library use: construct :class:`DatasetContract`, provide retrievers and a local
  reranker (or test doubles), then call :meth:`AblationHarness.run`.
* CLI use: pass a local JSON/JSONL manifest containing the corpus, query/qrels
  contract, and precomputed dense/BM25 rankings.  A local cross-encoder directory
  or deterministic per-query reranker scores are required for reranked conditions.

The CLI accepts only local files.  The local cross-encoder adapter requires a
directory that already exists and never asks Hugging Face or another service to
download weights.  The module imports the project's existing IR metric functions
when the project dependencies are installed; a mathematically equivalent stdlib
fallback keeps the harness unit-testable in a minimal analysis environment.

Manifest sketch (all paths are relative to the manifest unless absolute)::

    {
      "corpus": [{"doc_id": "d1", "text": "..."}],
      "queries": [{"query_id": "q1", "text": "..."}],
      "qrels": {"q1": {"d1": 1}},
      "rankings": {
        "dense": {"q1": ["d1", "d2"]},
        "bm25": {"q1": ["d2", "d1"]}
      },
      "top_k": 5,
      "candidate_pool_k": 50,
      "corpus_revision": "local-corpus-v1",
      "qrels_revision": "local-qrels-v1",
      "filter": {"name": "all-corpus", "revision": "v1"},
      "model_revisions": {
        "dense": "embedding-revision",
        "bm25": "rank-bm25-0.2.2",
        "reranker": "cross-encoder-revision"
      },
      "reranker_scores": {"q1": {"d1": 0.9, "d2": 0.1}}
    }

The output directory is create-only.  If it already exists, the CLI fails rather
than replacing any historical result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


# Running ``python scripts/retrieval_ablation.py`` puts ``scripts/`` rather than
# the project root on sys.path.  Resolve it from this file instead of using an
# absolute machine-specific path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


HARNESS_VERSION = "1.0.0"
DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_POOL_K = 50
DEFAULT_RRF_K = 60
METRIC_NAMES = (
    "recall_at_k",
    "precision_at_k",
    "hit_rate",
    "mrr_at_10",
    "ndcg_at_10",
)
PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "rank-bm25",
    "sentence-transformers",
    "torch",
    "transformers",
    "langchain-core",
    "langchain-chroma",
    "chromadb",
)


class AblationError(ValueError):
    """Raised when the controlled-ablation contract is invalid."""


def _canonical_json(value: Any) -> str:
    """Serialize metadata deterministically for fingerprints and manifests."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert common numeric/container values to JSON-safe stdlib values."""

    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - unusual third-party scalar
            pass
    return value


@dataclass(frozen=True)
class CorpusDocument:
    """One immutable corpus record used by every condition."""

    doc_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.doc_id).strip():
            raise AblationError("corpus document IDs must be non-empty")
        if not isinstance(self.text, str):
            raise AblationError("corpus document text must be a string")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": _json_safe(dict(self.metadata)),
        }


@dataclass(frozen=True)
class QueryRecord:
    """A stable query identity and its text."""

    query_id: str
    text: str

    def __post_init__(self) -> None:
        if not str(self.query_id).strip():
            raise AblationError("query IDs must be non-empty")
        if not isinstance(self.text, str) or not self.text.strip():
            raise AblationError("query text must be a non-empty string")

    def as_dict(self) -> Dict[str, str]:
        return {"query_id": self.query_id, "text": self.text}


@dataclass(frozen=True)
class FilterPolicy:
    """The one filtering policy shared by all five conditions.

    ``allowed_doc_ids`` and ``allowed_query_ids`` are optional explicit sets.  A
    policy with both set to ``None`` means the complete supplied contract.  The
    name/revision are part of the run metadata so a result cannot be confused with
    a run made using a different filtering rule.
    """

    name: str = "all-corpus"
    revision: str = "v1"
    allowed_doc_ids: Optional[frozenset] = None
    allowed_query_ids: Optional[frozenset] = None

    def allows_doc(self, doc_id: str) -> bool:
        return self.allowed_doc_ids is None or doc_id in self.allowed_doc_ids

    def allows_query(self, query_id: str) -> bool:
        return self.allowed_query_ids is None or query_id in self.allowed_query_ids

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "revision": self.revision,
            "allowed_doc_ids": (
                sorted(self.allowed_doc_ids) if self.allowed_doc_ids is not None else None
            ),
            "allowed_query_ids": (
                sorted(self.allowed_query_ids)
                if self.allowed_query_ids is not None
                else None
            ),
        }


@dataclass(frozen=True)
class DatasetContract:
    """Shared corpus/query/qrels contract for one controlled comparison."""

    corpus: Tuple[CorpusDocument, ...]
    queries: Tuple[QueryRecord, ...]
    qrels: Mapping[str, Mapping[str, float]]
    corpus_revision: str = "unspecified"
    qrels_revision: str = "unspecified"
    filter_policy: FilterPolicy = field(default_factory=FilterPolicy)

    def __post_init__(self) -> None:
        corpus = tuple(self.corpus)
        queries = tuple(q for q in self.queries if self.filter_policy.allows_query(q.query_id))
        if not corpus:
            raise AblationError("the corpus must not be empty")
        if not queries:
            raise AblationError("the shared query set must not be empty")

        corpus_ids = [doc.doc_id for doc in corpus]
        if len(corpus_ids) != len(set(corpus_ids)):
            raise AblationError("the corpus contains duplicate doc_id values")
        query_ids = [query.query_id for query in queries]
        if len(query_ids) != len(set(query_ids)):
            raise AblationError("the query set contains duplicate query_id values")

        raw_qrels: Mapping[str, Mapping[str, float]] = self.qrels
        normalized_qrels: Dict[str, Dict[str, float]] = {}
        corpus_id_set = set(corpus_ids)
        if self.filter_policy.allowed_doc_ids is not None:
            unknown_filtered_docs = set(self.filter_policy.allowed_doc_ids) - corpus_id_set
            if unknown_filtered_docs:
                raise AblationError(
                    "filter allowed_doc_ids reference unknown corpus IDs: "
                    + ", ".join(sorted(unknown_filtered_docs))
                )
        for query_id in query_ids:
            entries = raw_qrels.get(query_id, {})
            if not isinstance(entries, Mapping):
                raise AblationError("qrels values must map doc IDs to relevance labels")
            normalized: Dict[str, float] = {}
            for doc_id, grade in entries.items():
                doc_id = str(doc_id)
                if doc_id not in corpus_id_set:
                    raise AblationError(
                        f"qrels for {query_id!r} reference unknown corpus doc {doc_id!r}"
                    )
                try:
                    normalized[doc_id] = float(grade)
                except (TypeError, ValueError) as exc:
                    raise AblationError(
                        f"qrels relevance for {query_id!r}/{doc_id!r} is not numeric"
                    ) from exc
            normalized_qrels[query_id] = dict(sorted(normalized.items()))

        object.__setattr__(self, "corpus", corpus)
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "qrels", normalized_qrels)

    @property
    def query_ids(self) -> Tuple[str, ...]:
        return tuple(query.query_id for query in self.queries)

    @property
    def corpus_by_id(self) -> Dict[str, CorpusDocument]:
        return {doc.doc_id: doc for doc in self.corpus}

    def relevant_ids(self, query_id: str) -> List[str]:
        return [doc_id for doc_id, grade in self.qrels.get(query_id, {}).items() if grade > 0]

    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "corpus": [doc.as_dict() for doc in self.corpus],
                "queries": [query.as_dict() for query in self.queries],
                "qrels": {
                    query_id: dict(self.qrels.get(query_id, {}))
                    for query_id in self.query_ids
                },
                "corpus_revision": self.corpus_revision,
                "qrels_revision": self.qrels_revision,
                "filter": self.filter_policy.as_dict(),
            }
        )


@dataclass(frozen=True)
class RankedDocument:
    """A retriever or reranker output item."""

    doc_id: str
    score: Optional[float] = None


@dataclass(frozen=True)
class Candidate:
    """A filtered candidate with provenance from the first retrieval stage."""

    doc_id: str
    text: str
    source: str
    source_rank: int
    retrieval_score: Optional[float] = None
    fusion_score: Optional[float] = None
    sources: Tuple[str, ...] = ()
    rerank_score: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "source_rank": self.source_rank,
            "retrieval_score": self.retrieval_score,
            "fusion_score": self.fusion_score,
            "sources": list(self.sources),
            "rerank_score": self.rerank_score,
        }


@dataclass(frozen=True)
class ConditionSpec:
    """One of the five predeclared controlled conditions."""

    name: str
    label: str
    sources: Tuple[str, ...]
    rerank: bool
    candidate_pool: str
    description: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "sources": list(self.sources),
            "rerank": self.rerank,
            "candidate_pool": self.candidate_pool,
            "description": self.description,
        }


def define_conditions() -> Tuple[ConditionSpec, ...]:
    """Return the immutable, ordered five-condition ablation definition."""

    return (
        ConditionSpec(
            name="dense_only",
            label="dense-only",
            sources=("dense",),
            rerank=False,
            candidate_pool="dense top-k",
            description="Dense retriever returns top-k directly; no second stage.",
        ),
        ConditionSpec(
            name="dense_reranked",
            label="dense candidate pool + cross-encoder reranking",
            sources=("dense",),
            rerank=True,
            candidate_pool="dense[:candidate_pool_k]",
            description=(
                "Dense candidate pool is filtered once, then every candidate is "
                "scored by the same local cross-encoder before top-k truncation."
            ),
        ),
        ConditionSpec(
            name="bm25_only",
            label="BM25-only",
            sources=("bm25",),
            rerank=False,
            candidate_pool="BM25 top-k",
            description="BM25 retriever returns top-k directly; no second stage.",
        ),
        ConditionSpec(
            name="hybrid_no_rerank",
            label="hybrid without reranking",
            sources=("dense", "bm25"),
            rerank=False,
            candidate_pool=(
                "dedup(dense[:candidate_pool_k] U BM25[:candidate_pool_k]) "
                "+ deterministic RRF"
            ),
            description=(
                "Equal-width dense and BM25 pools are filtered, deduplicated, "
                "fused with RRF, and truncated to top-k."
            ),
        ),
        ConditionSpec(
            name="hybrid_reranked",
            label="hybrid + cross-encoder reranking",
            sources=("dense", "bm25"),
            rerank=True,
            candidate_pool="dedup(dense[:candidate_pool_k] U BM25[:candidate_pool_k])",
            description=(
                "The same hybrid union is passed in full to the local cross-encoder; "
                "only then is it truncated to top-k."
            ),
        ),
    )


CONDITION_SPECS = define_conditions()
CONDITION_BY_NAME = {spec.name: spec for spec in CONDITION_SPECS}


def _coerce_ranked_item(item: Any) -> RankedDocument:
    if isinstance(item, RankedDocument):
        return item
    if isinstance(item, Candidate):
        score = item.rerank_score if item.rerank_score is not None else item.fusion_score
        return RankedDocument(item.doc_id, score)
    if isinstance(item, str):
        return RankedDocument(item)
    if isinstance(item, Mapping):
        doc_id = item.get("doc_id", item.get("id"))
        if doc_id is None:
            raise AblationError(f"ranked item has no doc_id/id: {item!r}")
        score = item.get("score")
        return RankedDocument(str(doc_id), float(score) if score is not None else None)
    raise AblationError(f"unsupported ranked item: {item!r}")


def _coerce_ranked_items(items: Iterable[Any]) -> List[RankedDocument]:
    return [_coerce_ranked_item(item) for item in items]


class ReplayRetriever:
    """Replay precomputed ranked IDs without touching a model or the network."""

    def __init__(self, rankings: Mapping[str, Sequence[Any]], model_revision: str):
        self.rankings = {str(query_id): tuple(items) for query_id, items in rankings.items()}
        self.model_revision = model_revision

    def retrieve(self, query_id: str, query_text: str, k: int) -> List[RankedDocument]:
        del query_text
        if query_id not in self.rankings:
            raise AblationError(f"no replay ranking for query {query_id!r}")
        return _coerce_ranked_items(self.rankings[query_id][:k])


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


class LocalBM25Retriever:
    """Small deterministic BM25 implementation over the shared local corpus.

    It intentionally uses no dataset loader and no network.  The project also
    depends on ``rank-bm25``; this implementation keeps the harness runnable in a
    minimal offline test environment and makes the exact scoring contract explicit.
    """

    def __init__(
        self,
        corpus: Sequence[CorpusDocument],
        model_revision: str = "bm25-okapi-local-v1",
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if k1 <= 0 or not 0 <= b <= 1:
            raise AblationError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.corpus = tuple(corpus)
        self.model_revision = model_revision
        self.k1 = float(k1)
        self.b = float(b)
        self._tokens = [_tokenize(doc.text) for doc in self.corpus]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._avgdl = fmean(self._lengths) if self._lengths else 0.0
        document_frequency: Dict[str, int] = {}
        for tokens in self._tokens:
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        self._idf = {
            token: math.log(1.0 + (len(self.corpus) - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def retrieve(self, query_id: str, query_text: str, k: int) -> List[RankedDocument]:
        del query_id
        if k <= 0:
            return []
        query_tokens = _tokenize(query_text)
        scored: List[Tuple[float, int]] = []
        for index, tokens in enumerate(self._tokens):
            term_frequency: Dict[str, int] = {}
            for token in tokens:
                term_frequency[token] = term_frequency.get(token, 0) + 1
            length = self._lengths[index]
            denominator_norm = self.k1 * (
                1.0 - self.b + self.b * (length / self._avgdl if self._avgdl else 0.0)
            )
            score = 0.0
            for token in query_tokens:
                frequency = term_frequency.get(token, 0)
                if not frequency:
                    continue
                score += self._idf.get(token, 0.0) * (
                    frequency * (self.k1 + 1.0) / (frequency + denominator_norm)
                )
            scored.append((score, index))
        # Corpus order is the deterministic tie-break, matching the shared corpus
        # contract rather than relying on a heap/dict implementation detail.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [
            RankedDocument(self.corpus[index].doc_id, score)
            for score, index in scored[:k]
        ]


class DenseVectorStoreRetriever:
    """Adapter for the project's existing ``vectorstore.similarity_search`` API."""

    def __init__(
        self,
        vectorstore: Any,
        model_revision: str,
        document_id: Optional[Callable[[Any], str]] = None,
    ):
        self.vectorstore = vectorstore
        self.model_revision = model_revision
        if document_id is None:
            try:
                from module3.evaluation import doc_id_for
            except ImportError as exc:  # pragma: no cover - only minimal envs
                raise AblationError(
                    "document_id callback is required when project metrics are unavailable"
                ) from exc
            document_id = doc_id_for
        self.document_id = document_id

    def retrieve(self, query_id: str, query_text: str, k: int) -> List[RankedDocument]:
        del query_id
        docs = self.vectorstore.similarity_search(query_text, k=k)
        return [RankedDocument(str(self.document_id(doc))) for doc in docs]


class MappingScoreReranker:
    """Offline reranker driven by deterministic per-query scores from a manifest."""

    def __init__(self, scores: Mapping[str, Mapping[str, float]], model_revision: str):
        self.scores = {
            str(query_id): {str(doc_id): float(score) for doc_id, score in values.items()}
            for query_id, values in scores.items()
        }
        self.model_revision = model_revision
        self.calls: List[Dict[str, Any]] = []

    def rerank(
        self,
        query_id: str,
        query_text: str,
        candidates: Sequence[Candidate],
        top_k: int,
    ) -> List[Candidate]:
        del query_text
        query_scores = self.scores.get(query_id, {})
        self.calls.append(
            {"query_id": query_id, "candidate_ids": [candidate.doc_id for candidate in candidates]}
        )
        scored = [
            (query_scores.get(candidate.doc_id, float("-inf")), index, candidate)
            for index, candidate in enumerate(candidates)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            replace(candidate, rerank_score=score)
            for score, _index, candidate in scored[:top_k]
        ]


class LocalCrossEncoderReranker:
    """Local-only sentence-transformers cross-encoder adapter.

    ``model_path`` must already exist as a directory.  Passing a model name that
    would require a download is rejected before importing/loading the model.  This
    is a deliberate offline guard for reproducible evaluation.
    """

    def __init__(self, model_path: str, model_revision: str, device: str = "cpu"):
        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise AblationError(
                f"local cross-encoder path does not exist or is not a directory: {path}"
            )
        self.model_path = path
        self.model_revision = model_revision
        self.device = device
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - dependency is project-pinned
            raise AblationError("sentence-transformers is required for local reranking") from exc

        # Pass the offline guard through CrossEncoder's own argument.  Supplying
        # ``local_files_only`` inside ``automodel_args`` duplicates the keyword
        # that sentence-transformers forwards to AutoModel and raises a TypeError
        # on the pinned sentence-transformers 3.x stack.
        previous_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            self._model = CrossEncoder(
                str(self.model_path),
                device=self.device,
                local_files_only=True,
            )
        finally:
            if previous_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_offline
        return self._model

    def rerank(
        self,
        query_id: str,
        query_text: str,
        candidates: Sequence[Candidate],
        top_k: int,
    ) -> List[Candidate]:
        del query_id
        if not candidates:
            return []
        model = self._load()
        pairs = [(query_text, candidate.text) for candidate in candidates]
        scores = list(model.predict(pairs, show_progress_bar=False))
        if len(scores) != len(candidates):
            raise AblationError(
                "local cross-encoder returned a score count different from the candidate pool"
            )
        ranked = list(enumerate(zip(candidates, scores)))
        ranked.sort(key=lambda item: (-float(item[1][1]), item[0]))
        return [
            replace(candidate, rerank_score=float(score))
            for _index, (candidate, score) in ranked[:top_k]
        ]


def _fallback_metrics() -> Any:
    """Return project-compatible metric functions without third-party imports."""

    class Metrics:
        @staticmethod
        def compute_recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
            relevant_set = set(relevant)
            if not relevant_set or k <= 0:
                return 0.0
            return len(set(retrieved[:k]) & relevant_set) / len(relevant_set)

        @staticmethod
        def compute_precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
            if k <= 0:
                return 0.0
            relevant_set = set(relevant)
            return sum(doc_id in relevant_set for doc_id in retrieved[:k]) / k

        @staticmethod
        def compute_hit_rate(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
            return float(any(doc_id in set(relevant) for doc_id in retrieved))

        @staticmethod
        def compute_mrr(retrieved: Sequence[str], relevant: Sequence[str], k: Optional[int] = None) -> float:
            relevant_set = set(relevant)
            ranked = retrieved[:k] if k else retrieved
            for rank, doc_id in enumerate(ranked, start=1):
                if doc_id in relevant_set:
                    return 1.0 / rank
            return 0.0

        @staticmethod
        def compute_ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int = 10) -> float:
            relevant_set = set(relevant)
            if not relevant_set or k <= 0:
                return 0.0
            dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank, doc_id in enumerate(retrieved[:k], start=1)
                if doc_id in relevant_set
            )
            ideal_hits = min(len(relevant_set), k)
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            return dcg / idcg if idcg else 0.0

    return Metrics


_METRICS_API: Any = None


def _project_metrics() -> Any:
    """Load the project's metric functions once, with an offline stdlib fallback."""

    global _METRICS_API
    if _METRICS_API is not None:
        return _METRICS_API
    try:
        from module3 import evaluation

        _METRICS_API = evaluation
    except (ImportError, ModuleNotFoundError):
        _METRICS_API = _fallback_metrics()
    return _METRICS_API


def compute_ir_metrics(
    ranked_ids: Sequence[str],
    relevant_ids: Sequence[str],
    top_k: int,
) -> Dict[str, float]:
    """Compute the project's existing rank-aware metrics for one query."""

    metrics = _project_metrics()
    ranked = list(ranked_ids[:top_k])
    relevant = list(relevant_ids)
    return {
        "recall_at_k": float(metrics.compute_recall_at_k(ranked, relevant, k=top_k)),
        "precision_at_k": float(metrics.compute_precision_at_k(ranked, relevant, k=top_k)),
        "hit_rate": float(metrics.compute_hit_rate(ranked, relevant)),
        "mrr_at_10": float(metrics.compute_mrr(ranked, relevant, k=10)),
        "ndcg_at_10": float(metrics.compute_ndcg_at_k(ranked, relevant, k=10)),
    }


def collect_package_metadata(lockfile_path: Optional[Path] = None) -> Dict[str, Any]:
    """Capture package/runtime metadata once for all five conditions."""

    versions: Dict[str, Optional[str]] = {}
    for package_name in PACKAGE_NAMES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    metadata: Dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": versions,
    }
    if lockfile_path is None:
        lockfile_path = PROJECT_ROOT / "requirements.lock"
    if lockfile_path.is_file():
        metadata["requirements_lock"] = {
            "path": str(lockfile_path.relative_to(PROJECT_ROOT))
            if lockfile_path.is_relative_to(PROJECT_ROOT)
            else lockfile_path.name,
            "sha256": hashlib.sha256(lockfile_path.read_bytes()).hexdigest(),
        }
    return metadata


def _revision(obj: Any, default: str = "unspecified") -> str:
    value = getattr(obj, "model_revision", None)
    return str(value) if value not in (None, "") else default


class AblationHarness:
    """Run the five conditions against one immutable shared contract."""

    def __init__(
        self,
        contract: DatasetContract,
        dense_retriever: Any,
        bm25_retriever: Any,
        reranker: Optional[Any] = None,
        *,
        top_k: int = DEFAULT_TOP_K,
        candidate_pool_k: int = DEFAULT_CANDIDATE_POOL_K,
        rrf_k: int = DEFAULT_RRF_K,
        conditions: Optional[Sequence[str]] = None,
        model_revisions: Optional[Mapping[str, str]] = None,
        package_metadata: Optional[Mapping[str, Any]] = None,
    ):
        if top_k <= 0:
            raise AblationError("top_k must be positive")
        if candidate_pool_k < top_k:
            raise AblationError("candidate_pool_k must be at least top_k")
        if rrf_k <= 0:
            raise AblationError("rrf_k must be positive")
        self.contract = contract
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.top_k = int(top_k)
        self.candidate_pool_k = int(candidate_pool_k)
        self.rrf_k = int(rrf_k)

        requested = tuple(conditions) if conditions is not None else tuple(CONDITION_BY_NAME)
        unknown = [name for name in requested if name not in CONDITION_BY_NAME]
        if unknown:
            raise AblationError(f"unknown ablation conditions: {', '.join(unknown)}")
        if len(set(requested)) != len(requested):
            raise AblationError("ablation condition names must be unique")
        self.conditions = tuple(CONDITION_BY_NAME[name] for name in requested)
        if any(spec.rerank for spec in self.conditions) and reranker is None:
            raise AblationError("reranked conditions require one shared local reranker")

        revisions = {
            "dense": _revision(dense_retriever),
            "bm25": _revision(bm25_retriever),
            "reranker": _revision(reranker) if reranker is not None else None,
        }
        if model_revisions:
            revisions.update({str(key): str(value) for key, value in model_revisions.items()})
        self.model_revisions = revisions
        self.package_metadata = dict(
            collect_package_metadata() if package_metadata is None else package_metadata
        )

    def metadata(self) -> Dict[str, Any]:
        contract_fingerprint = self.contract.fingerprint()
        model_fingerprint = _fingerprint(self.model_revisions)
        package_fingerprint = _fingerprint(self.package_metadata)
        qrels_fingerprint = _fingerprint(
            {
                query_id: self.contract.qrels.get(query_id, {})
                for query_id in self.contract.query_ids
            }
        )
        return {
            "harness": {
                "name": "controlled-retrieval-ablation",
                "version": HARNESS_VERSION,
                "offline_only": True,
                "provider_calls": 0,
            },
            "conditions": [spec.as_dict() for spec in self.conditions],
            "top_k": self.top_k,
            "candidate_pool_k": self.candidate_pool_k,
            "rrf_k": self.rrf_k,
            "corpus_revision": self.contract.corpus_revision,
            "qrels_revision": self.contract.qrels_revision,
            "filter": self.contract.filter_policy.as_dict(),
            "query_count": len(self.contract.queries),
            "query_ids": list(self.contract.query_ids),
            "query_ids_fingerprint": _fingerprint(list(self.contract.query_ids)),
            "corpus_size": len(self.contract.corpus),
            "corpus_fingerprint": _fingerprint([doc.as_dict() for doc in self.contract.corpus]),
            "qrels_fingerprint": qrels_fingerprint,
            "contract_fingerprint": contract_fingerprint,
            "model_revisions": self.model_revisions,
            "model_revisions_fingerprint": model_fingerprint,
            "package_metadata": _json_safe(self.package_metadata),
            "package_metadata_fingerprint": package_fingerprint,
            "metrics": {
                "module": "module3.evaluation when available; stdlib-compatible fallback otherwise",
                "functions": list(METRIC_NAMES),
            },
        }

    def _source_candidates(
        self,
        retriever: Any,
        query: QueryRecord,
        source: str,
        requested_k: int,
    ) -> List[Candidate]:
        raw = retriever.retrieve(query.query_id, query.text, requested_k)
        ranked = _coerce_ranked_items(raw)
        corpus_by_id = self.contract.corpus_by_id
        candidates: List[Candidate] = []
        seen: Set[str] = set()
        for rank, item in enumerate(ranked, start=1):
            if item.doc_id in seen:
                continue
            if item.doc_id not in corpus_by_id:
                raise AblationError(
                    f"{source} retriever returned doc {item.doc_id!r} outside the shared corpus"
                )
            if not self.contract.filter_policy.allows_doc(item.doc_id):
                continue
            seen.add(item.doc_id)
            candidates.append(
                Candidate(
                    doc_id=item.doc_id,
                    text=corpus_by_id[item.doc_id].text,
                    source=source,
                    source_rank=rank,
                    retrieval_score=item.score,
                    sources=(source,),
                )
            )
        return candidates

    def _hybrid_candidates(self, query: QueryRecord) -> List[Candidate]:
        dense = self._source_candidates(
            self.dense_retriever, query, "dense", self.candidate_pool_k
        )
        bm25 = self._source_candidates(
            self.bm25_retriever, query, "bm25", self.candidate_pool_k
        )
        scores: Dict[str, float] = {}
        first_seen: Dict[str, int] = {}
        candidates: Dict[str, Candidate] = {}
        source_sets: Dict[str, Set[str]] = {}
        encounter = 0
        for source_candidates in (dense, bm25):
            for rank, candidate in enumerate(source_candidates, start=1):
                encounter += 1
                doc_id = candidate.doc_id
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
                first_seen.setdefault(doc_id, encounter)
                candidates.setdefault(doc_id, candidate)
                source_sets.setdefault(doc_id, set()).add(candidate.source)
        ordered_ids = sorted(scores, key=lambda doc_id: (-scores[doc_id], first_seen[doc_id]))
        return [
            replace(
                candidates[doc_id],
                source="hybrid_rrf",
                source_rank=rank,
                fusion_score=scores[doc_id],
                sources=tuple(sorted(source_sets[doc_id])),
            )
            for rank, doc_id in enumerate(ordered_ids, start=1)
        ]

    def _rerank(
        self,
        query: QueryRecord,
        candidates: Sequence[Candidate],
    ) -> List[Candidate]:
        if self.reranker is None:  # defensive; constructor already validates this
            raise AblationError("reranking requested without a reranker")
        raw = self.reranker.rerank(query.query_id, query.text, candidates, self.top_k)
        candidate_by_id = {candidate.doc_id: candidate for candidate in candidates}
        output: List[Candidate] = []
        seen: Set[str] = set()
        for item in raw:
            if isinstance(item, Candidate):
                doc_id = item.doc_id
                rerank_score = item.rerank_score
            else:
                ranked = _coerce_ranked_item(item)
                doc_id = ranked.doc_id
                rerank_score = ranked.score
            if doc_id not in candidate_by_id:
                raise AblationError(
                    f"reranker returned doc {doc_id!r} that was not in its candidate pool"
                )
            if doc_id in seen:
                raise AblationError(f"reranker returned duplicate doc {doc_id!r}")
            seen.add(doc_id)
            output.append(replace(candidate_by_id[doc_id], rerank_score=rerank_score))
            if len(output) >= self.top_k:
                break
        return output

    def _retrieve_condition(
        self,
        spec: ConditionSpec,
        query: QueryRecord,
    ) -> Tuple[List[Candidate], List[Candidate], Dict[str, Any]]:
        if spec.name == "dense_only":
            pool = self._source_candidates(self.dense_retriever, query, "dense", self.top_k)
        elif spec.name == "dense_reranked":
            pool = self._source_candidates(
                self.dense_retriever, query, "dense", self.candidate_pool_k
            )
        elif spec.name == "bm25_only":
            pool = self._source_candidates(self.bm25_retriever, query, "bm25", self.top_k)
        elif spec.name in {"hybrid_no_rerank", "hybrid_reranked"}:
            pool = self._hybrid_candidates(query)
        else:  # pragma: no cover - protected by condition validation
            raise AblationError(f"unsupported condition: {spec.name}")

        ranked = self._rerank(query, pool) if spec.rerank else pool[: self.top_k]
        stage = {
            "candidate_pool_size": len(pool),
            "candidate_pool_ids": [candidate.doc_id for candidate in pool],
            "candidate_pool_sources": sorted(
                {source for candidate in pool for source in (candidate.sources or (candidate.source,))}
            ),
            "reranker_applied": spec.rerank,
            "reranker_candidate_count": len(pool) if spec.rerank else 0,
        }
        return ranked, pool, stage

    def run(self) -> "AblationRun":
        """Execute every selected condition over the exact same query sequence."""

        run_metadata = self.metadata()
        rows: List[Dict[str, Any]] = []
        for spec in self.conditions:
            for query in self.contract.queries:
                ranked, _pool, stage = self._retrieve_condition(spec, query)
                ranked_ids = [candidate.doc_id for candidate in ranked[: self.top_k]]
                relevant_ids = self.contract.relevant_ids(query.query_id)
                metrics = compute_ir_metrics(ranked_ids, relevant_ids, self.top_k)
                row: Dict[str, Any] = {
                    "condition": spec.name,
                    "condition_label": spec.label,
                    "candidate_pool_definition": spec.candidate_pool,
                    "query_id": query.query_id,
                    "query": query.text,
                    "top_k": self.top_k,
                    "candidate_pool_k": self.candidate_pool_k,
                    "ranked_ids": ranked_ids,
                    "relevant_ids": relevant_ids,
                    "candidate_pool_ids": stage["candidate_pool_ids"],
                    "candidate_pool_size": stage["candidate_pool_size"],
                    "candidate_pool_sources": stage["candidate_pool_sources"],
                    "reranker_applied": stage["reranker_applied"],
                    "reranker_candidate_count": stage["reranker_candidate_count"],
                    "reranker_model_revision": self.model_revisions.get("reranker"),
                    "reranker_scores": {
                        candidate.doc_id: candidate.rerank_score
                        for candidate in ranked
                        if candidate.rerank_score is not None
                    },
                    "corpus_revision": self.contract.corpus_revision,
                    "qrels_revision": self.contract.qrels_revision,
                    "corpus_fingerprint": run_metadata["corpus_fingerprint"],
                    "qrels_fingerprint": run_metadata["qrels_fingerprint"],
                    "model_revisions_fingerprint": run_metadata["model_revisions_fingerprint"],
                    "package_metadata_fingerprint": run_metadata["package_metadata_fingerprint"],
                    "filter_fingerprint": _fingerprint(self.contract.filter_policy.as_dict()),
                    "contract_fingerprint": run_metadata["contract_fingerprint"],
                }
                row.update(metrics)
                rows.append(row)

        result = AblationRun(metadata=run_metadata, per_query=rows)
        result.validate()
        return result


@dataclass
class AblationRun:
    """Per-query rows plus deterministic summary and shared metadata."""

    metadata: Dict[str, Any]
    per_query: List[Dict[str, Any]]

    @property
    def summary(self) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for spec in self.metadata["conditions"]:
            name = spec["name"]
            rows = [row for row in self.per_query if row["condition"] == name]
            summary: Dict[str, Any] = {
                "condition": name,
                "condition_label": spec["label"],
                "candidate_pool_definition": spec["candidate_pool"],
                "n_queries": len(rows),
                "top_k": self.metadata["top_k"],
                "candidate_pool_k": self.metadata["candidate_pool_k"],
                "contract_fingerprint": self.metadata["contract_fingerprint"],
                "model_revisions_fingerprint": self.metadata["model_revisions_fingerprint"],
                "package_metadata_fingerprint": self.metadata["package_metadata_fingerprint"],
            }
            for metric in METRIC_NAMES:
                values = [float(row[metric]) for row in rows]
                summary[f"mean_{metric}"] = round(fmean(values), 8) if values else None
            summaries.append(summary)
        return summaries

    def validate(self) -> None:
        """Assert the controlled-comparison invariants before results are written."""

        expected_conditions = [spec["name"] for spec in self.metadata["conditions"]]
        expected_query_ids = list(self.metadata["query_ids"])
        expected_rows = len(expected_conditions) * len(expected_query_ids)
        if len(self.per_query) != expected_rows:
            raise AblationError(
                f"expected {expected_rows} per-query rows, got {len(self.per_query)}"
            )
        for condition in expected_conditions:
            rows = [row for row in self.per_query if row["condition"] == condition]
            actual_ids = [row["query_id"] for row in rows]
            if actual_ids != expected_query_ids:
                raise AblationError(
                    f"condition {condition!r} does not use the identical ordered query IDs"
                )
            if any(row["top_k"] != self.metadata["top_k"] for row in rows):
                raise AblationError(f"condition {condition!r} changed top_k")
            fingerprints = {
                (
                    row["contract_fingerprint"],
                    row["filter_fingerprint"],
                    row["model_revisions_fingerprint"],
                    row["package_metadata_fingerprint"],
                )
                for row in rows
            }
            if len(fingerprints) != 1:
                raise AblationError(f"condition {condition!r} changed shared metadata")
            for row in rows:
                if len(row["ranked_ids"]) > self.metadata["top_k"]:
                    raise AblationError(f"condition {condition!r} returned more than top_k")
                if len(row["ranked_ids"]) != len(set(row["ranked_ids"])):
                    raise AblationError(f"condition {condition!r} returned duplicate IDs")
                if not set(row["ranked_ids"]).issubset(set(row["candidate_pool_ids"])):
                    raise AblationError(
                        f"condition {condition!r} returned a document outside its candidate pool"
                    )

    def write_outputs(self, output_dir: Path) -> None:
        """Write create-only JSON/JSONL/CSV outputs to a new versioned directory."""

        output_dir = Path(output_dir)
        if output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite existing output directory: {output_dir}"
            )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir()

        _write_new_json(output_dir / "run_metadata.json", self.metadata)
        _write_new_jsonl(output_dir / "per_query.jsonl", self.per_query)
        _write_new_csv(output_dir / "condition_summary.csv", self.summary)


def _write_new_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_new_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(_json_safe(row)) + "\n")


def _write_new_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in columns})


def _resolve_source(source: Any, base_dir: Path) -> Any:
    if not isinstance(source, str):
        return source
    path = Path(source)
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        raise AblationError(f"manifest input file does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AblationError(f"invalid JSONL at {path}:{line_number}") from exc
        return rows
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AblationError(f"invalid JSON manifest input: {path}") from exc


def _records_payload(value: Any, key: str) -> Any:
    if isinstance(value, Mapping) and key in value:
        return value[key]
    if isinstance(value, Mapping) and "records" in value:
        return value["records"]
    return value


def _load_corpus(value: Any, base_dir: Path) -> Tuple[CorpusDocument, ...]:
    payload = _records_payload(_resolve_source(value, base_dir), "corpus")
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise AblationError("manifest corpus must be a list or JSONL file")
    documents = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise AblationError(f"corpus row {index} must be an object")
        doc_id = row.get("doc_id", row.get("id"))
        text = row.get("text", row.get("page_content", row.get("content")))
        if doc_id is None or text is None:
            raise AblationError(f"corpus row {index} needs doc_id/id and text/content")
        metadata = row.get("metadata", {})
        documents.append(CorpusDocument(str(doc_id), str(text), metadata))
    return tuple(documents)


def _load_queries(value: Any, base_dir: Path) -> Tuple[QueryRecord, ...]:
    payload = _records_payload(_resolve_source(value, base_dir), "queries")
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise AblationError("manifest queries must be a list or JSONL file")
    queries = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise AblationError(f"query row {index} must be an object")
        query_id = row.get("query_id", row.get("id"))
        text = row.get("text", row.get("question", row.get("query")))
        if query_id is None or text is None:
            raise AblationError(f"query row {index} needs query_id/id and text/question")
        queries.append(QueryRecord(str(query_id), str(text)))
    return tuple(queries)


def _load_qrels(value: Any, base_dir: Path) -> Dict[str, Dict[str, float]]:
    payload = _resolve_source(value, base_dir)
    if isinstance(payload, Mapping) and "qrels" in payload:
        payload = payload["qrels"]
    qrels: Dict[str, Dict[str, float]] = {}
    if isinstance(payload, Mapping):
        for query_id, entries in payload.items():
            if isinstance(entries, Mapping):
                qrels[str(query_id)] = {str(doc_id): float(grade) for doc_id, grade in entries.items()}
            elif isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
                qrels[str(query_id)] = {str(doc_id): 1.0 for doc_id in entries}
            else:
                raise AblationError(f"qrels for {query_id!r} must be a mapping or list")
        return qrels
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for index, row in enumerate(payload):
            if not isinstance(row, Mapping):
                raise AblationError(f"qrels row {index} must be an object")
            query_id = row.get("query_id", row.get("qid", row.get("query")))
            doc_id = row.get("doc_id", row.get("docid", row.get("id")))
            if query_id is None or doc_id is None:
                raise AblationError(f"qrels row {index} needs query_id/qid and doc_id/docid")
            grade = row.get("relevance", row.get("label", 1))
            qrels.setdefault(str(query_id), {})[str(doc_id)] = float(grade)
        return qrels
    raise AblationError("manifest qrels must be a mapping, list, or JSONL file")


def _load_rankings(value: Any, base_dir: Path) -> Dict[str, Tuple[Any, ...]]:
    payload = _resolve_source(value, base_dir)
    if isinstance(payload, Mapping) and "rankings" in payload:
        payload = payload["rankings"]
    if isinstance(payload, Mapping):
        output = {}
        for query_id, items in payload.items():
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
                raise AblationError(f"ranking for {query_id!r} must be a list")
            output[str(query_id)] = tuple(items)
        return output
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        output: Dict[str, Tuple[Any, ...]] = {}
        for index, row in enumerate(payload):
            if not isinstance(row, Mapping):
                raise AblationError(f"ranking row {index} must be an object")
            query_id = row.get("query_id", row.get("qid", row.get("query")))
            items = row.get("ranked_ids", row.get("ranking", row.get("doc_ids")))
            if query_id is None or items is None:
                raise AblationError(f"ranking row {index} needs query_id and ranked_ids")
            output[str(query_id)] = tuple(items)
        return output
    raise AblationError("manifest rankings must be a mapping, list, or JSONL file")


def _filter_from_manifest(value: Any) -> FilterPolicy:
    if value is None:
        return FilterPolicy()
    if not isinstance(value, Mapping):
        raise AblationError("manifest filter must be an object")
    allowed_doc_ids = value.get("allowed_doc_ids")
    allowed_query_ids = value.get("allowed_query_ids")
    return FilterPolicy(
        name=str(value.get("name", "all-corpus")),
        revision=str(value.get("revision", "v1")),
        allowed_doc_ids=frozenset(str(item) for item in allowed_doc_ids)
        if allowed_doc_ids is not None
        else None,
        allowed_query_ids=frozenset(str(item) for item in allowed_query_ids)
        if allowed_query_ids is not None
        else None,
    )


def build_harness_from_manifest(path: Path) -> AblationHarness:
    """Build a replay-based offline harness from one local JSON manifest."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise AblationError(f"manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AblationError(f"manifest is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise AblationError("manifest root must be an object")
    base_dir = path.parent
    corpus = _load_corpus(payload.get("corpus"), base_dir)
    queries = _load_queries(payload.get("queries"), base_dir)
    qrels = _load_qrels(payload.get("qrels", {}), base_dir)
    policy = _filter_from_manifest(payload.get("filter"))
    contract = DatasetContract(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        corpus_revision=str(payload.get("corpus_revision", "unspecified")),
        qrels_revision=str(payload.get("qrels_revision", "unspecified")),
        filter_policy=policy,
    )

    rankings = payload.get("rankings")
    if not isinstance(rankings, Mapping) or "dense" not in rankings or "bm25" not in rankings:
        raise AblationError("manifest rankings must contain dense and bm25 entries")
    model_revisions = payload.get("model_revisions", {})
    if not isinstance(model_revisions, Mapping):
        raise AblationError("model_revisions must be an object")
    dense = ReplayRetriever(
        _load_rankings(rankings["dense"], base_dir),
        str(model_revisions.get("dense", "dense-replay-v1")),
    )
    bm25 = ReplayRetriever(
        _load_rankings(rankings["bm25"], base_dir),
        str(model_revisions.get("bm25", "bm25-replay-v1")),
    )

    reranker: Optional[Any] = None
    if "reranker_scores" in payload:
        score_payload = _resolve_source(payload["reranker_scores"], base_dir)
        if not isinstance(score_payload, Mapping):
            raise AblationError("reranker_scores must be a query-to-doc-score mapping")
        reranker = MappingScoreReranker(
            score_payload,
            str(model_revisions.get("reranker", "cross-encoder-score-map-v1")),
        )
    elif payload.get("reranker_model_path"):
        reranker = LocalCrossEncoderReranker(
            str((base_dir / str(payload["reranker_model_path"])).resolve())
            if not Path(str(payload["reranker_model_path"])).is_absolute()
            else str(payload["reranker_model_path"]),
            str(model_revisions.get("reranker", "cross-encoder-local")),
            device=str(payload.get("reranker_device", "cpu")),
        )

    package_metadata = payload.get("package_metadata")
    if package_metadata is None:
        package_metadata = collect_package_metadata()
    if not isinstance(package_metadata, Mapping):
        raise AblationError("package_metadata must be an object")

    return AblationHarness(
        contract,
        dense,
        bm25,
        reranker,
        top_k=int(payload.get("top_k", DEFAULT_TOP_K)),
        candidate_pool_k=int(payload.get("candidate_pool_k", DEFAULT_CANDIDATE_POOL_K)),
        rrf_k=int(payload.get("rrf_k", DEFAULT_RRF_K)),
        conditions=payload.get("conditions"),
        model_revisions={str(key): str(value) for key, value in model_revisions.items()},
        package_metadata=package_metadata,
    )


def _default_output_dir() -> Path:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "outputs" / f"retrieval-ablation-{stamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="local JSON manifest")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="new output directory (default: versioned outputs/retrieval-ablation-<UTC>)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="run and print summaries without creating output files",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        harness = build_harness_from_manifest(args.manifest)
        result = harness.run()
        for row in result.summary:
            print(
                f"{row['condition']}: n={row['n_queries']} "
                f"hit={row['mean_hit_rate']:.4f} "
                f"mrr@10={row['mean_mrr_at_10']:.4f} "
                f"nDCG@10={row['mean_ndcg_at_10']:.4f}"
            )
        if not args.no_write:
            output_dir = args.output_dir or _default_output_dir()
            result.write_outputs(output_dir)
            print(f"wrote create-only outputs to {output_dir}")
        return 0
    except (AblationError, FileExistsError, OSError) as exc:
        print(f"retrieval ablation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised by CLI users
    raise SystemExit(main())
