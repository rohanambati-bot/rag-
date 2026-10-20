"""Bounded local reranker-depth benchmark.

This measures the current cross-encoder implementation on one fixed query set.
It never changes the configured store.  An unmanifested legacy store is allowed
only with ``--allow-unmanifested-legacy`` and is marked ``protocol_valid=false``;
those measurements must not be presented as release-quality corpus results.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHON_DOTENV_DISABLED", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def _p95(values: list[float]) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95)) if values else 0.0


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def run(args: argparse.Namespace) -> dict:
    from langchain_core.documents import Document
    from module1.data_preprocessing import get_embedding_model, load_vector_store
    from module2.rag_architectures import _CrossEncoderReranker, _reranker
    from module3.evaluation import (
        build_qrels, compute_hit_rate, compute_mrr, compute_ndcg_at_k, doc_id_for,
    )
    from configs.config import RERANKER_MODEL, TOP_K

    store_path = Path(args.store).resolve()
    manifest_present = (store_path / "_index_manifest.json").is_file()
    if not manifest_present and not args.allow_unmanifested_legacy:
        raise RuntimeError(
            f"Index manifest is missing at {store_path}; pass "
            "--allow-unmanifested-legacy only for explicitly non-release measurements."
        )

    embedder = get_embedding_model()
    if manifest_present:
        store = load_vector_store(embedder, str(store_path))
    else:
        store = load_vector_store(
            embedder,
            str(store_path),
            allow_unmanifested_legacy=True,
        )

    stored = store.get(include=["documents", "metadatas"])
    documents = [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(stored["documents"], stored["metadatas"])
    ]
    if not documents:
        raise RuntimeError("The benchmark store contains no documents")

    from rank_bm25 import BM25Okapi
    import re

    bm25 = BM25Okapi([re.findall(r"\w+", doc.page_content.lower()) for doc in documents])
    qrels = build_qrels(documents)
    dataset_path = ROOT / "data" / "synthetic_dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    queries = [
        row for row in dataset
        if row.get("question", "").strip() in qrels and qrels[row["question"].strip()]
    ][: args.samples]
    if len(queries) < args.samples:
        raise RuntimeError(
            f"Only {len(queries)} fixed queries have qrels in the selected corpus; "
            f"need {args.samples}."
        )

    reranker = _CrossEncoderReranker(device=args.device) if args.device else _reranker()
    if not reranker.is_available():
        raise RuntimeError("The configured cross-encoder is unavailable offline")

    # Warm the model once; the measured rows exclude model construction.
    reranker.rerank(queries[0]["question"], documents[: min(2, len(documents))], TOP_K)
    results = []
    for depth in args.depths:
        total_ms, pool_ms, rerank_ms = [], [], []
        hits5, hits10, mrr10, ndcg5, ndcg10 = [], [], [], [], []
        pool_sizes = []
        for row in queries:
            question = row["question"].strip()
            started = time.perf_counter()
            dense = store.similarity_search(question, k=depth)
            pool_started = time.perf_counter()
            scores = bm25.get_scores(re.findall(r"\w+", question.lower()))
            indices = np.argsort(np.asarray(scores))[::-1][:depth]
            pool = {doc.page_content[:200]: doc for doc in dense}
            for index in indices:
                doc = documents[int(index)]
                pool.setdefault(doc.page_content[:200], doc)
            pool_docs = list(pool.values())
            pool_sizes.append(len(pool_docs))
            pool_ms.append((time.perf_counter() - pool_started) * 1000)
            rerank_started = time.perf_counter()
            ranked = reranker.rerank(question, pool_docs, max(10, TOP_K))
            rerank_ms.append((time.perf_counter() - rerank_started) * 1000)
            total_ms.append((time.perf_counter() - started) * 1000)
            ids = [doc_id_for(doc) for doc in ranked]
            relevant = qrels[question]
            hits5.append(compute_hit_rate(ids[:5], relevant))
            hits10.append(compute_hit_rate(ids[:10], relevant))
            mrr10.append(compute_mrr(ids, relevant, k=10))
            ndcg5.append(compute_ndcg_at_k(ids, relevant, k=5))
            ndcg10.append(compute_ndcg_at_k(ids, relevant, k=10))
        results.append({
            "candidate_depth": depth,
            "mean_pool_size": round(_mean(pool_sizes), 2),
            "hit_rate_at_5": round(_mean(hits5), 4),
            "hit_rate_at_10": round(_mean(hits10), 4),
            "mrr_at_10": round(_mean(mrr10), 4),
            "ndcg_at_5": round(_mean(ndcg5), 4),
            "ndcg_at_10": round(_mean(ndcg10), 4),
            "median_total_ms": round(float(np.median(total_ms)), 2),
            "p95_total_ms": round(_p95(total_ms), 2),
            "median_pool_ms": round(float(np.median(pool_ms)), 2),
            "median_rerank_ms": round(float(np.median(rerank_ms)), 2),
            "p95_rerank_ms": round(_p95(rerank_ms), 2),
        })

    return {
        "schema": "reranker-depth-benchmark/v1",
        "status": "passed",
        "protocol_valid": manifest_present,
        "manifest_present": manifest_present,
        "store": str(store_path),
        "query_count": len(queries),
        "candidate_depths": args.depths,
        "top_k": TOP_K,
        "reranker_model": RERANKER_MODEL,
        "device": reranker.device,
        "peak_rss_mib": round(_rss_mib(), 2),
        "results": results,
        "limitations": ([] if manifest_present else [
            "Legacy store has no index manifest; metrics are not release-valid corpus evidence."
        ]),
    }


def main(argv=None) -> int:
    from configs.config import TOP_K

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="data/chroma_db_full")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--depths", type=int, nargs="+", default=[20, 50, 100, 200])
    parser.add_argument("--allow-unmanifested-legacy", action="store_true")
    parser.add_argument("--device", choices=["cpu", "mps"], default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.samples < 1 or any(depth < TOP_K for depth in args.depths):
        parser.error("samples must be positive and every depth must cover TOP_K")
    if args.output.exists():
        parser.error("output already exists; choose a fresh destination")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
