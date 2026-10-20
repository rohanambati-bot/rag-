#!/usr/bin/env python3
"""Cost-free, non-destructive replay of the existing retrieval experiment.

Requires the existing dataset/model caches and a built store; downloads and LLM
credentials are explicitly disabled. Writes NEW receipts only, never the historical
CSVs. The original chunk/qrels definitions are deliberately left unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ("GROQ_API_KEY", "MISTRAL_API_KEY", "OPENROUTER_API_KEY"):
    os.environ[key] = ""
for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[key] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def paired_statistics(raw: pd.DataFrame, iterations: int = 10000) -> list[dict]:
    """Paired query bootstrap + unrounded two-sided tests, fixed seed 42.

    BH is within each metric across ALL non-baseline architectures in this run,
    matching the project's comparison family. These are conditional on the
    available qrels; no missing judgements are invented.
    """
    from module3.evaluation import benjamini_hochberg

    rows = []
    for metric in ("ndcg_at_10", "mrr_at_10", "recall_at_k", "hit_rate"):
        pivot = raw.pivot(index="question", columns="architecture", values=metric)
        family = []
        for arch in pivot.columns:
            if arch == "Naive RAG":
                continue
            pair = pivot[["Naive RAG", arch]].dropna()
            base, treatment = pair.to_numpy().T
            delta = treatment - base
            rng = np.random.default_rng(42)
            means = np.array([
                rng.choice(delta, len(delta), replace=True).mean()
                for _ in range(iterations)
            ])
            nonzero = bool(np.any(delta))
            p = float(stats.wilcoxon(delta).pvalue) if nonzero else 1.0
            row = {
                "metric": metric, "architecture": arch, "n_pairs": len(pair),
                "baseline_mean": float(base.mean()),
                "mean": float(treatment.mean()), "mean_diff": float(delta.mean()),
                "paired_ci_lo": float(np.quantile(means, .025)),
                "paired_ci_hi": float(np.quantile(means, .975)),
                "wilcoxon_p": p,
                "paired_t_p": float(stats.ttest_rel(treatment, base).pvalue)
                if nonzero else 1.0,
                "dz": float(delta.mean() / delta.std(ddof=1))
                if delta.std(ddof=1) else None,
            }
            if metric == "hit_rate":
                rescued = int(((base == 0) & (treatment == 1)).sum())
                lost = int(((base == 1) & (treatment == 0)).sum())
                row.update(rescued=rescued, lost=lost,
                           mcnemar_exact_p=float(stats.binomtest(
                               min(rescued, lost), rescued + lost, .5
                           ).pvalue) if rescued + lost else 1.0)
            family.append(row)
        adjusted, _ = benjamini_hochberg([r["wilcoxon_p"] for r in family])
        for row, p in zip(family, adjusted):
            row["wilcoxon_p_bh"] = p
        rows.extend(family)
    return rows


def run(args) -> None:
    from datasets import load_dataset
    from huggingface_hub import try_to_load_from_cache
    from configs.config import (
        DATA_DIR, DATASET_NAME, DATASET_CONFIG, DATASET_SPLIT,
        EMBEDDING_MODEL, RERANKER_MODEL, RERANK_CANDIDATES, TOP_K,
    )
    from module1.data_preprocessing import (
        chunk_documents, get_embedding_model, load_msmarco_documents,
        load_msmarco_qa_pairs, load_vector_store, copy_vector_store,
    )
    import module2.rag_architectures as rag
    from module3.evaluation import EvaluationPipeline, build_qrels, doc_id_for

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    docs = load_msmarco_documents()
    chunks = chunk_documents(docs)
    qrels = build_qrels(chunks)
    pairs = load_msmarco_qa_pairs(max_samples=max(args.samples, 1000))
    idx = np.random.default_rng(42).choice(len(pairs), args.samples, replace=False)
    selected = [pairs[i] for i in idx]
    # Store the fixed order and reference answers before any scoring.
    subset = [{k: q[k] for k in ("id", "question", "ground_truth", "query_type")}
              for q in selected]
    (out / "evaluation_subset.json").write_text(json.dumps(subset, indent=2))
    answer_subset = subset[:50]
    (out / "answer_subset_v1.json").write_text(json.dumps(answer_subset, indent=2))
    provenance = {
        "runtime": platform.python_version(), "platform": platform.platform(),
        "dataset": DATASET_NAME, "config": DATASET_CONFIG, "split": DATASET_SPLIT,
        "dataset_rows": len(ds), "dataset_fingerprint": ds._fingerprint,
        "dataset_cache_revisions": sorted({Path(x["filename"]).parent.name
                                           for x in ds.cache_files}),
        "passages": len(docs), "chunks": len(chunks), "qrels_queries": len(qrels),
        "chunk_size_characters": 400, "overlap_characters": 50,
        "top_k": TOP_K, "candidates_per_source": RERANK_CANDIDATES,
        "query_selection": "numpy.default_rng(42).choice(first 1000 answerable QA, N, replace=False)",
        "subset_sha256": digest(subset), "answer_subset_v1_sha256": digest(answer_subset),
        "missing_qrels": [q["question"] for q in selected if q["question"] not in qrels],
        "exact_duplicate_chunks": len(chunks) - len({d.page_content for d in chunks}),
        "prefix_id_duplicates": len(chunks) - len({doc_id_for(d) for d in chunks}),
        "store_has_manifest": (ROOT / args.store / "_index_manifest.json").exists(),
        "models": {}, "timings_seconds": {},
    }
    for model in (EMBEDDING_MODEL, RERANKER_MODEL):
        cached = try_to_load_from_cache(model, "config.json")
        provenance["models"][model] = (
            cached.split("/snapshots/")[-1].split("/")[0]
            if isinstance(cached, str) else None
        )
    for module in ("sentence-transformers", "transformers", "torch", "chromadb", "rank-bm25"):
        from importlib.metadata import version
        provenance.setdefault("packages", {})[module] = version(module)

    t0 = time.perf_counter()
    embedder = get_embedding_model()
    provenance["timings_seconds"]["embedding_load"] = time.perf_counter() - t0
    # Chroma queries may write SQLite metadata. Query a private copy, not user data.
    with tempfile.TemporaryDirectory(prefix="validation-", dir=DATA_DIR) as temp:
        copied = Path(temp) / "store"
        copy_vector_store(str(ROOT / args.store), str(copied))
        t0 = time.perf_counter()
        store = load_vector_store(embedder, str(copied))
        provenance["timings_seconds"]["store_load"] = time.perf_counter() - t0
        stored = store.get(include=["documents", "metadatas"])
        expected = Counter(d.page_content for d in chunks)
        actual = Counter(stored["documents"])
        provenance["corpus_multiset_equal"] = expected == actual
        provenance["corpus_missing_chunks"] = sum((expected - actual).values())
        provenance["corpus_extra_chunks"] = sum((actual - expected).values())
        provenance["corpus_text_multiset_sha256"] = digest(sorted(actual.items()))
        if expected != actual:
            raise RuntimeError("Dense and sparse corpus differ; refusing a mixed-corpus comparison")
        t0 = time.perf_counter()
        architectures = rag.build_all_architectures(store, chunks, llm_free_only=True)
        provenance["timings_seconds"]["architecture_construction"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        if not rag._reranker().is_available():
            raise RuntimeError("Real cross-encoder unavailable; refusing a fallback benchmark")
        provenance["timings_seconds"]["reranker_load"] = time.perf_counter() - t0
        provenance["reranker_device"] = rag._reranker().device
        provenance["timings_seconds"]["startup_to_models_ready"] = time.perf_counter() - start
        pipeline = EvaluationPipeline(output_path=str(out / "summary.csv"))
        rows, rankings, failures, answers = [], [], [], []
        for key, arch in architectures.items():
            print(f"Evaluating {arch.name}: {len(selected)} fixed queries", flush=True)
            for i, q in enumerate(selected):
                t0 = time.perf_counter()
                try:
                    retrieved = arch.retrieve(q["question"])
                    elapsed = (time.perf_counter() - t0) * 1000
                    row = pipeline.evaluate_sample(
                        q["question"], q["ground_truth"], retrieved, "", arch.name,
                        qrels.get(q["question"]),
                    )
                    row["latency_ms"] = elapsed
                    rows.append(row)
                    rankings.append({"architecture": arch.name, "question": q["question"],
                                     "ranked_ids": [doc_id_for(d) for d in retrieved],
                                     "relevant_ids": sorted(qrels.get(q["question"], []))})
                    if i < 50 and key in ("naive", "rerank"):
                        generated_start = time.perf_counter()
                        answer = arch.generate(q["question"], retrieved)
                        answers.append({"architecture": arch.name, "question": q["question"],
                                        "ground_truth": q["ground_truth"], "answer": answer,
                                        "contexts": [d.page_content for d in retrieved],
                                        "citations": arch._citations(retrieved),
                                        "retrieval_ms": elapsed,
                                        "generation_ms": (time.perf_counter() - generated_start) * 1000})
                except Exception as exc:
                    failures.append({"architecture": arch.name, "question": q["question"],
                                     "exception_type": type(exc).__name__, "error": str(exc)})
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(selected)}", flush=True)
        raw = pd.DataFrame(rows)
        raw.to_csv(out / "raw.csv", index=False)
        grouped = {name: frame for name, frame in raw.groupby("architecture")}
        summary = pipeline.compare_architectures(grouped)
        (out / "rankings.json").write_text(json.dumps(rankings))
        (out / "failures.json").write_text(json.dumps(failures, indent=2))
        (out / "answers_extractive.json").write_text(json.dumps(answers, indent=2))
        pd.DataFrame(paired_statistics(raw)).to_csv(out / "paired_statistics.csv", index=False)
        provenance["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            1024 ** 2 if sys.platform == "darwin" else 1024
        )
        provenance["total_seconds"] = time.perf_counter() - start
        provenance["provider_calls"] = 0
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
        print(summary[["architecture", "n", "ndcg_at_10", "mrr_at_10", "recall_at_k", "hit_rate"]].to_string(index=False))
        print(f"Receipts: {args.out}; failures: {len(failures)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/validation-20261019")
    parser.add_argument("--store", default="data/chroma_db_full")
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()
    if not 5 <= args.samples <= 1000:
        parser.error("--samples must be between 5 and 1000")
    run(args)
