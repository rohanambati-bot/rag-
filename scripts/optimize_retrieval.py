#!/usr/bin/env python3
"""
Retrieval-accuracy diagnosis and tuning on a **dev split**, never on the eval set.

The standard eval samples 500 of the first 1000 MS MARCO QA pairs with a seeded
RNG (evaluate_architecture, seed=EVAL_RANDOM_SEED). This script reproduces that
sample deterministically, calls it the *eval* questions, and uses the remaining
~500 as the *dev* questions. All depth / fusion / model decisions are made on
the dev questions only; the final reported numbers must come from a standard
`main.py --mode eval` run touching the eval questions.

Modes (both share `bootstrap`):
  default           — rank-source diagnostics + miss taxonomy
  --rerankers       — cross-encoder model comparison on the same dev pool
                      (pool = dedup(dense<=50 U bm25<=50), the production
                      first stage). Candidates are listed in RERANKER_CANDIDATES.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import (  # noqa: E402
    DATA_DIR,
    EVAL_RANDOM_SEED,
    RERANK_CANDIDATES,
    TOP_K,
)

MAX_DEPTH = 200

# Cross-encoder candidates for `--rerankers` (dev-decided; production default
# stays whatever configs.RERANKER_MODEL is until a candidate wins on dev).
RERANKER_CANDIDATES = [
    "BAAI/bge-reranker-base",   # current production default
    "BAAI/bge-reranker-large",
    "BAAI/bge-reranker-v2-m3",
]


def bootstrap():
    """Load store/corpus/qrels and reproduce the eval/dev question split."""
    from module1.data_preprocessing import (
        chunk_documents,
        get_embedding_model,
        load_msmarco_documents,
        load_msmarco_qa_pairs,
        load_vector_store,
    )
    from module3.evaluation import build_qrels, doc_id_for

    store_dir = os.path.join(DATA_DIR, "chroma_db_full")
    embedding_model = get_embedding_model()
    vectorstore = load_vector_store(embedding_model, persist_dir=store_dir)
    docs = load_msmarco_documents()
    chunks = chunk_documents(docs)
    qrels = build_qrels(chunks)
    chunk_ids = [doc_id_for(c) for c in chunks]
    index_of_id = {cid: i for i, cid in enumerate(chunk_ids)}

    pairs = load_msmarco_qa_pairs(max_samples=1000)
    rng = np.random.default_rng(EVAL_RANDOM_SEED)
    n_eval = 500
    eval_idx = set(rng.choice(len(pairs), size=min(n_eval, len(pairs)), replace=False).tolist())
    eval_qs = [pairs[i] for i in sorted(eval_idx)]
    dev_qs = [q for i, q in enumerate(pairs) if i not in eval_idx]
    scored = [(q, qrels.get(q["question"].strip(), set()))
              for q in dev_qs]
    scored = [(q, rel) for q, rel in scored if rel]

    return {
        "vectorstore": vectorstore,
        "chunks": chunks,
        "qrels": qrels,
        "chunk_ids": chunk_ids,
        "index_of_id": index_of_id,
        "eval_qs": eval_qs,
        "dev_qs": dev_qs,
        "scored": scored,
    }


def mode_diagnosis(split: dict) -> None:
    from module3.evaluation import doc_id_for

    vectorstore, chunks, scored = split["vectorstore"], split["chunks"], split["scored"]
    chunk_ids = split["chunk_ids"]
    print(f"eval questions: {len(split['eval_qs'])} | dev questions: "
          f"{len(split['dev_qs'])} ({len(scored)} with qrels)\n")

    # ── Dense rankings at MAX_DEPTH (Chroma query = identical to production) ───
    print("Querying the vector store at depth", MAX_DEPTH, "…")
    dense_ids: dict[str, list[str]] = {}
    for q, _ in scored:
        docs_k = vectorstore.similarity_search(q["question"], k=MAX_DEPTH)
        dense_ids[q["question"]] = [doc_id_for(d) for d in docs_k]

    # ── BM25 rankings ──────────────────────────────────────────────────────────
    print("Ranking BM25 over the chunk corpus …")
    import re as _re

    from rank_bm25 import BM25Okapi

    tokenized = [_re.findall(r"\w+", c.page_content.lower()) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    bm25_ids: dict[str, list[str]] = {}
    for q, _ in scored:
        tokens = _re.findall(r"\w+", q["question"].lower())
        scores = bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1][:MAX_DEPTH]
        bm25_ids[q["question"]] = [chunk_ids[i] for i in order]

    # ── Metric helper ──────────────────────────────────────────────────────────
    def metrics(rankings: dict[str, list[str]], k: int) -> tuple[float, float, float]:
        """(hit@k, MRR capped at k, mean P@5) over scored dev questions."""
        recalls, rrs, precs = [], [], []
        for q, rel in scored:
            ranked = rankings[q["question"]][:k]
            recalls.append(float(any(cid in rel for cid in ranked)))
            rr = next((1.0 / rank for rank, cid in enumerate(ranked, start=1)
                       if cid in rel), 0.0)
            rrs.append(rr)
            if k >= 5:
                precs.append(sum(cid in rel for cid in ranked[:5]) / 5.0)
        n = len(scored)
        return sum(recalls) / n, sum(rrs) / n, (sum(precs) / n if precs else float("nan"))

    def rrf_fuse(depth: int, rrf_k: int = 60) -> dict[str, list[str]]:
        fused: dict[str, list[str]] = {}
        for q, _ in scored:
            question = q["question"]
            s: dict[str, float] = {}
            for lst in (dense_ids[question][:depth], bm25_ids[question][:depth]):
                for rank, cid in enumerate(lst):
                    s[cid] = s.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            fused[question] = [cid for cid, _ in sorted(s.items(), key=lambda x: -x[1])]
        return fused

    def weighted_fuse(depth: int, alpha: float) -> dict[str, list[str]]:
        """Rank-decay weighted sum: alpha*dense + (1-alpha)*bm25."""
        fused: dict[str, list[str]] = {}
        for q, _ in scored:
            question = q["question"]
            s: dict[str, float] = {}
            for rank, cid in enumerate(dense_ids[question][:depth]):
                s[cid] = s.get(cid, 0.0) + alpha * (1.0 - rank / 1000)
            for rank, cid in enumerate(bm25_ids[question][:depth]):
                s[cid] = s.get(cid, 0.0) + (1.0 - alpha) * (1.0 - rank / 1000)
            fused[question] = [cid for cid, _ in sorted(s.items(), key=lambda x: -x[1])]
        return fused

    # ── Configuration sweep on dev ─────────────────────────────────────────────
    rows: list[tuple[str, float, float, float]] = []

    def report(label: str, rankings: dict[str, list[str]], k: int) -> None:
        r, m, p = metrics(rankings, k)
        rows.append((label, r, m, p))

    report(f"dense top5 (Naive, pool k={MAX_DEPTH})",
           {q: v for q, v in dense_ids.items()}, TOP_K)
    for depth in (50, 100, 200):
        report(f"dense only, {depth}", {q: v[:depth] for q, v in dense_ids.items()}, depth)
    for depth in (50, 200):
        report(f"bm25 only, {depth}", {q: v[:depth] for q, v in bm25_ids.items()}, depth)
    for depth in (50, 100, 200):
        report(f"RRF60 dense+bm25 @ {depth}", rrf_fuse(depth), 10)
    for alpha in (0.3, 0.5, 0.7):
        report(f"weighted a={alpha} @50", weighted_fuse(50, alpha), 5)
    report(f"RRF60 dense+bm25 @ {RERANK_CANDIDATES} (current 1st stage)",
           rrf_fuse(RERANK_CANDIDATES), TOP_K)

    hdr = f"{'configuration':42s} {'R@k':>7s} {'MRR':>7s} {'P@5':>7s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for label, r, m, p in rows:
        print(f"{label:42s} {r:7.3f} {m:7.3f} {p:7.3f}")

    # ── Ceilings + miss taxonomy on dense-top5 misses ──────────────────────────
    missed = [(q, rel) for q, rel in scored
              if not any(cid in rel for cid in dense_ids[q["question"]][:TOP_K])]
    n = len(scored)
    print(f"\nmiss taxonomy — dense top5 misses on every relevant chunk for "
          f"{len(missed)}/{n} dev queries ({len(missed)/n:.1%}):")

    def best_rank(ids: list[str], rel: set, limit: int) -> int | None:
        return next((rank for rank, cid in enumerate(ids[:limit], start=1)
                     if cid in rel), None)

    def frac(pred) -> float:  # type: ignore[no-untyped-def]
        return sum(1 for q, rel in missed if pred(q, rel)) / max(len(missed), 1)

    def dense_rank(q, rel, limit):  # type: ignore[no-untyped-def]
        return best_rank(dense_ids[q["question"]], rel, limit)

    def bm25_rank(q, rel, limit):  # type: ignore[no-untyped-def]
        return best_rank(bm25_ids[q["question"]], rel, limit)

    print(f"  in dense pool  <=50:  {frac(lambda q, rel: dense_rank(q, rel, 50) is not None):6.1%}")
    print(f"  in dense pool 51..100: {frac(lambda q, rel: dense_rank(q, rel, 50) is None and dense_rank(q, rel, 100) is not None):6.1%}")
    print(f"  in dense pool 101..{MAX_DEPTH}: {frac(lambda q, rel: dense_rank(q, rel, 100) is None and dense_rank(q, rel, MAX_DEPTH) is not None):6.1%}")
    print(f"  not in dense pool <= {MAX_DEPTH} (paradigm miss): {frac(lambda q, rel: dense_rank(q, rel, MAX_DEPTH) is None):6.1%}")
    print(f"  in bm25 pool   <=50:  {frac(lambda q, rel: bm25_rank(q, rel, 50) is not None):6.1%}")
    print(f"  in bm25 pool 51..{MAX_DEPTH}: {frac(lambda q, rel: bm25_rank(q, rel, 50) is None and bm25_rank(q, rel, MAX_DEPTH) is not None):6.1%}")
    print(f"  in union(dense<=50, bm25<=50) [rerank candidate pool]: "
          f"{frac(lambda q, rel: dense_rank(q, rel, 50) is not None or bm25_rank(q, rel, 50) is not None):6.1%}")
    print(f"  in union(dense<={MAX_DEPTH}, bm25<={MAX_DEPTH}) [deep pool ceiling]: "
          f"{frac(lambda q, rel: dense_rank(q, rel, MAX_DEPTH) is not None or bm25_rank(q, rel, MAX_DEPTH) is not None):6.1%}")

    qtypes: dict[str, list[int]] = {}
    for q, rel in scored:
        qt = q.get("query_type", "unknown")
        hit = any(cid in rel for cid in dense_ids[q["question"]][:TOP_K])
        qtypes.setdefault(qt, [0, 0])
        qtypes[qt][0] += 1
        qtypes[qt][1] += int(not hit)
    print("\nby query_type (queries with qrels / dense-top5 misses):")
    for qt, (tot, miss) in sorted(qtypes.items()):
        if tot:
            print(f"  {qt:12s} {tot:4d}  {miss:4d}  ({miss/tot:.1%} missed)")


def mode_rerankers(split: dict) -> None:
    """Score candidate cross-encoders on the production first-stage dev pool."""
    import gc

    import torch
    from sentence_transformers import CrossEncoder

    from module2.rag_architectures import RerankRAG
    from module3.evaluation import doc_id_for

    vectorstore, chunks, scored = split["vectorstore"], split["chunks"], split["scored"]

    # Build the production pool once, shared by all candidates.
    print(f"Building first-stage pools (dense<= {RERANK_CANDIDATES} U bm25<= "
          f"{RERANK_CANDIDATES}) for {len(scored)} dev queries …")
    first_stage = RerankRAG(vectorstore=vectorstore, chunks=chunks)
    pool: dict[str, list[tuple[str, object]]] = {}
    sizes = []
    for q, _rel in scored:
        docs = first_stage._first_stage(q["question"])
        doc_ids = [doc_id_for(d) for d in docs]
        pool[q["question"]] = list(zip(doc_ids, docs))
        sizes.append(len(docs))
    print(f"pool size: mean {np.mean(sizes):.1f}, max {max(sizes)}")

    results: list[tuple[str, float, float, float, float]] = []
    for model_name in RERANKER_CANDIDATES:
        try:
            print(f"\nScoring {model_name} on {sum(sizes)} (query, passage) pairs …")
            model = CrossEncoder(model_name, device="mps")
            hits, rrs, precs, rescue = [], [], [], []
            for q, rel in scored:
                pairs_ = [(q["question"], d.page_content) for _cid, d in pool[q["question"]]]
                scores = model.predict(pairs_, batch_size=32, show_progress_bar=False)
                order = np.argsort(np.asarray(scores))[::-1][:TOP_K]
                top = [pool[q["question"]][i][0] for i in order]
                hits.append(float(any(cid in rel for cid in top)))
                rrs.append(next((1.0 / r for r, cid in enumerate(top, 1) if cid in rel), 0.0))
                precs.append(sum(cid in rel for cid in top) / TOP_K)
                if not any(cid in rel for cid in
                           [cid for cid, _ in pool[q["question"]][:TOP_K]]):
                    rescue.append(hits[-1])
            del model
            gc.collect()
            try:
                torch.mps.empty_cache()
            except (AttributeError, RuntimeError):
                pass
            n = len(scored)
            results.append((
                model_name,
                sum(hits) / n,
                sum(rrs) / n,
                sum(precs) / n,
                (sum(rescue) / len(rescue)) if rescue else float("nan"),
            ))
            print(f"  done: hit@5={results[-1][1]:.3f} MRR={results[-1][2]:.3f} "
                  f"P@5={results[-1][3]:.3f} rescue@5={results[-1][4]:.1%}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {model_name}: FAILED — {exc}")

    print("\n=== dev cross-encoder comparison (pool = production first stage) ===")
    hdr = f"{'model':32s} {'hit@5':>6s} {'MRR':>6s} {'P@5':>6s} {'rescue@5':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for name, h, m, p, rescue in results:
        print(f"{name:32s} {h:6.3f} {m:6.3f} {p:6.3f} {rescue:8.1%}")


if __name__ == "__main__":
    split = bootstrap()
    if "--rerankers" in sys.argv:
        mode_rerankers(split)
    else:
        mode_diagnosis(split)
