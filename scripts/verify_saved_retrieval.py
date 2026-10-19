"""Independent, offline retrospective supplement; never rewrites research outputs.

Binary qrels, nDCG@5 primary supplement. Historical nDCG@10 remains separately
recomputed on the saved (at most five) results. Three specified paired contrasts:
dense wide-pool truncation vs dense reranking, hybrid vs hybrid reranking, and
dense reranking vs hybrid reranking. Two-sided Monte Carlo sign-flip p-values
receive Holm familywise correction across these three contrasts. Percentile
bootstrap intervals are descriptive, not simultaneous or preregistered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def scores(ranked, relevant, k=5):
    if len(ranked) != len(set(ranked)):
        raise ValueError("duplicate ranked document")
    relevant = set(relevant)
    if not relevant:
        raise ValueError("unjudged query")
    hits = [doc in relevant for doc in ranked[:k]]
    dcg = sum(float(hit) / math.log2(i + 2) for i, hit in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return {
        f"ndcg_at_{k}": dcg / ideal,
        "mrr_at_10": next((1 / (i + 1) for i, doc in enumerate(ranked[:10]) if doc in relevant), 0),
        "recall_at_k": sum(hits) / len(relevant),
        "hit_rate": float(any(hits)), "precision_at_k": sum(hits) / k,
    }


def paired(left, right, seed=20261019, repetitions=10000):
    delta = np.asarray(right) - np.asarray(left)
    if not len(delta):
        raise ValueError("empty paired sample")
    rng = np.random.default_rng(seed)
    observed = float(delta.mean())
    boot = np.mean(rng.choice(delta, size=(repetitions, len(delta))), axis=1)
    null = np.mean(rng.choice([-1, 1], size=(repetitions, len(delta))) * delta, axis=1)
    return {"n": len(delta), "mean_delta_right_minus_left": observed,
            "bootstrap_95_percentile": np.quantile(boot, [.025, .975]).tolist(),
            "p_two_sided_monte_carlo": float((1 + np.count_nonzero(np.abs(null) >= abs(observed) - 1e-15)) / (repetitions + 1))}


def holm(pvalues):
    ordered = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted, previous = [0.0] * len(pvalues), 0.0
    for rank, index in enumerate(ordered):
        previous = max(previous, min(1.0, pvalues[index] * (len(pvalues) - rank)))
        adjusted[index] = previous
    return adjusted


def verify(source: Path, output: Path):
    if output.exists():
        raise FileExistsError("Refusing to overwrite supplement")
    contract = json.loads((source / "contract.json").read_text())
    metadata = json.loads((source / "run_metadata.json").read_text())
    rows = [json.loads(line) for line in (source / "per_query.jsonl").read_text().splitlines()]
    conditions = [row["name"] for row in metadata["conditions"]]
    query_ids = metadata["query_ids"]
    corpus = {doc["doc_id"] for doc in contract["corpus"]}
    grouped = {name: {} for name in conditions}
    recomputed, max_error = [], 0.0
    fingerprints = ["contract_fingerprint", "corpus_fingerprint", "qrels_fingerprint",
                    "filter_fingerprint", "model_revisions_fingerprint", "package_metadata_fingerprint"]
    expected = dict(metadata)
    expected["filter_fingerprint"] = hashlib.sha256(
        json.dumps(metadata["filter"], sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    for row in rows:
        condition, qid = row["condition"], row["query_id"]
        if qid in grouped[condition]:
            raise ValueError("duplicate query/condition")
        grouped[condition][qid] = row
        relevant = {doc for doc, grade in contract["qrels"][qid].items() if grade > 0}
        if relevant != set(row["relevant_ids"]):
            raise ValueError("qrels mismatch")
        pool, ranked = row["candidate_pool_ids"], row["ranked_ids"]
        if len(pool) != len(set(pool)) or not set(pool) <= corpus or not set(ranked) <= set(pool):
            raise ValueError("invalid candidate pool")
        if len(ranked) > 5 or row["top_k"] != 5:
            raise ValueError("supplement requires original top-five protocol")
        for key in fingerprints:
            if row[key] != expected[key]:
                raise ValueError(f"fingerprint mismatch: {key}")
        measured = scores(ranked, relevant)
        historical = dict(measured, ndcg_at_10=scores(ranked, relevant, 10)["ndcg_at_10"])
        for key in ["ndcg_at_10", "mrr_at_10", "recall_at_k", "hit_rate", "precision_at_k"]:
            max_error = max(max_error, abs(historical[key] - row[key]))
        recomputed.append({"condition": condition, "query_id": qid, **historical})
    if any(list(grouped[name]) != query_ids for name in conditions):
        raise ValueError("query order/coverage mismatch")
    if max_error > 1e-12:
        raise ValueError("historical metric mismatch")
    changed, same_hybrid = [], 0
    for qid in query_ids:
        dense = grouped["dense_reranked"][qid]
        derived = dense["candidate_pool_ids"][:5]
        if derived != grouped["dense_only"][qid]["ranked_ids"]:
            changed.append(qid)
        recomputed.append({"condition": "dense_same_pool_no_rerank_derived", "query_id": qid,
                           "ranked_ids": derived, "candidate_pool_ids": dense["candidate_pool_ids"],
                           **scores(derived, dense["relevant_ids"])})
        if grouped["hybrid_no_rerank"][qid]["candidate_pool_ids"] != grouped["hybrid_reranked"][qid]["candidate_pool_ids"]:
            raise ValueError("hybrid pool mismatch")
        same_hybrid += 1
    names = conditions + ["dense_same_pool_no_rerank_derived"]
    values = {name: [row["ndcg_at_5"] for row in recomputed if row["condition"] == name] for name in names}
    comparisons = [(names[-1], "dense_reranked"), ("hybrid_no_rerank", "hybrid_reranked"),
                   ("dense_reranked", "hybrid_reranked")]
    contrasts = [{"left": left, "right": right, **paired(values[left], values[right])} for left, right in comparisons]
    for row, adjusted in zip(contrasts, holm([row["p_two_sided_monte_carlo"] for row in contrasts])):
        row["p_holm_family_3"] = adjusted
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
              for name in ["contract.json", "run_metadata.json", "per_query.jsonl", "condition_summary.csv"]}
    summary = {"schema": "saved-retrieval-supplement/v1", "retrospective_not_preregistered": True,
               "source_hashes": hashes, "original_rows": len(rows), "queries_per_condition": len(query_ids),
               "maximum_historical_metric_error": max_error, "identical_hybrid_pools": same_hybrid,
               "derived_dense_differs_from_historical_query_ids": changed,
               "mean_ndcg_at_5": {name: float(np.mean(values[name])) for name in names},
               "paired_contrasts": contrasts,
               "limits": ["Inherited passage labels, not human chunk judgments.",
                          "No new retrieval/model run; dense common-pool baseline is derived from saved pool order.",
                          "No nDCG@10 depth-ten retrieval, semantic scoring or latency inference.",
                          "Bootstrap intervals are individual descriptive intervals; only p-values have Holm adjustment."]}
    output.mkdir(parents=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "per_query.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in recomputed))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.source, args.output)
