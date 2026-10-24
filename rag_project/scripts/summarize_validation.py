#!/usr/bin/env python3
"""Recompute receipt metrics and descriptive answer proxies without any model/API."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import string
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from module3.evaluation import (  # noqa: E402
    compute_hit_rate, compute_mrr, compute_ndcg_at_k, compute_recall_at_k,
)


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def token_f1(answer, reference):
    answer, reference = normalize(answer).split(), normalize(reference).split()
    if not answer or not reference:
        return float(answer == reference)
    overlap = sum((Counter(answer) & Counter(reference)).values())
    return 2 * overlap / (len(answer) + len(reference))


def summarize(path: Path) -> dict:
    raw = pd.read_csv(path / "raw.csv")
    ranks = json.loads((path / "rankings.json").read_text())
    recorded = raw.set_index(["architecture", "question"])
    differences = []
    duplicates = Counter()
    for row in ranks:
        ids, relevant = row["ranked_ids"], row["relevant_ids"]
        if len(set(ids)) != len(ids):
            duplicates[row["architecture"]] += 1
        if not relevant:
            continue
        expected = {
            "ndcg_at_10": compute_ndcg_at_k(ids, relevant, 10),
            "mrr_at_10": compute_mrr(ids, relevant, 10),
            "recall_at_k": compute_recall_at_k(ids, relevant, len(ids) or 1),
            "hit_rate": compute_hit_rate(ids, relevant),
        }
        actual = recorded.loc[(row["architecture"], row["question"])]
        differences.extend(abs(float(actual[k]) - v) for k, v in expected.items())
    answers = json.loads((path / "answers_extractive.json").read_text())
    answer_rows = []
    for architecture in sorted({row["architecture"] for row in answers}):
        group = [row for row in answers if row["architecture"] == architecture]
        valid, total, spans, abstentions = 0, 0, 0, 0
        for row in group:
            abstained = "could not find a relevant answer" in row["answer"]
            abstentions += int(abstained)
            spans += int(not abstained and any(row["answer"] in ctx for ctx in row["contexts"]))
            for citation in row["citations"]:
                total += 1
                valid += int(isinstance(citation.get("context_index"), int)
                             and 1 <= citation["context_index"] <= len(row["contexts"]))
        answer_rows.append({
            "architecture": architecture, "n": len(group),
            "normalized_exact_match": float(np.mean([
                normalize(row["answer"]) == normalize(row["ground_truth"]) for row in group
            ])),
            "reference_token_f1": float(np.mean([
                token_f1(row["answer"], row["ground_truth"]) for row in group
            ])),
            "citation_indices_valid": valid, "citation_count": total,
            "nonabstaining_answers_found_as_context_span": spans,
            "abstentions": abstentions,
            "mean_retrieval_ms": float(np.mean([row["retrieval_ms"] for row in group])),
            "mean_extractive_generation_ms": float(np.mean([row["generation_ms"] for row in group])),
        })
    historical = pd.read_csv(ROOT / "outputs/raw_eval_results.csv")
    metrics = ["ndcg_at_10", "mrr_at_10", "recall_at_k", "hit_rate"]
    joined = raw.merge(historical, on=["architecture", "question"], suffixes=("_new", "_historical"))
    comparison = {
        k: float((joined[f"{k}_new"] - joined[f"{k}_historical"]).abs().max())
        for k in metrics
    }
    latency = []
    for architecture, group in raw.groupby("architecture"):
        values = group.latency_ms.to_numpy()
        latency.append({"architecture": architecture, "n": len(group),
                        "first_query_ms": float(values[0]), "mean_ms": float(values.mean()),
                        "p50_ms": float(np.percentile(values, 50)),
                        "p95_ms": float(np.percentile(values, 95))})
    return {"ranked_metric_max_recompute_error": max(differences),
            "historical_paired_rows": len(joined),
            "historical_max_per_query_metric_difference": comparison,
            "queries_returning_duplicate_ids": dict(duplicates),
            "answer_proxies_NOT_semantic_quality": answer_rows,
            "retrieval_latency": latency,
            "limitations": ["No human/LLM judgement or semantic citation precision",
                            "Single reference, lexical proxies only",
                            "Process/disk warmness and host contention affect latency"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    path = ROOT / args.directory
    result = summarize(path)
    destination = path / "receipt_checks.json"
    with destination.open("x") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result, indent=2))
