#!/usr/bin/env python3
"""Run the five-condition ablation against the existing local MS MARCO store.

This is the real-data companion to :mod:`scripts.retrieval_ablation`.  It reads
only already-cached dataset/model files and ``data/chroma_db_full``; provider
credentials are cleared and Hugging Face offline mode is forced before project
modules are imported.  The output directory is create-only and contains a full
contract manifest, so a rerun cannot silently replace an earlier experiment.

The corpus is deduplicated by the project's existing 200-character content ID
before the shared contract is built.  That makes the five-way comparison
well-defined even though the legacy Chroma store contains duplicate IDs.  This
is a separately versioned protocol and must not be compared as a replacement
for the historical validation tables without saying so.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _configure_offline_environment() -> None:
    """Set CLI-only guards before importing project config/model loaders.

    Importing helper functions must not clear the caller's credentials or alter
    environment state (notably when the offline tests import this module).
    """
    for key in ("GROQ_API_KEY", "MISTRAL_API_KEY", "OPENROUTER_API_KEY"):
        os.environ[key] = ""
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _snapshot_revision(model_name: str) -> Tuple[Path, str]:
    """Return a cached snapshot directory and its immutable revision."""

    from huggingface_hub import try_to_load_from_cache

    config_path = try_to_load_from_cache(model_name, "config.json")
    if not isinstance(config_path, str):
        raise RuntimeError(
            f"cached config unavailable for {model_name!r}; refusing an online model load"
        )
    path = Path(config_path)
    marker = "/snapshots/"
    if marker not in config_path:
        raise RuntimeError(f"cached model path is not a snapshot: {config_path}")
    revision = config_path.split(marker, 1)[1].split("/", 1)[0]
    return path.parent, revision


def _reranker_device(configured: str) -> str:
    if configured:
        return configured
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _relative_or_name(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path.name


def _unique_corpus(chunks: Iterable[Any], doc_id_for: Any, corpus_document: Any) -> Tuple[Any, ...]:
    """Keep the first chunk for each stable content ID, preserving corpus order."""

    seen = set()
    unique = []
    for chunk in chunks:
        doc_id = str(doc_id_for(chunk))
        if doc_id in seen:
            continue
        seen.add(doc_id)
        unique.append(corpus_document(doc_id=doc_id, text=chunk.page_content))
    return tuple(unique)


def _retain_judged_pairs(selected: Sequence[Mapping[str, Any]], question_qrels: Mapping[str, Any]):
    """Keep only selected questions with at least one constructed qrel.

    The historical validation protocol excludes unjudged questions from metric
    denominators.  Applying the same rule here keeps the five conditions paired
    without turning missing labels into artificial retrieval failures.
    """

    judged = [
        pair for pair in selected
        if question_qrels.get(str(pair.get("question", "").strip()))
    ]
    excluded = [
        str(pair.get("id", ""))
        for pair in selected
        if pair not in judged
    ]
    return judged, excluded


def _build_contract(samples: int) -> Tuple[Any, Dict[str, Any]]:
    """Load the fixed local dataset and create the shared query/corpus/qrels contract."""

    import numpy as np
    from datasets import load_dataset

    from configs.config import (
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        DATASET_CONFIG,
        DATASET_NAME,
        DATASET_SPLIT,
        DATASET_SUBSET_SIZE,
    )
    from module1.data_preprocessing import (
        chunk_documents,
        load_msmarco_documents,
        load_msmarco_qa_pairs,
    )
    from module3.evaluation import build_qrels, doc_id_for
    from scripts.retrieval_ablation import (
        CorpusDocument,
        DatasetContract,
        FilterPolicy,
        QueryRecord,
    )

    if samples < 5 or samples > DATASET_SUBSET_SIZE:
        raise ValueError(f"samples must be between 5 and {DATASET_SUBSET_SIZE}")

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    docs = load_msmarco_documents(max_samples=DATASET_SUBSET_SIZE)
    chunks = chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    qa_pairs = load_msmarco_qa_pairs(max_samples=DATASET_SUBSET_SIZE)
    selected_indices = np.random.default_rng(42).choice(
        len(qa_pairs), samples, replace=False
    )
    selected_requested = [qa_pairs[int(index)] for index in selected_indices]

    question_qrels = build_qrels(chunks)
    selected, excluded_unjudged_ids = _retain_judged_pairs(selected_requested, question_qrels)
    if not selected:
        raise RuntimeError("the selected query sample contains no constructed qrels")
    corpus = _unique_corpus(chunks, doc_id_for, CorpusDocument)
    corpus_ids = {doc.doc_id for doc in corpus}
    qrels = {
        pair["id"]: {
            doc_id: 1.0
            for doc_id in sorted(question_qrels.get(pair["question"], set()))
            if doc_id in corpus_ids
        }
        for pair in selected
    }
    queries = tuple(QueryRecord(pair["id"], pair["question"]) for pair in selected)
    contract = DatasetContract(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        corpus_revision=(
            f"{DATASET_NAME}@{getattr(dataset, '_fingerprint', 'unknown')}"
            f"/{DATASET_CONFIG}/{DATASET_SPLIT}/first-{DATASET_SUBSET_SIZE}"
            f"/chunks-{CHUNK_SIZE}-{CHUNK_OVERLAP}/docid-dedup-first-v1"
        ),
        qrels_revision="ms-marco-is_selected-inherited-to-deduped-chunk-ids-v1",
        filter_policy=FilterPolicy(
            name="doc-id-prefix-dedup-first-occurrence",
            revision="v1",
        ),
    )

    cache_revisions = []
    for cache_file in getattr(dataset, "cache_files", []):
        filename = cache_file.get("filename", "")
        parts = Path(filename).parts
        if parts:
            cache_revisions.append(parts[-2] if len(parts) >= 2 else parts[-1])
    provenance = {
        "dataset": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "dataset_rows": len(dataset),
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "dataset_cache_revisions": sorted(set(cache_revisions)),
        "source_passages": len(docs),
        "source_chunks": len(chunks),
        "contract_corpus_documents": len(corpus),
        "duplicate_doc_ids_removed": len(chunks) - len(corpus),
        "query_pool_size": len(qa_pairs),
        "requested_query_count": len(selected_requested),
        "query_count": len(selected),
        "excluded_unjudged_query_count": len(excluded_unjudged_ids),
        "excluded_unjudged_query_ids": excluded_unjudged_ids,
        "query_selection": (
            "numpy.default_rng(42).choice(first 1000 answerable QA, N, replace=False)"
        ),
        "selected_query_ids": [query.query_id for query in queries],
        "selected_query_ids_fingerprint": _canonical_digest(contract.query_ids),
        "contract_fingerprint": contract.fingerprint(),
        "corpus_fingerprint": _canonical_digest([doc.as_dict() for doc in corpus]),
        "qrels_fingerprint": _canonical_digest(
            {query_id: qrels.get(query_id, {}) for query_id in contract.query_ids}
        ),
        "chunk_size_characters": CHUNK_SIZE,
        "overlap_characters": CHUNK_OVERLAP,
        "provider_calls": 0,
    }
    return contract, provenance


def _write_contract(output_dir: Path, contract: Any, provenance: Mapping[str, Any]) -> None:
    """Persist the exact shared contract without replacing an existing file."""

    payload = {
        "protocol": "controlled-retrieval-ablation-local-v1",
        "provenance": dict(provenance),
        "corpus_revision": contract.corpus_revision,
        "qrels_revision": contract.qrels_revision,
        "filter": contract.filter_policy.as_dict(),
        "corpus": [doc.as_dict() for doc in contract.corpus],
        "queries": [query.as_dict() for query in contract.queries],
        "qrels": {
            query_id: dict(contract.qrels.get(query_id, {}))
            for query_id in contract.query_ids
        },
    }
    with (output_dir / "contract.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_local_components(contract: Any, store_path: Path) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """Load the local dense store, sparse retriever and cross-encoder only."""

    from configs.config import EMBEDDING_MODEL, RERANKER_MODEL, RERANK_DEVICE
    from module1.data_preprocessing import get_embedding_model, load_vector_store
    from module3.evaluation import doc_id_for
    from scripts.retrieval_ablation import (
        DenseVectorStoreRetriever,
        LocalBM25Retriever,
        LocalCrossEncoderReranker,
        collect_package_metadata,
    )

    if not (store_path / "chroma.sqlite3").is_file():
        raise FileNotFoundError(f"local Chroma store not found: {store_path}")

    embedding_snapshot_dir, embedding_revision = _snapshot_revision(EMBEDDING_MODEL)
    reranker_snapshot_dir, reranker_revision = _snapshot_revision(RERANKER_MODEL)
    # Load the exact directory whose revision is recorded, not a mutable model
    # alias that might resolve to a different cached snapshot on another host.
    embedding_model = get_embedding_model(model_name=str(embedding_snapshot_dir))
    vectorstore = load_vector_store(embedding_model, str(store_path))
    stored = vectorstore.get(include=["documents"])
    stored_texts = list(stored.get("documents") or [])
    if not stored_texts:
        raise RuntimeError("local Chroma store returned no documents")

    expected = Counter(doc.text for doc in contract.corpus)
    # The contract is first-occurrence deduped, while the legacy store contains
    # duplicate rows.  Compare the store against the source contract only after
    # removing the same stable content IDs, not against a different corpus.
    stored_by_id = {}
    for text in stored_texts:
        doc_id = str(doc_id_for(type("StoredDocument", (), {"page_content": text})()))
        stored_by_id.setdefault(doc_id, text)
    actual = Counter(stored_by_id.values())
    if actual != expected:
        raise RuntimeError(
            "deduplicated dense-store text does not match the shared contract; "
            "refusing a mixed-corpus comparison"
        )

    dense = DenseVectorStoreRetriever(
        vectorstore,
        model_revision=f"{EMBEDDING_MODEL}@{embedding_revision}",
        document_id=doc_id_for,
    )
    bm25 = LocalBM25Retriever(
        contract.corpus,
        model_revision="bm25-okapi-local-v1:k1=1.5:b=0.75",
    )
    device = _reranker_device(RERANK_DEVICE)
    reranker = LocalCrossEncoderReranker(
        str(reranker_snapshot_dir),
        model_revision=f"{RERANKER_MODEL}@{reranker_revision}",
        device=device,
    )
    package_metadata = collect_package_metadata()
    package_metadata["experiment"] = {
        "runner": "scripts/run_retrieval_ablation_local.py",
        "protocol": "controlled-retrieval-ablation-local-v1",
        "store_path": _relative_or_name(store_path),
        "reranker_device": device,
        "dense_store_text_fingerprint": _canonical_digest(sorted(actual.items())),
        "contract_text_fingerprint": _canonical_digest(sorted(expected.items())),
    }
    return dense, bm25, reranker, package_metadata


def run(args: argparse.Namespace) -> int:
    _configure_offline_environment()
    from configs.config import RERANK_CANDIDATES, TOP_K
    from scripts.retrieval_ablation import AblationHarness

    output_dir = Path(args.out)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    started = time.perf_counter()
    contract, provenance = _build_contract(args.samples)
    store_path = (ROOT / args.store).resolve() if not Path(args.store).is_absolute() else Path(args.store)
    # Chroma may update SQLite metadata even for a similarity query.  A private
    # copy preserves the source store, including its unknown legacy manifest.
    with tempfile.TemporaryDirectory(prefix="ablation-", dir=ROOT / "data") as temp_dir:
        copied_store = Path(temp_dir) / "store"
        shutil.copytree(store_path, copied_store)
        dense, bm25, reranker, package_metadata = _load_local_components(contract, copied_store)
        package_metadata["experiment"]["store_path"] = _relative_or_name(store_path)
        harness = AblationHarness(
            contract,
            dense,
            bm25,
            reranker,
            top_k=TOP_K,
            candidate_pool_k=RERANK_CANDIDATES,
            package_metadata=package_metadata,
        )
        result = harness.run()
    provenance.update(
        {
            "runtime": platform.python_version(),
            "platform": platform.platform(),
            "store_path": _relative_or_name(store_path),
            "queried_private_store_copy": True,
            "store_manifest_present": (store_path / "_index_manifest.json").is_file(),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    )
    result.metadata["dataset_provenance"] = provenance
    result.metadata["execution"] = {
        "offline_only": True,
        "provider_calls": 0,
        "model_downloads": 0,
    }

    result.write_outputs(output_dir)
    _write_contract(output_dir, contract, provenance)
    for summary in result.summary:
        print(
            f"{summary['condition']}: n={summary['n_queries']} "
            f"hit={summary['mean_hit_rate']:.6f} "
            f"mrr@10={summary['mean_mrr_at_10']:.6f} "
            f"nDCG@10={summary['mean_ndcg_at_10']:.6f}"
        )
    print(f"wrote create-only outputs to {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--store", default="data/chroma_db_full")
    parser.add_argument("--out", default="outputs/retrieval-ablation-20261019")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:  # noqa: BLE001 - CLI reports a bounded local failure
        print(f"local retrieval ablation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI
    raise SystemExit(main())
