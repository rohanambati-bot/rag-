#!/usr/bin/env python3
"""
Train Set Processing — Shard-Based Approach
────────────────────────────────────────────
Reads the extracted train documents from _train_documents.jsonl,
splits them into shards, chunks + embeds each shard into its own
Chroma collection, then merges all shards into the final chroma_db_train.

Usage:
  # Process shard 0 (first 100K docs):
  python train_in_shards.py --shard 0

  # Process shard 3:
  python train_in_shards.py --shard 3

  # Process ALL shards sequentially:
  python train_in_shards.py --all

  # Merge all completed shards into final vector store:
  python train_in_shards.py --merge

  # Check status of all shards:
  python train_in_shards.py --status

Config:
  SHARD_SIZE = 100,000 documents per shard
  Total docs = ~808K → 9 shards (0-8)
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from configs.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, DATA_DIR,
    DATASET_NAME, DATASET_CONFIG, DATASET_REVISION, INDEX_REVISION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_shards")

# ─── Configuration ────────────────────────────────────────────────────────────
SHARD_SIZE = 100_000          # documents per shard
EMBED_BATCH_SIZE = 5000       # chunks to embed at once (memory-friendly)
TRAIN_DOCS_PATH = os.path.join(DATA_DIR, "_train_documents.jsonl")
SHARDS_DIR = os.path.join(DATA_DIR, "train_shards")
FINAL_TRAIN_DB = os.path.join(DATA_DIR, "chroma_db_train")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "shard_checkpoints")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def count_total_docs() -> int:
    """Count total lines in the train documents file."""
    count = 0
    with open(TRAIN_DOCS_PATH, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def get_num_shards() -> int:
    """Calculate how many shards we need."""
    total = count_total_docs()
    return (total + SHARD_SIZE - 1) // SHARD_SIZE


def load_shard_docs(shard_id: int) -> List[Document]:
    """Load documents for a specific shard from the JSONL file."""
    start_line = shard_id * SHARD_SIZE
    end_line = start_line + SHARD_SIZE

    docs: List[Document] = []
    with open(TRAIN_DOCS_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            if i >= end_line:
                break
            try:
                row = json.loads(line.strip())
                content = row.get("content", "").strip()
                if not content or len(content) < 30:
                    continue
                metadata = {k: v for k, v in row.items() if k != "content"}
                docs.append(Document(page_content=content, metadata=metadata))
            except json.JSONDecodeError:
                continue

    logger.info("Shard %d: loaded %d documents (lines %d–%d).", shard_id, len(docs), start_line, end_line - 1)
    return docs


def get_shard_checkpoint(shard_id: int) -> Dict[str, Any]:
    """Load checkpoint for a shard."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"shard_{shard_id}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"status": "pending", "chunks_done": 0, "total_chunks": 0}


def save_shard_checkpoint(shard_id: int, data: Dict[str, Any]):
    """Save checkpoint for a shard."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"shard_{shard_id}.json")
    from module1.data_preprocessing import atomic_write_json
    atomic_write_json(path, data)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load the embedding model."""
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ─── Core Processing ─────────────────────────────────────────────────────────

def process_shard(shard_id: int) -> None:
    """Publish one complete private stage; legacy partial shards are preserved."""
    from module1.data_preprocessing import build_lock_for
    shard_db_path = os.path.join(SHARDS_DIR, f"shard_{shard_id}")
    with build_lock_for(shard_db_path) as lock:
        _process_shard_locked(shard_id, shard_db_path, lock)


def _process_shard_locked(shard_id, shard_db_path, lock):
    from module1.data_preprocessing import create_staging_dir, run_isolated_index_builder
    from preprocess_full import _publish_vector_store
    checkpoint = get_shard_checkpoint(shard_id)
    if checkpoint["status"] == "done":
        if checkpoint["total_chunks"] and not os.path.isfile(os.path.join(shard_db_path, "chroma.sqlite3")):
            raise FileNotFoundError("Completed shard checkpoint has no persisted index")
        logger.info("Shard %d already completed (%d chunks). Skipping.", shard_id, checkpoint["total_chunks"])
        return

    # Step 1: Load documents for this shard
    logger.info("═══ Processing Shard %d ═══", shard_id)
    start_time = time.time()

    docs = load_shard_docs(shard_id)
    if not docs:
        logger.warning("Shard %d has no documents. Marking as done.", shard_id)
        save_shard_checkpoint(shard_id, {"status": "done", "chunks_done": 0, "total_chunks": 0})
        return

    # Step 2: Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    logger.info("Shard %d: %d docs → %d chunks.", shard_id, len(docs), len(chunks))

    stage = create_staging_dir(SHARDS_DIR, f".shard_{shard_id}.", shard_db_path, lock)
    chunks_done = run_isolated_index_builder(_embed_shard_stage, chunks, stage)
    _publish_vector_store(stage, shard_db_path, lock=lock)

    elapsed = time.time() - start_time
    save_shard_checkpoint(shard_id, {
        "status": "done",
        "chunks_done": chunks_done,
        "total_chunks": len(chunks),
        "elapsed_seconds": round(elapsed, 1),
    })
    logger.info("═══ Shard %d COMPLETE: %d chunks in %.1f min ═══", shard_id, chunks_done, elapsed / 60)


def _embed_shard_stage(chunks, stage):
    from module1.data_preprocessing import (
        corpus_identity, persistent_index_writer, write_index_manifest,
    )
    with persistent_index_writer(stage, get_embedding_model(), create=True) as store:
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            store.add_documents(chunks[start:start + EMBED_BATCH_SIZE])
        write_index_manifest(
            stage, num_chunks=len(chunks), dataset_name=DATASET_NAME,
            dataset_config=DATASET_CONFIG, dataset_revision=DATASET_REVISION,
            split="train", scope="train-shard", document_count=len(chunks),
            corpus_id=corpus_identity(chunks)["corpus_id"], index_revision=INDEX_REVISION,
        )
    return len(chunks)


def merge_shards() -> None:
    """Replace the train index from completed shards without touching live files."""
    from module1.data_preprocessing import (
        atomic_write_json, build_lock_for, create_staging_dir, run_isolated_index_builder,
    )
    from preprocess_full import _publish_vector_store
    with build_lock_for(FINAL_TRAIN_DB) as lock:
        stage = create_staging_dir(DATA_DIR, ".merged_train.", FINAL_TRAIN_DB, lock)
        total_merged = run_isolated_index_builder(_merge_shards_stage, stage)
        _publish_vector_store(stage, FINAL_TRAIN_DB, lock=lock)
        atomic_write_json(os.path.join(DATA_DIR, "_train_checkpoint.json"), {
            "phase": "done", "queries_done": 808731,
            "chunks_done": total_merged, "lines_done": total_merged,
        })


def _merge_shards_stage(stage):
    from module1.data_preprocessing import (
        corpus_identity, documents_from_vector_store, persistent_index_writer,
        write_index_manifest,
    )
    with persistent_index_writer(stage, get_embedding_model(), create=True) as final_vs:
        total = _merge_shards_into(final_vs)
        if not total:
            raise RuntimeError("No completed nonempty shards; refusing empty publication")
        documents = documents_from_vector_store(final_vs)
        identity = corpus_identity(documents)
        write_index_manifest(
            stage, num_chunks=total, dataset_name=DATASET_NAME,
            dataset_config=DATASET_CONFIG, dataset_revision=DATASET_REVISION,
            split="train", scope="full", document_count=total,
            corpus_id=identity["corpus_id"], index_revision=INDEX_REVISION,
        )
    return total


def _merge_shards_into(final_vs):
    from module1.data_preprocessing import load_vector_store
    logger.info("═══ MERGING SHARDS ═══")

    embedding_model = get_embedding_model()

    total_merged = 0
    shard_id = 0

    while True:
        checkpoint = get_shard_checkpoint(shard_id)
        if checkpoint["status"] == "pending" and shard_id > 0:
            break
        if checkpoint["status"] == "pending":
            raise RuntimeError("No completed shards found")
        if checkpoint["status"] != "done":
            raise RuntimeError(f"Shard {shard_id} is incomplete; refusing partial-corpus publication")

        shard_db_path = os.path.join(SHARDS_DIR, f"shard_{shard_id}")
        if checkpoint.get("total_chunks") == 0:
            shard_id += 1
            continue
        if not os.path.exists(shard_db_path):
            raise FileNotFoundError(f"Completed shard {shard_id} is missing; refusing partial corpus")

        # Load shard's vector store and get all documents
        shard_vs = load_vector_store(embedding_model, shard_db_path)

        # Get all documents from this shard
        shard_data = shard_vs.get(include=["documents", "metadatas", "embeddings"])
        if len(shard_data["documents"]) != checkpoint["total_chunks"]:
            raise RuntimeError(f"Shard {shard_id} count differs from its completion checkpoint")

        if shard_data["documents"]:
            docs_count = len(shard_data["documents"])
            logger.info("Shard %d: merging %d documents…", shard_id, docs_count)

            # Add to final store in batches
            for batch_start in range(0, docs_count, EMBED_BATCH_SIZE):
                batch_end = min(batch_start + EMBED_BATCH_SIZE, docs_count)

                batch_docs = shard_data["documents"][batch_start:batch_end]
                batch_metas = shard_data["metadatas"][batch_start:batch_end]
                batch_embeds = shard_data["embeddings"][batch_start:batch_end]

                final_vs._collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    embeddings=batch_embeds,
                    ids=[f"shard{shard_id}_{batch_start + i}" for i in range(len(batch_docs))],
                )

            total_merged += docs_count
            logger.info("  Merged shard %d. Total so far: %d", shard_id, total_merged)

        shard_id += 1

    logger.info("═══ MERGE COMPLETE: %d total vectors in chroma_db_train ═══", total_merged)

    return total_merged


def show_status() -> None:
    """Show the status of all shards."""
    print("\n" + "=" * 60)
    print("  SHARD PROCESSING STATUS")
    print("=" * 60)

    total_docs = count_total_docs()
    num_shards = get_num_shards()
    print(f"\n  Total train documents: {total_docs:,}")
    print(f"  Shard size: {SHARD_SIZE:,}")
    print(f"  Number of shards: {num_shards}")
    print()

    total_chunks_done = 0
    total_chunks_all = 0
    completed = 0

    print(f"  {'Shard':<8} {'Status':<14} {'Chunks':<20} {'Time':<10}")
    print(f"  {'─' * 8} {'─' * 14} {'─' * 20} {'─' * 10}")

    for shard_id in range(num_shards):
        cp = get_shard_checkpoint(shard_id)
        status = cp.get("status", "pending")
        chunks_done = cp.get("chunks_done", 0)
        total_chunks = cp.get("total_chunks", 0)
        elapsed = cp.get("elapsed_seconds", 0)

        total_chunks_done += chunks_done
        total_chunks_all += total_chunks

        if status == "done":
            completed += 1
            status_display = "✓ done"
        elif status == "in_progress":
            status_display = "⟳ in progress"
        else:
            status_display = "○ pending"

        time_display = f"{elapsed / 60:.1f} min" if elapsed else "—"
        chunks_display = f"{chunks_done:,} / {total_chunks:,}" if total_chunks else "—"

        print(f"  {shard_id:<8} {status_display:<14} {chunks_display:<20} {time_display:<10}")

    print(f"\n  Summary: {completed}/{num_shards} shards done, {total_chunks_done:,} chunks embedded")
    print("=" * 60 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Process MS MARCO train set in shards")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--shard", type=int, help="Process a specific shard (0-indexed)")
    group.add_argument("--all", action="store_true", help="Process ALL shards sequentially")
    group.add_argument("--merge", action="store_true", help="Merge all completed shards")
    group.add_argument("--status", action="store_true", help="Show status of all shards")
    group.add_argument("--count", action="store_true", help="Count total documents")

    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.count:
        total = count_total_docs()
        num_shards = get_num_shards()
        print(f"\nTotal documents: {total:,}")
        print(f"Shards needed: {num_shards} (at {SHARD_SIZE:,} docs/shard)\n")
    elif args.shard is not None:
        process_shard(args.shard)
    elif args.all:
        num_shards = get_num_shards()
        logger.info("Processing all %d shards sequentially…", num_shards)
        for shard_id in range(num_shards):
            process_shard(shard_id)
        logger.info("All shards processed. Run --merge to combine them.")
    elif args.merge:
        merge_shards()


if __name__ == "__main__":
    main()
