"""
MS MARCO v2.1 TRAIN SET Preprocessing
Processes the full 808K training queries (~4M passages)

REQUIREMENTS:
  - 16 GB RAM (will use ~12 GB)
  - Stable power for ~14 hours
  - ~50 GB free disk space

Run:
    python preprocess_train.py
"""

import os
import sys
import json
import time
import logging
import gc
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("preprocess_train")

sys.path.insert(0, str(Path(__file__).parent))

# Force offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from configs.config import (  # noqa: E402
    DATASET_NAME, DATASET_CONFIG,
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, DATA_DIR,
)

BATCH_SIZE = 2000          # Smaller batches to save RAM
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "chroma_db_train")
CHECKPOINT_FILE = os.path.join(DATA_DIR, "_train_checkpoint.json")
DOCS_FILE = os.path.join(DATA_DIR, "_train_documents.jsonl")
QA_FILE = os.path.join(DATA_DIR, "_train_qa_pairs.jsonl")


def _best_device() -> str:
    """Pick the fastest available torch device: CUDA → MPS (Apple Silicon) → CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"phase": "start", "queries_done": 0, "chunks_done": 0, "lines_done": 0}


def save_checkpoint(data):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)


def phase_extract():
    """Phase 1: Extract passages from train split."""
    checkpoint = load_checkpoint()
    start_from = checkpoint.get("queries_done", 0)

    if os.path.exists(DOCS_FILE) and start_from > 0:
        # Count existing docs
        with open(DOCS_FILE) as f:
            existing_count = sum(1 for _ in f)
        if existing_count > 3000000:
            print(f"       Extraction already done ({existing_count:,} passages). Skipping.")
            return existing_count
        print(f"       Resuming extraction from query {start_from:,} ({existing_count:,} passages so far)")

    print("\n[1/3] Loading MS MARCO v2.1 TRAIN split...")
    from datasets import load_dataset
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    total = len(dataset)
    print(f"       Loaded {total:,} queries")

    print(f"\n[2/3] Extracting passages (starting at {start_from:,})...")

    seen_passages = set()
    docs_count = 0

    # Load existing seen passages if resuming
    if start_from > 0 and os.path.exists(DOCS_FILE):
        print("       Loading existing passages for dedup...")
        with open(DOCS_FILE) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    seen_passages.add(d["content"][:150])
                    docs_count += 1
                except Exception:
                    pass
        print(f"       Loaded {docs_count:,} existing passages")
        # Free memory
        gc.collect()

    mode = "a" if start_from > 0 else "w"
    t0 = time.time()

    with open(DOCS_FILE, mode) as df, open(QA_FILE, mode) as qf:
        for idx in range(start_from, total):
            row = dataset[idx]

            answers = row.get("answers", [])
            if not answers or answers == ["No Answer Present."]:
                continue

            query = row.get("query", "").strip()
            if not query:
                continue

            passages_data = row.get("passages", {})
            passage_texts = passages_data.get("passage_text", [])
            is_selected = passages_data.get("is_selected", [])
            answer = answers[0]

            # Write QA pair
            qf.write(json.dumps({
                "question": query,
                "ground_truth": answer,
                "query_type": row.get("query_type", "unknown"),
            }) + "\n")

            # Write passages
            for p_idx, pt in enumerate(passage_texts):
                pt = pt.strip()
                if not pt or len(pt) < 30:
                    continue
                key = pt[:150]
                if key in seen_passages:
                    continue
                seen_passages.add(key)
                selected = is_selected[p_idx] if p_idx < len(is_selected) else 0

                df.write(json.dumps({
                    "content": pt,
                    "is_selected": selected,
                }) + "\n")
                docs_count += 1

            # Progress + checkpoint every 10K
            if (idx + 1) % 10000 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1 - start_from) / max(elapsed, 1)
                remaining = (total - idx - 1) / max(rate, 0.1)
                print(
                    f"       {idx+1:,}/{total:,} queries — "
                    f"{docs_count:,} passages — "
                    f"{rate:.0f} q/sec — ETA: {remaining/60:.0f} min"
                )
                save_checkpoint({"phase": "extracting", "queries_done": idx + 1, "chunks_done": 0, "lines_done": 0})

                # Free memory periodically
                if len(seen_passages) > 5000000:
                    seen_passages = set()  # Reset dedup (some dups ok)
                    gc.collect()

    save_checkpoint({"phase": "extracted", "queries_done": total, "chunks_done": 0, "lines_done": 0})
    print(f"       Done: {docs_count:,} passages from {total:,} queries")

    # Free dataset from memory
    del dataset
    del seen_passages
    gc.collect()

    return docs_count


def phase_embed():
    """Phase 2: Chunk + embed + index."""
    checkpoint = load_checkpoint()
    start_line = checkpoint.get("lines_done", 0)
    chunks_done = checkpoint.get("chunks_done", 0)

    # Count total lines
    print("\n[3/3] Chunking and embedding...")
    total_lines = 0
    with open(DOCS_FILE) as f:
        for _ in f:
            total_lines += 1
    print(f"       Total passages to process: {total_lines:,}")
    print(f"       Resuming from line: {start_line:,} ({chunks_done:,} chunks already done)")
    print(f"       Batch size: {BATCH_SIZE}")
    print(f"       Embedding model: {EMBEDDING_MODEL}")

    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    # Embed on the fastest available local device. On Apple Silicon, MPS is ~2.4x
    # faster than CPU for all-MiniLM-L6-v2, which matters a lot across ~4M passages.
    # Override with EMBEDDING_DEVICE=cpu if MPS ever misbehaves.
    device = os.getenv("EMBEDDING_DEVICE") or _best_device()
    print(f"       Embedding device: {device}")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Setup vector store
    import shutil
    if start_line == 0:
        if os.path.exists(VECTOR_STORE_DIR):
            shutil.rmtree(VECTOR_STORE_DIR, ignore_errors=True)
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    vectorstore = None
    batch_docs = []
    batch_num = chunks_done // BATCH_SIZE
    total_chunks = chunks_done
    t0 = time.time()

    with open(DOCS_FILE) as f:
        for line_num, line in enumerate(f):
            if line_num < start_line:
                continue

            try:
                data = json.loads(line)
            except Exception:
                continue

            doc = Document(
                page_content=data["content"],
                metadata={"is_selected": data.get("is_selected", 0)},
            )
            chunks = splitter.split_documents([doc])
            batch_docs.extend(chunks)

            if len(batch_docs) >= BATCH_SIZE:
                batch_num += 1
                try:
                    if vectorstore is None:
                        vectorstore = Chroma.from_documents(
                            documents=batch_docs,
                            embedding=embedding_model,
                            persist_directory=VECTOR_STORE_DIR,
                        )
                    else:
                        vectorstore.add_documents(batch_docs)
                except Exception as exc:
                    print(f"       ERROR at batch {batch_num}: {exc}")
                    print("       Saving checkpoint and exiting. Re-run to resume.")
                    save_checkpoint({
                        "phase": "embedding",
                        "queries_done": checkpoint.get("queries_done", 0),
                        "chunks_done": total_chunks,
                        "lines_done": line_num,
                    })
                    return total_chunks

                total_chunks += len(batch_docs)
                batch_docs = []
                gc.collect()  # Free memory

                elapsed = time.time() - t0
                rate = (total_chunks - chunks_done) / max(elapsed, 1)
                remaining_lines = total_lines - line_num
                eta = remaining_lines / max(rate, 0.1)

                print(
                    f"       Batch {batch_num}: {total_chunks:,} chunks — "
                    f"{rate:.0f}/sec — ETA: {eta/60:.0f} min"
                )

                # Checkpoint every 10 batches
                if batch_num % 10 == 0:
                    save_checkpoint({
                        "phase": "embedding",
                        "queries_done": checkpoint.get("queries_done", 0),
                        "chunks_done": total_chunks,
                        "lines_done": line_num + 1,
                    })

    # Final batch
    if batch_docs and vectorstore:
        vectorstore.add_documents(batch_docs)
        total_chunks += len(batch_docs)
    elif batch_docs:
        vectorstore = Chroma.from_documents(
            documents=batch_docs,
            embedding=embedding_model,
            persist_directory=VECTOR_STORE_DIR,
        )
        total_chunks += len(batch_docs)

    try:
        vectorstore.persist()
    except Exception:
        pass

    save_checkpoint({"phase": "done", "chunks_done": total_chunks, "lines_done": total_lines})
    print(f"       Done: {total_chunks:,} chunks indexed")
    return total_chunks


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 70)
    print("  MS MARCO v2.1 TRAIN SET — FULL PREPROCESSING")
    print("  808,731 queries · ~4M passages · ~14 hours")
    print("  RESUMABLE — safe to interrupt and re-run")
    print("=" * 70)

    checkpoint = load_checkpoint()

    t_start = time.time()

    # Phase 1: Extract
    if checkpoint.get("phase") not in ("extracted", "embedding", "done"):
        phase_extract()
    else:
        print("\n  Passage extraction already complete. Skipping to embedding.")

    # Phase 2: Embed
    if checkpoint.get("phase") != "done":
        total_chunks = phase_embed()
    else:
        total_chunks = checkpoint.get("chunks_done", 0)
        print(f"\n  Already done! {total_chunks:,} chunks indexed.")

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("  TRAIN SET PREPROCESSING COMPLETE")
    print(f"  Total time: {total_time/3600:.1f} hours")
    print(f"  Chunks indexed: {total_chunks:,}")
    print(f"  Vector store: {VECTOR_STORE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
