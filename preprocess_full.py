"""
Full MS MARCO v2.1 Preprocessing — Processes the ENTIRE validation set.
This will take several hours. Progress is saved in checkpoints.

Run:
    python preprocess_full.py
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("preprocess_full")

sys.path.insert(0, str(Path(__file__).parent))

from configs.config import (  # noqa: E402
    DATASET_NAME, DATASET_CONFIG, DATASET_SPLIT,
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL,
    VECTOR_STORE_DIR, SYNTHETIC_DATA_PATH, DATA_DIR,
    NUM_SYNTHETIC_SAMPLES,
)

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 5000          # Process documents in batches of 5000
CHECKPOINT_FILE = os.path.join(DATA_DIR, "_preprocess_checkpoint.json")


def selected_context(passage_texts: list, is_selected: list) -> str:
    """
    Join the passages the MS MARCO annotator marked relevant for a query.

    Written into the synthetic QA pairs as ``context`` so token-overlap metrics
    have a non-empty ground-truth context (they previously stored the empty
    string, leaving answer_recall / IoU unscoreable for those pairs).
    """
    return " ".join(
        passage_texts[i].strip()
        for i in range(len(passage_texts))
        if i < len(is_selected) and is_selected[i] == 1 and passage_texts[i].strip()
    )


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"queries_processed": 0, "docs_saved": 0, "phase": "loading"}


def save_checkpoint(data):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 70)
    print("  FULL MS MARCO v2.1 PREPROCESSING")
    print("  This will process ALL 101,093 validation queries (~1M passages)")
    print("  Estimated time: 2-6 hours depending on your machine")
    print("=" * 70)

    checkpoint = load_checkpoint()
    start_from = checkpoint.get("queries_processed", 0)

    if start_from > 0:
        print(f"\n  Resuming from query {start_from} (checkpoint found)")

    # ── Phase 1: Load dataset ─────────────────────────────────────────────────
    print("\n[1/4] Loading MS MARCO v2.1 validation split...")
    t0 = time.time()

    from datasets import load_dataset
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    total_queries = len(dataset)
    print(f"       Loaded {total_queries:,} queries in {time.time()-t0:.1f}s")

    # ── Phase 2: Extract passages ─────────────────────────────────────────────
    print(f"\n[2/4] Extracting passages from all queries (starting at {start_from})...")

    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs_file = os.path.join(DATA_DIR, "_all_documents.jsonl")
    qa_file = os.path.join(DATA_DIR, "_all_qa_pairs.jsonl")

    seen_passages = set()
    docs_count = checkpoint.get("docs_saved", 0)
    qa_count = 0

    # Count existing if resuming
    if start_from > 0 and os.path.exists(docs_file):
        with open(docs_file, "r") as f:
            for line in f:
                seen_passages.add(json.loads(line)["content"][:200])

    t1 = time.time()

    # Open files in append mode for resuming
    mode = "a" if start_from > 0 else "w"
    with open(docs_file, mode) as df, open(qa_file, mode) as qf:
        for idx in range(start_from, total_queries):
            row = dataset[idx]

            # Skip unanswerable
            answers = row.get("answers", [])
            if not answers or answers == ["No Answer Present."]:
                continue

            query = row.get("query", "").strip()
            if not query:
                continue

            query_type = row.get("query_type", "unknown")
            passages_data = row.get("passages", {})
            passage_texts = passages_data.get("passage_text", [])
            is_selected = passages_data.get("is_selected", [])
            urls = passages_data.get("url", [])
            answer = answers[0]

            # Write QA pair, including the human-selected context so downstream
            # token-overlap metrics can score against a real ground-truth context.
            qf.write(json.dumps({
                "id": f"msmarco_{idx}",
                "question": query,
                "ground_truth": answer,
                "query_type": query_type,
                "context": selected_context(passage_texts, is_selected),
            }) + "\n")
            qa_count += 1

            # Write passages
            for p_idx, passage_text in enumerate(passage_texts):
                passage_text = passage_text.strip()
                if not passage_text or len(passage_text) < 30:
                    continue

                key = passage_text[:200]
                if key in seen_passages:
                    continue
                seen_passages.add(key)

                selected = is_selected[p_idx] if p_idx < len(is_selected) else 0
                url = urls[p_idx] if p_idx < len(urls) else ""

                df.write(json.dumps({
                    "content": passage_text,
                    "query": query,
                    "answer": answer,
                    "query_type": query_type,
                    "is_selected": selected,
                    "url": url,
                }) + "\n")
                docs_count += 1

            # Progress every 5000 queries
            if (idx + 1) % 5000 == 0:
                elapsed = time.time() - t1
                rate = (idx + 1 - start_from) / elapsed
                remaining = (total_queries - idx - 1) / max(rate, 0.1)
                print(
                    f"       Progress: {idx+1:,}/{total_queries:,} queries "
                    f"({docs_count:,} passages) — "
                    f"{rate:.0f} queries/sec — "
                    f"ETA: {remaining/60:.0f} min"
                )
                save_checkpoint({
                    "queries_processed": idx + 1,
                    "docs_saved": docs_count,
                    "phase": "extracting",
                })

    print(f"       Done: {docs_count:,} unique passages from {total_queries:,} queries")
    print(f"       QA pairs: {qa_count:,}")
    save_checkpoint({"queries_processed": total_queries, "docs_saved": docs_count, "phase": "chunking"})

    # ── Phase 3: Chunk + Embed + Index ────────────────────────────────────────
    print(f"\n[3/4] Chunking and embedding {docs_count:,} passages...")
    print(f"       Chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")
    print(f"       Embedding model: {EMBEDDING_MODEL}")
    print("       This is the slow part — embedding on CPU...")

    # Force offline mode so it doesn't need internet
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Clear old vector store
    import shutil
    new_vector_dir = os.path.join(DATA_DIR, "chroma_db_full")
    if os.path.exists(new_vector_dir):
        try:
            shutil.rmtree(new_vector_dir)
        except Exception:
            pass
    os.makedirs(new_vector_dir, exist_ok=True)
    # Override for this run
    vector_store_dir = new_vector_dir

    # Process in batches to avoid OOM
    total_chunks = 0
    batch_docs = []
    batch_num = 0

    t2 = time.time()
    with open(docs_file, "r") as f:
        for line_num, line in enumerate(f):
            data = json.loads(line)
            doc = Document(
                page_content=data["content"],
                metadata={
                    "query": data.get("query", ""),
                    "answers": [data.get("answer", "")],
                    "query_type": data.get("query_type", ""),
                    "is_selected": data.get("is_selected", 0),
                    "source": "ms_marco_v2.1",
                },
            )
            chunks = splitter.split_documents([doc])
            batch_docs.extend(chunks)

            # Index every BATCH_SIZE chunks
            if len(batch_docs) >= BATCH_SIZE:
                batch_num += 1
                if batch_num == 1:
                    # First batch: create the vectorstore
                    vectorstore = Chroma.from_documents(
                        documents=batch_docs,
                        embedding=embedding_model,
                        persist_directory=vector_store_dir,
                    )
                else:
                    # Subsequent batches: add to existing
                    vectorstore.add_documents(batch_docs)

                total_chunks += len(batch_docs)
                elapsed = time.time() - t2
                rate = total_chunks / elapsed
                remaining_docs = docs_count - line_num
                eta = remaining_docs / max(rate, 0.1)

                print(
                    f"       Batch {batch_num}: {total_chunks:,} chunks indexed — "
                    f"{rate:.0f} chunks/sec — ETA: {eta/60:.0f} min"
                )
                batch_docs = []

    # Final batch
    if batch_docs:
        if batch_num == 0:
            vectorstore = Chroma.from_documents(
                documents=batch_docs,
                embedding=embedding_model,
                persist_directory=vector_store_dir,
            )
        else:
            vectorstore.add_documents(batch_docs)
        total_chunks += len(batch_docs)

    # Persist
    try:
        vectorstore.persist()
    except Exception:
        pass  # newer Chroma auto-persists

    # Record build parameters so load_vector_store can detect a stale/mismatched index.
    from module1.data_preprocessing import write_index_manifest
    write_index_manifest(vector_store_dir, num_chunks=total_chunks)

    print(f"       Done: {total_chunks:,} chunks indexed in {(time.time()-t2)/60:.1f} min")

    # Replace only the full-set store; other stores (e.g. the train-split store
    # built by `train_in_shards.py`) are never touched. The earlier code rebuilt
    # into a fresh dir and then `rmtree`+`rename`d it into VECTOR_STORE_DIR,
    # which by default pointed at the train store — silently destroying it.
    old_full_db = vector_store_dir
    try:
        if os.path.exists(old_full_db):
            shutil.rmtree(old_full_db)
    except Exception:
        print(f"       Warning: Could not remove old {old_full_db}, using new path instead.")

    print(f"       Vector store at: {vector_store_dir}")
    if VECTOR_STORE_DIR != vector_store_dir:
        print(
            "       NOTE: VECTOR_STORE_DIR is set to a different path — "
            "point `--store` at this new store to evaluate it."
        )

    # ── Phase 4: Synthetic dataset ────────────────────────────────────────────
    print(f"\n[4/4] Creating synthetic evaluation dataset ({NUM_SYNTHETIC_SAMPLES} samples)...")

    qa_pairs = []
    with open(qa_file, "r") as f:
        for i, line in enumerate(f):
            if i >= NUM_SYNTHETIC_SAMPLES:
                break
            qa_pairs.append(json.loads(line))

    synthetic = []
    for pair in qa_pairs:
        synthetic.append({
            "id": pair["id"],
            "question": pair["question"],
            "ground_truth": pair["ground_truth"],
            "query_type": pair.get("query_type", "unknown"),
            # Present in checkpoints written by phase 2; older checkpoint files
            # may omit it, in which case the empty string preserves prior format.
            "context": pair.get("context", ""),
        })

    os.makedirs(os.path.dirname(SYNTHETIC_DATA_PATH), exist_ok=True)
    with open(SYNTHETIC_DATA_PATH, "w") as f:
        json.dump(synthetic, f, indent=2)
    print(f"       Saved {len(synthetic)} synthetic Q&A pairs")

    # ── Done ──────────────────────────────────────────────────────────────────
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("  PREPROCESSING COMPLETE")
    print(f"  Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"  Queries processed: {total_queries:,}")
    print(f"  Unique passages: {docs_count:,}")
    print(f"  Chunks indexed: {total_chunks:,}")
    print(f"  Synthetic Q&A: {len(synthetic)}")
    print(f"  Vector store: {vector_store_dir}")
    print("=" * 70)

    # Cleanup checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    save_checkpoint({"phase": "done", "total_chunks": total_chunks, "total_passages": docs_count})


if __name__ == "__main__":
    main()
