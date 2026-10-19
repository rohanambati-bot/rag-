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
import shutil
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


def _atomic_write_json(path, data) -> None:
    """Use the shared same-directory atomic JSON writer."""
    from module1.data_preprocessing import atomic_write_json
    atomic_write_json(path, data)


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
    _atomic_write_json(CHECKPOINT_FILE, data)


def _recover_previous_vector_store(final_dir: str) -> None:
    """Restore or discard a backup left by an interrupted publication."""
    final_dir = os.path.abspath(final_dir)
    backup_dir = f"{final_dir}.previous"
    if not os.path.exists(backup_dir):
        return
    if os.path.exists(final_dir):
        shutil.rmtree(backup_dir)
    else:
        os.replace(backup_dir, final_dir)


def _create_staging_dir(data_dir: str, prefix: str, target_dir: str, lock=None) -> str:
    """Create a staging directory with ownership metadata for safe cleanup."""
    from module1.data_preprocessing import create_staging_dir
    return create_staging_dir(data_dir, prefix, target_dir, lock=lock)


def _cleanup_staged_vector_stores(
    data_dir: str,
    *,
    target_dir: str | None = None,
    prefix: str = ".chroma_db_full.",
    lock=None,
) -> list:
    """Remove only marked stale stages while the target build lock is held.

    A direct caller briefly acquires the same lock; if another process is
    actively building, cleanup returns without touching any staging directory.
    Unmarked directories are intentionally left in place because their ownership
    cannot be established safely.
    """
    from module1.data_preprocessing import (
        BuildLockError,
        build_lock_for,
        cleanup_stale_staging_dirs,
    )

    target_dir = target_dir or os.path.join(data_dir, "chroma_db_full")
    if lock is None:
        try:
            with build_lock_for(target_dir, timeout=0) as cleanup_lock:
                return _cleanup_staged_vector_stores(
                    data_dir,
                    target_dir=target_dir,
                    prefix=prefix,
                    lock=cleanup_lock,
                )
        except BuildLockError:
            logger.info("Skipping staging cleanup while %s is actively building.", target_dir)
            return []

    return cleanup_stale_staging_dirs(
        data_dir,
        prefix=prefix,
        target_path=target_dir,
        active_lock_token=getattr(lock, "_token", None),
    )


def _publish_vector_store_locked(staged_dir: str, final_dir: str) -> None:
    from module1.data_preprocessing import index_publication_guard
    with index_publication_guard(final_dir):
        _publish_vector_store_without_readers(staged_dir, final_dir)


def _publish_vector_store_without_readers(staged_dir: str, final_dir: str) -> None:
    """Publish a completed store without discarding the last good build.

    Chroma writes a directory rather than a single file, so replacing a populated
    store takes two directory renames.  Keep the previous directory as a backup
    during that small window and restore it if the staged directory cannot be
    promoted.  This means an interrupted or failed rebuild leaves the previous
    store loadable and a later run can try again.
    """
    staged_dir = os.path.abspath(staged_dir)
    final_dir = os.path.abspath(final_dir)
    if staged_dir == final_dir:
        raise ValueError("staged and final vector-store paths must differ")
    if not os.path.isdir(staged_dir):
        raise FileNotFoundError(f"Staged vector store not found: {staged_dir}")

    os.makedirs(os.path.dirname(final_dir), exist_ok=True)
    backup_dir = f"{final_dir}.previous"
    _recover_previous_vector_store(final_dir)

    moved_previous = False
    try:
        if os.path.exists(final_dir):
            os.replace(final_dir, backup_dir)
            moved_previous = True
        os.replace(staged_dir, final_dir)
    except Exception:
        if moved_previous and not os.path.exists(final_dir) and os.path.exists(backup_dir):
            os.replace(backup_dir, final_dir)
        raise

    # The new store is now live.  A stale backup is no longer needed, but a
    # cleanup failure must not turn a successful publish into a failed build.
    if os.path.exists(backup_dir):
        try:
            shutil.rmtree(backup_dir)
        except OSError as exc:
            logger.warning("Could not remove previous vector store %s: %s", backup_dir, exc)

    # The marker is useful while building but must not become part of the live
    # Chroma directory. A cleanup failure is harmless after publication.
    try:
        os.unlink(os.path.join(final_dir, ".build-staging.json"))
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Could not remove staging marker from %s: %s", final_dir, exc)


def _publish_vector_store(staged_dir: str, final_dir: str, *, lock=None) -> None:
    """Publish a store while holding the cross-process lock for ``final_dir``."""
    if lock is not None:
        _publish_vector_store_locked(staged_dir, final_dir)
        return
    from module1.data_preprocessing import build_lock_for
    with build_lock_for(final_dir):
        _publish_vector_store_locked(staged_dir, final_dir)


def main():
    """Run the full rebuild under a bounded cross-process builder lock."""
    from module1.data_preprocessing import build_lock_for

    final_vector_dir = os.path.join(DATA_DIR, "chroma_db_full")
    with build_lock_for(final_vector_dir) as build_lock:
        return _run_locked(final_vector_dir, build_lock)


def _run_locked(final_vector_dir: str, build_lock):
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
    print(f"       Chunk size: {CHUNK_SIZE} characters, overlap: {CHUNK_OVERLAP} characters")
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

    # Build beside the live store.  The existing full-set store remains available
    # until the new Chroma directory and its manifest are complete and published.
    _cleanup_staged_vector_stores(
        DATA_DIR,
        target_dir=final_vector_dir,
        lock=build_lock,
    )
    # Recovery also renames/removes directories: use the same reader barrier.
    from module1.data_preprocessing import index_publication_guard
    with index_publication_guard(final_vector_dir):
        _recover_previous_vector_store(final_vector_dir)
    vector_store_dir = _create_staging_dir(
        DATA_DIR,
        ".chroma_db_full.",
        final_vector_dir,
        lock=build_lock,
    )

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
    if batch_num == 0 and not batch_docs:
        shutil.rmtree(vector_store_dir, ignore_errors=True)
        raise RuntimeError("No valid passages were produced; refusing to publish an empty index")
    try:
        vectorstore.persist()
    except AttributeError:
        pass  # newer Chroma auto-persists and may not expose persist()

    # Record build parameters so load_vector_store can detect a stale/mismatched index.
    from module1.data_preprocessing import write_index_manifest
    write_index_manifest(vector_store_dir, num_chunks=total_chunks)

    manifest_path = os.path.join(vector_store_dir, "_index_manifest.json")
    try:
        with open(manifest_path) as manifest_file:
            json.load(manifest_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Full validation index manifest was not written correctly: {manifest_path}"
        ) from exc

    _publish_vector_store(vector_store_dir, final_vector_dir, lock=build_lock)
    vector_store_dir = final_vector_dir

    print(f"       Done: {total_chunks:,} chunks indexed in {(time.time()-t2)/60:.1f} min")

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

    _atomic_write_json(SYNTHETIC_DATA_PATH, synthetic)
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
