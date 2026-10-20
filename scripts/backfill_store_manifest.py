#!/usr/bin/env python3
"""Backfill an index manifest onto a pre-manifest-era Chroma store.

Stores built before the manifest system existed carry no `_index_manifest.json`,
so the (correctly strict) fail-closed loader refuses them. This script rebuilds
the manifest from the *ground truth*: it reconstructs the corpus exactly the
main pipeline does, reads the store's own documents back, and only writes a
manifest if the two identities agree. On any disagreement it refuses to write —
the manifest must never bless a store whose provenance cannot be proven.

Usage: python scripts/backfill_store_manifest.py [store_dir] [num_chunks]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATASET_SUBSET_SIZE,
)


def main() -> None:
    from module1.data_preprocessing import (
        chunk_documents,
        corpus_identity,
        documents_from_vector_store,
        get_embedding_model,
        load_msmarco_documents,
        load_vector_store,
        write_index_manifest,
    )
    from configs.config import DATA_DIR

    store_dir = os.path.realpath(
        sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_DIR, "chroma_db_full")
    )
    expected_queries = DATASET_SUBSET_SIZE

    embedding_model = get_embedding_model()
    store = load_vector_store(
        embedding_model, persist_dir=store_dir, allow_unmanifested_legacy=True
    )
    stored_docs = documents_from_vector_store(store)
    store_identity = corpus_identity(stored_docs)

    docs = load_msmarco_documents(max_samples=expected_queries)
    chunks = chunk_documents(docs)
    rebuilt_identity = corpus_identity(chunks)

    if store_identity != rebuilt_identity:
        raise SystemExit(
            "REFUSED to write manifest: reconstructed corpus identity "
            f"{rebuilt_identity} != store identity {store_identity} — the manifest "
            "would bless a corpus it cannot prove. Rebuild the store instead."
        )

    document_count = len({d.metadata.get("query", "") for d in chunks})
    write_index_manifest(
        store_dir,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        num_chunks=len(stored_docs),
        scope="dev",
        query_count=expected_queries,
        document_count=document_count,
        corpus_id=store_identity["corpus_id"],
    )
    print(f"manifest written to {store_dir}/_index_manifest.json")
    print(f"  identity={store_identity['corpus_id']}")
    print(f"  chunks={store_identity['chunk_count']} query_count={expected_queries}")


if __name__ == "__main__":
    main()
