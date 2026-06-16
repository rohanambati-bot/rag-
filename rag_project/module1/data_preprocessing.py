# module1/data_preprocessing.py
"""
Module 1 — Data Preprocessing & Synthetic Data Generation
─────────────────────────────────────────────────────────
Steps:
  1. Load MS MARCO v2.1 (HuggingFace datasets) — passage ranking QA
  2. Chunk documents with RecursiveCharacterTextSplitter
  3. Generate dense embeddings (all-MiniLM-L6-v2)
  4. Index into Chroma vector store
  5. Generate synthetic Q&A queries via LLM (simple / paraphrased / ambiguous)
  6. Persist everything for downstream modules

Dataset: MS MARCO v2.1
  - Real Bing search queries
  - ~10 passages per query with is_selected labels
  - Human-written answers
  - 800K+ samples (we use a configurable subset)
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional

from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import (
    DATASET_NAME, DATASET_CONFIG, DATASET_SPLIT, DATASET_SUBSET_SIZE,
    SQUAD_SPLIT, SQUAD_SUBSET_SIZE,
    CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, EMBEDDING_DEVICE, VECTOR_STORE_DIR,
    SYNTHETIC_DATA_PATH, SYNTHETIC_BATCH_SIZE,
    NUM_SYNTHETIC_SAMPLES, MISTRAL_API_KEY, LLM_MODEL,
    SYNTHETIC_QUERY_TYPES, DATA_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATASET LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_msmarco_documents(
    split: str = DATASET_SPLIT,
    max_samples: int = DATASET_SUBSET_SIZE,
) -> List[Document]:
    """
    Load MS MARCO v2.1 and convert to LangChain Documents.

    MS MARCO structure per row:
      - query (str): the Bing search question
      - passages (dict):
          - passage_text (list[str]): ~10 passages
          - is_selected (list[int]): 1 if passage contains the answer
          - url (list[str]): source URLs
      - answers (list[str]): human-written answers
      - query_type (str): numeric, entity, description, location, person

    We extract:
      - Each selected passage → Document (page_content)
      - Metadata: query, answer, query_type, url
    """
    logger.info("Loading MS MARCO v2.1 (%s split)…", split)
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)

    docs: List[Document] = []
    seen_passages: set = set()
    count = 0

    for row in dataset:
        # Skip rows with no answers
        answers = row.get("answers", [])
        if not answers or answers == ["No Answer Present."]:
            continue

        query = row.get("query", "").strip()
        query_type = row.get("query_type", "unknown")
        passages_data = row.get("passages", {})
        passage_texts = passages_data.get("passage_text", [])
        is_selected = passages_data.get("is_selected", [])
        urls = passages_data.get("url", [])

        # Get the best answer
        answer = answers[0] if answers else ""

        # Extract ALL passages for this query (both selected and unselected)
        # This gives us a realistic retrieval corpus
        for idx, passage_text in enumerate(passage_texts):
            passage_text = passage_text.strip()
            if not passage_text or len(passage_text) < 30:
                continue

            # Deduplicate
            passage_key = passage_text[:200]
            if passage_key in seen_passages:
                continue
            seen_passages.add(passage_key)

            selected = is_selected[idx] if idx < len(is_selected) else 0
            url = urls[idx] if idx < len(urls) else ""

            doc = Document(
                page_content=passage_text,
                metadata={
                    "query": query,
                    "answers": [answer],
                    "query_type": query_type,
                    "is_selected": selected,
                    "url": url,
                    "source": "ms_marco_v2.1",
                },
            )
            docs.append(doc)

        count += 1
        if count >= max_samples:
            break

    logger.info(
        "Loaded %d passages from %d MS MARCO queries (selected + unselected).",
        len(docs), count,
    )
    return docs


def load_msmarco_qa_pairs(
    split: str = DATASET_SPLIT,
    max_samples: int = DATASET_SUBSET_SIZE,
) -> List[Dict[str, Any]]:
    """
    Load MS MARCO as Q&A pairs for evaluation.
    Returns list of dicts with: question, ground_truth, query_type, context.
    """
    logger.info("Loading MS MARCO Q&A pairs (%s split, max %d)…", split, max_samples)
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)

    qa_pairs: List[Dict[str, Any]] = []
    count = 0

    for row in dataset:
        answers = row.get("answers", [])
        if not answers or answers == ["No Answer Present."]:
            continue

        query = row.get("query", "").strip()
        if not query:
            continue

        passages_data = row.get("passages", {})
        passage_texts = passages_data.get("passage_text", [])
        is_selected = passages_data.get("is_selected", [])

        # Get selected passages as ground-truth context
        selected_passages = [
            passage_texts[i]
            for i in range(len(passage_texts))
            if i < len(is_selected) and is_selected[i] == 1
        ]
        context = " ".join(selected_passages) if selected_passages else ""

        qa_pairs.append({
            "id": f"msmarco_{count}",
            "question": query,
            "ground_truth": answers[0],
            "query_type": row.get("query_type", "unknown"),
            "context": context,
            "all_passages": passage_texts,
        })

        count += 1
        if count >= max_samples:
            break

    logger.info("Loaded %d Q&A pairs from MS MARCO.", len(qa_pairs))
    return qa_pairs


# Legacy SQuAD loader (kept for backward compatibility)
def load_squad_documents(
    split: str = SQUAD_SPLIT,
    max_samples: int = SQUAD_SUBSET_SIZE,
) -> List[Document]:
    """
    Load SQuAD 2.0 (legacy). Use load_msmarco_documents() for the primary dataset.
    """
    logger.info("Loading SQuAD 2.0 %s split (legacy)…", split)
    dataset = load_dataset("rajpurkar/squad_v2", split=split)

    docs: List[Document] = []
    seen_contexts: set = set()
    count = 0

    for row in dataset:
        if not row["answers"]["text"]:
            continue
        ctx = row["context"].strip()
        if ctx in seen_contexts:
            continue
        seen_contexts.add(ctx)

        doc = Document(
            page_content=ctx,
            metadata={
                "id": row["id"],
                "title": row["title"],
                "question": row["question"],
                "answers": row["answers"]["text"],
            },
        )
        docs.append(doc)
        count += 1
        if count >= max_samples:
            break

    logger.info("Loaded %d unique answerable contexts from SQuAD.", len(docs))
    return docs


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DOCUMENT CHUNKING
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_documents(
    docs: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into chunks while preserving metadata.
    RecursiveCharacterTextSplitter respects sentence / paragraph boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunked = splitter.split_documents(docs)
    logger.info(
        "Split %d docs into %d chunks (size=%d, overlap=%d).",
        len(docs), len(chunked), chunk_size, chunk_overlap,
    )
    return chunked


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EMBEDDING GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Return a HuggingFace embedding model on the configured device."""
    logger.info("Loading embedding model: %s (device=%s)", model_name, EMBEDDING_DEVICE)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VECTOR STORE INDEXING
# ═══════════════════════════════════════════════════════════════════════════════

def build_vector_store(
    chunks: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    persist_dir: str = VECTOR_STORE_DIR,
) -> Chroma:
    """Index chunks into Chroma and persist to disk."""
    os.makedirs(persist_dir, exist_ok=True)
    logger.info("Building Chroma vector store at %s (%d chunks)…", persist_dir, len(chunks))
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )
    # NOTE: chromadb >= 0.4 persists automatically; explicit persist() is a no-op.
    logger.info("Vector store built and persisted (%d vectors).", len(chunks))
    return vectorstore


def load_vector_store(
    embedding_model: HuggingFaceEmbeddings,
    persist_dir: str = VECTOR_STORE_DIR,
) -> Chroma:
    """Load an already-persisted Chroma vector store."""
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SYNTHETIC DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

SYNTHETIC_PROMPT_TEMPLATES = {
    "simple": (
        "Given the following passage, generate ONE clear, specific question whose answer "
        "can be found directly in the passage. Return ONLY the question.\n\nPassage:\n{context}"
    ),
    "paraphrased": (
        "Given the following passage, generate ONE paraphrased or rephrased question that "
        "asks the same thing as a direct question but in a different way. The answer must "
        "be in the passage. Return ONLY the question.\n\nPassage:\n{context}"
    ),
    "ambiguous": (
        "Given the following passage, generate ONE ambiguous or multi-hop question that "
        "requires understanding the passage deeply. Return ONLY the question.\n\nPassage:\n{context}"
    ),
}


def generate_synthetic_queries(
    docs: List[Document] = None,
    qa_pairs: List[Dict[str, Any]] = None,
    num_samples: int = NUM_SYNTHETIC_SAMPLES,
    batch_size: int = SYNTHETIC_BATCH_SIZE,
    output_path: str = SYNTHETIC_DATA_PATH,
    api_key: str = MISTRAL_API_KEY,
    model: str = LLM_MODEL,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic Q&A dataset.

    Strategy:
      - If qa_pairs provided (from MS MARCO), use real queries + augment with LLM rewrites
      - If docs provided, generate questions from passages using LLM
      - Falls back to using MS MARCO's own queries as-is (free, no API needed)

    Returns list of dicts:
      { "id", "context", "question", "query_type", "ground_truth" }
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Resume if checkpoint exists
    existing: List[Dict] = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        logger.info("Resuming from %d existing samples.", len(existing))
        if len(existing) >= num_samples:
            return existing[:num_samples]

    samples = existing[:]

    # Strategy 1: Use MS MARCO Q&A pairs directly (FREE — no API needed)
    if qa_pairs:
        logger.info("Using MS MARCO Q&A pairs as synthetic dataset (free mode).")
        for i, pair in enumerate(qa_pairs[:num_samples]):
            if i < len(existing):
                continue
            samples.append({
                "id": pair.get("id", f"msmarco_{i}"),
                "context": pair.get("context", ""),
                "question": pair["question"],
                "query_type": pair.get("query_type", "unknown"),
                "ground_truth": pair.get("ground_truth", ""),
            })
        _save_json(samples, output_path)
        logger.info("Synthetic dataset saved (%d samples, from MS MARCO queries).", len(samples))
        return samples

    # Strategy 2: Generate from docs using LLM (requires API key)
    if docs and api_key:
        from mistralai import Mistral
        client = Mistral(api_key=api_key)
        docs_subset = docs[:num_samples]

        for i, doc in enumerate(docs_subset):
            if i < len(existing):
                continue

            ctx = doc.page_content
            ground_truth = (
                doc.metadata.get("answers", [""])[0]
                if doc.metadata.get("answers")
                else ""
            )
            query_type = SYNTHETIC_QUERY_TYPES[i % len(SYNTHETIC_QUERY_TYPES)]
            prompt = SYNTHETIC_PROMPT_TEMPLATES[query_type].format(context=ctx)

            try:
                resp = client.chat.complete(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                question = resp.choices[0].message.content.strip()
            except Exception as exc:
                logger.warning("LLM call failed for doc %d: %s", i, exc)
                question = doc.metadata.get("query", f"What does this passage describe?")

            samples.append({
                "id": doc.metadata.get("id", str(i)),
                "context": ctx,
                "question": question,
                "query_type": query_type,
                "ground_truth": ground_truth,
            })

            if (i + 1) % batch_size == 0:
                _save_json(samples, output_path)
                logger.info("Checkpoint: %d / %d samples saved.", len(samples), num_samples)
                time.sleep(0.5)

    # Strategy 3: Use docs metadata queries as-is (free fallback)
    elif docs:
        logger.info("No API key — using passage metadata queries as synthetic dataset.")
        for i, doc in enumerate(docs[:num_samples]):
            if i < len(existing):
                continue
            samples.append({
                "id": doc.metadata.get("id", str(i)),
                "context": doc.page_content,
                "question": doc.metadata.get("query", "What is described in this passage?"),
                "query_type": doc.metadata.get("query_type", "unknown"),
                "ground_truth": (
                    doc.metadata.get("answers", [""])[0]
                    if doc.metadata.get("answers")
                    else ""
                ),
            })

    _save_json(samples, output_path)
    logger.info("Synthetic dataset saved to %s (%d samples).", output_path, len(samples))
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PARENT-CHILD CHUNK BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_parent_child_chunks(
    docs: List[Document],
    parent_chunk_size: int = 1200,
    child_chunk_size: int = 300,
    overlap: int = 50,
) -> Dict[str, Any]:
    """
    For each parent chunk, create smaller child chunks.
    Returns { "parents": [...], "children": [...], "child_to_parent": {child_id: parent_id} }
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size, chunk_overlap=overlap
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=overlap
    )

    parents = parent_splitter.split_documents(docs)
    children: List[Document] = []
    child_to_parent: Dict[str, str] = {}

    for p_idx, parent in enumerate(parents):
        parent_id = f"parent_{p_idx}"
        parent.metadata["parent_id"] = parent_id
        child_docs = child_splitter.split_documents([parent])
        for c_idx, child in enumerate(child_docs):
            child_id = f"child_{p_idx}_{c_idx}"
            child.metadata["child_id"] = child_id
            child.metadata["parent_id"] = parent_id
            children.append(child)
            child_to_parent[child_id] = parent_id

    logger.info(
        "Parent-Child split: %d parents → %d children.", len(parents), len(children)
    )
    return {"parents": parents, "children": children, "child_to_parent": child_to_parent}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_module1(
    generate_synthetic: bool = True,
    dataset: str = "msmarco",
) -> Dict[str, Any]:
    """
    Full Module 1 pipeline. Returns artefacts for downstream modules.

    Args:
        generate_synthetic: Whether to generate/load synthetic Q&A dataset
        dataset: "msmarco" (default) or "squad" (legacy)
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load documents based on dataset choice
    if dataset == "msmarco":
        logger.info("═══ Loading MS MARCO v2.1 ═══")
        docs = load_msmarco_documents()
        qa_pairs = load_msmarco_qa_pairs()
    else:
        logger.info("═══ Loading SQuAD 2.0 (legacy) ═══")
        docs = load_squad_documents()
        qa_pairs = None

    # Chunk
    chunks = chunk_documents(docs)

    # Embed + index
    embedding_model = get_embedding_model()
    vectorstore = build_vector_store(chunks, embedding_model)

    # Synthetic data
    synthetic_data: List[Dict] = []
    if generate_synthetic:
        synthetic_data = generate_synthetic_queries(
            docs=docs,
            qa_pairs=qa_pairs,
        )

    # Parent-child
    pc = build_parent_child_chunks(docs)

    logger.info("═══ Module 1 complete ═══")
    logger.info("  Documents: %d", len(docs))
    logger.info("  Chunks: %d", len(chunks))
    logger.info("  Synthetic Q&A: %d", len(synthetic_data))
    logger.info("  Parent chunks: %d", len(pc["parents"]))
    logger.info("  Child chunks: %d", len(pc["children"]))

    return {
        "docs": docs,
        "chunks": chunks,
        "embedding_model": embedding_model,
        "vectorstore": vectorstore,
        "synthetic_data": synthetic_data,
        "parent_child": pc,
        "qa_pairs": qa_pairs,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  Module 1 — Data Preprocessing (MS MARCO v2.1)")
    print("=" * 60)
    artifacts = run_module1(generate_synthetic=True, dataset="msmarco")
    print(f"\n  Documents loaded: {len(artifacts['docs'])}")
    print(f"  Chunks created:   {len(artifacts['chunks'])}")
    print(f"  Synthetic Q&A:    {len(artifacts['synthetic_data'])}")
    print(f"  Parent chunks:    {len(artifacts['parent_child']['parents'])}")
    print(f"  Child chunks:     {len(artifacts['parent_child']['children'])}")
    print("\n  ✓ Ready for Module 2 (RAG architectures)")
