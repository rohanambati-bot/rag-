#!/usr/bin/env python3
"""
Interactive RAG Query Interface
--------------------------------
Usage:
    python query.py             # uses best architecture (CRAG) by default
    python query.py --arch naive
    python query.py --arch hybrid
    python query.py --arch parentchild
    python query.py --arch crag
"""

import argparse
import logging
import sys

logging.disable(logging.CRITICAL)  # suppress all INFO/WARNING noise

def main():
    parser = argparse.ArgumentParser(description="Ask questions to your RAG system")
    parser.add_argument("--arch", choices=["naive", "hybrid", "parentchild", "crag"],
                        default="crag", help="RAG architecture to use (default: crag)")
    args = parser.parse_args()

    # Check API key — the RAG architectures resolve keys via configs.config
    # (Groq → Mistral → extractive fallback), so accept EITHER key here.
    from configs.config import GROQ_API_KEY, MISTRAL_API_KEY, OPENROUTER_API_KEY
    if not (GROQ_API_KEY or MISTRAL_API_KEY or OPENROUTER_API_KEY):
        print("ERROR: No LLM API key set (GROQ_API_KEY, MISTRAL_API_KEY, or OPENROUTER_API_KEY).")
        print("Set one before running:")
        print("  export GROQ_API_KEY=your_key_here     (free, recommended)")
        print("  export MISTRAL_API_KEY=your_key_here")
        sys.exit(1)

    print("Loading RAG system, please wait...")

    from module1.data_preprocessing import (
        load_vector_store, get_embedding_model, chunk_documents,
        load_msmarco_documents,
    )
    from module2.rag_architectures import NaiveRAG, HybridMultiQueryRAG, ParentChildRAG, CorrectiveRAG

    embedding_model = get_embedding_model()
    vectorstore = load_vector_store(embedding_model)

    # One corpus everywhere: the BM25 index for hybrid/parent-child must cover the
    # same MS MARCO chunks as the dense store. (Previously this bootstrapped from
    # SQuAD, so sparse retrieval searched a different corpus than dense.)
    chunks = chunk_documents(load_msmarco_documents())

    arch_map = {
        "naive":       ("Naive RAG",          NaiveRAG(vectorstore=vectorstore)),
        "hybrid":      ("Hybrid Multi-Query",  HybridMultiQueryRAG(vectorstore=vectorstore,
                            chunks=chunks)),
        "parentchild": ("Parent-Child RAG",    ParentChildRAG(vectorstore=vectorstore,
                            chunks=chunks)),
        "crag":        ("Corrective RAG (CRAG)", CorrectiveRAG(vectorstore=vectorstore)),
    }

    arch_name, rag = arch_map[args.arch]

    print(f"\n{'='*55}")
    print(f"  RAG Query Interface — {arch_name}")
    print(f"{'='*55}")
    print("  Type your question and press Enter.")
    print("  Type 'exit' or press Ctrl+C to quit.")
    print(f"{'='*55}\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        try:
            result = rag.run(query)
            print(f"\nAnswer: {result['answer']}")
            print(f"  (retrieved {len(result.get('context_docs', []))} chunks, "
                  f"latency: {result.get('latency_ms', 0):.0f}ms)\n")
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()
