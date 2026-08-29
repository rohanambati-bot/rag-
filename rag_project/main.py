#!/usr/bin/env python3
# main.py
"""
Production-Grade RAG System — Capstone Project Entry Point
────────────────────────────────────────────────────────────
Usage:
  python main.py --mode full          # Run entire pipeline end-to-end
  python main.py --mode eval          # Evaluation only (loads pre-built vectorstore)
  python main.py --mode cicd          # CI/CD quality gate check
  python main.py --mode visualize     # Generate plots from saved CSV
  python main.py --mode demo          # Demo plots with benchmark data (no API needed)
  python main.py --mode cache-demo    # Demonstrate semantic caching

Env vars required for full pipeline:
  MISTRAL_API_KEY — Mistral API key
  REDIS_HOST      — Redis hostname (optional, defaults to localhost)
  REDIS_PORT      — Redis port    (optional, defaults to 6379)
"""

import argparse
import json
import logging
import os
import sys

import pandas as pd

from configs.config import (
    SYNTHETIC_DATA_PATH, EVAL_RESULTS_PATH, OUTPUTS_DIR, MISTRAL_API_KEY
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("main")

os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _require_api_key() -> None:
    from configs.config import GROQ_API_KEY
    if not MISTRAL_API_KEY and not GROQ_API_KEY:
        logger.error(
            "Neither GROQ_API_KEY nor MISTRAL_API_KEY is set. Export one before running:\n"
            "  export GROQ_API_KEY=your_key_here   (free, recommended)\n"
            "  export MISTRAL_API_KEY=your_key_here"
        )
        sys.exit(1)


def _load_synthetic_data():
    if not os.path.exists(SYNTHETIC_DATA_PATH):
        logger.error("Synthetic dataset not found at %s. Run --mode full first.", SYNTHETIC_DATA_PATH)
        sys.exit(1)
    with open(SYNTHETIC_DATA_PATH) as f:
        return json.load(f)


# ─── Pipeline Modes ───────────────────────────────────────────────────────────

def run_full_pipeline(args) -> None:
    """End-to-end: data → embeddings → RAG → eval → plots."""
    _require_api_key()
    logger.info("═══ FULL PIPELINE START ═══")

    # Module 1
    from module1.data_preprocessing import run_module1
    artifacts = run_module1(generate_synthetic=True)

    # Module 2 — build all architectures
    from module2.rag_architectures import build_all_architectures, NaiveRAG
    from langchain_community.vectorstores import Chroma

    # Build child vectorstore for Parent-Child RAG — persist + reuse across runs
    pc = artifacts["parent_child"]
    _child_dir = os.path.join("data", "chroma_db_children")
    if os.path.exists(os.path.join(_child_dir, "chroma.sqlite3")):
        logger.info("Found existing child vector store at %s — loading instead of rebuilding.", _child_dir)
        child_vectorstore = Chroma(
            persist_directory=_child_dir,
            embedding_function=artifacts["embedding_model"],
        )
    else:
        os.makedirs(_child_dir, exist_ok=True)
        child_vectorstore = Chroma.from_documents(
            documents=pc["children"],
            embedding=artifacts["embedding_model"],
            persist_directory=_child_dir,
        )

    # Build knowledge graph for GRACE-Ω
    from module1.data_preprocessing import KnowledgeGraphBuilder
    logger.info("Building knowledge graph for GRACE-Ω...")
    kg_builder = KnowledgeGraphBuilder()
    chunk_dicts = [
        {
        "id": str(i),
        "text": doc.page_content,
        "pid": str(i),
        "chunk_idx": 0
        }
        for i, doc in enumerate(artifacts["chunks"])
    ]
    kg_builder.build(chunk_dicts)
    logger.info("Knowledge graph built successfully.")

    architectures = build_all_architectures(
        vectorstore=artifacts["vectorstore"],
        chunks=artifacts["chunks"],
        parent_child=pc,
        child_vectorstore=child_vectorstore,
        kg_builder=kg_builder,
    )

    # Module 3 — evaluate
    synthetic_data = artifacts["synthetic_data"] or _load_synthetic_data()
    _run_evaluation(architectures, synthetic_data)

    # Module 4 — visualize
    _run_visualization()

    logger.info("═══ FULL PIPELINE COMPLETE ═══")


def run_eval_mode(args) -> None:
    """Load pre-built vectorstore and evaluate all architectures."""
    _require_api_key()
    from module1.data_preprocessing import load_vector_store, get_embedding_model, chunk_documents, load_squad_documents
    from module2.rag_architectures import build_all_architectures

    embedding_model = get_embedding_model()
    vectorstore = load_vector_store(embedding_model)
    docs = load_squad_documents()
    chunks = chunk_documents(docs)

    architectures = build_all_architectures(
        vectorstore=vectorstore, chunks=chunks
    )

    synthetic_data = _load_synthetic_data()
    _run_evaluation(architectures, synthetic_data)


def _run_evaluation(architectures: dict, synthetic_data: list) -> None:
    from module3.evaluation import EvaluationPipeline, RAGASEvaluator

    ragas_evaluator = RAGASEvaluator() if MISTRAL_API_KEY else None
    pipeline = EvaluationPipeline(ragas_evaluator=ragas_evaluator)

    all_results: dict = {}
    all_raw_rows = []

    for name, rag in architectures.items():
        arch_display = getattr(rag, "name", name)
        logger.info("Evaluating: %s", arch_display)
        df = pipeline.evaluate_architecture(
            architecture_name=arch_display,
            rag_system=rag,
            synthetic_dataset=synthetic_data,
            max_samples=200,
        )
        all_results[arch_display] = df
        all_raw_rows.append(df)

    if all_raw_rows:
        raw_df = pd.concat(all_raw_rows, ignore_index=True)
        raw_csv = os.path.join(OUTPUTS_DIR, "raw_eval_results.csv")
        raw_df.to_csv(raw_csv, index=False)
        logger.info("Raw results saved to %s", raw_csv)

    summary = pipeline.compare_architectures(all_results)
    print("\n=== EVALUATION SUMMARY ===")
    print(summary.to_string(index=False))
    
    from module4.visualization import generate_summary_report
    generate_summary_report(summary)

    raw_csv = os.path.join(OUTPUTS_DIR, "raw_eval_results.csv")
    _run_significance_tests(raw_csv)

def _run_visualization() -> None:
    """Generate all plots + summary report from the saved eval CSV."""
    from module4.visualization import generate_all_plots, generate_summary_report

    if not os.path.exists(EVAL_RESULTS_PATH):
        logger.warning("No eval CSV found — skipping visualization.")
        return

    summary_df = pd.read_csv(EVAL_RESULTS_PATH)
    raw_csv = os.path.join(OUTPUTS_DIR, "raw_eval_results.csv")
    raw_df = pd.read_csv(raw_csv) if os.path.exists(raw_csv) else None

    paths = generate_all_plots(summary_df, raw_df)
    generate_summary_report(summary_df)
    logger.info("Generated %d plots in outputs/plots/", len(paths))


def _run_significance_tests(raw_csv_path: str) -> None:
    """Run statistical significance tests and save results."""
    if not os.path.exists(raw_csv_path):
        logger.warning("No raw results found for significance testing.")
        return
    
    from module3.evaluation import test_pomega_significance
    df = pd.read_csv(raw_csv_path)
    sig_df = test_pomega_significance(df, baseline_arch="Naive RAG")
    
    if not sig_df.empty:
        sig_path = os.path.join(OUTPUTS_DIR, "significance_tests.csv")
        sig_df.to_csv(sig_path, index=False)
        logger.info("Significance tests saved to %s", sig_path)
        print("\n=== STATISTICAL SIGNIFICANCE (vs Naive RAG baseline) ===")
        print(sig_df.to_string(index=False))


def run_cicd_mode(args) -> None:
    """Run CI/CD quality gate against a pre-built RAG system."""
    _require_api_key()
    from module1.data_preprocessing import load_vector_store, get_embedding_model
    from module2.rag_architectures import CorrectiveRAG
    from phase3.cicd_pipeline import run_quality_gate

    embedding_model = get_embedding_model()
    vectorstore = load_vector_store(embedding_model)
    crag = CorrectiveRAG(vectorstore=vectorstore)
    synthetic_data = _load_synthetic_data()

    report = run_quality_gate(
        rag_system=crag,
        synthetic_dataset=synthetic_data,
        architecture_name="CRAG",
        max_samples=getattr(args, "samples", 30),
        commit_hash=os.getenv("GITHUB_SHA", "local"),
    )

    sys.exit(0 if report.passed else 1)


def run_visualize_mode(args) -> None:
    """Generate all plots from a saved eval CSV."""
    if not os.path.exists(EVAL_RESULTS_PATH):
        logger.warning("No eval CSV found — generating demo plots instead.")
        from module4.visualization import demo_plots
        demo_plots()
        return

    _run_visualization()


def run_demo_mode(args) -> None:
    """Generate demo plots from benchmark data (no API key needed)."""
    from module4.visualization import demo_plots
    demo_plots()
    print("\nDemo plots generated in outputs/plots/")


def run_cache_demo(args) -> None:
    """Demonstrate the semantic cache with in-memory fallback."""
    from phase3.cicd_pipeline import SemanticCache

    print("\n=== Semantic Cache Demo ===\n")
    cache = SemanticCache()  # uses in-memory fallback if Redis not available

    queries = [
        ("What is the capital of France?", "The capital of France is Paris.", ["Paris is the capital..."]),
        ("Tell me the capital city of France.", None, None),      # should cache hit
        ("What is deep learning?", "Deep learning uses neural networks.", ["Neural networks..."]),
        ("Explain deep learning briefly.", None, None),           # should cache hit
        ("What is the weather today?", "It's sunny.", ["Weather data..."]),
    ]

    for query, answer, contexts in queries:
        cached_result = cache.lookup(query)
        if cached_result:
            print(f"  CACHE HIT  [{cached_result['similarity_score']:.3f}] → {query[:50]}")
            print(f"             ↳ {cached_result['answer'][:60]}\n")
        else:
            print(f"  CACHE MISS → {query[:50]}")
            if answer:
                cache.store(query, answer, contexts or [])
                print(f"             ↳ Stored in cache\n")

    print("Cache stats:", cache.stats())


# ─── Argument Parser ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production RAG Evaluation Framework — Capstone Phase 3"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "eval", "cicd", "visualize", "demo", "cache-demo"],
        default="demo",
        help="Pipeline mode to run",
    )
    parser.add_argument(
        "--samples", type=int, default=30,
        help="Number of synthetic samples for CI/CD gate (default: 30)",
    )
    args = parser.parse_args()

    mode_map = {
        "full":       run_full_pipeline,
        "eval":       run_eval_mode,
        "cicd":       run_cicd_mode,
        "visualize":  run_visualize_mode,
        "demo":       run_demo_mode,
        "cache-demo": run_cache_demo,
    }

    runner = mode_map.get(args.mode)
    if runner:
        runner(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
