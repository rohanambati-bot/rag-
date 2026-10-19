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

Env vars for the full/eval/cicd pipelines (at least ONE LLM key is required):
  GROQ_API_KEY    — Groq API key (free tier, recommended)
  MISTRAL_API_KEY — Mistral API key (optional fallback)
  REDIS_HOST      — Redis hostname (optional, defaults to localhost)
  REDIS_PORT      — Redis port    (optional, defaults to 6379)
"""

import argparse
import json
import logging
import os
import shutil
import sys

import pandas as pd

from configs.config import (
    SYNTHETIC_DATA_PATH, EVAL_RESULTS_PATH, OUTPUTS_DIR, MISTRAL_API_KEY, GROQ_API_KEY,
    OPENROUTER_API_KEY, DATA_DIR, EMBEDDING_MODEL, CHILD_CHUNK_SIZE, CHUNK_OVERLAP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("main")

os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _require_api_key() -> None:
    from configs.config import GROQ_API_KEY, MISTRAL_API_KEY, OPENROUTER_API_KEY
    if not MISTRAL_API_KEY and not GROQ_API_KEY and not OPENROUTER_API_KEY:
        logger.error(
            "No LLM API key found (GROQ_API_KEY, MISTRAL_API_KEY, or OPENROUTER_API_KEY). "
            "Export one before running:\n"
            "  export GROQ_API_KEY=your_key_here   (free, recommended)\n"
            "  export OPENROUTER_API_KEY=your_key_here\n"
            "  export MISTRAL_API_KEY=your_key_here"
        )
        sys.exit(1)


def _store_dir(args) -> str:
    """
    Resolve which Chroma store to evaluate against.

    ``--store`` accepts a shorthand (``full`` / ``train``) or an explicit path, so
    the large train-split index built by ``preprocess_train.py`` is reachable
    without editing config.
    """
    from configs.config import DATA_DIR, VECTOR_STORE_DIR

    choice = getattr(args, "store", None)
    if not choice:
        return VECTOR_STORE_DIR
    # The shorthands must be distinct regardless of the VECTOR_STORE_DIR default or
    # env-override: README documents `full` → data/chroma_db_full and
    # `train` → data/chroma_db_train. (Previously `full` mapped to
    # VECTOR_STORE_DIR, which by default resolved to the *train* store — so
    # `--store full` and `--store train` silently both selected the train corpus.)
    shorthand = {
        "full": os.path.join(DATA_DIR, "chroma_db_full"),
        "train": os.path.join(DATA_DIR, "chroma_db_train"),
    }
    path = shorthand.get(choice, choice)
    if not os.path.exists(path):
        logger.error("Vector store not found: %s", path)
        logger.error(
            "Build it with `python preprocess_full.py` (full) or "
            "`python train_in_shards.py --all --merge` (train)."
        )
        sys.exit(1)
    logger.info("Using vector store: %s", path)
    return path


def _filter_llm_free(architectures: dict) -> dict:
    """
    Keep only architectures whose retrieval path makes no LLM API calls.

    ``--retrieval-only`` removes answer generation and LLM judging, but several
    architectures call the LLM *inside* retrieval (HyDE expansion, multi-query
    rewriting, complexity classification, CRAG relevance grading), and GRACE-Ω also
    summarises RAPTOR clusters when constructed. Those runs are therefore NOT free.
    This filter selects the genuinely zero-cost subset so a large-n sweep can run
    without API spend or rate limits.
    """
    kept, dropped = {}, []
    for key, arch in architectures.items():
        is_free = getattr(arch, "llm_free_retrieval", False)
        (kept.__setitem__(key, arch) if is_free else dropped.append(key))

    logger.info(
        "LLM-free filter: keeping %d architecture(s) %s; excluding %d that call the "
        "LLM during retrieval %s.",
        len(kept), sorted(kept), len(dropped), sorted(dropped),
    )
    if not kept:
        logger.error("No LLM-free architectures available — nothing to evaluate.")
        sys.exit(1)
    return kept


def _load_eval_dataset(requested: int) -> list:
    """
    Return an evaluation set of at least ``requested`` questions where possible.

    The committed synthetic dataset holds only 50 questions, which is far below the
    ~50+ topics IR practice expects for a meaningful comparison and was the binding
    constraint on statistical power (not the ``--samples`` flag). MS MARCO already
    ships real user queries paired with human-written answers, so when more samples
    are requested than the synthetic file contains we draw straight from the dataset
    instead. That needs no LLM, so scaling up costs nothing.
    """
    synthetic = _load_synthetic_data()

    # Anything that is not a sized list (or is already big enough) passes straight
    # through unchanged.
    if not isinstance(synthetic, list) or len(synthetic) >= requested:
        return synthetic

    try:
        from module1.data_preprocessing import load_msmarco_qa_pairs
    except ImportError:
        logger.warning(
            "load_msmarco_qa_pairs unavailable — evaluating on the %d available "
            "synthetic questions.", len(synthetic),
        )
        return synthetic

    logger.info(
        "Synthetic set has %d questions but %d requested — drawing real MS MARCO "
        "query/answer pairs instead (no LLM needed).",
        len(synthetic), requested,
    )
    pairs = load_msmarco_qa_pairs(max_samples=max(requested, 1000))
    if len(pairs) < requested:
        logger.warning(
            "Only %d MS MARCO pairs available; evaluating on those.", len(pairs)
        )
    return pairs or synthetic


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
    from module2.rag_architectures import build_all_architectures
    from langchain_community.vectorstores import Chroma

    # Build child vectorstore for Parent-Child RAG — persist + reuse across runs.
    # Path is anchored to DATA_DIR, not CWD, so the pipeline behaves identically
    # when launched from another working directory.
    pc = artifacts["parent_child"]
    _child_dir = os.path.join(DATA_DIR, "chroma_db_children")
    child_manifest = os.path.join(_child_dir, "_index_manifest.json")

    def _child_store_is_reusable() -> bool:
        if not (
            os.path.exists(os.path.join(_child_dir, "chroma.sqlite3"))
            and os.path.exists(child_manifest)
        ):
            return False
        try:
            with open(child_manifest, encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
            return (
                manifest.get("embedding_model") == EMBEDDING_MODEL
                and manifest.get("chunk_size") == CHILD_CHUNK_SIZE
                and manifest.get("chunk_overlap") == CHUNK_OVERLAP
                and manifest.get("num_chunks") == len(pc["children"])
            )
        except (OSError, AttributeError, TypeError, json.JSONDecodeError):
            return False

    if _child_store_is_reusable():
        logger.info("Found existing child vector store at %s — loading instead of rebuilding.", _child_dir)
        child_vectorstore = Chroma(
            persist_directory=_child_dir,
            embedding_function=artifacts["embedding_model"],
        )
    else:
        logger.warning("Child vector store is missing or stale; rebuilding %s.", _child_dir)
        from module1.data_preprocessing import build_lock_for, create_staging_dir
        from module1.data_preprocessing import write_index_manifest
        from preprocess_full import _cleanup_staged_vector_stores, _publish_vector_store

        with build_lock_for(_child_dir) as child_lock:
            # Another process may have completed the same rebuild while this one
            # waited. Re-read the manifest only after acquiring the lock.
            if _child_store_is_reusable():
                child_vectorstore = Chroma(
                    persist_directory=_child_dir,
                    embedding_function=artifacts["embedding_model"],
                )
            else:
                _cleanup_staged_vector_stores(
                    DATA_DIR,
                    target_dir=_child_dir,
                    prefix=".chroma_db_children.",
                    lock=child_lock,
                )
                staged_child_dir = create_staging_dir(
                    DATA_DIR,
                    ".chroma_db_children.",
                    _child_dir,
                    lock=child_lock,
                )
                try:
                    Chroma.from_documents(
                        documents=pc["children"],
                        embedding=artifacts["embedding_model"],
                        persist_directory=staged_child_dir,
                    )
                    write_index_manifest(
                        staged_child_dir,
                        embedding_model=EMBEDDING_MODEL,
                        chunk_size=CHILD_CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                        num_chunks=len(pc["children"]),
                    )
                    _publish_vector_store(
                        staged_child_dir, _child_dir, lock=child_lock
                    )
                except Exception:
                    shutil.rmtree(staged_child_dir, ignore_errors=True)
                    raise
                # The staged directory was renamed during publication; reopen it
                # at the stable path rather than retaining an old pathname handle.
                child_vectorstore = Chroma(
                    persist_directory=_child_dir,
                    embedding_function=artifacts["embedding_model"],
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
    retrieval_only = getattr(args, "retrieval_only", False)
    # Retrieval-only runs never call an LLM, so they need no API key at all.
    if not retrieval_only:
        _require_api_key()

    from module1.data_preprocessing import load_vector_store, get_embedding_model, chunk_documents, load_msmarco_documents
    from module2.rag_architectures import build_all_architectures
    from module3.evaluation import build_qrels

    embedding_model = get_embedding_model()
    vectorstore = load_vector_store(embedding_model, persist_dir=_store_dir(args))
    docs = load_msmarco_documents()
    chunks = chunk_documents(docs)

    # Human relevance judgements (MS MARCO is_selected) → rank-aware IR metrics.
    qrels = build_qrels(chunks)

    llm_free = getattr(args, "llm_free", False)
    architectures = build_all_architectures(
        vectorstore=vectorstore, chunks=chunks, llm_free_only=llm_free
    )
    if llm_free:
        # Constructed selectively (GRACE-Ω's RAPTOR build costs LLM calls), then
        # asserted so a future architecture cannot silently reintroduce API spend.
        architectures = _filter_llm_free(architectures)

    requested = getattr(args, "samples", None) or 200
    eval_set = _load_eval_dataset(requested)
    _run_evaluation(
        architectures,
        eval_set,
        qrels=qrels,
        max_samples=requested,
        retrieval_only=retrieval_only,
    )


def _run_evaluation(
    architectures: dict,
    synthetic_data: list,
    qrels: dict | None = None,
    max_samples: int = 200,
    retrieval_only: bool = False,
) -> None:
    from module3.evaluation import EvaluationPipeline, RAGASEvaluator

    # Retrieval-only runs are LLM-free by design: no generator, no judge. This is
    # what makes large-n (n>=500) experiments free and rate-limit-proof.
    if retrieval_only:
        logger.info(
            "RETRIEVAL-ONLY mode: skipping generation and LLM judging. "
            "Rank-aware IR metrics only (no API calls)."
        )
        ragas_evaluator = None
    else:
        ragas_evaluator = RAGASEvaluator() if (
            OPENROUTER_API_KEY or GROQ_API_KEY or MISTRAL_API_KEY
        ) else None

    # One-call judge health check before the long evaluation run. Runs ONLY when
    # an evaluator was actually constructed (a key was present); the no-key path
    # (regression 3.2) leaves ragas_evaluator as None and is skipped entirely, so
    # extractive answering + token-overlap metrics still run without a judge and
    # without crashing. A mis-set/retired judge model now fails loudly here rather
    # than silently dropping every LLM-judged metric across the whole run.
    if ragas_evaluator is not None:
        try:
            probe_score = ragas_evaluator.health_check()
            logger.info("Judge health check passed (probe score=%.3f).", probe_score)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "\n"
                "════════════════════════════════════════════════════════════════\n"
                " JUDGE HEALTH CHECK FAILED — LLM-judged metrics will be MISSING.\n"
                " %s\n"
                " Fix JUDGE_PROVIDER / JUDGE_MODEL (see .env.example) before trusting\n"
                " these results; the judge model may be mis-set or retired.\n"
                "════════════════════════════════════════════════════════════════",
                exc,
            )
            raise RuntimeError(
                "Judge health check failed; refusing to spend the evaluation budget."
            ) from exc

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
            max_samples=max_samples,
            qrels=qrels,
            retrieval_only=retrieval_only,
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
    """
    Paired significance tests vs the Naive baseline, across every available metric.

    Tests are question-paired and Benjamini-Hochberg corrected within each metric
    family (see module3.evaluation.test_metric_significance).
    """
    if not os.path.exists(raw_csv_path):
        logger.warning("No raw results found for significance testing.")
        return

    from module3.evaluation import test_metric_significance
    df = pd.read_csv(raw_csv_path)

    # Rank-aware IR metrics first (they exist only when qrels were supplied), then
    # the token-overlap and LLM-judged metrics.
    candidate_metrics = [
        "ndcg_at_10", "mrr_at_10", "recall_at_k", "hit_rate",
        "precision_omega", "answer_recall", "iou",
        "faithfulness", "context_precision", "context_recall", "answer_relevance",
    ]
    frames = []
    for metric in candidate_metrics:
        if metric not in df.columns or df[metric].dropna().empty:
            continue
        out = test_metric_significance(df, baseline_arch="Naive RAG", metric=metric)
        if not out.empty:
            frames.append(out)

    if not frames:
        logger.warning("No metrics available for significance testing.")
        return

    sig_df = pd.concat(frames, ignore_index=True)
    sig_path = os.path.join(OUTPUTS_DIR, "significance_tests.csv")
    sig_df.to_csv(sig_path, index=False)
    logger.info("Significance tests saved to %s", sig_path)
    print("\n=== STATISTICAL SIGNIFICANCE (paired, BH-corrected, vs Naive RAG) ===")
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
    cache = SemanticCache(namespace="cli-cache-demo-v1")  # availability fallback, never a serving namespace

    queries = [
        ("What is the capital of France?", "The capital of France is Paris.", ["Paris is the capital..."]),
        ("What's the capital of France?", None, None),   # should cache hit (sim 0.99)
        ("What is deep learning?", "Deep learning uses neural networks.", ["Neural networks..."]),
        ("What exactly is deep learning?", None, None),  # should cache hit (sim 0.98)
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
                print("             ↳ Stored in cache\n")

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
        help="Number of samples (CI/CD gate default 30; eval mode default 200)",
    )
    parser.add_argument(
        "--store", default=None,
        help="Vector store to evaluate: 'full', 'train', or an explicit path",
    )
    parser.add_argument(
        "--retrieval-only", action="store_true", dest="retrieval_only",
        help=(
            "Skip answer generation and LLM judging; report rank-aware IR metrics "
            "only. NOTE: some architectures still call the LLM during retrieval "
            "(HyDE, query rewriting, routing, CRAG grading) — add --llm-free for a "
            "genuinely zero-cost run."
        ),
    )
    parser.add_argument(
        "--llm-free", action="store_true", dest="llm_free",
        help=(
            "Evaluate only architectures whose retrieval makes no LLM API calls "
            "(naive, parent_child, rerank). Combine with --retrieval-only for a "
            "large-n sweep at zero API cost."
        ),
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
