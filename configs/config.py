# configs/config.py
# Central configuration for the RAG evaluation framework

import os
from dotenv import load_dotenv
from security_boundary import is_loopback_literal, valid_api_key

# ─── Load .env as early as possible ─────────────────────────────────────────────
# Every entry point (main.py CLI modes and the standalone research scripts) imports
# this module, so loading here guarantees .env values are present before the
# os.getenv() calls below. override=False (python-dotenv's default) means any
# shell-exported variable wins over the same key in .env, preserving shell
# precedence (Req 3.9). An absent .env file is a silent no-op (returns False).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
# Default retrieval corpus for eval/cicd/API/UI: the full MS MARCO VALIDATION set
# built by `preprocess_full.py` (README "Build the dataset / vector store").
# `data/chroma_db_train` is a separate, larger train-split store written by
# `train_in_shards.py`; it must never be loaded by default.
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", os.path.join(DATA_DIR, "chroma_db_full"))
SYNTHETIC_DATA_PATH = os.path.join(DATA_DIR, "synthetic_dataset.json")
EVAL_RESULTS_PATH = os.path.join(OUTPUTS_DIR, "eval_results.csv")

# ─── Dataset ───────────────────────────────────────────────────────────────────
# MS MARCO v2.1 (passage ranking / QA)
DATASET_NAME = "microsoft/ms_marco"
DATASET_CONFIG = "v2.1"
DATASET_SPLIT = "validation"
# Dataset revisions are part of the on-disk index contract.  The default is the
# dataset/config revision used by this project; deployments that pin a Hub
# revision can override it without changing source code.
DATASET_REVISION = os.getenv("DATASET_REVISION", DATASET_CONFIG)
# Number of validation queries whose passages form the retrieval corpus for the
# in-tree `--mode full` build (data/chroma_db_full: 9858 passages / 11510 chunks).
# The complete 101,093-query validation set is built by `preprocess_full.py`.
DATASET_SUBSET_SIZE = 1000

# Persistent-index schema revision.  This is deliberately separate from the
# random build_id in _index_manifest.json: readers can reject indexes produced
# by an incompatible writer while still retaining every last-good build.
INDEX_REVISION = os.getenv("INDEX_REVISION", "rag-index-v2")

# Legacy SQuAD settings (kept for backward compat / smoke tests)
SQUAD_SPLIT = "validation"
SQUAD_SUBSET_SIZE = 500

# ─── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 400                   # characters per chunk (RecursiveCharacterTextSplitter)
CHUNK_OVERLAP = 50                 # character overlap between consecutive chunks
PARENT_CHUNK_SIZE = 1200           # larger parent chunk for Parent-Child RAG
CHILD_CHUNK_SIZE = 300             # fine-grained child chunk

# ─── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
def _default_embedding_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", _default_embedding_device())  # "cpu" | "cuda" | "mps"

# ─── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5
BM25_WEIGHT = 0.4
DENSE_WEIGHT = 0.6

# ─── Reranking (cross-encoder second stage) ────────────────────────────────────
# A bi-encoder scores query and passage independently, so similarity != relevance.
# A cross-encoder reads (query, passage) jointly and re-orders the first-stage
# candidates. Standard practice: retrieve a wide candidate set, rerank, keep top-k.
# Runs locally (CPU/MPS/CUDA) so it adds no API cost.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "50"))  # first-stage depth
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "")  # "" → auto-detect (cuda → mps → cpu)

# ─── LLM ───────────────────────────────────────────────────────────────────────
# Provider priority: Groq (free) → Mistral → extractive fallback.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-small-latest")

# Generation params (used by the LLM wrapper).
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_RATE_LIMIT_SECONDS = float(os.getenv("LLM_RATE_LIMIT_SECONDS", "3.0"))
# Reasoning-family models (gpt-oss) spend hidden "reasoning" completion tokens
# BEFORE the answer; a 512 cap truncates them mid-reasoning (finish_reason=
# "length", empty content) and the answer gate honestly abstains every time.
# The answer contract therefore gets its own headroom independent of LLM_MAX_TOKENS.
LLM_ANSWER_MAX_TOKENS = int(os.getenv("LLM_ANSWER_MAX_TOKENS", "2048"))
# Bound on how long a generation call may wait for a provider slot before the
# answer gate abstains (""). Kept at 2 s for serving latency guarantees.
# OFFLINE EVALUATION must NOT abstain — abstention makes faithfulness an
# artefact of queue pressure, not answer quality — so evaluated commands may
# raise this (e.g. LLM_ANSWER_MAX_WAIT_SECONDS=30).
LLM_ANSWER_MAX_WAIT_SECONDS = float(os.getenv("LLM_ANSWER_MAX_WAIT_SECONDS", "2.0"))
# Bound the amount of retrieved text sent to a generator. Retrieval metrics still
# score the complete returned documents; this only protects generation latency and
# provider context windows from unusually large parent chunks.
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))

# Judge LLM (separate from generator to reduce self-evaluation bias).
# If set, RAGAS uses this provider/model for grading; otherwise falls back to
# the generator LLM with a warning logged.
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "groq")          # "groq" | "mistral" | "openrouter"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")  # GPT-OSS family (was llama-3.1-70b-versatile)

# ─── OpenRouter (OpenAI-compatible gateway) ─────────────────────────────────────
# When OPENROUTER_API_KEY is set, the generator (_LLM) routes through OpenRouter's
# OpenAI-compatible endpoint instead of calling Groq directly; set
# JUDGE_PROVIDER=openrouter to route the RAGAS judge through it too. This sidesteps
# Groq's per-model daily token cap. Model IDs are taken from GROQ_MODEL (generator)
# and JUDGE_MODEL (judge), interpreted as OpenRouter model slugs.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# ─── Synthetic Data Generation ─────────────────────────────────────────────────
SYNTHETIC_BATCH_SIZE = 50
SYNTHETIC_QUERY_TYPES = ["simple", "paraphrased", "ambiguous"]
NUM_SYNTHETIC_SAMPLES = 12

# ─── RAGAS / Quality Gate Thresholds ──────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.75
CONTEXT_PRECISION_THRESHOLD = 0.70
ANSWER_RELEVANCE_THRESHOLD = 0.80
NOISE_RATIO_MAX = 0.25

# ─── Evaluation methodology ────────────────────────────────────────────────────
EVAL_DEFAULT_SAMPLES = int(os.getenv("EVAL_DEFAULT_SAMPLES", "200"))
EVAL_RANDOM_SEED = int(os.getenv("EVAL_RANDOM_SEED", "42"))
EVAL_BOOTSTRAP_ITERS = int(os.getenv("EVAL_BOOTSTRAP_ITERS", "1000"))

# ─── Semantic Cache (Redis) ────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CACHE_SIMILARITY_THRESHOLD = 0.92
CACHE_TTL_SECONDS = 3600

# ─── API security ──────────────────────────────────────────────────────────────
# Authentication FAILS CLOSED. Protected routes require a matching X-API-Key, and if
# API_KEY is unset the service refuses those requests instead of serving them openly.
# Local development can opt out explicitly with ALLOW_UNAUTHENTICATED=true, which is
# a deliberate, visible choice rather than a silent default.
API_KEY = os.getenv("API_KEY", "")
API_BIND_HOST = os.getenv("HOST", "127.0.0.1")
ALLOW_UNAUTHENTICATED = os.getenv("ALLOW_UNAUTHENTICATED", "false").lower() in {
    "1", "true", "yes", "on",
}
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",") if o.strip()
]
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "2000"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))  # 20 MB

# Per-request wall-clock budget. Without this a slow upstream LLM ties up a worker
# thread indefinitely and the service degrades instead of shedding load.
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
# Bound on concurrent blocking work (retrieval/LLM run in a thread pool).
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "8"))

# ─── Observability ─────────────────────────────────────────────────────────────
# JSON logs are machine-parseable for aggregation; plain text is easier to read
# locally. Metrics are exposed at /metrics for Prometheus scraping.
LOG_FORMAT = os.getenv("LOG_FORMAT", "json").lower()      # "json" | "text"
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() in {
    "1", "true", "yes", "on",
}
SERVICE_NAME = os.getenv("SERVICE_NAME", "rag-eval-api")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "2.0.0")


def validate_runtime_config(require_llm: bool = False) -> list:
    """
    Validate configuration at startup and return a list of human-readable problems.

    Failing fast on boot is preferable to discovering a misconfiguration on the first
    production request. Returns (rather than raises) so callers can decide whether a
    given problem is fatal for their entrypoint.
    """
    problems: list = []

    if not API_KEY and not ALLOW_UNAUTHENTICATED:
        problems.append(
            "API_KEY is not set. Protected endpoints will reject all requests. "
            "Set API_KEY, or set ALLOW_UNAUTHENTICATED=true for local development."
        )
    if API_KEY and len(API_KEY) < 16:
        problems.append("API_KEY is shorter than 16 characters; use a longer secret.")
    if API_KEY and not valid_api_key(API_KEY):
        problems.append("API_KEY must contain only printable non-space ASCII characters (max 4096).")
    if ALLOW_UNAUTHENTICATED and not is_loopback_literal(API_BIND_HOST):
        problems.append("ALLOW_UNAUTHENTICATED requires a numeric loopback HOST.")
    if "*" in ALLOWED_ORIGINS and API_KEY:
        problems.append(
            "ALLOWED_ORIGINS contains '*' while authentication is enabled; "
            "list explicit origins instead."
        )
    if require_llm and not (OPENROUTER_API_KEY or GROQ_API_KEY or MISTRAL_API_KEY):
        problems.append(
            "No LLM key configured (OPENROUTER_API_KEY / GROQ_API_KEY / MISTRAL_API_KEY)."
        )
    if not 0.0 < CACHE_SIMILARITY_THRESHOLD <= 1.0:
        problems.append("CACHE_SIMILARITY_THRESHOLD must be in (0, 1].")
    if RATE_LIMIT_PER_MINUTE <= 0:
        problems.append("RATE_LIMIT_PER_MINUTE must be positive.")
    if REQUEST_TIMEOUT_SECONDS <= 0:
        problems.append("REQUEST_TIMEOUT_SECONDS must be positive.")
    if MAX_CONTEXT_CHARS <= 0:
        problems.append("MAX_CONTEXT_CHARS must be positive.")
    if RERANK_CANDIDATES < TOP_K:
        problems.append("RERANK_CANDIDATES must be at least TOP_K.")

    return problems

# ─── CI/CD ─────────────────────────────────────────────────────────────────────
CICD_REPORT_PATH = os.path.join(OUTPUTS_DIR, "cicd_report.json")

# ─── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
