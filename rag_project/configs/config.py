# configs/config.py
# Central configuration for the RAG evaluation framework

import os

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "chroma_db_full")
SYNTHETIC_DATA_PATH = os.path.join(DATA_DIR, "synthetic_dataset.json")
EVAL_RESULTS_PATH = os.path.join(OUTPUTS_DIR, "eval_results.csv")

# ─── Dataset ───────────────────────────────────────────────────────────────────
# MS MARCO v2.1 (passage ranking / QA)
DATASET_NAME = "microsoft/ms_marco"
DATASET_CONFIG = "v2.1"
DATASET_SPLIT = "validation"
DATASET_SUBSET_SIZE = 101093      # FULL validation set (101K queries)

# Legacy SQuAD settings (kept for backward compat / smoke tests)
SQUAD_SPLIT = "validation"
SQUAD_SUBSET_SIZE = 500

# ─── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 400                   # tokens per chunk
CHUNK_OVERLAP = 50                 # token overlap between consecutive chunks
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

# Judge LLM (separate from generator to reduce self-evaluation bias).
# If set, RAGAS uses this provider/model for grading; otherwise falls back to
# the generator LLM with a warning logged.
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "groq")          # "groq" | "mistral"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.1-70b-versatile")

# ─── Synthetic Data Generation ─────────────────────────────────────────────────
SYNTHETIC_BATCH_SIZE = 50
SYNTHETIC_QUERY_TYPES = ["simple", "paraphrased", "ambiguous"]
NUM_SYNTHETIC_SAMPLES = 300

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
API_KEY = os.getenv("API_KEY", "")  # if set, all /query, /ab-test, /evaluate, /cache require X-API-Key
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",") if o.strip()
]
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "2000"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))  # 20 MB

# ─── CI/CD ─────────────────────────────────────────────────────────────────────
CICD_REPORT_PATH = os.path.join(OUTPUTS_DIR, "cicd_report.json")

# ─── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
