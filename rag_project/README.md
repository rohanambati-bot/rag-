# RAG Evaluation Framework
### Synthetic-Data-Driven Evaluation of Retrieval-Augmented Generation Architectures

**Capstone UE23CS320B | Phase 3 | Project ID: 1**
**Guide:** Dr. Monika Goyal
**Team:** Manoj R · Lakshay Mittal · Keerthan P V · Rohan A

---

## Overview

A self-contained framework that:

- Implements and benchmarks **4 RAG architectures** — Naive, Hybrid Multi-Query, Parent-Child, Corrective RAG (CRAG).
- Builds a Q&A evaluation set from **MS MARCO v2.1** (with a SQuAD 2.0 fallback for smoke tests).
- Scores each architecture with **token-overlap retrieval metrics** plus **LLM-judged RAGAS-style metrics** (Faithfulness, Context Precision, Context Recall, Answer Relevance) using a **judge LLM decoupled from the generator** to reduce self-preference bias.
- Reports per-architecture means with **95% bootstrap confidence intervals**.
- Wraps the whole pipeline in a **CI/CD quality gate** (GitHub Actions) and an optional **Redis-backed semantic cache**.

This is a research/engineering framework, not a deployable product.

---

## Pipeline at a glance

```
User Query
   │
   ▼
Semantic Cache (Redis)  ─── cache hit → answer
   │  (cache miss)
   ▼
Retriever (Hybrid BM25 + Dense / Parent-Child / CRAG)
   │
   ▼
Generator LLM  (Groq Llama-3.1-8B  →  Mistral fallback  →  extractive fallback)
   │
   ▼
Judge LLM  (separate model — Groq Llama-3.1-70B by default)
   │
   ▼
Quality Gate  /  CI Result
```

---

## Project Structure

```
rag_project/
├── configs/
│   └── config.py              # All thresholds, paths, model names, security knobs
│
├── module1/
│   └── data_preprocessing.py  # MS MARCO / SQuAD loading, chunking, embeddings,
│                              # Chroma indexing, parent–child split, synthetic queries
│
├── module2/
│   └── rag_architectures.py   # NaiveRAG, HybridMultiQueryRAG, ParentChildRAG, CorrectiveRAG
│
├── module3/
│   └── evaluation.py          # Token-overlap metrics + LLM-judged RAGAS-style + bootstrap CIs
│
├── module4/
│   └── visualization.py       # Matplotlib/Seaborn plots + Markdown report
│
├── module5/
│   ├── document_upload.py     # Validated user document upload (PDF/DOCX/TXT/MD/CSV)
│   ├── confidence.py          # Per-answer confidence scoring
│   ├── comparison.py          # 4-way live architecture comparison
│   └── failure_analysis.py    # Failure-point detection (Barnett et al. 2024 taxonomy)
│
├── phase3/
│   └── cicd_pipeline.py       # Quality gate, SemanticCache, CachedRAG wrapper, TruLensMonitor
│
├── .github/workflows/
│   └── rag_quality_gate.yml   # GitHub Actions CI/CD workflow
│
├── tests/                     # Pytest unit tests (metrics, security, parsing)
├── api.py                     # FastAPI REST backend (auth + rate limit + CORS)
├── app.py                     # Streamlit multi-page UI
├── main.py                    # CLI entry point
├── docker-compose.yml         # Redis + API + UI (least-privilege defaults)
├── Dockerfile                 # Multi-stage, non-root
├── requirements.txt           # Pinned versions
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set API keys (at least one)
```bash
# Linux / macOS
export GROQ_API_KEY=your_groq_key            # free tier; preferred
export MISTRAL_API_KEY=your_mistral_key      # optional fallback
export API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# Windows (PowerShell)
$env:GROQ_API_KEY    = "your_groq_key"
$env:MISTRAL_API_KEY = "your_mistral_key"
$env:API_KEY         = (python -c "import secrets; print(secrets.token_urlsafe(32))")
```

### 3. Run the demo (no API key needed)
```bash
python main.py --mode demo
```

### 4. Run the full pipeline
```bash
python main.py --mode full
```

### 5. CI/CD quality gate only
```bash
python main.py --mode cicd --samples 50
```

---

## Pipeline Modes

| Mode         | Description                                                              |
|--------------|--------------------------------------------------------------------------|
| `full`       | End-to-end: data → embeddings → RAG → eval → plots                       |
| `eval`       | Evaluation only (loads pre-built vectorstore)                            |
| `cicd`       | CI/CD quality gate (exits 0 = pass, 1 = fail)                            |
| `visualize`  | Generate plots + report from saved CSVs                                  |
| `demo`       | Demo plots from synthetic data (no API key needed)                       |
| `cache-demo` | Demonstrate semantic cache with similarity lookup                        |

---

## Benchmark results

> **Honest disclosure**: The current `outputs/eval_results.csv` is a small
> preliminary run (n=50 per architecture, single LLM family). It is **not yet
> publication-quality** for the reasons documented in
> `outputs/evaluation_report.md`. A larger run (n ≥ 200) with the decoupled
> judge LLM is required before reporting headline numbers.

To reproduce the most recent run:

```bash
python main.py --mode eval --samples 200
python main.py --mode visualize
```

---

## CI/CD Quality Gates

GitHub Actions runs `main.py --mode cicd` on every push/PR to `main` and rejects
the build if any threshold is violated:

| Metric             | Threshold |
|--------------------|----------:|
| Faithfulness       |   ≥ 0.75  |
| Context Precision  |   ≥ 0.70  |
| Answer Relevance   |   ≥ 0.80  |
| Noise Ratio        |   ≤ 0.25  |

Thresholds are configurable in `configs/config.py`.

---

## Datasets

| Dataset    | Size                     | Purpose                                  |
|------------|--------------------------|------------------------------------------|
| MS MARCO v2.1 | 101 K validation queries | Primary retrieval + Q&A corpus          |
| SQuAD 2.0     | 500 contexts (subset)    | Fallback / smoke tests when MS MARCO is unavailable |

---

## Failure-mode taxonomy

`module5/failure_analysis.py` maps each detected failure to one of the seven
RAG failure points from Barnett et al. (IEEE/ACM CAIN 2024):

| ID  | Failure Point             | Mitigation present in this framework        |
|-----|---------------------------|---------------------------------------------|
| FP1 | Query content missed      | Hybrid BM25 + Dense (Reciprocal Rank Fusion) |
| FP2 | Missing content           | CRAG keyword-overlap relevance grading      |
| FP3 | Not in top-K              | Parent-Child indexing                        |
| FP4 | Not extracted             | Query rewriting (Hybrid arch, configurable) |
| FP5 | Wrong format              | Quality-gate Answer-Relevance threshold     |
| FP6 | Specificity mismatch      | Quality-gate Faithfulness threshold         |
| FP7 | Incomplete synthesis      | Generator LLM (Llama-3.1 / Mistral)          |

---

## Technologies

| Category        | Tools                                                          |
|-----------------|----------------------------------------------------------------|
| Embeddings      | `sentence-transformers/all-MiniLM-L6-v2`                       |
| Vector store    | Chroma (dev), Redis (semantic cache)                           |
| Generator LLM   | Groq Llama-3.1-8B-Instant (free) → Mistral Small (fallback)    |
| Judge LLM       | Groq Llama-3.1-70B-Versatile (default) — decoupled from generator |
| Retrieval       | LangChain, rank-bm25, Reciprocal Rank Fusion                    |
| Evaluation      | Custom RAGAS-style + bootstrap CIs                              |
| CI/CD           | GitHub Actions                                                  |
| Visualization   | Matplotlib, Seaborn                                             |
| API / UI        | FastAPI (auth + rate-limit) · Streamlit                         |
| Containerization| Docker (multi-stage, non-root) · Docker Compose                 |

---

## Security defaults

- The FastAPI backend requires an `X-API-Key` header (`API_KEY` env var). When
  unset, the API runs in **open dev mode** and logs a warning on every request.
- Per-IP rate limiting via `slowapi` (default `RATE_LIMIT_PER_MINUTE = 30`,
  `5/minute` for `/evaluate`).
- CORS is restricted to `ALLOWED_ORIGINS` (no `*` with credentials).
- File uploads are bounded by `MAX_UPLOAD_BYTES` (default 20 MB) and validated
  by extension allow-list **and** magic-byte sniffing.
- Question length capped by `MAX_QUESTION_LENGTH` (default 2 000 chars).
- Redis bound to `127.0.0.1` in `docker-compose.yml`; container runs as non-root
  with `cap_drop: ALL` and `security_opt: no-new-privileges`.

---

## Team

| Member          | ID            | Role                                                    |
|-----------------|---------------|---------------------------------------------------------|
| Manoj R         | PES2UG23CS328 | System Architect — Hybrid search, dynamic chunking      |
| Lakshay Mittal  | PES2UG23CS300 | Data Engineer — Vector DB, semantic cache               |
| Keerthan P V    | PES2UG23CS272 | ML Engineer — Prompt engineering, RAGAS scoring          |
| Rohan A         | PES2UG24CS819 | DevOps / QA — CI/CD, security, monitoring                |
