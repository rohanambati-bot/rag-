# RAG Evaluation Framework
### Synthetic-Data-Driven Evaluation of Retrieval-Augmented Generation Architectures

**Capstone UE23CS320B | Phase 3 | Project ID: 1**
**Guide:** Dr. Monika Goyal
**Team:** Manoj R · Lakshay Mittal · Keerthan P V · Rohan A

---

## Overview

A self-contained framework that:

- Implements and benchmarks a **suite of 12 RAG architectures** — Naive, CRAG, RAG-Fusion, Hybrid Multi-Query, Adaptive, Parent-Child, **Rerank** (cross-encoder two-stage), and the project's own **GRACE-Ω** engine (with and without reranking) plus three single-variable GRACE-Ω ablations. `full` mode registers all 12; `eval` mode registers the 9 that don't need the knowledge graph (see [RAG architectures](#rag-architectures)).
- Builds a Q&A evaluation set from **MS MARCO v2.1** (with a SQuAD 2.0 fallback for smoke tests).
- Scores each architecture on three metric families:
  - **Rank-aware IR metrics graded against MS MARCO's human relevance labels** —
    nDCG@10, MRR@10, Recall@k, Hit Rate, Precision@k. These need **no LLM**, so they
    scale to thousands of queries at zero API cost.
  - **Token-overlap metrics** — Answer-Recall, Token-IoU, Noise Ratio, PΩ.
  - **LLM-judged RAGAS-style metrics** — Faithfulness, Context Precision, Context
    Recall, Answer Relevance, using a **judge LLM decoupled from the generator** to
    reduce self-preference bias, guarded by a one-call health check.
- Reports per-architecture means with **95% bootstrap confidence intervals**, plus
  **question-paired significance tests with Benjamini-Hochberg correction** for
  multiple comparisons.
- Wraps the whole pipeline in a **CI/CD quality gate** (GitHub Actions) and an optional **Redis-backed semantic cache**.
- Ships a write-up (`docs/paper.md`) and bibliography (`docs/references.bib`).

This is a research/engineering framework, not a deployable product.

### Documentation map

| Document | Purpose |
|---|---|
| `docs/paper.md` | Full write-up: method, results, discussion (§5.3/5.4 pending Run 2) |
| `docs/PANEL_GUIDE.md` | Presentation talking points, demo script, anticipated Q&A |
| `docs/ROADMAP.md` | Prioritised improvements and the exact Run 2 command |
| `docs/references.bib` | 27 citations |
| `RUNBOOK.md` | Operational runbook |
| `Makefile` | `make help` lists common tasks (test, lint, eval, api, loadtest) |

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
Judge LLM  (separate model — Groq GPT-OSS-20B by default)
   │
   ▼
Quality Gate  /  CI Result
```

---

## RAG architectures

`module2/rag_architectures.py::build_all_architectures()` registers the set
below. The registration is conditional on what the caller passes in:

- **Always:** `naive`, `crag`, `rag_fusion`.
- **When `chunks` are provided:** adds `hybrid`, `adaptive`, `grace_omega`,
  `parent_child`, `rerank`, `grace_rerank`.
- **When `chunks` *and* a knowledge-graph builder are provided:** adds the three
  GRACE-Ω ablations `grace_no_gate`, `grace_no_kg`, `grace_no_hyde`.

So `full` mode (chunks + KG builder) registers the **full 12-architecture set**,
while `eval` mode (chunks, no KG builder) registers the **9** that don't need the
knowledge graph.

| Key             | Class                | Display name            | `full` | `eval` | LLM-free retrieval |
|-----------------|----------------------|-------------------------|:------:|:------:|:------------------:|
| `naive`         | `NaiveRAG`           | Naive RAG               |   ✓    |   ✓    | ✓ |
| `crag`          | `CorrectiveRAG`      | CRAG                    |   ✓    |   ✓    | — (LLM grading) |
| `rag_fusion`    | `RAGFusion`          | RAG-Fusion              |   ✓    |   ✓    | — (query variants) |
| `hybrid`        | `HybridMultiQueryRAG`| Hybrid Multi-Query      |   ✓    |   ✓    | — (query rewriting) |
| `adaptive`      | `AdaptiveRAG`        | Adaptive RAG            |   ✓    |   ✓    | — (classification) |
| `grace_omega`   | `GRACEOmegaRAG`      | GRACE-Ω                 |   ✓    |   ✓    | — (HyDE + routing) |
| `parent_child`  | `ParentChildRAG`     | Parent-Child RAG        |   ✓    |   ✓    | ✓ |
| `rerank`        | `RerankRAG`          | Rerank RAG              |   ✓    |   ✓    | ✓ (local model) |
| `grace_rerank`  | `GRACEOmegaRAG`      | GRACE-Ω + Rerank        |   ✓    |   ✓    | — |
| `grace_no_gate` | `GRACEOmegaNoGate`   | GRACE-Ω (no PΩ gate)    |   ✓    |   —    | — |
| `grace_no_kg`   | `GRACEOmegaNoKG`     | GRACE-Ω (no KG verify)  |   ✓    |   —    | — |
| `grace_no_hyde` | `GRACEOmegaNoHyDE`   | GRACE-Ω (no HyDE)       |   ✓    |   —    | — |

**Two-stage retrieval.** `rerank` retrieves a wide candidate set (dense + BM25,
`RERANK_CANDIDATES`=50 by default) and re-orders it with a cross-encoder
(`BAAI/bge-reranker-base`) that scores each (query, passage) pair jointly, keeping the
top-k. The cross-encoder is a local model, so this adds no API cost.
`grace_rerank` applies the same second stage on top of GRACE-Ω, giving a
single-variable A/B against `grace_omega`.

**LLM-free retrieval** marks architectures whose *retrieval path* makes no LLM API
call. `--llm-free` restricts a run to these, which is what makes large-n sweeps free.
Note that `--retrieval-only` alone is **not** free for the others: HyDE expansion,
multi-query rewriting, complexity classification and CRAG grading all call the LLM
during retrieval, and GRACE-Ω additionally summarises RAPTOR clusters at construction.

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
│   └── rag_architectures.py   # NaiveRAG, CorrectiveRAG (CRAG), RAGFusion, HybridMultiQueryRAG,
│                              # AdaptiveRAG, ParentChildRAG, GRACEOmegaRAG + 3 GRACE-Ω ablations;
│                              # build_all_architectures() registers them
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
│   ├── comparison.py          # Side-by-side comparison across registered architectures
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

### Prerequisites
- **Python 3.12** is required. The pinned dependencies in `requirements.txt`
  (e.g. `numpy==2.1.3`, `torch==2.5.1`) need Python ≥ 3.10, and the project
  targets 3.12 — both the `Dockerfile` (`python:3.12-slim`) and the CI workflow
  (`PYTHON_VERSION: "3.12"`) build on it.

### 1. Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

The pipeline needs **at least one** LLM key — `GROQ_API_KEY` **or**
`MISTRAL_API_KEY` (`main.py` exits with an error if neither is set). The
`demo` and `cache-demo` modes need no key at all.

| Variable          | Required?                    | Used for                                                                                     |
|-------------------|------------------------------|----------------------------------------------------------------------------------------------|
| `GROQ_API_KEY`    | Primary (free tier)          | Generator (`GROQ_MODEL=llama-3.1-8b-instant`) **and** judge (`JUDGE_PROVIDER=groq`, `JUDGE_MODEL=openai/gpt-oss-20b`) |
| `MISTRAL_API_KEY` | Optional fallback            | Generator fallback (`LLM_MODEL=mistral-small-latest`) when Groq is unset/unavailable          |
| `API_KEY`         | Optional                     | If set, the FastAPI service (`api.py`) requires an `X-API-Key` header; if unset it runs in open dev mode |

You can provide these two ways.

**Option A — shell export**
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

**Option B — `.env` file**
```bash
cp .env.example .env      # then edit .env and fill in your keys
```
`configs/config.py` auto-loads `.env` at import time via `python-dotenv`, so
every entry point (CLI modes and the standalone scripts) picks it up. A
shell-exported variable **takes precedence** over the same key in `.env`.
`.env` is gitignored; `.env.example` is the tracked template — never commit real
keys.

### 3. Run the demo (no API key needed)
```bash
python main.py --mode demo
```

### 4. Run the full pipeline (builds everything, then evaluates)
```bash
python main.py --mode full
```
`full` downloads MS MARCO, builds the vector store **and** the knowledge graph,
registers every architecture, evaluates them, and writes plots. It needs an LLM
key but **no** pre-built data — it produces the data.

### 5. Evaluate / run the CI/CD gate against a pre-built store
```bash
python main.py --mode eval --samples 200
python main.py --mode cicd --samples 50
```
Unlike `full`, the `eval` and `cicd` modes **load** an existing
`data/chroma_db_full` vector store and `data/synthetic_dataset.json` — build
them first (see [Build the dataset / vector store](#build-the-dataset--vector-store)).
If `data/synthetic_dataset.json` is missing, the run stops with a clear
`Run --mode full first` error.

---

## Build the dataset / vector store

The `full` mode builds its data inline, but `eval` and `cicd` expect a
pre-built vector store at `data/chroma_db_full` plus `data/synthetic_dataset.json`.
These artifacts are gitignored, so you build them locally with one of the
standalone scripts below.

### `preprocess_full.py` — full validation-set build (recommended)
```bash
python preprocess_full.py
```
Processes the **entire MS MARCO v2.1 validation set** (101,093 queries, ~1M
passages) and produces the two artifacts `eval`/`cicd` need:
- `data/chroma_db_full` — the Chroma vector store (`VECTOR_STORE_DIR`).
- `data/synthetic_dataset.json` — `NUM_SYNTHETIC_SAMPLES` (300) Q&A pairs
  (`SYNTHETIC_DATA_PATH`).

It is **slow** (~2–6 hours on CPU) but **checkpoint-resumable**: progress is
saved to `data/_preprocess_checkpoint.json`, so re-running continues where it
left off.

### `train_in_shards.py` — shard-based train-set build (optional, larger)
Builds a separate, much larger store from the MS MARCO **train** documents
(`data/_train_documents.jsonl`) in shards of `SHARD_SIZE` (100,000) docs, writing
each shard under `data/train_shards/`, then merging into `data/chroma_db_train`.

```bash
python train_in_shards.py --count       # count total docs / how many shards
python train_in_shards.py --status      # show per-shard progress
python train_in_shards.py --shard 0     # process one shard (0-indexed)
python train_in_shards.py --all         # process every shard sequentially
python train_in_shards.py --merge       # merge finished shards into data/chroma_db_train
```
Note this produces `data/chroma_db_train`, **not** the `data/chroma_db_full`
that `eval`/`cicd` load by default — use `preprocess_full.py` for those modes.

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

### What each mode requires

| Mode         | LLM key? | Pre-built store (`data/chroma_db_full`)? | Saved eval CSV (`outputs/eval_results.csv`)? |
|--------------|:--------:|:----------------------------------------:|:--------------------------------------------:|
| `full`       | Yes      | No — builds it                           | No — writes it                               |
| `eval`       | Yes*     | Yes                                      | No — writes it                               |
| `cicd`       | Yes      | Yes                                      | No                                           |
| `visualize`  | No       | No                                       | Yes (falls back to demo plots if absent)     |
| `demo`       | No       | No                                       | No                                           |
| `cache-demo` | No       | No                                       | No                                           |

\* `eval` needs no LLM key when run with `--retrieval-only --llm-free`.

`cicd` also requires `data/synthetic_dataset.json`. `eval` uses it when present and
otherwise draws real MS MARCO query/answer pairs. See
[Build the dataset / vector store](#build-the-dataset--vector-store).

### `eval` mode flags

| Flag | Effect |
|---|---|
| `--samples N` | Number of questions (default 200). If the synthetic set is smaller, real MS MARCO QA pairs are used instead so large n is reachable. |
| `--store full\|train\|PATH` | Which Chroma store to evaluate. `full` → `data/chroma_db_full`, `train` → `data/chroma_db_train`, or an explicit path. |
| `--retrieval-only` | Skip answer generation and LLM judging; report rank-aware IR metrics only. |
| `--llm-free` | Evaluate only architectures whose retrieval makes no LLM call. Combine with `--retrieval-only` for a genuinely zero-cost run. |

### Metrics reference

| Family | Metrics | Needs LLM? | Graded against |
|---|---|:---:|---|
| Rank-aware IR | nDCG@10, MRR@10, Recall@k, Precision@k, Hit Rate | No | MS MARCO human `is_selected` labels (qrels) |
| Token overlap | Answer-Recall, Token-IoU, Noise Ratio, PΩ, noise fingerprint | No | Gold answer string |
| LLM-judged | Faithfulness, Context Precision, Context Recall, Answer Relevance | Yes | Judge LLM (decoupled from generator) |

Rank-aware metrics appear only when qrels are available for the evaluated questions.
Statistical comparisons (`outputs/significance_tests.csv`) are question-paired, run
both a paired t-test and Wilcoxon signed-rank test, report Cohen's d_z, and apply
Benjamini-Hochberg correction across each metric's comparison family. Architectures
that return identical context to the baseline are flagged rather than reported as
undefined test statistics.

---

## Benchmark results

**Retrieval evaluation, n=500 MS MARCO validation queries** (496 with paired
relevance judgements), graded against human `is_selected` labels. LLM-free, so
reproducible at zero API cost. Full write-up in [`docs/paper.md`](docs/paper.md).

| Architecture | nDCG@10 | MRR@10 | Recall@k | Hit Rate | Latency |
|---|---|---|---|---|---|
| **Rerank RAG** | **0.699** | **0.689** | **0.857** | **0.950** | 1538 ms |
| Naive RAG | 0.632 | 0.611 | 0.800 | 0.889 | 12 ms |
| Parent-Child RAG | 0.632 | 0.611 | 0.801 | 0.889 | 8 ms |
| CRAG (lexical grader) | 0.575 | 0.570 | 0.702 | 0.786 | 11 ms |

Headline findings:

- **Cross-encoder reranking gives the only significant retrieval gains**: nDCG@10
  +0.067 (BH-corrected p = 3.0×10⁻⁶, d_z = 0.218), MRR@10 +0.077, Recall +0.057. It
  cuts the rate of queries where a known-relevant passage is missed entirely from
  **11.1% → 5.0%**, recovering 55% of the baseline's misses — at 129× latency.
- **PΩ does not track ground truth.** Across architectures it correlates *negatively*
  with nDCG@10 (Spearman ρ = −0.200) and ranks the weakest system first, because its
  denominator rewards returning fewer tokens even when relevant passages are lost.
  Reported as a negative result; PΩ should not be used to select architectures.
- **Lexical filtering hurts.** CRAG with keyword grading is significantly *worse* than
  the baseline on every rank-aware metric.
- Earlier committed results (n=12, no rank-aware metrics, positionally-paired tests,
  nothing significant) should not be cited; see `docs/paper.md` Appendix C.

Reproduce:

```bash
# Retrieval metrics, no API key needed, n=500
python main.py --mode eval --retrieval-only --llm-free --samples 500 --store full

# Post-hoc analysis (PΩ validity, miss rates, latency trade)
python analyze_results.py

# Full pipeline with generation + LLM judge (costs API calls)
python main.py --mode eval --samples 200
python main.py --mode visualize
```

---

## Production serving

The HTTP API (`api.py`) is built to be operated, not just demoed.

**One retrieval stack.** `serving.py` owns the embedding model, vector store, chunk
corpus, architecture registry and cache. The API, the Streamlit UI and the CLI all
read from it, so the served system is the measured system. (Previously the API and UI
bootstrapped from **SQuAD** while the CLI evaluated **MS MARCO**, meaning served BM25
indexed a different corpus than the dense store, and only 4 of 12 architectures were
reachable over HTTP.)

**Auth fails closed.** Protected routes require `X-API-Key`, compared with
`hmac.compare_digest` to avoid timing leaks. With no `API_KEY` configured the routes
return **503**, not open access. Local development opts out explicitly:

```bash
export API_KEY=$(openssl rand -hex 24)      # production
export ALLOW_UNAUTHENTICATED=true           # local only, deliberate
```

**Health probes are split.**

| Endpoint | Meaning | Use |
|---|---|---|
| `GET /healthz` | Process is alive | Liveness probe (never gates on warmup) |
| `GET /readyz` | Stack is usable; 503 until warm | Readiness probe / load-balancer |
| `GET /metrics` | Prometheus exposition | Scrape target |

The port binds immediately and warmup runs in the background, so probes answer during
startup instead of timing out.

**Observability.** JSON logs with a correlation id on every line; `X-Request-ID` is
accepted from the caller or generated, and echoed in the response. Secrets are
redacted. Metrics include `rag_http_requests_total`,
`rag_http_request_duration_seconds`, `rag_pipeline_duration_seconds`,
`rag_cache_events_total`, `rag_cache_entries`, `rag_inflight_requests`,
`rag_errors_total`.

**Load behaviour.** Retrieval and generation are blocking, so they run in a bounded
thread pool (`MAX_CONCURRENT_REQUESTS`) with a per-request budget
(`REQUEST_TIMEOUT_SECONDS`). At capacity the service returns 503; over budget, 504 —
it sheds load rather than degrading silently. Expensive architectures (GRACE-Ω builds
an LLM-generated RAPTOR tree, ~130 s) are constructed lazily on first request, off the
event loop, so unrelated traffic is unaffected — measured at 1.7-20 ms response times
while a build is in progress.

### Semantic cache

Rewritten in `phase3/semantic_cache.py`. The previous version ran
`KEYS rag_cache:*` plus one `GET` per entry on **every lookup**, then scored
candidates in a Python loop — `KEYS` blocks Redis, and lookup latency grew with cache
size.

Now: an in-process vector index (hnswlib ANN, with a vectorised exact fallback), Redis
used only for write-through persistence and a single `SCAN`-based warm start, bounded
capacity with LRU eviction, TTL expiry, and thread safety.

| Cached entries | Old (ms/lookup) | New (ms/lookup) | Speedup |
|---:|---:|---:|---:|
| 100 | 0.87 | 0.08 | 10× |
| 1,000 | 8.54 | 0.28 | 30× |
| 5,000 | 46.74 | 0.51 | 91× |
| 20,000 | 182.65 | 0.84 | **217×** |

Measured with the old algorithm reproduced in-process — i.e. *excluding* the Redis
round-trips it also paid, so the real-world gap is larger. End-to-end, a repeated
query returns in 26 ms versus 17,407 ms uncached.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `API_KEY` | *(unset)* | Required for protected routes; unset ⇒ 503 |
| `ALLOW_UNAUTHENTICATED` | `false` | Explicit local-dev bypass |
| `REQUEST_TIMEOUT_SECONDS` | `60` | Per-request wall-clock budget |
| `MAX_CONCURRENT_REQUESTS` | `8` | Bound on concurrent blocking work |
| `LOG_FORMAT` | `json` | `json` or `text` |
| `METRICS_ENABLED` | `true` | Expose `/metrics` |
| `CACHE_MAX_ENTRIES` | `50000` | Cache capacity before LRU eviction |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder |
| `RERANK_CANDIDATES` | `50` | First-stage depth before reranking |

`configs.config.validate_runtime_config()` runs at startup and logs every
misconfiguration (missing/short key, wildcard CORS with auth, bad thresholds).

```bash
# Run the API locally
API_KEY=$(openssl rand -hex 24) python api.py

# Or the full stack (Redis + API + UI); API_KEY is mandatory
docker compose up --build
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
| Architectures   | Naive, CRAG, RAG-Fusion, Hybrid Multi-Query, Adaptive, Parent-Child, GRACE-Ω (+ 3 ablations) |
| Generator LLM   | Groq Llama-3.1-8B-Instant (free) → Mistral Small (fallback)    |
| Judge LLM       | Groq GPT-OSS-20B (`openai/gpt-oss-20b`, default) — decoupled from generator to reduce self-preference bias |
| Retrieval       | LangChain, rank-bm25, Reciprocal Rank Fusion                    |
| Evaluation      | Custom RAGAS-style + bootstrap CIs                              |
| CI/CD           | GitHub Actions                                                  |
| Visualization   | Matplotlib, Seaborn                                             |
| API / UI        | FastAPI (auth + rate-limit) · Streamlit                         |
| Containerization| Docker (multi-stage, non-root) · Docker Compose                 |

---

## Honest scope

What this project is, stated plainly so nothing here oversells:

**Solid.** The evaluation framework (qrels-graded rank-aware metrics, question-paired
BH-corrected statistics, decoupled judge, 220 automated tests including property-based
tests), the data pipeline (4.1M passages, checkpoint-resumable), and the serving layer
described above.

**Not claimed.** This has not been load-tested under production traffic, has no
staging environment, no distributed tracing, no autoscaling story and no on-call
runbook beyond `RUNBOOK.md`. It is a research framework with a genuinely deployable
prototype — not a system that has survived production.

**Known limitations.**

- Chroma is a local, single-node store; there is no horizontal scaling story.
- The cache index is per-process. Multiple workers share Redis persistence but each
  keeps its own in-memory index, so hit rates are per-worker until warm.
- `GRACE-Ω` costs ~130 s to construct and ~8.5 s per query; it is not viable for
  latency-sensitive serving, and §5 of `docs/paper.md` shows a plain cross-encoder
  cascade outperforming it on retrieval.
- LLM-judged metrics have not been validated against human labels (Cohen's kappa on a
  labelled subset is the outstanding check).

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
