# RUNBOOK — Live Run (build data + run the pipeline end-to-end)

> Current security/deployment scope: see
> [the 19 October security continuation](docs/VALIDATION_2026-10-19_SECURITY_V3.md).
> Use `make test` for offline checks without loading private `.env` files.
> Compose now starts **Redis + authenticated API only**; use `make ui` on the host
> for the unauthenticated loopback-only UI. Do not tunnel/proxy it. Stop every
> participating index reader before publication/rebuild. Docker/Linux replay is
> not yet verified. Commands below that write research outputs or call providers
> are deliberate research operations, not part of the safe validation command.

A copy-paste runbook for the RAG Evaluation Framework. Follow it top to bottom
on a Python 3.12 machine with a Groq API key. It includes a **FAST smoke path**
so you can prove everything works in minutes before committing to the multi-hour
full build.

- Run every command from the project root: `rag_project/`.
- This runbook does not change any code. It only tells you what to run.
- For concepts, architectures, and results, see [README.md](README.md). This
  runbook is the operational counterpart.

---

## 1. Prerequisites

- **Python 3.12.14** (required). The resolved deps in `requirements.lock`
  (`numpy==2.1.3`, `torch==2.5.1`) need Python ≥ 3.10, and the project targets
  3.12: the `Dockerfile` uses `python:3.12.14-slim` and CI
  (`.github/workflows/rag_quality_gate.yml`) sets `PYTHON_VERSION: "3.12.14"`.
- **git** — to clone/manage the repo.
- **Disk** — several GB free. The ML stack (`torch`, `transformers`,
  `sentence-transformers`) is large, and the full Chroma store at
  `data/chroma_db_full` grows with the corpus (hundreds of MB to a few GB).
- **Internet** — needed on first run for two things:
  - HuggingFace downloads the embedder `sentence-transformers/all-MiniLM-L6-v2`
    the first time the embedding model is used (cached afterwards).
  - Groq API calls for the generator and the judge LLM.

Check your Python:

```bash
python --version   # expect Python 3.12.x
```

---

## 2. Environment setup

```bash
# from rag_project/
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --requirement requirements.lock
```

Optional (recommended if you will run `--mode full` / GRACE-Ω):

```bash
python -m spacy download en_core_web_sm
```

Why: the knowledge-graph builder used by GRACE-Ω (`full` mode only) calls
`spacy.load("en_core_web_sm")`. If the model is missing it **auto-downloads it
on first use**; if spaCy itself is unavailable it silently falls back to a regex
entity extractor. Pre-installing avoids a mid-run download and keeps entity
extraction deterministic. `eval`/`cicd` modes don't build the knowledge graph,
so they don't need it.

---

## 3. API keys

At least **one** LLM key is required for `full` / `eval` / `cicd`
(`main.py` exits unless `GROQ_API_KEY`, `OPENROUTER_API_KEY`, or
`MISTRAL_API_KEY` is set). The `demo` and `cache-demo` modes need no key.

| Variable          | Required?           | Used for                                                                 |
|-------------------|---------------------|--------------------------------------------------------------------------|
| `GROQ_API_KEY`    | Primary (free tier) | Generator (`GROQ_MODEL=llama-3.1-8b-instant`) **and** judge (`JUDGE_PROVIDER=groq`, `JUDGE_MODEL=openai/gpt-oss-20b`) |
| `MISTRAL_API_KEY` | Optional fallback   | Generator fallback (`LLM_MODEL=mistral-small-latest`) |
| `API_KEY`         | Protected HTTP routes | Unset means 503 unless `ALLOW_UNAUTHENTICATED=true` explicitly enables local open access |

Get a free Groq key at <https://console.groq.com> → API Keys → Create.

**Option A — `.env` file (recommended):**

```bash
cp .env.example .env
# then edit .env and set GROQ_API_KEY=...   (MISTRAL_API_KEY / API_KEY optional)
```

`configs/config.py` auto-loads `.env` at import time, so every entry point picks
it up. `.env.example` also documents the judge settings
(`JUDGE_PROVIDER=groq`, `JUDGE_MODEL=openai/gpt-oss-20b`) and other knobs.

**Option B — shell export (wins over `.env` for the same key):**

```bash
export GROQ_API_KEY=your_groq_key
export MISTRAL_API_KEY=your_mistral_key            # optional
export API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")  # optional
```

---

## 4. Zero-cost sanity checks (no key, no data)

Prove the install works before touching the LLM or building data.

```bash
# Pure plots from built-in benchmark numbers — no key, no data, no embedder.
python main.py --mode demo

# Semantic cache demo — no key. First run downloads the embedder
# (all-MiniLM-L6-v2). If Redis isn't running it logs a warning and uses an
# in-memory fallback (still works).
python main.py --mode cache-demo

# Unit + property tests (offline, no key). Needs the installed requirements
# (pytest, hypothesis are pinned in requirements.lock).
pytest -q
```

`demo` writes plots to `outputs/plots/` and a report to
`outputs/evaluation_report.md`. The tests under `tests/` are all offline (they
patch/monkeypatch network calls); none needs an API key.

---

## 5. FAST smoke path (recommended first real run)

Goal: exercise the real key end-to-end in minutes without the multi-hour full
build.

**Important knob note:** `DATASET_SUBSET_SIZE` and `NUM_SYNTHETIC_SAMPLES` in
`configs/config.py` are **hardcoded — there is no environment-variable
override** for them. To shrink a run you must edit `configs/config.py`.

### 5a. Shrink the corpus + synthetic set

Edit `configs/config.py`:

```python
DATASET_SUBSET_SIZE = 200      # was 101093 (full validation set)
NUM_SYNTHETIC_SAMPLES = 20     # was 300
```

### 5b. Run the full pipeline on the reduced set

```bash
python main.py --mode full
```

`full` mode builds data inline (respecting the two knobs above), builds the
knowledge graph, registers all 12 architectures, evaluates them, and writes
plots. On a fresh checkout this produces a small store + small synthetic set in
a few minutes.

> Reuse caveat: `full` mode **reuses** an existing `data/chroma_db_full`
> (and `data/chroma_db_children`) if present, and **resumes/reuses**
> `data/synthetic_dataset.json`. For a genuinely small fresh build, make sure
> those don't already exist (a fresh clone won't have them — they're
> gitignored). To force a rebuild, remove them first.

### 5c. (Alternative) Build only the data, then run the fastest gate

This mirrors what CI does and is handy if you want a store without a full eval:

```bash
# Build data/chroma_db_full + data/synthetic_dataset.json (uses the reduced knobs)
python -c "from module1.data_preprocessing import run_module1; run_module1(generate_synthetic=True)"

# Fastest real check: single-architecture (CRAG) gate on 20 samples
python main.py --mode cicd --samples 20
```

> `--samples` applies to **both** `cicd` and `eval`. In `eval` mode, when the
> committed synthetic set has fewer questions than requested, real MS MARCO
> query/answer pairs are drawn so large n is reachable (see README "eval mode
> flags").

### 5d. Expected artifacts after 5b

```
data/chroma_db_full/            # Chroma vector store (VECTOR_STORE_DIR)
data/chroma_db_children/        # child store for Parent-Child RAG
data/synthetic_dataset.json     # NUM_SYNTHETIC_SAMPLES Q&A pairs (SYNTHETIC_DATA_PATH)
outputs/raw_eval_results.csv    # per-sample rows
outputs/eval_results.csv        # per-architecture summary + bootstrap CIs (EVAL_RESULTS_PATH)
outputs/evaluation_report.md    # Markdown summary
outputs/significance_tests.csv  # paired tests vs Naive RAG (when enough samples)
outputs/plots/*.png             # all plots
```

### 5e. Revert the smoke edits before the real build

Restore only the two values you deliberately edited, using the values you recorded
before the smoke run. **Do not use checkout/reset/clean to discard a file that may
contain unrelated uncommitted work.** Prefer the non-destructive validation scripts
in [`docs/VALIDATION_2026-10-19.md`](docs/VALIDATION_2026-10-19.md).

---

## 6. Full data build

The `full` mode builds data inline, but `eval` and `cicd` **load** a pre-built
store at `data/chroma_db_full` plus `data/synthetic_dataset.json`. Build them
with one of the scripts below.

### 6a. `preprocess_full.py` — full validation-set build (recommended)

```bash
python preprocess_full.py
```

- Processes the **entire MS MARCO v2.1 validation set** (101,093 queries,
  ~1M passages). Expect **~2–6 hours** on CPU.
- **Checkpoint-resumable**: progress is saved to
  `data/_preprocess_checkpoint.json`; re-running continues where it left off.
- Writes `data/chroma_db_full` and `data/synthetic_dataset.json`
  (`NUM_SYNTHETIC_SAMPLES` Q&A pairs).
- Note: this script always processes the full validation split — it **ignores**
  `DATASET_SUBSET_SIZE`.
- Note: its embedding phase runs in **offline mode** (`HF_HUB_OFFLINE=1`), so
  the embedder must already be cached. Warm the cache first by running the
  §5 FAST smoke `--mode full` once (it downloads `all-MiniLM-L6-v2`), or launch
  the UI/`cache-demo` once.

### 6b. `train_in_shards.py` — shard-based train build (optional, larger)

Builds a separate, larger store from the MS MARCO **train** documents at
`data/_train_documents.jsonl`, in shards of `SHARD_SIZE` (100,000) docs under
`data/train_shards/`, then merges into `data/chroma_db_train`.

```bash
python train_in_shards.py --count        # count total docs / shards needed
python train_in_shards.py --status       # per-shard progress
python train_in_shards.py --shard 0      # process one shard (0-indexed)
python train_in_shards.py --all          # process every shard sequentially
python train_in_shards.py --merge        # merge finished shards -> data/chroma_db_train
```

> This produces `data/chroma_db_train`, **not** the `data/chroma_db_full` that
> `eval`/`cicd` load by default. Use `preprocess_full.py` for those modes.

---

## 7. Full evaluation + quality gate

All commands assume `data/chroma_db_full` and `data/synthetic_dataset.json`
exist (§6).

```bash
# End-to-end: build data -> 12 architectures -> eval -> plots.
python main.py --mode full

# Evaluation only: loads the pre-built store; evaluates the 9 no-KG architectures.
python main.py --mode eval --samples 200

# CI/CD quality gate on CRAG: exits 0 (pass) or 1 (fail).
python main.py --mode cicd --samples 50

# Plots + report from the saved eval CSV (or demo plots if none exist).
python main.py --mode visualize
```

What each writes:

- `full` → `outputs/eval_results.csv`, `outputs/raw_eval_results.csv`,
  `outputs/evaluation_report.md`, `outputs/significance_tests.csv`, and
  `outputs/plots/*.png`.
- `eval` → same CSVs/report as `full` **but no plots**. Run `--mode visualize`
  afterward to render `outputs/plots/`.
- `cicd` → `outputs/cicd_report.json` and a pass/fail exit code.
- `visualize` → `outputs/plots/*.png` + `outputs/evaluation_report.md`.

Gotchas verified against the code:

- **`--samples` is honoured by `eval` too.** When requested samples exceed the
  committed synthetic set, real MS MARCO QA pairs are substituted (§7), so
  `--samples 500` works without editing config. The default `make eval-retrieval`
  target passes `--samples 500` explicitly.
- **`cicd` judge accepts any one LLM key.** The gate (`phase3/cicd_pipeline.py`
  `run_quality_gate`) constructs the LLM judge when `OPENROUTER_API_KEY`,
  `GROQ_API_KEY` **or** `MISTRAL_API_KEY` is set, and `RAGASEvaluator` defaults to
  `JUDGE_PROVIDER=groq` — so a Groq-only setup works. With **no** key at all,
  `cicd` produces no faithfulness / context precision / answer relevance columns,
  so those thresholds evaluate as 0.0 and the gate fails.
- **Before any judged run, probe the judge once.** `RAGASEvaluator.health_check()`
  issues one trivial prompt and raises loudly if the model id is retired or
  mis-configured — far better than 2,000 silently-failed calls:

  ```bash
  python -c "import sys; sys.path.insert(0,'.');
  from module3.evaluation import RAGASEvaluator
  print(RAGASEvaluator().health_check())"
  ```

  Verified working: provider=groq, model=qwen/qwen3.8-27b (score 1.0 on the
  probe). A failed probe should be fixed in `.env` before launching eval.

---

## 7.5 Accuracy-optimization workflow (dev split — never tune on the eval set)

`scripts/optimize_retrieval.py` diagnoses retrieval accuracy and compares
candidate models on a **dev split only**. The standard eval deterministically
samples 500 of the first 1,000 MS MARCO QA pairs (seed = `EVAL_RANDOM_SEED`);
this script reproduces that exact sample (verified: 500/500 overlap with the
questions in `outputs/raw_eval_results.csv`) and tunes on the remaining ~498
questions with qrels.

```bash
# Rank-source diagnosis: where do misses come from (coverage vs ordering)?
TOKENIZERS_PARALLELISM=false python scripts/optimize_retrieval.py

# Cross-encoder comparison on the production first-stage pool
TOKENIZERS_PARALLELISM=false python scripts/optimize_retrieval.py --rerankers
```

Findings recorded 2026-10-18 (dev n=498):

- First-stage coverage is not the bottleneck: on 100% of dev misses the
  relevant chunk sits inside the production rerank pool (dense≤50 ∪ BM25≤50);
  99.8% sit inside dense≤50 alone. Deeper candidate pools buy nothing.
- RRF fusion ordering is worse than dense-only ordering at top-5 (hit 0.813 vs
  0.878) — the cross-encoder is what recovers it. Candidate *ordering* before
  the cross-encoder is irrelevant (it rescores the whole pool); only pool
  *membership* matters.
- Dense top-5 misses concentrate in DESCRIPTION (13.4%) and ENTITY (15.4%)
  query types; NUMERIC 9.6%.

Cross-encoder comparison (2026-10-18, dev n=498, pool = production first
stage, mean 84.7 docs/query — `--rerankers` mode):

| model | hit@5 | MRR | P@5 | rescue@5 |
|---|---|---|---|---|
| BAAI/bge-reranker-base (production default) | **0.944** | 0.694 | **0.212** | **66.7%** |
| BAAI/bge-reranker-large | 0.944 | 0.704 | 0.211 | 61.7% |

Decision: **keep `bge-reranker-base`**. Hit@5 identical, the MRR delta sits
exactly at the pre-declared +0.01 adoption threshold, P@5 and miss-rescue are
marginally worse, and the large model costs ~2.5x cross-encoder latency. The
reranker is not the binding constraint on accuracy at this pool size.
(`rescue@5` = fraction of dev queries with no relevant chunk in the pool's
default ordering that the cross-encoder still promotes into its top-5.)
Note: Hugging Face downloads of the large model stalled via the xet transfer;
`HF_HUB_DISABLE_XET=1` forces plain HTTP and lets it complete.

Proven workflow for a model upgrade:

1. Run the diagnosis + comparison on dev (`--rerankers`).
2. Adopt a candidate only if it clears a pre-declared margin on dev
   (≥ +0.01 hit@5 or MRR used here) — never by looking at eval metrics first.
3. If adopted: set `RERANKER_MODEL` in `configs/config.py` (and `.env.example`
   comment), re-run the standard n=500 eval, update README/paper tables.
4. Commit the dev evidence and the eval run separately, clearly labelled.

Note: cross-encoder downloads can stall with HF's xet transfer on some
networks; `HF_HUB_ENABLE_XET=0` forces plain HTTP and is set for these runs.


---

## 8. Verify the two deferred acceptance checks

These are tasks 6.3 and 10.2 from `.kiro/specs/eval-pipeline-fixes/tasks.md`,
deferred to a live Python 3.12 runtime.

### 8a. Task 6.3 — single-corpus retrieval in eval mode

Confirm dense, sparse (BM25), and graph retrievers all operate over the one
MS MARCO corpus in `eval` mode. `run_eval_mode` loads `data/chroma_db_full`
(dense) and builds BM25/graph inputs from `load_msmarco_documents()` — the same
MS MARCO corpus — so there's no SQuAD/MS MARCO split.

```bash
python main.py --mode eval --samples 200 2>&1 | tee outputs/eval_run.log
```

Confirm in the log:

```bash
grep -i "Loading MS MARCO" outputs/eval_run.log        # corpus source for chunks (BM25/graph)
grep -Ei "squad" outputs/eval_run.log                  # expect NO SQuAD loading in eval mode
```

The point of 6.3 is corpus consistency: the dense retriever (Chroma store) and
the BM25 / graph inputs (from `chunks`) are all derived from the **same**
MS MARCO corpus — no retriever falls back to SQuAD in eval mode. `eval`
registers the 9 no-KG architectures; the dedicated knowledge-graph retriever is
fully built in `full` mode (which constructs the `KnowledgeGraphBuilder`).

### 8b. Task 10.2 — full pipeline records judged metrics, full arch set, judge health

Run the full pipeline with a Groq key and confirm three things:

```bash
python main.py --mode full 2>&1 | tee outputs/full_run.log
```

1. **Judge health check passes.** `main.py` runs a one-call probe before the
   long run and logs exactly:

   ```
   Judge health check passed (probe score=0.XXX).
   ```

   ```bash
   grep "Judge health check passed" outputs/full_run.log
   ```

   If the judge is mis-set/retired you'll instead see a
   `JUDGE HEALTH CHECK FAILED` banner (see §10).

2. **RAGAS metrics are recorded** — faithfulness, context precision, context
   recall, answer relevance:

   ```bash
   head -1 outputs/eval_results.csv        # columns include the four metrics below
   python -c "import pandas as pd; print([c for c in pd.read_csv('outputs/eval_results.csv').columns if c in {'faithfulness','context_precision','context_recall','answer_relevance'}])"
   ```

3. **Expected architecture set is present.** `full` registers **12**
   architectures (`eval` registers **9**, or 4 in the legacy `--llm-free` mode,
   which includes explicitly lexical CRAG). Check the `architecture` column:

   ```bash
   python -c "import pandas as pd; df=pd.read_csv('outputs/eval_results.csv'); print(len(df)); print(df['architecture'].tolist())"
   ```

   Expect the display names from the README table, including Rerank RAG and
   GRACE-Ω + Rerank in addition to the three GRACE-Ω ablations.

---

## 9. Serve the API + UI

### 9a. FastAPI backend

```bash
uvicorn api:app --reload --port 8000
```

- App object is `app` in `api.py`; it listens on port **8000**.
- Interactive docs: <http://localhost:8000/docs> (Swagger),
  <http://localhost:8000/redoc>.
- Endpoints: `GET /health`, `GET /architectures`, `POST /query`,
  `POST /ab-test`, `POST /evaluate` (rate-limited to 5/min),
  `GET /cache/stats`, `POST /cache/lookup`, `DELETE /cache`.
- **Auth:** when `API_KEY` is set, protected endpoints require an `X-API-Key`
  header. When unset, protected routes return **503** (fails closed) unless
  `ALLOW_UNAUTHENTICATED=true` — do not expose it with that bypass on.

Example call (protected route with a key set):

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"question": "what is a corporation", "architecture": "crag"}'
```

`/evaluate` needs `data/synthetic_dataset.json` (returns 503 if missing).
Valid API architecture keys are the 9 no-knowledge-graph entries listed by
`GET /architectures` (including `naive`, `crag`, `hybrid`, `parent_child`, and
`rerank`). GRACE-Ω variants are constructed lazily on first use.

### 9b. Streamlit UI

```bash
streamlit run app.py
```

Opens on port **8501**. No API key required (extractive fallback works offline).
If `data/chroma_db_full` doesn't exist yet, the UI shows an error telling you to
build the store first (`python preprocess_full.py` or `--mode full`) — it
deliberately does **not** silently index a different corpus.

### 9c. Docker (Redis + API + UI)

```bash
docker compose up --build       # or: docker compose up
```

Starts three services from `docker-compose.yml`, all bound to localhost:

- `redis` — `redis:7-alpine` on `127.0.0.1:6379`
- `api` — `uvicorn api:app` on `127.0.0.1:8000`
- `ui` — `streamlit run app.py` on `127.0.0.1:8501`

Keys are read from your host environment / `.env` (`GROQ_API_KEY`,
`MISTRAL_API_KEY`, `API_KEY`, `ALLOWED_ORIGINS`). Set `API_KEY` before exposing
the API beyond localhost. Tear down with:

```bash
docker compose down
```

---

## 10. Troubleshooting

**Judge health check failed / LLM-judged metrics missing.**
Usually a mis-set or retired `JUDGE_MODEL`. The config default is
`openai/gpt-oss-20b`; set `JUDGE_PROVIDER` / `JUDGE_MODEL` (in `.env`) to a model
your account can access. The current CI workflow does not invoke a judge or set a
judge model; its demo plots are not evidence that these thresholds passed.

**`Synthetic dataset not found at ... Run --mode full first`.**
`eval`/`cicd` need `data/synthetic_dataset.json`. Build it via `--mode full`
(§5/§7), `preprocess_full.py` (§6a), or the `run_module1(...)` one-liner (§5c).

**Missing vector store for `eval`/`cicd`.**
They load `data/chroma_db_full`. Build it first (§6a). Remember
`train_in_shards.py` builds `data/chroma_db_train`, which these modes do **not**
load.

**`cicd` gate fails with missing LLM-judged metrics.**
If no LLM key is configured at all, `run_quality_gate` cannot construct the
judge: judged metrics are absent and their thresholds evaluate as 0.0, so the
gate fails. Set `GROQ_API_KEY` (free), `OPENROUTER_API_KEY`, or
`MISTRAL_API_KEY`.

**Groq rate limits (HTTP 429).**
The generator throttles calls using `LLM_RATE_LIMIT_SECONDS` (default `3.0`
seconds between calls). Increase it (in `.env`) if you still hit limits:
`LLM_RATE_LIMIT_SECONDS=6.0`.

**Redis not running.**
`SemanticCache` logs a warning and falls back to an in-memory dict, so runs
still work. For persistence, start Redis: `docker compose up redis`
(binds `127.0.0.1:6379`, matching the default `REDIS_HOST`/`REDIS_PORT`).

**`preprocess_full.py` fails to load the embedder.**
It runs the embedder in offline mode. Warm the model cache first with the §5
FAST smoke `--mode full` (downloads `all-MiniLM-L6-v2`), then re-run
`preprocess_full.py`.
