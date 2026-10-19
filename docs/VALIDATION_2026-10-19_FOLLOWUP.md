# Validation follow-up — 19 October 2026

This supplements, rather than replaces, [the original validation report](VALIDATION_2026-10-19.md).
It describes the uncommitted working tree on `fix/audit-repairs`, starting from
HEAD `76287c8`. It is not a security sign-off, release or clean-Linux rebuild.
No generation/judge provider requests, fabricated human labels, deployment,
push, reset or cleanup of historical evidence was performed.

## 1. Ingestion and persistence verification

The new focused ingestion/cache/index checks were run against the existing
project virtual environment, Python **3.12.14**:

- Ingestion, upload isolation, cache lifecycle, semantic cache, full-preprocessing
  persistence, API and integrity selection: **128 passed, 6 warnings**.
- Additional cache namespaces, generation boundaries and audit regressions:
  **41 passed**.
- Scoped Ruff and compilation: passed. No repair was needed for those focused
  tests. These counts overlap the full suite; do not add them to the suite count.

### Enforced upload limits

`module5/document_upload.py` now has bounded extracted text as well as the
existing 20 MiB encoded-upload limit:

| Boundary | Default |
|---|---:|
| Extracted UTF-8 text per upload | 8 MiB |
| Extracted characters per upload | 2,000,000 |
| Extracted UTF-8 text per document/page | 2 MiB |
| Extracted characters per document/page | 500,000 |
| Parsed documents | 10,000 |
| PDF pages | 200 |
| DOCX ZIP entries | 1,024 |
| DOCX total declared uncompressed size | 50 MiB |
| DOCX declared size per member | 10 MiB |
| DOCX compression ratio | 100:1 |

DOCX inspection happens without extraction to disk and rejects unsafe ZIP member
paths and malformed ZIPs before `python-docx` receives them. PDF handles and
fallback streams are closed on failure. A failed replacement preserves the
previous upload and owned collection. Real ephemeral Chroma tests verify
separate uploader collections; parser-limit tests use local fixtures.

These are application limits, **not** an OS sandbox or hard CPU/RSS deadline.
Native parser vulnerabilities and decompression/XML/PDF work before text is
returned remain relevant. Do not accept hostile public uploads on the strength
of these tests alone.

### Cache and index lifecycle

- Redis writes and clears increment an architecture/source-scoped generation.
  Warm workers check the token and reload local entries when it changes.
  `generation_poll_interval=0` checks each lookup, a positive interval bounds
  polling frequency, and `None` disables polling. `refresh_generation()` forces
  a check. Zero TTL uses persistent Redis `SET`.
- A Redis outage retains process-local entries. This is availability fallback,
  **not** an immediate cross-worker privacy/revocation guarantee. No live Redis
  multi-worker test was available; deterministic Redis doubles were exercised.
- `BuildLock` serialises participating builders with bounded waits. Atomic JSON
  writes protect manifests/checkpoints. Marked staging cleanup preserves active
  or unknown-owned directories; failed publication restores the previous store.
- This does not make two directory renames an atomic reader transaction. Stop
  serving processes before rebuilding and restart them after publication.

## 2. Partial dependency remediation

The before/after audit receipts are separate and preserved:

- Before: `outputs/dependency-audit-20261019.json` — **148 advisory matches across
  20 packages**.
- After: `outputs/dependency-audit-20261019-after-safe-upgrades.json` — **56 known
  advisory matches across 9 packages**, still a failing audit.

These are advisory matches, not 148 or 56 independently proven exploitable CVEs.
Aliases occur in the raw JSON. The scanner output does not supply trustworthy
per-advisory severity or full affected-version ranges, so neither was invented.
Reachability below is source-based triage, not exploit testing or a suppression.

### Applied versions

| Package | Before | After |
|---|---:|---:|
| FastAPI | 0.115.6 | 0.147.0 |
| Starlette | 0.41.3 | 1.9.0 |
| Streamlit | 1.40.2 | 1.66.0 |
| Pillow | 11.3.0 | 12.3.0 |
| pypdf | 5.1.0 | 6.21.1 |
| python-multipart | 0.0.19 | 0.0.32 |
| python-dotenv | 1.0.1 | 1.2.2 |
| multidict | 6.7.1 | 6.9.1 |
| oauthlib | 3.3.1 | 4.0.0 |
| Tornado | 6.5.8 | 6.5.10 |
| urllib3 | 2.7.0 | 2.8.0 |
| pytest | 8.3.4 | 9.0.3 |

FastAPI was upgraded to admit the fixed Starlette series. The resolver also
added `itsdangerous==2.2.0` and advanced the already-ranged direct NetworkX
dependency from 3.6.1 to 3.7. `requirements.lock` was regenerated from `pip freeze`
and checked for exact agreement with that environment. Direct security pins
were added to `requirements.txt`; no advisory was ignored to improve the count.

The targeted installed parser/API/upload selection passed **80 tests** after the
upgrade. The final full-suite and `pip check` results are in section 5.

### Still unresolved

| Package | Matches | Current boundary / follow-up |
|---|---:|---|
| Chroma 1.5.9 | 4 | The recorded code-injection/authorisation advisories concern Chroma server endpoints and model-repository input. This project uses embedded Chroma, not an exposed Chroma service; do not deploy that server as a workaround. The receipt lists no fixed versions. |
| datasets 3.1.0 | 1 | Folder-builder path traversal. Dataset loading is used; untrusted folder-based dataset metadata must not be accepted. A tested 5.1.0 upgrade conflicts with the retained Transformers stack's Hugging Face Hub `<1` requirement. |
| fsspec 2024.9.0 | 1 | Reference-filesystem template processing. No direct Kerchunk/reference-file feature was found, but this is not a clean scan. datasets 3.1.0 caps fsspec at 2024.9.0; migrate them together. |
| LangChain 0.3.30 | 1 | Path-loading/search helpers; these vulnerable helpers were not found in the application's direct call paths. A compatible major-stack migration is still needed. |
| langchain-core 0.3.86 | 2 | Prompt-file config loading and image token-count URL fetching; this project uses Document/Embeddings and plain prompt strings, not those directly named paths. |
| langchain-text-splitters 0.3.11 | 1 | URL/redirect handling in an HTML splitter. The application uses RecursiveCharacterTextSplitter, not the URL loader. |
| sentence-transformers 3.3.1 | 1 | Model import/trust bypass. Model loading **is** used; require trusted pinned weights and avoid attacker-selected model directories/repositories. |
| Torch 2.5.1 | 19 | Includes checkpoint deserialisation and native operation issues. This project loads local models; major-version testing and trusted weights remain necessary. |
| Transformers 4.46.3 | 26 | Includes config/checkpoint loading and specialised processor/conversion paths. Core model loading is reachable; do not dismiss the whole package as unused. |

Exact advisory IDs, aliases, descriptions and supplied fix versions remain in
the after-scan JSON. Version availability was checked, but a large ML/LangChain
migration was **not** applied without compatibility evidence. Retaining the
research model stack is not a vulnerability exemption. Some advisories list no
fix version. A clean scan is not claimed.

`pip-audit==2.9.0` was run from an isolated tool directory with `--disable-pip
--no-deps --requirement requirements.lock`. The direct spaCy model URL is skipped
by this scanner. Audit lookup needs internet; the test suite does not. No image
or operating-system vulnerability scan was run.

## 3. UI access boundary

The new `.streamlit/config.toml` sets the normal `streamlit run app.py` entry
point to **127.0.0.1**, keeps CORS/XSRF protections enabled and caps an encoded
upload at 20 MiB. Two tests check the TOML and the actual Streamlit config parser.

The UI directly calls the Python RAG stack. **The FastAPI API key does not
authenticate Streamlit.** Docker still binds inside its container on `0.0.0.0`,
but Compose publishes API, UI and Redis ports only on host loopback. Do not
override those host bindings, tunnel the UI publicly or put an unauthenticated
reverse proxy in front of it. This is a local-development boundary, not a public
multi-tenant authentication system.

Uploaded collections are session-owned and not inserted into the global chat
cache. Shared-corpus caches are architecture/source scoped, not tenant scoped;
do not extend them to personalised/private data without an explicit privacy and
authorisation design.

## 4. Controlled five-way retrieval experiment

New implementation:

- `scripts/retrieval_ablation.py` — shared contract, candidate provenance,
  create-only receipts, five conditions and local/replay adapters.
- `scripts/run_retrieval_ablation_local.py` — real local data/model runner.
- `tests/test_retrieval_ablation.py` — offline experimental-boundary checks.
- `tests/fixtures/retrieval_ablation_smoke.json` — explicitly fake smoke fixture;
  its scores are never research evidence.

The real run is in **`outputs/retrieval-ablation-local-20261019-v2/`**:
`contract.json`, `run_metadata.json`, `per_query.jsonl` and `condition_summary.csv`.
All historical validation and evaluation summaries were left in place.

### Contract and interpretation

- Same seeded 500 selected question IDs as the historical sample; the four
  selected questions without qrels are **excluded**, leaving **496 judged queries
  per condition**, 2,480 result rows. IDs such as `msmarco_83` are the loader's
  stable positions within the answerable subset, not official MS MARCO IDs.
- Same first 1,000 answerable validation-query corpus: 9,858 source passages,
  11,510 source chunks, 400-character chunks/50-character overlap.
- This new protocol keeps the first occurrence of each existing content-prefix
  ID: 74 duplicate IDs are removed, leaving **11,436 unique corpus records**.
  Corpus/qrels/filter/model/package fingerprints are recorded. This is a
  separately versioned protocol, not an undisclosed edit to the historical run.
- Human passage `is_selected` labels are still inherited onto chunks; there are
  no new human chunk relevance annotations. No gold text/labels are fed into the
  embedding model or cross-encoder; the contract corpus metadata is empty.
- MiniLM snapshot `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU; BGE reranker
  snapshot `2cfc18c9415c912f9d8155881c133215df768a70`, MPS. Research model package
  versions remain those in the earlier validation; the new lock hash is recorded.
- BM25 uses the harness's explicit local Okapi scoring (`k1=1.5`, `b=0.75`,
  positive `log(1 + ...)` IDF and stable corpus-order ties), **not** a claim to
  reproduce every detail of production `rank-bm25`. Hybrid is unweighted RRF,
  constant 60, over the deduplicated union of dense top-50 and BM25 top-50.
- Reranking scores the entire supplied candidate pool before truncation. Every
  hybrid reranked/non-reranked pair had the **identical ordered candidate pool**
  in the saved results (496/496). Dense top-k was the wide dense-pool prefix for
  496/496 queries. Model revisions and all shared fingerprints also agree.
- Metrics use `module3.evaluation` and a fixed final `k=5`. MRR@10/nDCG@10 see
  only up to five returned documents, not an actual depth-10 run. Deduplication
  leaves one dense-only list (`msmarco_505`) with three results and no backfill;
  its Precision@5 denominator is still five. This accounts for a small nDCG
  difference from the old duplicate-credit baseline.

### Descriptive results, n=496

| Condition | nDCG@10 (depth ≤5) | MRR@10 (depth ≤5) | Recall@5 | Hit@5 |
|---|---:|---:|---:|---:|
| Dense-only | 0.631370 | 0.611391 | 0.799765 | 0.889113 |
| Dense pool + cross-encoder | 0.698600 | 0.688575 | 0.855376 | 0.947581 |
| BM25-only | 0.415000 | 0.379738 | 0.601613 | 0.683468 |
| Hybrid without reranking | 0.553363 | 0.526949 | 0.736593 | 0.826613 |
| Same hybrid pool + cross-encoder | 0.698972 | 0.688878 | 0.856720 | 0.949597 |

On this fixed corpus, reranking the **same hybrid pool** raises nDCG by about
0.145610 and Hit@5 by 0.122984. Adding BM25 to the already-reranked dense pool
changes Hit@5 by only 1/496 and nDCG by about 0.000372. This is descriptive
evidence, not a significance/generalisation or production-adoption claim. No
new hyperparameter or production retriever setting was selected from this run.

All five metrics for every row were recomputed from the saved rankings and qrels:
maximum difference **0.0**. There are no duplicate returned IDs or IDs outside
their candidate pools. No provider generation/judge calls or model downloads
were made. The 1,060.988-second total is a single local run, not a latency study.

### Preserved intermediate runs and latest runner check

`outputs/retrieval-ablation-local-20261019/` is retained as an intermediate run:
it included the four unjudged questions as zero-score rows and is **not** the
reported denominator. The `-v2` run corrects that error without overwriting it.
The first real smoke exposed a duplicate `local_files_only` keyword in the
CrossEncoder adapter; the adapter now uses its top-level argument and has a
regression test.

The full `-v2` run queried the existing Chroma directory without rebuilding it.
Chroma can update SQLite metadata while querying, so a byte-for-byte unchanged
source store is **not** claimed. The latest runner uses a private temporary copy
and explicitly loads the embedding snapshot directory. Its copied-store path
was exercised on five real queries in
`outputs/retrieval-ablation-local-pinned-copy-smoke-20261019/`, reproducing the earlier
five-query metric values. The complete 496-query experiment was not repeated
solely for the copied-store safety change.

## 5. Final checks and reproduction

Working-tree checks after the feature groups:

| Check | Actual result |
|---|---|
| `make test` | **336 passed, 1 skipped, 17 warnings** |
| `make lint` | Passed |
| `.venv/bin/python -m pip check` | No broken requirements |
| `.venv/bin/python -m compileall -q .` | Passed |
| `git diff --check` | Passed |
| Dependency audit | **Still failing: 56 matches across 9 packages** |

The skipped full chunk ablation is opt-in and overwrites its normal evidence
paths; it was not enabled. Warnings include five PyMuPDF binding deprecations,
eleven pre-existing SciPy precision warnings and one LangChain Chroma deprecation.
The new experiment tests and real retrieval runs are additional evidence, not
external-provider answer-quality tests.

From the project root, with the pinned runtime and installed lockfile:

```bash
make test
make lint
.venv/bin/python -m pip check
.venv/bin/python -m compileall -q .
git diff --check

# Explicitly FAKE, fast offline harness smoke; choose a fresh destination.
.venv/bin/python scripts/retrieval_ablation.py \
  --manifest tests/fixtures/retrieval_ablation_smoke.json \
  --output-dir outputs/retrieval-ablation-fixture-repeat-1

# Real local cached dataset, model snapshots and built store required.
# The script refuses an existing output directory; no historical CSV is replaced.
.venv/bin/python scripts/run_retrieval_ablation_local.py \
  --samples 500 --out outputs/retrieval-ablation-real-repeat-1

# Local-only UI. These options do not add application authentication.
.venv/bin/python -m streamlit run app.py

# Isolated audit tooling; internet required for installation/advisory lookup.
.venv/bin/python -m pip --isolated install \
  --target .venv/audit-tools-next pip-audit==2.9.0
PYTHONPATH=.venv/audit-tools-next .venv/bin/python -m pip_audit \
  --disable-pip --no-deps --progress-spinner off --format json \
  --output outputs/dependency-audit-next.json --requirement requirements.lock
```

The existing `.venv` provides Python 3.12.14. This machine's bare pyenv `python`
shim cannot locate that version; the exact project interpreter above works.
Follow the README runtime installation instructions on a new machine rather than
copying this host's interpreter or absolute paths.

## 6. Remaining blockers and next work

1. **Security:** compatible ML/LangChain/datasets/fsspec migrations and advisory
   range/severity verification remain open. Keep the application local-only.
2. **Linux/container reproducibility:** Docker is unavailable on this host.
   Docker build/startup, Compose connectivity and clean-Linux installation remain
   unverified. The lockfile is still a macOS environment freeze; Linux-only Torch
   dependencies are not fully enclosed. Do not call this a clean Linux rebuild.
3. **Redis:** no local `redis-server`; a real multi-process invalidation/outage/
   recovery test is still required.
4. **Research:** add an explicitly prespecified statistical comparison family,
   retain this versioned dedup protocol and use a genuinely held-out test before
   selecting/adopting changes. This corpus is still a validation subset with
   inherited chunk labels and in-domain model-training exposure.
5. **Answer quality:** the fixed human-label subset, annotation rubric, judge
   calibration and authorised provider budget are still missing. No semantic
   correctness/faithfulness/citation precision or Cohen's kappa result exists.
6. **Adversarial boundaries:** deterministic prompt-injection, contradictory-
   evidence, malformed-response and claim-level citation checks still need a
   focused implementation. Parser isolation and hard request/SDK deadlines are
   not supplied by this work.

No commits, pushes, merges or deployments were made in this continuation.
