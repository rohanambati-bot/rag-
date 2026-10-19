# Independent validation — 19 October 2026

This review started from the existing working tree, not a clean checkout. Prior
test/benchmark statements were treated as hypotheses. No provider generation or
judge requests were authorized or made during this validation. Research labels,
sampling, chunking and retrieval model choices were not tuned or changed.
Source repairs remain in the working tree; these receipts describe that tree,
not certification of a clean HEAD or a released build.

## 1. Starting state and checks

- Branch `fix/audit-repairs`; HEAD `25b10201935ab3000c119deb6e327a7c37eae302`.
- Initially 29 modified tracked files, 4 untracked files, nothing staged. The full
  tracked diff and all four untracked files were read before additional edits.
- `make test`: **247 passed, 1 skipped, 11 warnings in 9.89 s**.
- `make lint`: passed. `python -m pip check`: no broken requirements.
- `git diff --check`: passed. Compose/workflow YAML parse successfully; that is
  syntax validation, **not** a Docker build or proof that CI passed remotely.
- Runtime: Python **3.12.14**, macOS arm64. Linux rebuild remains unverified.

## 2. Verified defects and bounded improvements

1. **Upload isolation, high severity.** Distinct uploaders used Chroma's default
   ephemeral collection, exposing each other's documents. Reindexing accumulated
   old content and clear did not delete it. UUID-named owned collections now isolate
   uploads; publication is staged, failed replacements retain the prior upload,
   and replacement/clear delete only owned collections. Real local Chroma tests
   cover isolation, cleanup, failure and preservation of unrelated collections.
2. **Cross-architecture cache contamination.** A Naive answer was returned with
   `architecture="rerank"` for the same query. API/UI serving now separates caches
   by architecture and a restart-time source/model contract. Redis namespaces are
   separated too; unsafe legacy unscoped entries are not loaded by serving. The
   source signature is size/mtime, **not a cryptographic corpus identity**. Stop
   readers and restart after any store publication; no hot-reload guarantee exists.
   Cache capacity is per architecture scope. Zero TTL now uses Redis SET rather
   than invalid SETEX(0), and same-process writes/clears serialize write-through.
3. **Invalid keyless answers.** Empty evidence extracted the prompt's instruction
   preamble as an answer. Actual evidence headers are now matched, no/blank
   evidence abstains before any provider call, and context separators count toward
   the cap. This changes keyless answer behavior, **not retrieval results**. It does
   not make lexical extraction a semantic correctness/faithfulness evaluator.
4. **Actual lazy-model race.** Locking the reranker wrapper did not protect lazy
   CrossEncoder weight construction. A controlled concurrent reproduction built
   two models. Per-instance model-load locking now attempts initialization once.
5. **Silent stale-handle rebuild.** A same-process Chroma rebuild could publish new
   files but return the old corpus through cached path-bound handles. Reloaded
   content is now checked before deleting the backup; mismatch fails with a
   fresh-process instruction and rolls back. This is fail-safe detection, **not a
   hot-swap implementation**. Empty-directory loads no longer create empty stores;
   non-object JSON manifests no longer raise AttributeError.
6. **Evaluation cost/inference integrity.** A supplied judge was still called by
   `evaluate_architecture(..., retrieval_only=True)`; judging is now explicitly
   suppressed. A failed main-CLI judge preflight now stops rather than continuing
    paid evaluation. P-values retain full precision through BH correction instead
    of first rounding small nonzero values to zero. Metric definitions are unchanged.
7. **Authentication input boundary.** A non-ASCII incorrect API-key header raised
   TypeError instead of 401; constant-time byte comparison now rejects it normally.

## 3. Exact experiment contract and limitations

The dataset is `microsoft/ms_marco`, config `v2.1`, validation split (101,093 rows).
The locally cached dataset revision is
`a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a`, fingerprint `ec570a8962845fb8`.
The standard experiment uses the **first 1,000 answerable queries**, yielding
**9,858 deduplicated passages and 11,510 400-character/50-overlap chunks**.
Despite its directory name, this local `chroma_db_full` is **not** the full
101,093-query corpus. Its manifest is absent, so build provenance is incomplete.

The dense store's complete text multiset exactly equals the reconstructed sparse
corpus: no missing or extra chunks. Both systems receive the same seeded 500
questions (`numpy.default_rng(42).choice(1000, 500, replace=False)`); all 500 match
the historical per-query CSV. No answer text or selection label is included in
the text sent to the embedding model or cross-encoder. Metadata contains labels
and gold answers, so future retrievers must not consume that metadata as features.

Models used from existing caches:

- MiniLM: `sentence-transformers/all-MiniLM-L6-v2`, snapshot
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, normalized embeddings, CPU.
- Cross-encoder: `BAAI/bge-reranker-base`, snapshot
  `2cfc18c9415c912f9d8155881c133215df768a70`, Apple MPS.
- Chroma 1.5.9 (L2 collection), sentence-transformers 3.3.1,
  transformers 4.46.3, torch 2.5.1, rank-bm25 0.2.2.

Naive returns dense top-5. Rerank scores the union of dense top-50 and BM25 top-50,
deduplicated by 200-character prefix, and returns five. Its first-stage ordering
is dense-first union, **not RRF**. This is a comparison of two complete systems;
it does **not** isolate the cross-encoder from BM25 or candidate-pool widening.

Metrics follow `module3/evaluation.py` literally:

- IDs are a lowercased, stripped, first-200-character content hash.
- Qrels inherit passage-level `is_selected` onto **every** child chunk. Relevant
  chunk labels are not independently annotated, and the official full MS MARCO
  passage-ranking leaderboard protocol is not reproduced.
- Recall@k uses **k = number returned**, so k=5 for Naive/Rerank. Precision uses
  the same variable denominator; lexical CRAG sometimes returns fewer documents.
- nDCG@10/MRR@10 receive only five retrieved documents for Naive/Rerank: no ranks
  6–10 are retrieved. The labels do not mean these are depth-10 retrieval runs.
- 994 of 1,000 queries have nonempty constructed qrels. Of the sampled 500,
  **496 are scored**; four unjudged questions are excluded, not scored as failures:
  `how old do you have to be to be a server`, `how to bake 2 5lb chickens`,
  `temperature in clearwater, fl`, `temperature by cricket chirps`.
- Corpus dedup discards 125 passage occurrences, including **55 distinct suffixes**
  sharing a prefix and **2 cross-query positive occurrences**. Chunking leaves 74
  exact duplicate chunk texts/IDs. These defects were **not silently corrected**
  in this replay because doing so changes the experiment and requires versioned
  qrels/corpus/results. Duplicate credit is a potential metric weakness; bounds
  held in the inspected historical run.
- Eval/dev questions are disjoint under the documented split, but both index
  passages from the same validation corpus. There is no separate independently
  collected held-out test here. The [MiniLM model card at the used snapshot](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/raw/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md)
  explicitly lists **9,144,553 MS MARCO training triplets**. That establishes
  in-domain training exposure, not proof of overlap with these validation queries.
  The [BGE snapshot card](https://huggingface.co/BAAI/bge-reranker-base/raw/2cfc18c9415c912f9d8155881c133215df768a70/README.md)
  describes multilingual pair training but does not resolve exact query overlap.
  Claiming clean out-of-domain/test-set generalization is invalid.

## 4. Answer evaluation scope

Historical `outputs/raw_eval_results.csv` contains 2,000 rows and **zero generated
answers**: it cannot establish answer correctness or faithfulness. The replay fixes
and hashes a 50-question `answer_subset_v1.json` **before** evaluating keyless
extractive answers. Those receipts are explicitly not paid-model results.

Metadata-backed citations prove an index/source relationship to returned context,
not that every cited source supports an answer claim. All retrieved documents are
listed, including documents beyond a generation context cap or irrelevant to the
answer. No semantic citation precision, faithfulness or human-supported answer
correctness score is established by structural citation validity or token overlap.

There is no human-labelled judge-validation sample in the repository. Cohen's
kappa cannot responsibly be calculated from invented human labels, continuous
judge scores without a prespecified discretization, or labels assigned by the same
judge. Next: freeze the subset, blind two human annotators to architecture, define
categorical claim-support/abstention labels and adjudication, then run the chosen
judge with an approved budget. Report raw agreement and Cohen's kappa for an
appropriate two-rater categorical design (weighted kappa for ordinal labels),
with confidence intervals and prevalence limitations. A health-check prompt only
proves an endpoint responds numerically; it does not validate a judge.

## 5. Security/deployment evidence

`pip-audit==2.9.0` queried the default PyPI advisory service against the lockfile:
**148 advisory matches across 20 packages**, nonzero exit. Full result:
`outputs/dependency-audit-20261019.json`. One URL requirement (`en_core_web_sm`)
was skipped. Matches require reachability/alias triage; they are not 148 proven
remotely exploitable bugs. Examples include pypdf (49), transformers (26), torch
(19), Pillow (18), Starlette (7), python-multipart (6), Chroma (4). Some fixes
require major-version stack migrations; Chroma matches list no fix versions.
No mass upgrade or suppression was made to hide this result. The audit tool and
its dependencies were installed only under `.venv/audit-tools`, not over the app's
dependencies; `pip check` is a separate compatibility check, not a vulnerability
scan. No Docker/OS image vulnerability scan was performed.
Secret exclusions and logging were reviewed, but a full Git-history secret scan
and a separate comprehensive static-security scan were not performed.

Docker commands fail exactly with `/bin/bash: docker: command not found`.
Consequently neither an image build, container startup, Compose service connectivity
nor a clean Linux lockfile installation has been verified. The lockfile was frozen
on macOS: Linux-only dependencies (e.g. CUDA/transitive wheels) are not proven fully
locked. Model/dataset revisions are recorded by the replay but not pinned/enclosed
by the default loaders. Existing dependencies/models/data are required for offline
replay; a bare clone is not a self-contained benchmark dataset/model distribution.

Remaining risks:

- Shared static API key, no per-user authorization/RBAC or tenant namespaces.
  Streamlit uses the Python serving stack directly, not the protected API; it is
  not an authenticated public multi-tenant frontend. Keep localhost bindings.
- Local Chroma is single-node. Publishing uses two renames, not an atomic reader
  transaction. Do not rebuild while serving; concurrent builders/checkpoint writers
  remain uncoordinated. A second full-build cleanup can remove another's staging
  directory. Checkpoint/manifest writes are not transactional/fsynced.
- Redis is write-through and startup snapshots, **not live invalidation**. Tests
  reproduce a warm worker returning an entry after another worker clears Redis.
  Do not use cache clear as an immediate privacy/revocation guarantee. TTL expiry
  and LRU are local; the Redis index set can retain expired members.
  Similarity thresholds also do not prove that an answer is reusable for changed
  numbers, entities or dates. Provider-failure extractive fallbacks are not tagged
  distinctly in cached answers. Freshness/false-hit calibration remains open.
- Per-scope capacity multiplies worker memory; vector buffers alone are not a
  memory ceiling (Python floats, answers, contexts and model memory add overhead).
  Expiry compaction and exact-key refresh can rebuild ANN under a lock.
- Worker threads continue after a response timeout/cancellation; tests prove the
  semaphore stays held, not that work is interrupted. Request budgets currently
  apply to separate stages, not a single hard end-to-end deadline. SDK timeouts,
  process isolation and shutdown/drain behavior need dedicated tests.
- Extension/magic-byte/encoded-size validation does not bound decompressed DOCX
  expansion or PDF page/CPU cost. Vulnerable parsers make untrusted uploads risky.
- Prompt injection remains model/instruction based. The extractive fallback can
  quote malicious evidence sharing query keywords. There is no adversarial safety
  guarantee. Judge prompts and permissive numeric parsing also need validation.
- Logger redaction covers named structured extras, not arbitrary message or
  traceback contents. Public readiness errors and high-cardinality route labels
  need hardening. Do not log private prompts, provider payloads or credentials.
- TLS, reverse-proxy limits, distributed rate limits, token/cost accounting and
  deployment secret-management are not supplied by this local prototype.

## 6. Reproduce without overwriting historical evidence

From the project root, after installing the pinned runtime and local environment:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --requirement requirements.lock
make test
make lint
python -m pip check
git diff --check

# Existing cached dataset/models and data/chroma_db_full required; never pays APIs.
# Choose a NEW --out directory each time; existing receipts are refused.
python scripts/validate_research.py --samples 500 --out outputs/validation-repeat-1
python scripts/summarize_validation.py outputs/validation-repeat-1

# Bounded local HTTP subprocess, isolated copied store, blank provider credentials.
python scripts/validate_serving.py --out outputs/serving-repeat-1.json
python scripts/validate_serving.py --stages-only --out outputs/stages-repeat-1.json

# Scanner is separate from the runtime environment; advisory lookup needs internet.
python -m pip --isolated install --target .venv/audit-tools pip-audit==2.9.0
PYTHONPATH=.venv/audit-tools python -m pip_audit --disable-pip --no-deps \
  --progress-spinner off --format json --output outputs/dependency-audit-repeat-1.json \
  --requirement requirements.lock
```

Later, on a clean Linux host with Docker, a populated validation store and writable
data/output mounts (UID 1001), using placeholders rather than real keys in files:

```bash
docker compose config --quiet
docker build --no-cache -t rag-validation .
# The image can be tested without a model/dataset download or API key.
docker run --rm --network none --entrypoint python rag-validation -m pytest -q
# Compose startup requires the prebuilt store; a bare clone cannot be ready.
API_KEY="$(openssl rand -hex 24)" docker compose up --build -d
docker compose ps
curl --fail http://127.0.0.1:8000/healthz
curl --fail http://127.0.0.1:8000/readyz
curl --fail http://127.0.0.1:8501/_stcore/health
docker compose down
```

Do not execute `main.py --mode demo`/`make eval-retrieval` over working evidence:
they overwrite historical output paths. Demo plots use built-in numbers and are
not regenerated research evidence. The new scripts use private copied stores and
new output paths, never rebuild/rename the user's index, and do not read `.env`
credentials into provider clients.

## 7. Fresh retrieval replay and statistical uncertainty

Receipts are in `outputs/validation-20261019/`. The run completed all **2,000
architecture/query attempts with zero exceptions**; 500 per architecture, 496
judged per architecture. The other four rows remain unjudged. `rankings.json`
records ranked/relevant IDs so scores can be independently recomputed without a
model. Recomputed scores agree within 1.12e-16; comparison against the historical
CSV finds **zero per-query difference** for all four requested metrics on all
available scored rows. The historical CSV alone could not establish this because
it did not save rankings. Summary:

| System | nDCG@10 (returned depth 5) | MRR@10 (returned depth 5) | Recall@5 | Hit@5 |
|---|---:|---:|---:|---:|
| Naive | 0.632380 | 0.611391 | 0.799765 | 0.889113 |
| Rerank | 0.698972 | 0.688878 | 0.856720 | 0.949597 |
| Parent-Child fallback | 0.631902 | 0.611391 | 0.800773 | 0.889113 |
| CRAG, lexical/no rewrite | 0.574988 | 0.570329 | 0.701647 | 0.786290 |

The last two rows are controls, not faithful evaluation of an independently built
parent-child index or LLM-graded CRAG. One Naive query returns duplicate IDs;
Rerank has no returned duplicates. No dedup-induced metric correction was applied.

Paired query bootstrap: 10,000 resamples, seed 42, percentile 95% interval on the
**difference** (not overlap of two separate system CIs). Two-sided Wilcoxon
signed-rank tests and paired t-tests are saved unrounded. BH adjusts the three
baseline comparisons within each metric, not across the four metrics jointly.

| Rerank minus Naive | Mean difference | Paired bootstrap 95% CI | Wilcoxon p (BH) |
|---|---:|---|---:|
| nDCG@10 | +0.066592 | [+0.039591, +0.093234] | 2.6742e-6 |
| MRR@10 | +0.077487 | [+0.044892, +0.110384] | 1.9599e-5 |
| Recall@5 | +0.056956 | [+0.030746, +0.082661] | 4.0179e-5 |
| Hit@5 | +0.060484 | [+0.032258, +0.088710] | 6.6836e-5 |

Hit outcomes: baseline 441/496, rerank 471/496; **42 rescues, 12 lost baseline
hits**, exact two-sided McNemar p=5.20945e-5. The net miss reduction is 30/55
(54.5%), whereas the rescue rate is 42/55 (76.4%). Older wording conflated these.
The paired nDCG effect size d_z is 0.2181. Bootstrap/t-tests treat queries as units;
shared passages/topic dependence, inherited imperfect qrels and the selection of
answerable queries limit inference. Wilcoxon is a rank/symmetry-null test, not a
direct test of the arithmetic mean. These are conditional within-experiment
results, not proof of out-of-sample superiority or semantic answer quality.

## 8. Generated-answer proxies and measured performance

`answer_subset_v1.json` contains the fixed first 50 selected questions, with hash
`dee91026ad53d1699df69eabc6a6c1aaf6e8bb314d1910fbba6eb10c993bdd15`.
Only the real **keyless extractive generator** was exercised, not a mocked answer
or a live provider. `receipt_checks.json` reports:

| Descriptive proxy, n=50 each | Naive | Rerank |
|---|---:|---:|
| Normalized exact match to single reference | 0.020 | 0.020 |
| Multiset reference-token F1 | 0.2726 | 0.2602 |
| Valid returned-context citation indices | 250/250 | 250/250 |
| Non-abstaining answer is a literal retrieved-context span | 50/50 | 50/50 |
| Abstentions | 0 | 0 |

These do **not** establish factual correctness, entailment, semantic citation
precision or calibration. There are no unanswerable examples in this subset;
abstention is verified only in deterministic empty/unrelated-evidence regressions.
No meaningful judge agreement/kappa can be reported. The quality of a paid
generator remains unevaluated; provider token usage is not retained by the wrapper,
so actual paid-model cost per query is unavailable.

Performance receipts:

- Fresh 500-query retrieval replay (cached model files): Naive mean **7.43 ms**,
  median 7.14, p95 8.85; Rerank mean **1370.57 ms**, median 1286.16, p95 1939.66.
  Reranker weights were loaded before query timing; its first inference was
  2250.86 ms. Peak process RSS was **1536.13 MiB**; MPS allocations are not a
  separately sampled GPU-memory peak. Total run 702.65 s. This is **not** a
  before/after speedup against historical runs under other host load.
- Stage profile, n=10, five fixed queries repeated twice, cached weights/MPS:
  dense top-5 mean **10.26 ms**, wide dense+BM25 first stage **23.67 ms**, actual
  cross-encoder **1233.78 ms**. Candidate pools were 86–91 documents; first
  cross-encoder inference 1886.69 ms, last 1127.62 ms. Receipt:
  `outputs/retrieval-stages-20261019.json`. Unit tests ran concurrently with this
  short profile, so these are noisy diagnostic timings, not an isolated benchmark.
- Fresh real HTTP subprocess on a private copied store, one worker, semaphore=2,
  local Redis disabled, provider credentials blank. Process start to liveness
  **1.07 s**, to readiness **5.26 s** (cached models/OS disk cache, not first-ever
  install/download). RSS snapshots: **620.34 MiB** ready, **1338.13 MiB** after
  loading/using the reranker. These are snapshots, not a measured memory peak.

| HTTP workload | Client concurrency | Requests | Errors | p50 ms | p95 ms | requests/s |
|---|---:|---:|---:|---:|---:|---:|
| Naive, cache bypass | 1 | 20 | 0 | 10.28 | 13.13 | 91.51 |
| Naive, cache bypass | 2 | 20 | 0 | 15.44 | 19.35 | 123.98 |
| Naive, cache bypass | 4 | 20 | 0 | 30.86 | 35.09 | 126.00 |
| Naive, exact cached repeat | 1 | 20 | 0 | 9.12 | 11.65 | 104.18 |
| Naive, exact cached repeat | 2 | 20 | 0 | 14.35 | 19.31 | 131.33 |
| Naive, exact cached repeat | 4 | 20 | 0 | 27.64 | 33.31 | 142.27 |
| Rerank, cache bypass | 2 | 10 | 0 | 2297.67 | 3917.44 | 0.776 |

The HTTP workloads include real retrieval and keyless answer generation, not
network LLM generation. Rate limiting was deliberately raised to 100,000/min for
measurement; production defaults differ. Requests repeat five fixed questions;
the cached scenario is an intentionally primed exact repeat, **not a natural hit
rate estimate**. Cache stats are 60 hits/1 miss (98.36% including priming); each
measured cached batch hit 20/20. Unauthorized query returned 401. Receipt:
`outputs/serving-validation-20261019.json`. The local test subprocess was stopped;
this was not a deployment or a graceful-drain certification.

Synthetic deterministic 32-dimensional cache lookup microbenchmarks, 100 repeats:
mean **0.031 / 0.042 / 0.051 ms** at **100 / 1,000 / 5,000 entries**. These omit
real transformer embedding cost and Redis persistence. Actual HTTP cached queries
still embed the query, explaining their materially greater latency. No distributed
Redis test or production-capacity extrapolation is supported. External-provider
requests were zero; total compute/hosting cost per query was **not measured**.

## 9. Final checks and priorities

After implementation and documentation groups, the checks were repeated:

- `make test`: **297 passed, 1 skipped, 12 warnings** (recorded runs 9.11–10.99 s).
- `python -m pytest -q -rs`: **297 passed, 1 skipped, 12 warnings in 9.48 s**.
- `make lint`: passed. `python -m pip check`: passed.
- Python compilation of all three new validation scripts: passed.
- `git diff --check`: passed. Tracked historical summaries/significance/report/
  plots have no diff. No resets, clean operations, merges, pushes or deployments.
- The skipped full chunk ablation is opt-in via `RUN_DEFERRED_ABLATION=1`; it
  overwrites its normal output paths, so it was not invoked over research evidence.
- Warnings: the original 11 SciPy near-constant-difference precision warnings, plus
  one LangChain Chroma deprecation warning from the real upload isolation test.
- Regressions added: **50** (247 → 297), including actual ephemeral Chroma and
  controlled concurrency, not 50 new full integration or external-model tests.

Priority next steps:

1. **Before public exposure:** triage/fix the dependency audit (particularly upload
   parsers/web frameworks), bound decompressed uploads and parser work, protect the
   direct UI, and test a clean Linux image. Retain localhost-only access meanwhile.
2. **Research integrity:** version immutable query IDs, corpus/qrels/model revisions
   and content fingerprints; design a corrected dedup/label experiment as a
   separately versioned protocol. Do not silently replace the replicated baseline.
3. **Causal retrieval claim:** compare dense-only wide-pool and dense+BM25 same-pool
   systems with/without the cross-encoder on a prespecified protocol; decide any
   adoption on dev only, then run a separate held-out test once.
4. **Answer quality:** obtain blinded human labels on the frozen subset, approve a
   generator/judge budget, preserve raw usage and parse failures, add semantic
   correctness/citation/abstention rubrics, then assess judge agreement and kappa.
5. **Persistence/cache operations:** serialize builders and checkpoint publication,
   validate collection counts/content/embedding contracts, stop readers for index
   replacement, and define cross-worker invalidation requirements before adding
   infrastructure. Namespaces are not a substitute for invalidation or tenancy.
6. **Serving evidence:** run repeated idle-host concurrency/memory sweeps with
   production rate limits, real request distributions and approved provider usage;
   add a true end-to-end deadline, SDK budgets and tested shutdown draining.
