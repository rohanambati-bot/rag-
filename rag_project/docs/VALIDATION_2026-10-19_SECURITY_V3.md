# Security and integration continuation — 19 October 2026

## Status and preserved scope

This is a continuation of `VALIDATION_2026-10-19_FOLLOWUP.md`, not a replacement
for its historical protocol/results. It is **not a security or production sign-off**.
Work was performed with Python 3.12.14 on macOS arm64, with provider credentials
blanked for tests/experiments. No generator/judge calls, human labels, deployment,
commits, pushes, merges, resets or destructive workspace cleanup were performed.

Initial and final branch/HEAD: `fix/audit-repairs`, `76287c8`
(`test: record reproducible RAG validation evidence`). The workspace was already
dirty, with substantial tracked edits and untracked tests/research receipts. All
were retained; the final staged diff is empty. Immediately before adding this
report, `git diff --stat` showed 33 modified tracked files, 2,486 insertions and
543 deletions **including inherited work**. Those are not this continuation's
exclusive line counts. New scripts, tests and receipts remain untracked too.
This report is another new untracked file. Reproduce live state with
`git status --short`, `git diff --stat`, `git diff --cached --stat` and
`git log -1 --oneline`; no claim is made that HEAD contains these changes.

Preserved historical files include `outputs/eval_results.csv`,
`outputs/significance_tests.csv`, `outputs/evaluation_report.md`, `outputs/plots/`,
all existing validation/ablation directories and intermediate receipts. New
outputs use separate names. The retrieval supplement records SHA-256 hashes of
its four historical input files. Existing reports have not been rewritten.

## 1. Priority security findings

These are engineering priorities, **not invented advisory CVSS classifications**.
The scanner receipt does not establish trustworthy severity/affected-range data.

| Finding | Result and boundary |
|---|---|
| UI bypassed API authentication and shared answer caches | UI is explicitly unauthenticated/local-only, validates effective settings/session headers, and owns per-session/per-architecture answer caches. No remote UI authentication was invented. |
| Partial API protection/malformed credentials | Nonpublic routes are authenticated before body parsing, including docs/OpenAPI, architectures, metrics and cache routes. Invalid/missing server configuration fails closed; Unicode, oversized, duplicate and malformed headers are tested. |
| Failed shared clear reported success | API now returns 503 when Redis invalidation is not confirmed; UI warns instead of claiming a successful shared clear. |
| Redis stale snapshots/reconnect/restart epochs | Generation-bracketed snapshot loading, fresh random epochs, atomic namespace clear and reconnect are exercised with actual separate processes. Serving/UI choose misses during Redis failure. |
| Renaming an index under active Chroma handles | Participating readers retain POSIX lifetime leases. Publication refuses active readers; arrivals during publication/rollback are refused before opening Chroma. This enforces stop-serving rebuilds, not hot swapping. |
| Vulnerable coupled ML/loader dependencies | 56 advisories remain open. No untested major migration or advisory suppression was substituted for a verified fix. |
| Prompt injection/citation semantics | Deterministic diagnostics expose existing weaknesses; lexical attack echoes, contradictory evidence selection and malformed model-output passthrough remain. No complete injection defense or semantic correctness claim. |

API public routes remain `/`, `/health`, `/healthz` and `/readyz`; probes sanitize
internal warmup errors. Correlation IDs are bounded/sanitized. Generic client
errors do not echo underlying exceptions. Local unauthenticated API opt-out
requires a numeric loopback configured bind, loopback peer and no forwarded
headers; production/container commands disable implicit proxy-header trust.

UI guard limitations: checks guard application execution, not every Streamlit
HTTP endpoint. Loopback is not authentication. A tunnel/proxy rewriting traffic
to appear local is not reliably detectable and is unsupported. Sessions have
UUID-scoped caches and uploader collections, but there are no authenticated
tenant identities. Redis entries persist until TTL; disconnect is not deletion.

## 2. Dependency remediation and unresolved advisories

Fresh final receipt: `outputs/dependency-audit-20261019-v3-final.json`.
Current triage: `outputs/dependency-security-v3-final-audit/` (`report.md`,
`report.json`, `advisory_matrix.csv`, `evidence.json`). Earlier triage/replay/decision
receipts remain intact. The tool records transitive alias deduplication,
installed versions, scanner-supplied fixes, shortest dependency paths from
installed metadata, static import evidence and explicit open dispositions.

| Installed package | Open advisories |
|---|---:|
| chromadb 1.5.9 | 4 |
| datasets 3.1.0 | 1 |
| fsspec 2024.9.0 | 1 |
| langchain 0.3.30 | 1 |
| langchain-core 0.3.86 | 2 |
| langchain-text-splitters 0.3.11 | 1 |
| sentence-transformers 3.3.1 | 1 |
| torch 2.5.1 | 19 |
| transformers 4.46.3 | 26 |
| **Total** | **56** |

Initial and final audits both contain 56 matches/9 packages; deduplication leaves
56 unique advisories. Eighteen rows have no scanner-supplied fixed version.
Installed/lock/audit consistency issues: zero. Severity/full affected ranges are
explicitly unknown, not fetched or verified upstream in this continuation.
Static import evidence is not an exploit test. Embedded Chroma use does not
prove server advisories irrelevant, and absent direct imports do not establish
transitive unreachability. The direct spaCy model URL is an audit coverage gap.

No further dependency fixes were installed here. The earlier reduction from
148 matches/20 packages to 56/9 belongs to the preceding continuation. Existing
requirements/lock were left unchanged because the remaining migrations are
coupled: datasets constrains fsspec, torchvision pins Torch, Sentence Transformers
excludes fixed Transformers 5.x, and LangChain-HuggingFace excludes fixed core
1.x. Removing direct LangChain alone retains its community dependency. No new
isolated candidate installation or clean-Linux resolution was performed.

## 3. Files changed in this continuation

This groups the current continuation's implementation, not every inherited diff.

| Files | Purpose |
|---|---|
| `api.py`, `security_boundary.py`, `configs/config.py` | Early fail-closed auth, safe probes/errors, effective exposure checks, failed-clear receipt handling. |
| `app.py`, `.streamlit/config.toml`, `.env.example` | Local-only guard, per-session cache/upload ownership, replacement digest and clear warnings; UI selects fail-closed cache policy. |
| `phase3/semantic_cache.py` | Redis epochs, snapshot/concurrency handling, atomic clear, reconnect, explicit failure policy/receipts. |
| `module1/data_preprocessing.py`, `preprocess_full.py` | Reader/publication leases and manifest build identities. |
| `serving.py`, `main.py` | Source/configuration/implementation-scoped serving cache; separate CLI demo namespace. |
| `module2/rag_architectures.py` | Extracted existing prompt-budget logic into `_generation_evidence()` for exact prompt-visible slice diagnostics; no claimed generation-quality fix. |
| `scripts/dependency_security_report.py` | Offline evidence-backed advisory matrix and replay. |
| `scripts/validate_live_lifecycle.py` | Owned temporary Redis/Chroma resources and real-process verification. |
| `scripts/evaluate_answer_boundaries.py`, `tests/fixtures/answer_boundaries_v1.json` | Frozen 64-case deterministic suite, structural diagnostics and blank human review. |
| `scripts/verify_saved_retrieval.py` | Independent saved-ranking metrics and retrospective paired supplement. |
| `scripts/validate_serving.py` | Correctly distinguishes local microbenchmark fallback from serving outage misses; blanks credentials and avoids automatic private dotenv reads. |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `Makefile` | API-only container default, no exposed UI service, local project installs, offline tests, native UI command and excluded outputs/private board. |
| `README.md`, `RUNBOOK.md`, this report | Updated run/deployment boundaries, reproduction commands and honest remaining blockers. |

New tests: `test_dependency_security_report.py`, `test_exposure_security.py`,
`test_live_lifecycle.py`, `test_answer_boundaries_eval.py`,
`test_deployment_boundary.py`, `test_saved_retrieval_verification.py`.
Extended tests cover auth/UI, cache/index lifecycle/namespaces, generation slices
and semantic-cache behavior. `test_env_loading.py` deliberately re-enables dotenv
only for explicit temporary fixture files, so the full suite can otherwise
disable automatic private `.env` reads. The session integration test uses the
same explicit no-telemetry ephemeral Chroma settings as the upload tests.

## 4. Final checks and failures encountered

| Check | Actual result |
|---|---|
| `make test` | **522 passed, 3 skipped, 17 warnings**, 10.93 s in final implementation run. |
| `make lint` | Passed. |
| `.venv/bin/python -m pip check` | No broken requirements. |
| `git diff --check` | Passed. |
| Fresh pip-audit against `requirements.lock` | **Fail: 56 advisories/9 packages**. |
| Final offline advisory transformation | 56 unique open advisories, 0 consistency issues. |
| Real process lifecycle runner | Redis and index-reader sections passed. |
| Answer-boundary runner/replay | 64 cases, 547 assertions, 0 regression failures. |
| Saved-ranking supplement | 2,480 rows; maximum historical metric error 0.0. |
| `docker --version` | `/bin/bash: docker: command not found`. Build/run not attempted. |

Offline skips: one opt-in expensive chunk ablation, two opt-in real lifecycle
tests. The latter were run independently through the live runner. Warnings:
five PyMuPDF SWIG deprecations, one community Chroma deprecation, eleven existing
SciPy precision-loss warnings for near-identical paired samples.

The first combined run exposed two dotenv fixture failures and eleven ephemeral
Chroma settings errors (503 passed, 3 skipped). Those were fixed as described in
section 3; the final suite is not that failed run. The first independent retrieval
supplement attempt rejected missing top-level `filter_fingerprint`; the verifier
was corrected to hash the actual filter metadata, then rerun with zero error.
An attempted final-audit replay using older evidence was correctly refused on
input hash mismatch; new final evidence was captured instead of overriding the
check. An intermediate HTTP profile (`serving-validation-20261019-v3.json`)
contained noisy, artificially disconnected microbenchmark writes; it is preserved
but superseded by `-v4` for microbenchmark timing. No failure was deleted or
relabeled a pass.

## 5. Real Redis and index integrations

Receipt: `outputs/live-lifecycle-20261019-v3.json`. This was rerun in the main
session against Redis **7.2.16**, built only under ignored project-local tooling.
Tarball SHA-256 verified against the Redis project's `redis/redis-hashes` listing:
`960a8ec15e34ff40e57ff16837b26b33bd81f2da6d24497bb63de532a323a18e`.

- **10 distinct Python worker processes**, **96 concurrent writes**, **16
  overlapping clears**; not threads or mocked Redis.
- Update/clear visibility, namespace isolation, outage misses, explicitly
  stale-local availability fallback, failed-clear receipt, initial-outage
  reconnect, nonpersistent restart epoch recovery, concurrent readers/writers,
  and final empty-state verification all passed.
- Real temporary Chroma stores: **2 active readers blocked publication**;
  **4 arriving readers were refused** across publication/rollback gaps; newly
  published corpus was visible afterwards; injected promotion failure retained
  the previous corpus.
- The runner starts/verifies its own Redis PID and terminates only owned child
  processes. It uses temporary stores, never an existing live Redis or corpus.

Limits: leases apply to participating POSIX/local-filesystem clients, not direct
`Chroma(...)` opens or arbitrary network filesystems. Readers must exit before
publication; there is no atomic hot swap. Serving cache identities now include
source content/manifest, configuration and architecture implementation digests,
but optional model revisions are not actual immutable-weight fingerprints.
Non-serving consumers retaining `redis_failure_policy="local"` can serve stale
answers during outages. Polling can be disabled; that is not revocation safety.

**Still open:** generation begun before clear can later repopulate its answer.
A hard privacy-revocation guarantee requires a request-start revocation epoch
and atomic conditional store/fence. The current clear operation does not cancel
in-flight work or provide that guarantee. API/UI now expose failed clear receipts
honestly, but do not claim privacy-safe erasure of in-flight generations.

## 6. Docker and clean Linux

Static deployment tests pass: non-root API-only image, libgomp runtime package,
public API liveness probe, proxy-header trust disabled, loopback host publications,
and no unauthenticated Compose opt-out. Compose pins Redis to `7.2.16-alpine`.
The UI service was removed because a published container UI requires a
non-loopback internal bind that its unauthenticated runtime guard rejects.
`make ui` is the supported host-local UI path. Runtime version and existing
requirements lock remain project-local; installs now target `.venv`.

**Not verified:** clean Linux resolution/install, container build, container
startup/readiness, non-root volume permissions and fresh model/data downloads.
The lock remains a macOS freeze, not hash-complete cross-platform dependency
closure; platform-only Torch dependencies are not independently locked. Docker
is absent. Image tags are not immutable digest pins, and model/dataset caches
used for profiling were pre-existing host caches. This remains a release blocker.

## 7. Answer/adversarial evaluation

Frozen output: `outputs/answer-boundaries-v1-20261019/`; independent rerun:
`outputs/answer-boundaries-v1-20261019-replay/`. Dataset/schema/runner version
`1.0.0`, schema `answer-boundaries.dataset/v1`.

Dataset SHA-256:
`4ee85c94a08c6d577cd7fe2c71df0705704f997cdf41026b650b35ea8eda69d0`.
Schema SHA-256:
`1bb85f4f01fe497cbcfb4fdcbed1dad9629ddf4731ec024434e513e29fa83681`.

64 authored synthetic cases: eight each for supported, unsupported, abstention,
citation probes, contradiction, evidence scoping, retrieved instructions and
exfiltration requests. **55 exercise real extractive generation; 9 use scripted
invocation-seam answers.** All 547 structural/provenance/lexical regression
assertions pass. These are expected-behavior checks, **not 64 safe/correct answers**.

Observations (not quality/attack success scores): 15 exact abstention strings,
9 cases with observed synthetic attack/canary literals, 10 cases containing
retrieved documents outside prompt-visible evidence. Scripted diagnostics flag
2 out-of-range references, 1 out-of-scope reference, 3 malformed citations,
5 missing citations and 1 empty answer; counts overlap. Diagnostics are an
offline checker, not a production enforcement gate. Retrieval citations may
include documents excluded from generation, so structural citation existence
is not claim-level support.

`human_review.jsonl` provides 64 anonymized records, each with two blank reviewer
slots; `human_review_key.json` maps them back for later analysis. Answer correctness,
faithfulness, citation support, appropriate abstention, contradiction handling,
instruction resistance and exfiltration resistance remain **unscored/null**.
No human judgments, kappa, live-model quality, actual secret leakage or judge
ground truth were invented. No provider or judge calls were made.

### Independent retrieval supplement

`outputs/retrieval-supplement-20261019-v3/` independently recomputes binary IR
metrics without importing production metric functions. All **2,480 saved rows**
(496 queries × five conditions) agree with historical metric values: maximum
absolute error **0.0**. Query order, qrels against the contract, corpus/pool
membership, duplicate IDs and metadata fingerprint agreement are checked.
All 496 paired hybrid candidate pools are identical.

This is a separate, explicitly **retrospective** nDCG@5 supplement. Historical
nDCG@10 remains untouched and was computed on at most five saved results, not
a new depth-ten retrieval. For this data the means coincide, but their definitions
remain distinct. Original labels are inherited passage labels, not new human
chunk judgments. No reranker/model/dataset was rerun for this supplement.

| Saved or derived condition | Mean nDCG@5 |
|---|---:|
| dense_only (historical direct top-five pool) | 0.631369541 |
| dense_reranked | 0.698599932 |
| bm25_only | 0.414999842 |
| hybrid_no_rerank | 0.553362635 |
| hybrid_reranked | 0.698972402 |
| dense_same_pool_no_rerank_derived | 0.631901936 |

Original dense-only and reranked supplied pools were top-five versus top-fifty;
they were not literally a common pool. The additional baseline truncates the
**saved dense-reranked candidate order** to five, so it shares the exact wide
pool with that reranked condition. It is derived evidence, not a fresh experiment.
It differs from historical dense-only for exactly `msmarco_505` (earlier top-five
deduplication returned only three results). The historical result is preserved.

Three specified retrospective paired contrasts use 10,000 seeded paired
bootstrap draws for individual descriptive intervals and 10,000 two-sided
sign-flip draws for p-values, with Holm familywise correction across the three:

| Right minus left | Mean delta | Individual bootstrap 95% interval | Holm p |
|---|---:|---|---:|
| Dense rerank − same-pool derived dense | 0.0666980 | [0.0396948, 0.0934427] | 0.000299970 |
| Hybrid rerank − hybrid | 0.1456098 | [0.1170005, 0.1734195] | 0.000299970 |
| Hybrid rerank − dense rerank | 0.0003725 | [−0.0012224, 0.0023398] | 1.0 |

The last comparison does not support a hybrid-over-dense reranking advantage
on this sample. Intervals are not simultaneous; the family was not preregistered.
Monte Carlo p-values have finite resolution and are not exact asymptotic tests.

## 8. Performance before/after and profiling limits

Current HTTP receipt: `outputs/serving-validation-20261019-v4.json`.
Stage receipt: `outputs/stage-profile-20261019-v3.json`.
Both use temporary private copies of the persisted store, cached local models,
five fixed questions and blank provider credentials. Device: embedding CPU,
reranker MPS. HTTP service work limit is two, rate limit raised for the bounded
workload. No latency-driven candidate reduction or production algorithm change
was made; the cross-encoder remains the measured retrieval bottleneck.

| Observation | Current measurement |
|---|---:|
| Process start → liveness / readiness | 1.072 s / 5.058 s |
| HTTP naive, concurrency 2, cache disabled, n=20, p50 / p95 | 16.053 / 18.042 ms |
| HTTP rerank, concurrency 2, cache disabled, n=10, p50 / p95 | 2,029.339 / 3,747.748 ms |
| HTTP repeat requests, cache requested, Redis down, concurrency 2, n=20, p50 | 41.078 ms; **0 hits**, intentionally fail closed |
| Dense top-five stage, n=10, p50 | 10.817 ms |
| Hybrid first-stage candidates, n=10, p50 | 21.594 ms; 86–91 candidates |
| Cross-encoder stage, n=10, p50 | 1,021.589 ms |
| Stage process peak RSS | 995.469 MiB |

All 130 measured HTTP requests succeeded; one additional priming request was
made, and an unauthenticated query returned 401. Repeated-cache runs are **not**
cached-hit latency: Redis is intentionally unavailable and all 61 requested
lookups miss. Outage errors are recorded, not hidden. The cache's internal
average lookup statistic currently omits early fail-closed returns (reports
0.0 here); use the external HTTP timer, not that incomplete statistic.

The local synthetic 32-dimensional cache microbenchmark, 100 lookups per size,
measures explicit local availability fallback with Redis disabled after initial
connection checking: means 0.028739 / 0.042641 / 0.052918 ms at 100/1,000/5,000
entries. These are not serving outage behavior, real embedding cost or live
Redis round trips. One initialization connection error per size is retained.

Previous follow-up reported ~15.44 ms naive and ~2,297.67 ms rerank HTTP p50 at
concurrency two. Current descriptive numbers are **not a measured improvement**:
warmup/cache state, stronger safety behavior and exact conditions were not
controlled in a paired before/after benchmark. Historical cached-hit figures
must not be compared to new fail-closed outage misses. No speedup is claimed.
HTTP timers include middleware, admission/queueing, retrieval, extractive
generation and serialization together; separate per-stage generation,
serialization and queue costs, live-provider generation latency/cost, sustained
capacity and a healthy-Redis end-to-end benchmark remain unmeasured.

## 9. Ranked remaining risks and next steps

1. **Dependency risk and clean-Linux closure:** migrate the coupled ML, loader
   and LangChain stacks in an isolated environment, pin platform-complete
   dependencies and immutable model revisions, enrich severities/ranges from
   authoritative advisories, then run the real corpus tests and Linux image.
2. **Deployment/identity:** no authenticated UI, tenant authorization or complete
   proxy/TLS trust policy. Keep the UI strictly local and untunnelled. Verify
   the API-only image on a clean Linux Docker host before release.
3. **Revocation:** add request-start fencing/conditional stores before claiming
   clears revoke in-flight generations. Do not enable stale local fallback or
   disabled generation polling for privacy-sensitive shared answers.
4. **Answer safety:** malformed inline references are not enforced, retrieved
   instructions can be echoed and contradictory evidence can be mishandled.
   Obtain real blinded human labels; only then report correctness, faithfulness,
   citation support, appropriate abstention or agreement statistics. A prompt
   instruction alone is not a security boundary.
5. **Ingestion/native execution:** parser limits and ZIP preflight do not sandbox
   native parsers or guarantee CPU/RSS deadlines. Isolate hostile inputs and
   untrusted model artifacts in appropriately restricted processes/containers.
6. **Reader integration coverage:** stop-serving POSIX leases do not protect
   arbitrary direct Chroma clients or prove hot-swap correctness. Integrate all
   readers/builders with the publication contract before broadening that claim.
7. **Performance evidence:** add healthy-Redis full-path benchmarks, correct
   early-miss telemetry and use equivalent repeated before/after workloads;
   preserve rankings and fixed candidate budgets for any optimization.

## 10. Exact reproduction commands

Run from the project root using Python 3.12.14 and project-local dependencies:

```bash
python3.12 -m venv .venv             # only for a fresh checkout/environment
.venv/bin/python -m pip install --requirement requirements.lock
make test
make lint
.venv/bin/python -m pip check
git diff --check
git status --short
git diff --cached --stat
```

The following destinations must be new; do not overwrite existing evidence:

```bash
PYTHON_DOTENV_DISABLED=1 .venv/bin/python scripts/evaluate_answer_boundaries.py \
  --output-dir outputs/answer-boundaries-v1-new-run
.venv/bin/python scripts/verify_saved_retrieval.py \
  --source outputs/retrieval-ablation-local-20261019-v2 \
  --output outputs/retrieval-supplement-new-run
PYTHON_DOTENV_DISABLED=1 .venv/bin/python scripts/dependency_security_report.py \
  --audit outputs/dependency-audit-20261019-v3-final.json \
  --output-dir outputs/dependency-security-new-run
```

Fresh audit (network/advisory access needed; tooling install is not part of the
offline test command). Tooling is separate from runtime dependencies:

```bash
.venv/bin/python -m pip install --target .venv/audit-tools-v3 pip-audit==2.9.0
test ! -e outputs/dependency-audit-new-run.json && \
  PYTHONPATH=.venv/audit-tools-v3 .venv/bin/python -m pip_audit \
  --disable-pip --no-deps --progress-spinner off --format json \
  --output outputs/dependency-audit-new-run.json --requirement requirements.lock
```

Actual live lifecycle reproduction with the already built project-local binary:

```bash
PYTHON_DOTENV_DISABLED=1 .venv/bin/python scripts/validate_live_lifecycle.py \
  --redis-server .venv/lifecycle-tools-t3-20261019/redis-7.2.16/src/redis-server \
  --output outputs/live-lifecycle-new-run.json
LIVE_LIFECYCLE=1 \
  REDIS_SERVER=.venv/lifecycle-tools-t3-20261019/redis-7.2.16/src/redis-server \
  PYTHON_DOTENV_DISABLED=1 .venv/bin/python -m pytest -q tests/test_live_lifecycle.py
```

For a clean machine, provide Redis 7.2.16 or build its verified tarball in a new
project-local tooling directory. Compiler/make are required; no global install:

```bash
mkdir .venv/lifecycle-repro-v3
curl --fail --location https://download.redis.io/releases/redis-7.2.16.tar.gz \
  --output .venv/lifecycle-repro-v3/redis.tar.gz
.venv/bin/python -c "import hashlib; from pathlib import Path; assert hashlib.sha256(Path('.venv/lifecycle-repro-v3/redis.tar.gz').read_bytes()).hexdigest() == '960a8ec15e34ff40e57ff16837b26b33bd81f2da6d24497bb63de532a323a18e'" && \
  tar -xzf .venv/lifecycle-repro-v3/redis.tar.gz -C .venv/lifecycle-repro-v3
make -C .venv/lifecycle-repro-v3/redis-7.2.16 -j4 MALLOC=libc BUILD_TLS=no OPT=-O2 redis-server
PYTHON_DOTENV_DISABLED=1 .venv/bin/python scripts/validate_live_lifecycle.py \
  --redis-server .venv/lifecycle-repro-v3/redis-7.2.16/src/redis-server \
  --output outputs/live-lifecycle-fresh-build.json
```

Stop on download/digest/build failure. The runner reports Redis absence as an
explicit skip if no binary is supplied; that is not a passing Redis verification.

Profiles require the existing corpus and pre-cached model/dataset files and do
not download or call providers. Their outputs are create-only:

```bash
PYTHON_DOTENV_DISABLED=1 .venv/bin/python scripts/validate_serving.py \
  --stages-only --out outputs/stage-profile-new-run.json
PYTHON_DOTENV_DISABLED=1 .venv/bin/python scripts/validate_serving.py \
  --out outputs/serving-validation-new-run.json
```

Local run/deployment instructions are in README: `make ui` for the host-only UI,
`API_KEY=<generated-secret> .venv/bin/python api.py` for authenticated local API,
and `docker compose up --build` with `API_KEY` exported for API + Redis. Prepare
the intended corpus first and stop readers before rebuilding. Docker commands
here are reproduction instructions **not commands successfully run here**.
