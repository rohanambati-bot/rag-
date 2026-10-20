# F1 — atomic request-start generation fencing

Date: 2026-10-19. Python 3.12.14. This is a scoped engineering receipt, not a
production/security sign-off. No commits, pushes, deployments, provider calls,
private dotenv reads, dependency installs, or changes to the shared environment
were performed. The main-owned `tests/test_audit_regressions.py` fixture repair
was not edited. Historical receipts were retained.

## Contract and implemented boundary

- `begin_generation()` captures a random shared revocation epoch **before** lazy
  architecture construction, retrieval, generation, and generation-related query
  embedding. Tokens also bind cache instance/process, namespace, local epoch, and
  index/config identity.
- Ordinary successful stores change a separate random **snapshot version**. They
  do not revoke unrelated outstanding tokens. Atomic clear changes both the
  snapshot and revocation epochs. Restart/missing authority creates a new random
  epoch; persisted entries carry their publication epoch and cannot be recertified
  under a recreated authority.
- `publish_generation()` compares the captured revocation epoch and publishes
  entry/index/snapshot in **one Redis Lua execution**. A clear between the final
  Python read and this script causes rejection. There is no read-then-write
  publication race. Shared-mode publication does not optimistically insert local
  rows; local rows are loaded only from a generation-bracketed snapshot.
- `lookup_generation()` verifies authority before and after scoring. Standalone
  `lookup()` under `redis_failure_policy="miss"` also uses that boundary. Polling
  opt-outs cannot bypass supported request-token lookups.
- API `/query`, including `use_cache=false`, `/ab-test`, and `/cache/lookup` use
  request tokens. The UI chat, cache lookup, A/B/comparison, diagnostics, and
  uncached private-upload response paths capture/validate tokens. CachedRAG uses
  token publication by default. API failures are sanitized HTTP 503; UI chat
  suppresses obsolete answers and shows its generic failure message; other UI
  paths abort before rendering a revoked answer.
- Serving identity covers source content, manifest, configuration, architecture
  implementation, and model contract. Manifest-only changes are checked even
  when SQLite is unchanged. Source/manifest stat keys include ctime. A changed
  identity rejects old resources; this remains stop-serving/restart publication,
  not hot swapping. Private uploads bind the retained uploader vectorstore
  identity and upload digest without changing uploader implementation.
- A Redis outage or lost publication acknowledgment rejects the response. An
  ambiguous write **may have committed**, but is never retried/acknowledged as
  success. A server-side clear with a lost acknowledgment still fences another
  worker with stale local bookkeeping. A failed clear rotates the caller's local
  epoch but does not claim confirmed shared revocation.

**Chosen response policy:** fail closed. Uncached `/query` and A/B answers are
rejected when their captured authority is obsolete or unavailable. Consequently
healthy shared Redis is required for these supported answer-serving paths even
when answer reuse is disabled.

The response linearization point is its final authority validation. A clear
ordered before that validation rejects the response. Bytes/results already
linearized before clear cannot be retracted, and expensive provider work is not
cancelled. This is a cache/current-response guarantee, not physical erasure of
every historical artifact, old Redis namespace, prompt, log, or returned answer.

## Compatibility and unsupported direct-store generation boundaries

The new Redis namespace is `rag_cache:v4`; old v3 entries are never treated as
current v4 entries, and existing historical keys are not destructively migrated.
The following compatibility paths are explicitly outside request-start fencing:

- `SemanticCache.store()` / `store_with_citations()` called **after** external
  generation: capture at store time cannot identify pre-clear work. The atomic
  adapter still publishes conditionally, but such a caller must migrate to
  `begin_generation` **before** generation to obtain the end-to-end guarantee.
- `CachedRAG(..., allow_legacy_cache=True)` is explicit compatibility opt-in.
  Supported API/UI defaults never enable it, and reject adapters without the token
  interface rather than falling back to three-argument stores.
- Legacy Redis adapters lacking Lua retain only their explicit compatibility
  behavior. Strict serving cannot start generation or confirm a shared clear
  through them. `redis_failure_policy="local"` / disabled polling availability
  lookups remain potentially stale across workers and are not serving defaults.
- The CLI cache demo in `main.py`, the synthetic local microbenchmark in
  `scripts/validate_serving.py`, existing lifecycle-runner direct stores / generic
  command dispatch in `scripts/validate_live_lifecycle.py`, and direct-store cache
  unit fixtures are compatibility consumers. These files were not edited by F1.

Custom CachedRAG callers must supply a truthful namespace and index/config
identity provider for every mutable source they use. Arbitrary direct Redis
writers, clients bypassing the API, unobserved mutations that deliberately reuse
the same identity, and nonparticipating index readers are outside this contract.
Redis Cluster cross-slot Lua operation is not supported; verification used
standalone Redis 7.2.16.

## Files changed by F1

Production: `phase3/semantic_cache.py`, `phase3/cicd_pipeline.py`, `api.py`,
`app.py`, `serving.py`.

Existing cache/API tests: `tests/test_api_production.py`,
`tests/test_cache_namespaces.py`, `tests/test_cache_index_lifecycle.py` (Redis
double only), `tests/test_exposure_security.py` (UI cache fixture identity only).

New: `tests/test_generation_fencing.py` (31 deterministic cases),
`scripts/validate_generation_fencing.py`, this report, and create-only receipts:
`outputs/security-v4-fencing-20261019.json`,
`outputs/security-v4-fencing-20261019-v2.json`,
`outputs/security-v4-fencing-20261019-v3.json`.

No F1 edits to module2, uploader/parser, preprocessing/index builders, `main.py`,
dependencies, shared documentation, Docker, or the existing live lifecycle runner.
Other owners concurrently changed their assigned files.

## Actual checks

Final focused offline command:

```bash
PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 GROQ_API_KEY= MISTRAL_API_KEY= \
OPENROUTER_API_KEY= OPENAI_API_KEY= .venv/bin/python -m pytest -q \
  tests/test_generation_fencing.py tests/test_semantic_cache.py \
  tests/test_cache_namespaces.py tests/test_cache_index_lifecycle.py \
  tests/test_api_production.py tests/test_exposure_security.py
```

Result: **152 passed, 1 warning, 6.06 s**. The warning is the existing community
Chroma deprecation. Events/barriers establish the new race-test order; there are
no new arbitrary synchronization sleeps. Cases include current cached-hit
scoring races, same-process clears, concurrent clears, overlapping source scopes,
outage/initial outage, lost acknowledgments, restart, missing authority with old
persisted entries, numeric-looking epochs, foreign tokens, API cache bypass,
lazy constructors, UI cache bypass, and private upload replacement/clear.

Final real-process command (destination was new when executed):

```bash
PYTHON_DOTENV_DISABLED=1 .venv/bin/python scripts/validate_generation_fencing.py \
  --redis-server .venv/lifecycle-tools-t3-20261019/redis-7.2.16/src/redis-server \
  --output outputs/security-v4-fencing-20261019-v3.json
```

Result: **49 assertions passed**, **11 distinct Python worker processes**, **4
barrier-synchronized concurrent clears**, Redis **7.2.16**. All resources were
temporary and owned. Redis readiness used its ready log event plus PID verification;
generation/publication ordering used multiprocessing events, pipes, and a barrier.
Only owned child processes were stopped. Expected injected outage/lost-clear
warnings were visible. The final receipt includes SHA-256 for the implementation,
runner, and new test source; earlier receipts are retained.

```bash
.venv/bin/python -m ruff check --select E,F,W --ignore E501 \
  phase3/semantic_cache.py phase3/cicd_pipeline.py api.py serving.py app.py \
  tests/test_generation_fencing.py tests/test_cache_index_lifecycle.py \
  tests/test_cache_namespaces.py tests/test_api_production.py \
  tests/test_exposure_security.py scripts/validate_generation_fencing.py
git diff --check
```

Both passed. The first focused attempt had two fixture/response-string failures
(119 passed); those were corrected and subsequent focused checks passed. The
first multi-file UI patch attempt was refused without modifying files; relevant
files were reread before retrying.

An additional `make test` during concurrent A1/P1 implementation was **not green**:
**62 failed, 542 passed, 3 skipped, 17 warnings** (14.17 s). Failures were 61 old
answer-boundary evaluation expectations and one document-ingestion per-upload
error expectation, in other owners' changing files. Those files were not edited
by F1. This receipt does not claim a final whole-workspace pass; the coordinator
must rerun after integration. The supplied 526-passed/3-skipped baseline and
main-owned CI fixture fix were not replaced or reset.
