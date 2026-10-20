# F1 final receipt — atomic generation fencing

This final receipt supersedes the intermediate F1 counts in
`security-v4-fencing-20261019.md`. That checkpoint and all four JSON runner receipts
are retained. The contract, compatibility boundaries, direct-store inventory,
whole-workspace failure record, and no-provider/no-dependency-change restrictions
documented in the checkpoint still apply.

## Final changes and behavior

- Added request-start `GenerationToken`, strict lookup/validation, and atomic Lua
  conditional publication. Clear rotates a random revocation epoch separately
  from the snapshot version that ordinary stores change. Tokens bind
  instance/process, namespace, index/config identity, and local/shared epoch.
- Shared publication cannot optimistically populate local state. Persisted entries
  carry their revocation epoch, so recreating missing authority cannot recertify
  old persisted entries. Outage/uncertain publication rejects responses; a lost
  clear acknowledgment cannot permit a remote stale worker to publish.
- API `/query` (including cache bypass), `/ab-test`, and cache lookup use tokens.
  UI chat, A/B/comparison, diagnostics, private uploads, and cache lookup capture
  and validate tokens. CachedRAG is strict by default; old adapters require its
  explicit `allow_legacy_cache=True` opt-in, which is outside this guarantee.
- **Final stale-worker hardening:** ServingContext binds **all** architecture
  identities when resources first open/build. It cannot create a previously cold
  architecture cache under a new index/config identity while retaining an old
  loaded corpus or architecture stack. UI caches use the same bound identity.
  Changed resources require a serving restart, as before.

The chosen response policy is fail closed, including uncached answers. Supported
answer serving now requires verifiable Redis authority even with cache reuse
disabled. The final authority check is the response linearization point: a clear
ordered before it rejects the response. Already-returned/previously-linearized
answers cannot be retracted; generation/provider work is not cancelled. The
boundary is participating v4 namespaces and truthful identity providers, not
arbitrary direct Redis writers, physical erasure of historical artifacts, or hot
index replacement. Standalone Redis was verified; Cluster cross-slot Lua was not.

## Exact F1 files

Production:
`phase3/semantic_cache.py`, `phase3/cicd_pipeline.py`, `api.py`, `app.py`, `serving.py`.

Existing tests:
`tests/test_api_production.py`, `tests/test_cache_namespaces.py`,
`tests/test_cache_index_lifecycle.py` (atomic Redis double only),
`tests/test_exposure_security.py` (UI identity fixture only).

New code/tests:
`scripts/validate_generation_fencing.py`, `tests/test_generation_fencing.py`
(**32 new deterministic cases**).

New evidence:
`outputs/security-v4-fencing-20261019.json`,
`outputs/security-v4-fencing-20261019-v2.json`,
`outputs/security-v4-fencing-20261019-v3.json`,
`outputs/security-v4-fencing-20261019-v4.json`,
`outputs/security-v4-fencing-20261019.md`, and this final receipt.

F1 did not edit `tests/test_audit_regressions.py`, module2, uploader/parser,
preprocessing/index builders, `main.py`, dependencies, shared docs/Docker, the
existing lifecycle runner, or the shared virtual environment's dependencies.

## Final verification

```bash
PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 GROQ_API_KEY= MISTRAL_API_KEY= \
OPENROUTER_API_KEY= OPENAI_API_KEY= .venv/bin/python -m pytest -q \
  tests/test_generation_fencing.py tests/test_semantic_cache.py \
  tests/test_cache_namespaces.py tests/test_cache_index_lifecycle.py \
  tests/test_api_production.py tests/test_exposure_security.py
```

**153 passed, 1 existing Chroma deprecation warning, 5.96 s.**

```bash
PYTHON_DOTENV_DISABLED=1 .venv/bin/python scripts/validate_generation_fencing.py \
  --redis-server .venv/lifecycle-tools-t3-20261019/redis-7.2.16/src/redis-server \
  --output outputs/security-v4-fencing-20261019-v4.json
```

**51 assertions passed; 11 distinct Python worker processes; 4 barrier-synchronized
concurrent clears; Redis 7.2.16; Python 3.12.14.** Temporary owned resources only;
ready-log/PID verification, events, pipes and barriers establish ordering. The
runner contains no arbitrary synchronization sleeps. The v4 JSON has final source
SHA-256s. Its two new assertions capture a separate remote worker token immediately
before a server-committed clear with a lost acknowledgment, then verify that
remote publication is fenced.

```bash
.venv/bin/python -m ruff check --select E,F,W --ignore E501 \
  phase3/semantic_cache.py phase3/cicd_pipeline.py api.py serving.py app.py \
  tests/test_generation_fencing.py tests/test_cache_index_lifecycle.py \
  tests/test_cache_namespaces.py tests/test_api_production.py \
  tests/test_exposure_security.py scripts/validate_generation_fencing.py
git diff --check
```

Scoped Ruff and whitespace checks passed. A mid-integration `make test` was
**62 failed, 542 passed, 3 skipped, 17 warnings**: 61 failures in the other owner's
old answer-boundary evaluation expectations and one in document-ingestion
per-upload error expectations. Those tests/implementations were changing in
parallel and were not edited by F1. A final integrated full-suite pass is the
coordinator's remaining check; the focused results are not represented as a full
workspace pass. The supplied baseline fixture repair was preserved.

## Direct stores still outside request-start fencing

The CLI cache demo (`main.py`), local synthetic cache microbenchmark
(`scripts/validate_serving.py`), existing lifecycle-runner fixture stores/command
dispatch (`scripts/validate_live_lifecycle.py`), direct-store unit fixtures, and
explicit legacy CachedRAG mode remain compatibility consumers. Direct `.store()`
or `.store_with_citations()` after external generation cannot identify work that
began before clear; migrate such generation callers to request-start tokens.
Supported API/UI/CachedRAG defaults do not use that fallback. Legacy local
availability lookups may remain stale during outage; serving selects strict mode.
