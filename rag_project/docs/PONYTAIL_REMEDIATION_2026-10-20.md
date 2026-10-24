# Ponytail remediation — 20 October 2026

## Scope and starting state

This is a remediation pass over the current workspace, not a replacement for the
historical validation reports. The repository was already dirty when this pass
resumed. I preserved tracked edits, untracked tests, datasets, checkpoints and
receipts; I did not reset, clean, commit, push, merge or deploy.

- Repository: `rag-`
- Branch: `fix/audit-repairs`
- Starting reported working-tree revision: `7890bc6`
- Current branch is one commit ahead of `origin/fix/audit-repairs`; no new commit
  was created during this pass.
- Runtime: `.venv/bin/python`, Python `3.12.14` (`.python-version`)
- Ponytail: already present as the workspace-local `.opencode/skills/ponytail`
  skill and the OmniRush Ponytail skill was loaded. It was not reinstalled or
  duplicated.

The starting workspace already contained the earlier security continuation:
generation fencing, parser isolation, answer validation, API/session hardening,
index lifecycle work, reports and tests. Those changes remain inherited work and
are not represented as a clean baseline commit.

## Findings and remediation

### 1. Unsafe Streamlit answer rendering — confirmed and already remediated

The risk was real: retrieved documents, uploaded documents and provider answers
are untrusted content. The current UI routes answer/context/cache text through
`_render_untrusted_markdown`, which calls Streamlit with
`unsafe_allow_html=False`. Static application chrome and CSS remain the only raw
HTML paths. The A/B page no longer interpolates answer text into an unsafe HTML
wrapper.

This fix was present in the inherited working tree when this pass resumed. The
existing regression suite covers HTML, script markup, event handlers, external
image/error attributes, entities, Unicode and ordinary Markdown, including the
placeholder path:

- `tests/test_streamlit_ui_security.py`
- `app.py::_render_untrusted_markdown`

This does **not** claim that every possible Streamlit/browser issue is eliminated.

### 2. OpenRouter UI consistency — confirmed and already remediated

The current UI imports and checks `OPENROUTER_API_KEY` alongside Groq and Mistral,
uses the same provider priority as the runtime, displays only presence booleans,
and enables RAGAS controls for OpenRouter-only configuration. This was also
present in the inherited working tree and is covered by the same focused tests.

### 3. Dense/BM25/graph/qrels corpus mismatch — confirmed and fixed

The mismatch was confirmed in the code paths: the full builder could write a
dense store while serving/evaluation rebuilt a configured development subset for
sparse and graph retrieval. Full builds also used generic manifest defaults.

The current fix uses the existing index manifest as a fail-closed contract:

- `module1/data_preprocessing.py` now records/validates dataset name/config,
  dataset revision, split, scope, query/document/chunk counts, chunk settings,
  index revision and a canonical chunk-corpus identity.
- Dense loading rejects missing/malformed/incomplete manifests and count drift.
- `documents_from_vector_store()` reconstructs the exact persisted dense chunk
  view, including the metadata required for qrels.
- `serving.py` derives sparse/graph/qrels side views from the persisted dense
  corpus and verifies the identity; it no longer silently pairs the dense store
  with a freshly loaded unrelated corpus.
- `main.py` preserves the historical MS MARCO development protocol while using
  the persisted exact chunk view for a full-scope manifest. Unknown scopes fail
  closed.
- `preprocess_full.py` writes a `scope="full"` manifest with the streamed corpus
  identity and query/document counts.
- `train_in_shards.py` writes train shard/full manifests with their own split and
  scope instead of masquerading as development indexes.
- Failed implicit reuse no longer silently rebuilds an existing incompatible
  store; explicit staging/publication remains responsible for replacement and
  the last valid store is retained.

Regression coverage includes matching and malformed manifest cases, interrupted
publication, stale index retention, eval-mode corpus wiring, lifecycle barriers,
and full-build batch construction. The large real full-corpus build was not run
in this pass.

### 4. Failed evaluation samples disappearing — confirmed and fixed

`EvaluationPipeline.evaluate_architecture()` previously logged an exception and
dropped the row. It now emits one structured row per selected sample with
`sample_status` (`success`, `failed` or `skipped`), sample index, expected count,
exception class and bounded diagnostic text. Summary output reports expected,
successful, failed and skipped counts, failure rate and `evaluation_valid`.
Metrics use successful rows only but a partial run is explicitly invalid and the
quality gate cannot pass it.

The Streamlit evaluation page now warns about incomplete coverage. Its failure
analysis batch path records execution failures and reports that the analysis is
partial instead of silently swallowing exceptions.

The API uses `os.devnull` for request-scoped evaluation summaries, so concurrent
HTTP evaluations do not overwrite the CLI's historical `outputs/eval_results.csv`.
CLI persistence remains unchanged. Summary writes use same-directory temporary
files plus `os.replace`.

### 5. Knowledge-graph co-occurrence direction — confirmed and fixed

The graph builder inserted only `e1 -> e2` edges. Co-occurrence is symmetric, so
the builder now inserts both directions with duplicate checks. A focused test
proves order-independent traversal and no duplicate edge inflation. No separate
directed relation type existed in this builder, so no intentionally directed
edges were changed.

### 6. Thread-safe provider pacing — confirmed and fixed

The previous timestamp checks were instance-local and used wall-clock time in the
legacy provider methods. `_ProviderSlotScheduler` in
`module2/rag_architectures.py` now reserves provider-specific slots under a small
lock, uses `time.monotonic()`, and sleeps outside the lock. Answer calls retain
the bounded-abstention behavior for excessive waits. OpenRouter, Groq and Mistral
legacy paths use the shared scheduler.

The regression test verifies distinct reservations for concurrent provider slots
and separate provider schedules. No real provider call was made.

### 7. Warmup/shutdown race — confirmed and fixed

Cancelling an asyncio wrapper does not stop a synchronous executor function. The
serving context now has a lifecycle generation. Startup passes the generation to
warmup; shutdown invalidates it before clearing resources. Warmup checks the
generation before publishing readiness and serializes final publication with
resource teardown.

Tests cover stale warmup after shutdown and normal current-generation readiness.

### 8. CI/CD summary status — confirmed and fixed

The old formatter tested whether a metric name appeared in a list of complete
failure strings, so failed metrics could display a success mark. `CICDReport` now
stores structured `metric_status`; the summary uses that same status as the gate,
handles missing/non-finite values as failures, and records sample coverage.

The regression test covers a faithfulness value of `0.1000` below threshold and
asserts the displayed cross rather than a check mark.

### 9. Request-specific evaluation outputs — confirmed and fixed

The API and CLI previously shared the default evaluation output path through
`EvaluationPipeline`. HTTP evaluation now uses a non-persisting request-local
pipeline, while CLI behavior continues to write its documented artifact. The
pipeline's explicit file writer is atomic for callers that do persist. A focused
concurrent writer test proves the final file is one complete result, not a mixed
or partial CSV.

### 10. Unused helper cleanup — reviewed, not deleted

`app.py` resource helpers are used by the UI. `compute_ensemble_confidence()` and
`rank_by_latency()` remain public-ish comparator helpers with no safe evidence that
they are outside documented/notebook use. Deleting them would not be a necessary
remediation, so they were retained.

## Verification

### Offline checks

```bash
make test
make lint
.venv/bin/python -m pip check
```

Final results in this pass:

- `make test`: **701 passed, 3 skipped, 17 warnings**
- `make lint`: passed
- `pip check`: `No broken requirements found`
- `git diff --check`: passed
- Focused remediation tests: **8 passed** in the final remediation-only run
  (the broader focused suites also passed; the exact combined run included 57
  passed tests)

Warnings are existing PyMuPDF SWIG/Chroma deprecations and SciPy precision-loss
warnings in significance tests.

### Live/provider-free checks

The following used temporary resources and no paid provider/judge calls:

```bash
PYTHON_DOTENV_DISABLED=1 .venv/bin/python \
  scripts/validate_live_lifecycle.py \
  --redis-server .venv/lifecycle-tools-t3-20261019/redis-7.2.16/src/redis-server \
  --output outputs/security-v4-index-live-lifecycle-20261020-integrated.json

PYTHON_DOTENV_DISABLED=1 .venv/bin/python \
  scripts/validate_generation_fencing.py \
  --redis-server .venv/lifecycle-tools-t3-20261019/redis-7.2.16/src/redis-server \
  --output outputs/security-v4-fencing-20261020-integrated.json

PYTHON_DOTENV_DISABLED=1 .venv/bin/python \
  scripts/validate_parser_isolation.py \
  --output outputs/security-v4-parser-20261020-integrated.json
```

Receipts:

- `security-v4-index-live-lifecycle-20261020-integrated.json`: Redis 7.2.16 and
  index publication/rollback barriers passed; 10 Redis worker processes had
  distinct PIDs, with four gap-arrival refusals.
- `security-v4-fencing-20261020-integrated.json`: **51 assertions passed**, 11
  worker processes, outage/restart/clear/publication fencing covered.
- `security-v4-parser-20261020-integrated.json`: **25/25 cases passed**.

Docker was not available (`docker: command not found`), so the Docker build,
clean Linux dependency resolution, non-root volume permissions and container
readiness remain unverified.

## Remaining limitations

- Dependency audit remains open: **56 known advisory matches across 9 packages**
  in `outputs/dependency-audit-20261019-v4-final.json`; no unverified major
  dependency migration was attempted.
- Full-corpus preprocessing and a real full-corpus evaluation were not run due to
  their size and local resource cost. The full workflow is manifest-aware, but
  that path still needs a clean Linux/Docker and production-scale rehearsal.
- Index lifecycle is a participating POSIX reader/build-lock contract, not hot
  swapping for arbitrary direct Chroma clients or network filesystems.
- Parser isolation bounds subprocess resources and protocol exposure; it is not a
  complete native-code, filesystem, network or privilege sandbox.
- Generation fencing cannot retract a response already linearized before a clear
  and does not cancel provider work.
- Answer semantic entailment, faithfulness, citation quality, human review and
  adversarial robustness remain unscored/partially unknown. The answer mode is
  deliberately extractive and deterministic tests are not semantic proof.
- No live provider, RAGAS judge, Docker or deployment check was claimed.

## Final workspace state

The final workspace remains intentionally dirty with inherited security/audit
changes, new remediation edits, tests and create-only receipts. Historical reports
and outputs were not overwritten. Inspect the exact state with:

```bash
git status --short --branch
git diff --stat
git diff --name-only
git log --oneline -8
```

No commit or push was performed. The next safe step is a human review of the
combined diff, followed by an explicit request if these local changes should be
committed or published.
