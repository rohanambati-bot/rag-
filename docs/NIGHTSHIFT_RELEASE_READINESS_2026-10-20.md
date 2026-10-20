# Nightshift release-readiness report — 2026-10-20

## Decision

**Not release-ready.** The application passes the available offline regression
suite, lint, dependency-consistency checks and the existing local API/UI smoke
checks, but the release gates are still blocked by unresolved dependency
advisories, the missing manifest for the measured full Chroma store, and the
absence of Docker/clean-Linux verification on this host.

This is a readiness report, not a security sign-off. The dependency evidence is
an offline transformation of a supplied scanner receipt; it is not a fresh
advisory lookup and does not independently verify severity, affected ranges or
exploitability.

## Scope and workspace state

- Repository: `rag_project`
- Branch: `fix/audit-repairs`
- HEAD when the continuation started: `7890bc65d97a5aa38feb929c201a48f9eb1eea2e`
- The report also includes the continuation's uncommitted working-tree changes;
  record the final commit hash after review and commit.
- Runtime: Python `3.12.14`, `.venv/bin/python`
- Platform: macOS arm64; Docker and Podman are unavailable
- The worktree was already dirty and contains inherited remediation changes,
  generated evidence and source/test edits. This continuation did not reset,
  clean, merge, deploy or push it.
- No provider, judge, network advisory lookup or paid API call was used.

## Changes completed in this continuation

### RAPTOR cache identity

`module2/rag_architectures.py::_raptor_cache_key` now hashes the complete
ordered corpus, including every chunk's text and metadata, the chunk count and
the embedding model. Length-prefixed records avoid ambiguity when records are
concatenated. The previous `chunks[:300]` shortcut was removed.

`tests/test_ponytail_remediation.py` now checks both metadata changes and a
change to chunk 301, so a later-corpus edit cannot silently reuse a stale
summary tree.

### Shared persistent-store boundary

The legacy reranker benchmark no longer constructs `langchain_chroma.Chroma`
directly. `scripts/benchmark_reranker_depth.py` uses the shared guarded loader
with `allow_unmanifested_legacy=True` only after the explicit
`--allow-unmanifested-legacy` benchmark option. Normal `load_vector_store`
loads remain manifest-validated and fail closed.

### Dependency triage and documentation

- `scripts/dependency_security_report.py` now describes the current compatible
  pair `torch==2.6.0` / `torchvision==0.21.0` rather than the superseded
  2.5.1/0.20.1 pair.
- The README's Python 3.12 lock example now reports `torch==2.6.0`.
- A fresh triage receipt was written to
  `outputs/dependency-security-20261020-nightshift-final-v2/`.

## Verification results

Commands run in the project environment:

| Check | Result |
|---|---|
| `make test` | **704 passed, 3 skipped, 17 warnings** |
| `make lint` | **passed** (`ruff`, E/F/W selection) |
| `.venv/bin/python -m pip check` | **passed**; no broken requirements |
| `git diff --check` | **passed** |
| RAPTOR/remediation focused tests | **11 passed** |
| Lifecycle inventory + remediation focused tests | **22 passed** |
| Dependency-report tests | **10 passed** |
| Local API smoke | Previously passed: health `200`, unauthenticated query `401`, readiness `503` for the unmanifested local store |
| Local Streamlit smoke | Previously passed: `/_stcore/health` became healthy |
| Docker/container smoke | **Not run**; Docker and Podman are unavailable |

The three skipped tests are part of the existing suite. Warnings include PDF
native-parser deprecations, a LangChain Chroma deprecation in a test fixture,
and expected SciPy precision-loss warnings on near-identical significance-test
inputs.

## Dependency security status

Receipt: `outputs/dependency-audit-20261020-nightshift-after-safe-upgrades.json`

Triage: `outputs/dependency-security-20261020-nightshift-final-v2/report.md`

- Scanner matches: **59**
- Alias-deduplicated advisories: **59**
- Package/advisory rows: **59**
- Affected packages: **9**
- Open rows: **59**
- Rows with scanner-supplied fix versions: **41**
- Rows without a supplied fix: **18**
- Installed/lock/audit consistency issues: **0**
- Coverage gaps: **1** (`en-core-web-sm==3.8.0` is a direct GitHub URL not
  found on PyPI by the offline scanner)

Every current row remains `open`. Severity, complete affected ranges and
exploit reachability are unknown in this evidence. The most important active
families are ChromaDB, datasets/fsspec, LangChain 0.x, Sentence Transformers,
Torch and Transformers. The supplied fixes require coordinated migrations in
several cases; they were not promoted blindly because model loading,
integration APIs, native operations and Linux wheels still need verification.

### Current package-level interpretation

- **ChromaDB 1.5.9:** embedded use is present, but the receipt's server and
  model-repository findings are not exploit-verified against this deployment
  boundary. No scanner fix was supplied.
- **datasets 3.1.0 / fsspec 2024.9.0:** dataset loading is reachable, and the
  retained datasets constraint excludes the supplied fsspec fix. Upgrade them
  together with the Hugging Face stack.
- **LangChain 0.3.x:** the project imports document, splitter, embedding and
  Chroma integrations. The supplied 1.x fixes conflict with the retained 0.x
  integration set.
- **sentence-transformers 3.3.1:** `SentenceTransformer` and `CrossEncoder`
  are used. The supplied 5.6.0 candidate requires a coupled model-stack test.
- **Torch 2.6.0 / torchvision 0.21.0:** the pair is internally compatible,
  but 22 Torch advisory rows remain open. Later scanner fixes span multiple
  Torch releases and some rows have no supplied fix.
- **Transformers 4.46.3:** model loading is reachable through the embedding
  stack. Some fixes are in 5.x, which conflicts with the retained
  Sentence-Transformers 3.3.1 constraint.

## Performance and corpus evidence

The reranker depth receipts are retained at:

- `outputs/nightshift-reranker-depth-20261020-8q.json`
- `outputs/nightshift-reranker-depth-20261020-8q-replay.json`
- `outputs/nightshift-reranker-depth-20261020-cpu-4q.json`

They are explicitly marked `protocol_valid: false` because
`data/chroma_db_full` has no `_index_manifest.json`. The measurements are
useful for local engineering direction but are not release-quality corpus
evidence. The measured MPS sample showed median total latency increasing from
about 574 ms at candidate depth 20 to about 1,301 ms at depth 50 and about
2,418 ms at depth 100; the sample quality metrics did not improve between the
tested 20/50 settings. No production reranker setting was changed from this
small, legacy-protocol probe.

The RAPTOR key repair removes a correctness risk, but no controlled startup or
query benchmark has yet isolated the contribution of RAPTOR reuse, HyDE,
knowledge-graph access or routing to end-to-end GRACE-Ω efficiency.

## Release blockers and known limits

1. **Dependency audit is not clean.** All 59 current rows remain open. Do not
   claim vulnerability-free status.
2. **Linux/Docker is unverified.** The repository has a Dockerfile and CI
   container-smoke job, but neither can run on this host. The lock is a Python
   3.12 macOS freeze, not a proven universal Linux lock.
3. **The full dense store is legacy/unmanifested.** The strict loader correctly
   rejects `data/chroma_db_full` without its manifest. Rebuilding or changing
   the approximately 71 MB store must be an explicit, preserved workflow.
4. **The training store is large and must be preserved.**
   `data/chroma_db_train` is approximately 19 GB, with an approximately 1.4 GB
   `_train_documents.jsonl`; no casual rebuild was attempted.
5. **No semantic faithfulness proof exists.** Existing answer-boundary tests
   verify bounded extractive claims and controlled abstention, not human-rated
   faithfulness or citation support.
6. **Parser isolation is bounded but not a complete sandbox.** The existing
   evidence covers resource and process limits, not every possible parser or
   kernel escape path.
7. **No live provider or judge run was made.** Paid generation, RAGAS-style
   judging, large-scale MS MARCO preprocessing and production deployment remain
   unverified.

## Next steps

1. On clean Linux, install `requirements.lock`, build the Docker image and run
   the container-smoke job with Redis and the authenticated API.
2. Obtain and verify authoritative advisory ranges/fixes, then migrate one
   coordinated dependency family at a time in an isolated environment:
   datasets/fsspec/Hugging Face, LangChain integrations, and the
   Sentence-Transformers/Torch/Transformers stack.
3. Produce a manifest-backed evaluation corpus and rerun the reranker depth
   benchmark with a fixed query/qrels protocol. Keep the legacy receipts as
   historical evidence.
4. Measure GRACE-Ω startup, cache reuse and query latency with a controlled
   offline workload before changing architecture defaults.
5. Review the dirty worktree, stage only intentional files, commit each logical
   change, and rerun `make test`, `make lint`, `pip check` and `git diff --check`
   from the resulting clean candidate.

## Evidence index

- Previous remediation record: `docs/PONYTAIL_REMEDIATION_2026-10-20.md`
- Latest dependency triage: `outputs/dependency-security-20261020-nightshift-final-v2/`
- Latest raw audit: `outputs/dependency-audit-20261020-nightshift-after-safe-upgrades.json`
- Answer-quality v2 evidence: `outputs/security-v4-answer-quality-v2-20261019/`
- Parser isolation evidence: `outputs/security-v4-parser-20261020-integrated.json`
- Generation-fencing evidence: `outputs/security-v4-fencing-20261020-integrated.json`
- Index lifecycle evidence: `outputs/security-v4-index-live-lifecycle-20261020-integrated.json`
- API/UI smoke and CI/container details: `.github/workflows/rag_quality_gate.yml`,
  `docs/VALIDATION_2026-10-19_SECURITY_V3.md`
