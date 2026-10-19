# Dependency security v3 — bounded remediation decision

## Result

Requirements and lockfile were left unchanged by T1. The baseline receipt still
contains **56 matches, 56 alias-deduplicated advisories, 56 package/advisory rows
across 9 packages**, with every row open. **18 rows have no scanner-supplied fix**.
No dependency candidate was resolved, downloaded, installed, or proposed as tested.
The shared environment was used only for checks and metadata inspection, not
dependency installation. No vulnerability was suppressed.

Canonical evidence is in `outputs/dependency-security-v3-final/`:

- `report.json`: per-advisory IDs and aliases, audited/installed versions, supplied
  fixes, unknown severity/range, active metadata paths, static import evidence,
  existing-constraint blockers, reachability caveats and open dispositions.
- `advisory_matrix.csv`: the same reviewable package/advisory matrix; structured
  cells use JSON to retain aliases and path evidence without ambiguity.
- `evidence.json`: installed distribution versions/Requires-Dist, marker
  environment, hashes of 39 source files and the three inputs, source imports,
  and unverified URL artifact requirements. No environment credentials or source
  file contents are captured.
- `report.md`: human-readable package decisions and advisory matrix.

`outputs/dependency-security-v3-replay/` reproduces all four canonical files
byte-for-byte from captured evidence. The earlier
`outputs/dependency-security-v3-report/` is preserved as an intermediate receipt.
The report is an offline transformation of the existing audit, not a fresh scan.

The installed/audited/locked versions agree: **0 consistency issues**. All 375
active inspected dependency edges have installed versions satisfying their
declared version constraints. The scanner's `en-core-web-sm` URL skip remains a
coverage gap; the report does not claim to verify that artifact's bytes.

## Why constraints were retained

These decisions use the existing follow-up report
`docs/VALIDATION_2026-10-19_FOLLOWUP.md`, actual runtime AST imports and installed
Requires-Dist metadata. They are compatibility blockers, not exploit tests.

| Package | Open rows | Concrete blocker / next work |
|---|---:|---|
| chromadb 1.5.9 | 4 | No supplied fix. Embedded vector-store integrations are used; independently confirm upstream applicability of the named server/model-repository endpoints. |
| datasets 3.1.0 | 1 | Dataset loading is used. Supplied 5.0.1 is a major migration; the earlier follow-up's 5.1.0 candidate conflicts with retained Hub `<1`. That prior result is not evidence about all 5.x releases. Test the loader/Hub/fsspec stack together. |
| fsspec 2024.9.0 | 1 | datasets 3.1.0 explicitly requires `fsspec[http]<=2024.9.0,>=2023.1.0`, excluding supplied fix 2026.6.0. Also reached through Torch and Hub. |
| langchain 0.3.30 | 1 | Supplied fix 1.3.9 is outside direct `<0.4`. No direct `langchain` import was found, but removing that direct requirement would retain the runtime dependency declared by used `langchain-community`. |
| langchain-core 0.3.86 | 2 | Supplied 1.x fixes conflict with both direct `<1.0` and `langchain-huggingface==0.2.0`'s `<1.0.0`. Document and embedding interfaces are used. |
| langchain-text-splitters 0.3.11 | 1 | Supplied fix 1.1.2 is outside direct `<0.4`; verify coordinated integration/parser behavior before relaxing it. RecursiveCharacterTextSplitter is used, which does not prove all advisory paths unreachable. |
| sentence-transformers 3.3.1 | 1 | Supplied fix 5.6.0 changes the retained model stack. SentenceTransformer and CrossEncoder are actually loaded; removal would break features. |
| torch 2.5.1 | 19 | `torchvision==0.20.1` requires `torch==2.5.1`; Torch device/model paths are used. Torchvision has no direct runtime import in the inventoried scope, but its removal only unlocks a candidate migration, does not remediate Torch by itself, and needs isolated model/native tests. Five Torch rows have no supplied fix. |
| transformers 4.46.3 | 26 | A same-major 4.x upgrade could be partial remediation but cannot clear this receipt: supplied fixes also include 5.0.0/5.3.0/5.10.0, excluded by Sentence Transformers 3.3.1's `<5.0.0`. Tokenizers is also coupled to the retained version. Model loading is used indirectly; nine rows have no supplied fix. |

The report compares each supplied fix to the direct and incoming metadata
constraints. That static comparison is not candidate resolution, upstream range
verification, or proof that the supplied version is a safe global upgrade target.
Severity and complete affected ranges remain explicitly **unknown** for every row.

## Actual checks

- Focused new report suite: **10 passed** (`tests/test_dependency_security_report.py`).
- Scoped Ruff (`E,F,W`, excluding E501): **passed**.
- `.venv/bin/python -m pip check`: **No broken requirements found**.
- Canonical report generation: **passed**, 56 open rows, 0 consistency issues.
- Evidence replay: **all four files byte-identical**.
- Scoped `git diff --check`: **passed**.
- Full `make test` during concurrent working-tree changes: **384 passed, 1 skipped,
  1 failed, 11 errors, 17 warnings**. The failure was
  `tests/test_ui_local_boundary.py::test_actual_streamlit_override_is_checked_and_not_silently_allowed[env]`
  (Streamlit returned 127.0.0.1 rather than expected override 0.0.0.0). Eleven
  upload-isolation setup errors reported an existing ephemeral Chroma instance
  with different settings. These are outside T1's file scope; the full suite is
  not reported as passing and must be rerun after the owning agents' changes.

## Safe reproduction command

From the project root, use a fresh output destination (existing destinations are
refused). This uses no advisory lookup, model loading, dependency installation,
private configuration or provider calls:

```bash
PYTHON_DOTENV_DISABLED=1 .venv/bin/python scripts/dependency_security_report.py \
  --audit outputs/dependency-audit-20261019-v3-baseline.json \
  --evidence outputs/dependency-security-v3-final/evidence.json \
  --output-dir outputs/dependency-security-v3-next-replay
```

Before any later requirements proposal, create a fresh ignored project-local
`.venv-security-v3-*` environment, place its pip cache/downloads inside that
ignored path, resolve/install an explicitly chosen coupled candidate set, run
the actual complete offline project test command with that interpreter and run
`pip check`. Preserve resolver/test receipts in fresh versioned paths and
regenerate the lock from the verified candidate only after those checks pass.
The macOS metadata evidence here does not verify clean Linux installation.
