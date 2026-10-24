# P1 — bounded upload-parser subprocesses

Date: 19 October 2026. Host execution: Python 3.12.14, macOS 26.6.2 arm64.
This extends the ingestion gap in `docs/VALIDATION_2026-10-19_SECURITY_V3.md`;
it is resource-isolation evidence, not a native-code sandbox/security sign-off.

## Implemented flow

`DocumentUploader.parse_file()` validates only encoded length, extension and
signature in the caller. It admits at most **two parser invocations per process**
with a nonblocking semaphore; excess invocations receive a generic busy error.
There is no retained parser queue. All accepted PDF, DOCX, TXT, MD and CSV parsing,
including ZIP preflight, runs in a fresh actual child process by default.
There is no unrestricted fallback to the caller's process.

The child is launched as `python -I module5/parser_worker.py` by file path, with
an owned private temporary working directory and a minimal environment. It does
not import `module5/__init__.py`, application configuration, Torch, Chroma or the
LangChain stack. Provider credentials and caller HOME/PYTHONPATH are not inherited.
The input file has a fixed name; the original filename is only bounded metadata.
The parent/child protocol is bounded JSON plus a bounded input file, never pickle.

Before reading untrusted input or importing PDF/DOCX/ZIP libraries, the child sets
CPU, core-file, file-size and open-descriptor limits. Linux also gets hard
`RLIMIT_AS`. macOS uses parent-side libproc resident-memory monitoring because
macOS AS/RSS rlimits are unreliable and DATA was rejected by the host kernel.
Failure to establish/read the required memory monitor fails closed.

Default budgets (trusted constructor configuration can choose other finite values):

| Budget | Default |
|---|---:|
| Encoded upload | 20 MiB (existing MAX_UPLOAD_BYTES) |
| Extracted text per upload | 8 MiB UTF-8 / 2,000,000 characters |
| Extracted text per document/page/row | 2 MiB UTF-8 / 500,000 characters |
| Parsed documents / PDF pages | 10,000 / 200 |
| DOCX ZIP entries | 1,024 |
| DOCX expanded total / member | 50 MiB / 10 MiB |
| DOCX member compression ratio | 100:1 |
| Wall deadline / process CPU limit | 15 seconds / 8 seconds |
| Worker memory | 512 MiB |
| Worker stdout / request JSON | 16 MiB / 16 KiB |
| Worker descriptors / core files | 64 / disabled |
| RSS polling interval | 20 ms |

DOCX checks include unsafe names, symlinks, duplicates, encrypted entries, entry
counts, member/total expansion and ratios; all members are streamed through a
bounded buffer and CRC-checked before python-docx. No ZIP member is extracted to
disk. Text/document budgets are rechecked by the parent on returned records.
Child stderr is discarded and only fixed error codes become public messages.
The supervisor bounds stdout independently of child cooperation.

All completion/failure paths kill the owned process group, wait/reap the direct
worker, close its pipe and remove the owned temporary directory. The supervisor
handles macOS exit/libproc/group-signal transition races and pipe EOF without
losing the wall deadline. Index replacement remains staged: parsing, protocol
conversion, splitting and new-store population complete before publication;
failed parsing or indexing retains the previous active collection and filename.

## Actual checks

```bash
PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 GROQ_API_KEY= MISTRAL_API_KEY= \
  OPENROUTER_API_KEY= OPENAI_API_KEY= .venv/bin/python -m pytest -q \
  tests/test_parser_isolation.py tests/test_document_ingestion_security.py \
  tests/test_upload_isolation.py tests/test_document_upload.py
```

**81 passed, 6 warnings, 9.03 s.** Warnings are five PyMuPDF SWIG deprecations
and the existing community Chroma deprecation. Coverage includes valid real
PDF/DOCX/text/MD/CSV children, limits before native imports, finite 64 MiB RSS
probes, CPU termination, timeout, crash, bounded stdout, malformed JSON/pickle
rejection, fail-closed monitor errors, malformed archives, closed stdout,
macOS group-signal races, 20 immediate flood rejections with two blocked workers,
kill/reap/temporary cleanup and real Chroma replacement preservation.

Two prior tests that monkeypatch native libraries now explicitly exercise the
same private helpers in-process; their assertions are unchanged. Independent
tests require the production entry point to cross a real process boundary and
make every caller-side parser helper raise if used.

```bash
PYTHON_DOTENV_DISABLED=1 .venv/bin/python scripts/validate_parser_isolation.py \
  --output outputs/security-v4-parser-20261019-v1.json
```

**25/25 cases passed, 25 actual worker processes**. The receipt records PIDs,
applied limits, observed RSS, protocol sizes, implementation hashes and cleanup
for every direct worker. Native import hooks verified limits first and absence
of provider environment/ML stack. The standalone transaction case uses an owned
local store double; the focused tests above exercise actual local Chroma.
Attempting the identical command again returned exit 2 with
`output already exists; choose a new evidence filename`; the receipt was retained.

```bash
.venv/bin/python -m ruff check --select E,F,W --ignore E501 \
  module5/document_upload.py module5/parser_isolation.py module5/parser_worker.py \
  tests/test_document_ingestion_security.py tests/test_upload_isolation.py \
  tests/test_parser_isolation.py scripts/validate_parser_isolation.py
git diff --check
```

Both passed. The full project suite and Linux/container execution are left to
the integrating session. No dependency installation, shared configuration,
private dotenv reads, provider calls, commits, pushes or deployment occurred.

### Failures encountered and fixed

The initial focused run had 26 failures because macOS rejected lowering DATA to
512 MiB. Using monitored RSS rather than an unreliable DATA limit restored all
existing 41 tests. A later run exposed transient libproc absence and EPERM when
signaling a terminating macOS group. Bounded exit detection and exact-PID fallback
fixed this; dedicated real-process regression tests cover both exited and live
permission-error paths. These intermediate failures were not relabeled passes.

## Remaining boundaries

- Linux code is implemented (`/proc` RSS + hard AS) but not executed in this
  macOS receipt. Windows and other unsupported platforms explicitly fail closed.
- RSS monitoring can overshoot between samples; a short-lived allocation can
  escape observation. Linux hard AS gives an additional kernel bound. RSS is
  measured for the parser PID, not aggregate descendant memory; inherited
  CPU/address-space limits are per process, not cgroup/job-wide budgets.
- Process groups are terminated, but this is not a syscall, filesystem, network,
  privilege or native-code-execution sandbox. Native compromise could access
  same-user resources; stronger OS/container isolation remains separate work.
- Admission is per application process, not cluster-wide. Upstream body
  buffering, document splitting, embedding and vector indexing are outside its
  parser resource budget. Trusted private helper callers can parse in-process.
- Fixtures are deliberately finite/local, not comprehensive document fuzzing or
  dependency exploit coverage. No vulnerable dependency was declared fixed.
