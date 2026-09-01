# Design Document — eval-pipeline-fixes

## Overview

This design specifies the concrete fixes for the six independent defects captured in
`bugfix.md`. The defects combine to prevent the RAG evaluation framework from producing
valid benchmark results: LLM-judged metrics silently vanish, whole architectures are
dropped from the comparison, dense/sparse retrievers search different corpora, two
research scripts crash, and `.env` values never reach configuration.

The guiding principle is **minimal, targeted change**: each fix touches only the file and
function responsible for its defect, and every fix is checked against the
regression-prevention clauses (3.1–3.10) so existing behavior is preserved.

Language: **Python** (matches the existing codebase — LangChain, Chroma, Groq/Mistral SDKs,
pandas, matplotlib/seaborn).

### Defect-to-fix map

| Bug | Requirement | File | Function / location | Nature of fix |
|-----|-------------|------|---------------------|---------------|
| 1 | 2.1 | `configs/config.py`, `.env.example` | `JUDGE_MODEL` default | Change judge model default to a live cross-family Groq model |
| 2 | 2.2 | `module2/rag_architectures.py` | `build_all_architectures()` | Un-gate Parent-Child from the KG block |
| 3 | 2.3 | `main.py` | `_run_evaluation()` | Construct RAGAS when Groq **or** Mistral key present |
| 4 | 2.4 | `main.py` | `run_eval_mode()` | Load the MS MARCO corpus (not SQuAD) to match the vectorstore |
| 5 | 2.5 | `paradox_analysis.py` | bottom of module | Remove dead re-read block + duplicate imports |
| 5 | 2.6 | `chunk_ablation_experiment.py` | dataset load | Read from `SYNTHETIC_DATA_PATH` |
| 6 | 2.7 | `configs/config.py` | module top | Load `.env` via `python-dotenv` before `os.getenv` |

### Out of scope

Provisioning the Python 3.12 runtime (the analysis machine currently runs 3.9) is a separate
environment task and requires no code change here. Runtime verification of the full pipeline
is therefore deferred until the runtime is available (see Testing & Verification).

---

## Architecture

The pipeline runs in stages that the fixes touch at these seams:

```
                    configs/config.py  ◄── BUG 6: load_dotenv() must run here first
                          │  (reads os.getenv for keys, models, paths)
                          ▼
  main.py ──► run_full_pipeline / run_eval_mode
                │                      │
                │  BUG 4: eval mode    │
                │  corpus must be      ▼
                │  MS MARCO      module1/data_preprocessing.py
                │                (load_msmarco_documents / chunk_documents)
                ▼
        module2/rag_architectures.build_all_architectures  ◄── BUG 2: register parent_child
                │                                                on chunks, not on kg_builder
                ▼
        module3/evaluation.EvaluationPipeline
                │   uses RAGASEvaluator  ◄── BUG 1: judge model default must be live+cross-family
                │                             BUG 3: evaluator must be built on Groq OR Mistral key
                ▼
        outputs/raw_eval_results.csv , outputs/eval_results.csv
                │
                ├──► paradox_analysis.py            ◄── BUG 5a: drop dead summary re-read
                └──► chunk_ablation_experiment.py   ◄── BUG 5b: read SYNTHETIC_DATA_PATH
```

Two independent config knobs drive most behavior and must stay decoupled:

- **Generator LLM** — `GROQ_MODEL` = `llama-3.1-8b-instant` (Llama family). Unchanged (3.1).
- **Judge LLM** — `JUDGE_PROVIDER` / `JUDGE_MODEL`, deliberately a **different model family**
  to avoid self-preference bias.

---

## Components and Interfaces

### BUG 1 — Judge model default is a retired model (Req 2.1, regression 3.1, 3.5)

**Current behavior.** `configs/config.py` defaults the judge to
`JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.1-70b-versatile")`. That Groq model has been
retired, so every judge call raises, `RAGASEvaluator._score` catches it and returns `None`,
and faithfulness / context precision / context recall / answer relevance are dropped —
producing output indistinguishable from having no API key.

**Fix.** Change the default judge model to a **live, cross-family** Groq model. The generator
stays on the Llama-family instant model; the judge defaults to the GPT-OSS-family model. Set
the default consistently in both `configs/config.py` and the committed `.env.example`.

`configs/config.py`:

```python
# Judge LLM (separate from generator to reduce self-evaluation bias).
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "groq")          # "groq" | "mistral"
JUDGE_MODEL    = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")  # GPT-OSS family (was llama-3.1-70b-versatile)
```

`.env.example` (keep generator, only the judge default moves):

```dotenv
GROQ_MODEL=llama-3.1-8b-instant          # generator — unchanged
JUDGE_PROVIDER=groq
JUDGE_MODEL=openai/gpt-oss-20b           # judge — live, different family from generator
```

**No evaluator code change is required.** `RAGASEvaluator._init_groq()` constructs
`Groq(api_key=...)` and `_call()` passes `self.model` straight through to
`chat.completions.create(model=self.model, ...)`, so the Groq path already handles arbitrary
model IDs. With `JUDGE_PROVIDER="groq"` and a Groq key present, the first branch of
`RAGASEvaluator.__init__` selects Groq with the new `JUDGE_MODEL`.

**Decoupling preserved (3.1).** `JUDGE_MODEL` (`openai/gpt-oss-20b`) differs from `GROQ_MODEL`
(`llama-3.1-8b-instant`) in both provider-family and identity, so the judge remains decoupled
from the generator.

**Precedence preserved (3.5).** The existing `__init__` branch order (explicit
`JUDGE_PROVIDER`+key → other provider+key → same-provider-different-model → warn-and-fallback)
is untouched; only the default string changes.

**Operational note — silent judge failures.** Because `_score` swallows exceptions and
returns `None` on any failure (bad model id, rate limit, network), a mis-set judge model
degrades silently exactly as this defect did. Recommend a **one-call judge health check**
before long evaluation runs: issue a single trivial judge prompt (e.g. score a toy
context/answer pair) and confirm a numeric result comes back; abort or warn loudly if it does
not. This is a small guard, not a behavior change, and keeps 3.2's no-key fallback intact.

---

### BUG 2 — Architectures wrongly dropped (Req 2.2, regression 3.4)

**Current behavior.** In `build_all_architectures()`, `ParentChildRAG` is registered *inside*
the `if chunks and kg_builder:` block, so eval mode (chunks present, no KG builder) omits
Parent-Child even though it depends only on chunks. The three GRACE ablation variants are
correctly gated on the KG builder but share the same mis-nested block.

**Fix.** Register `parent_child` at the `if chunks:` level (depends only on chunks). Keep the
three GRACE ablations gated on `kg_builder`. Corrected structure:

```python
def build_all_architectures(
    vectorstore,
    chunks=None,
    parent_child=None,
    child_vectorstore=None,
    kg_builder=None,
) -> Dict[str, BaseRAG]:
    architectures: Dict[str, BaseRAG] = {
        "naive":      NaiveRAG(vectorstore=vectorstore),
        "crag":       CorrectiveRAG(vectorstore=vectorstore),
        "rag_fusion": RAGFusion(vectorstore=vectorstore),
    }
    if chunks:
        architectures["hybrid"] = HybridMultiQueryRAG(vectorstore=vectorstore, chunks=chunks)
        architectures["adaptive"] = AdaptiveRAG(vectorstore=vectorstore, chunks=chunks)
        architectures["grace_omega"] = GRACEOmegaRAG(
            vectorstore=vectorstore, chunks=chunks, kg_builder=kg_builder,
        )

        # Parent-Child depends ONLY on chunks — register whenever chunks exist.
        architectures["parent_child"] = ParentChildRAG(
            vectorstore=vectorstore,
            chunks=chunks,
            parent_child=parent_child,
            child_vectorstore=child_vectorstore,
        )

        # GRACE ablation variants require the knowledge graph.
        if kg_builder:
            architectures["grace_no_gate"] = GRACEOmegaNoGate(
                vectorstore=vectorstore, chunks=chunks, kg_builder=kg_builder)
            architectures["grace_no_kg"] = GRACEOmegaNoKG(
                vectorstore=vectorstore, chunks=chunks, kg_builder=kg_builder)
            architectures["grace_no_hyde"] = GRACEOmegaNoHyDE(
                vectorstore=vectorstore, chunks=chunks, kg_builder=None)  # KG not needed
    return architectures
```

**Resulting registration sets:**

- **Full mode** (chunks + `kg_builder`): `naive`, `crag`, `rag_fusion`, `hybrid`, `adaptive`,
  `grace_omega`, `parent_child`, `grace_no_gate`, `grace_no_kg`, `grace_no_hyde` — 10 total.
- **Eval mode** (chunks, no `kg_builder`): `naive`, `crag`, `rag_fusion`, `hybrid`,
  `adaptive`, `grace_omega`, `parent_child` — 7 total (the three ablations stay gated).

**Note on `grace_omega` in eval mode.** `grace_omega` is registered whenever `chunks` are
present (it accepts `kg_builder=None` and its KG-dependent steps degrade gracefully). This is
existing behavior and is what regression clause 3.4 protects ("previously working
architectures ... GRACE-Ω"). The defect and its fix concern **only** `parent_child` (add on
chunks) and the three ablations (keep on KG); `grace_omega` is intentionally left registered
in both modes. This differs slightly from an eval-mode list that omits `grace_omega`; the
requirements (2.2 + 3.4) and the existing code agree that `grace_omega` stays, so the design
keeps it to satisfy 3.4 and avoid an out-of-scope behavior change.

**Regression 3.4 preserved:** the six previously working architectures remain registered in
full mode; only `parent_child`'s availability is broadened.

---

### BUG 3 — RAGAS gated on the wrong key (Req 2.3, regression 3.2, 3.5)

**Current behavior.** `main.py._run_evaluation()` builds the evaluator only when a Mistral key
is present:

```python
ragas_evaluator = RAGASEvaluator() if MISTRAL_API_KEY else None
```

Users on the documented free tier (Groq only) get `None`, so all LLM-judged metrics are absent
and the quality gate fails on missing data.

**Fix.** Construct the evaluator when **either** key is set, and import `GROQ_API_KEY`.

Top-of-file import in `main.py`:

```python
from configs.config import (
    SYNTHETIC_DATA_PATH, EVAL_RESULTS_PATH, OUTPUTS_DIR,
    MISTRAL_API_KEY, GROQ_API_KEY,   # + GROQ_API_KEY
)
```

Inside `_run_evaluation()`:

```python
ragas_evaluator = RAGASEvaluator() if (GROQ_API_KEY or MISTRAL_API_KEY) else None
pipeline = EvaluationPipeline(ragas_evaluator=ragas_evaluator)
```

`RAGASEvaluator()` (no args) already resolves provider/model from config, so construction alone
is enough. Its own `__init__` still raises if it is somehow built with no key — but the guard
above only constructs it when a key exists, so that path is not reached.

**Regression 3.2 preserved:** when neither key is set, the expression is `None`, the pipeline
runs with `ragas_evaluator=None`, LLM-judged columns are simply absent (the evaluate loop skips
the RAGAS block), and the extractive generator fallback still answers — no crash.

**Regression 3.5 preserved:** with both keys set, provider/model selection is still decided by
`RAGASEvaluator.__init__` precedence; this change only widens *when* the evaluator is built.

---

### BUG 4 — Mixed corpora in eval mode (Req 2.4, regression 3.8)

**Current behavior.** `main.py.run_eval_mode()` loads the persisted vectorstore (built from
**MS MARCO** in full mode) for dense retrieval, but builds the chunk list — used by BM25 /
hybrid / GRACE — from **SQuAD**:

```python
docs = load_squad_documents()
chunks = chunk_documents(docs)
```

Dense and sparse/graph retrievers then search different corpora, making hybrid and graph
results meaningless.

**Fix.** Build the chunk set from the same MS MARCO documents that back the vectorstore.

Import change:

```python
from module1.data_preprocessing import (
    load_vector_store, get_embedding_model, chunk_documents, load_msmarco_documents,
)
```

Body of `run_eval_mode()`:

```python
embedding_model = get_embedding_model()
vectorstore = load_vector_store(embedding_model)
docs = load_msmarco_documents()          # was load_squad_documents()
chunks = chunk_documents(docs)

architectures = build_all_architectures(vectorstore=vectorstore, chunks=chunks)
```

Now dense (vectorstore), sparse (BM25 over `chunks`), and graph retrievers all operate over one
shared MS MARCO corpus.

**Corpus-match precondition.** The persisted store at `VECTOR_STORE_DIR` must have been built
from MS MARCO (the config default: `DATASET_NAME = "microsoft/ms_marco"`, `run_module1(dataset="msmarco")`).
If a store was ever built from a different dataset, it must be rebuilt so the loader and the
store agree. `load_msmarco_documents()` uses the same `DATASET_*` config as the builder, so the
two stay aligned by construction.

**Regression 3.8 preserved:** full mode already builds every retriever over the single MS MARCO
corpus (`run_module1` → `chunks` → `build_all_architectures`); this fix only brings eval mode
in line and does not alter the full-mode path.

---

### BUG 5a — paradox_analysis.py crashes on stale re-read (Req 2.5, regression 3.6)

**Current behavior.** The top half of `paradox_analysis.py` is defensive and self-contained:
it loads `outputs/raw_eval_results.csv` (which has `precision_omega`, `architecture`,
`query_type`), checks required columns, derives a `hit_rate` proxy, flags paradoxes, prints
rates, and writes the heatmap. But the bottom of the file then **unconditionally** re-reads the
**summary** CSV and accesses columns it does not contain:

```python
df = pd.read_csv("outputs/eval_results.csv")            # summary schema
df["paradox"] = (df["hit_rate"] >= 1.0) & (df["precision_omega"] < 0.4)  # KeyError: hit_rate
pivot = df.pivot_table(values="paradox", index="architecture", columns="query_type", ...)  # KeyError: query_type
```

`eval_results.csv` (written by `compare_architectures`) has aggregate columns
(`architecture`, `n`, `answer_recall`, `iou`, `precision_omega`, `noise_ratio`, judged metric
means, `*_ci_lo/_ci_hi`) — no `hit_rate`, no `query_type`. The re-read raises `KeyError` and the
script crashes. The file also duplicates its `pandas`/`seaborn`/`matplotlib` imports.

**Fix.** Delete the dead bottom block entirely (everything from the second
`df = pd.read_csv("outputs/eval_results.csv")` to end of file) and remove the duplicated import
lines, keeping a single import header and only the defensive top half. The retained top half
already produces the paradox rate and the heatmap from the raw results with column-existence
guards.

Resulting shape of the file:

```python
# paradox_analysis.py
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RAW_PATH = "outputs/raw_eval_results.csv"
OLD_PATH = "outputs/eval_results.csv"

# ... load raw (fallback to old), check required columns,
#     derive hit_rate proxy + query_type default,
#     flag paradoxes, print rates, write outputs/plots/paradox_heatmap.png ...
# (no second read of eval_results.csv, no duplicate imports)
```

**Regression 3.6 preserved:** against valid raw results the retained defensive path still
computes the overall paradox rate and writes the paradox heatmap.

---

### BUG 5b — chunk_ablation_experiment.py can't find the dataset (Req 2.6, regression 3.7)

**Current behavior.** The script hard-codes a path/filename the pipeline never writes:

```python
SYNTHETIC_PATH = "outputs/synthetic_data.json"
if not os.path.exists(SYNTHETIC_PATH):
    print(f"ERROR: {SYNTHETIC_PATH} not found. Run main.py --mode full first.")
    sys.exit(1)
```

The pipeline actually writes the synthetic dataset to `SYNTHETIC_DATA_PATH`
(`configs/config.py` → `data/synthetic_dataset.json`, used by `generate_synthetic_queries` and
`main._load_synthetic_data`). So the script always reports the file missing and exits before
running the ablation.

**Fix.** Read the path from config instead of hard-coding it:

```python
from configs.config import SYNTHETIC_DATA_PATH

if not os.path.exists(SYNTHETIC_DATA_PATH):
    print(f"ERROR: {SYNTHETIC_DATA_PATH} not found. Run main.py --mode full first.")
    sys.exit(1)

with open(SYNTHETIC_DATA_PATH) as f:
    data = json.load(f)
```

(The `sys.path.insert(0, ".")` already present lets the script import `configs.config`.)

**Regression 3.7 preserved:** once the dataset is found, the existing ablation logic (chunk-size
sweep, PΩ vs Precision@K, divergence CSV + `divergence_proof.png`) runs unchanged.

---

### BUG 6 — .env never loaded (Req 2.7, regression 3.9, 3.10)

**Current behavior.** The project ships `.env.example` and depends on `python-dotenv`
(`requirements.txt` → `python-dotenv==1.0.1`), but no startup code calls `load_dotenv()`.
`configs/config.py` reads everything through `os.getenv`, so keys/judge settings placed in a
`.env` file never reach configuration and the framework falls back as if nothing were set —
which also defeats the corrected judge default from BUG 1.

**Fix.** Load `.env` at the **top of `configs/config.py`**, before any `os.getenv` call. Every
entry point (each `main.py` CLI mode and both standalone research scripts) imports
`configs.config`, so loading here guarantees the values are present when config is read.

```python
# configs/config.py
import os
from dotenv import load_dotenv

# ── Load .env as early as possible ──────────────────────────────────────────
# All entry points import this module, so loading here ensures .env values are
# available before the os.getenv() calls below. override=False (the default)
# means shell-exported variables win over .env, preserving shell precedence (3.9).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# ... existing BASE_DIR / paths / os.getenv(...) definitions follow unchanged ...
```

**Why this location.** It is the single choke point every entry point already passes through,
so it fixes `main.py`, `paradox_analysis.py`, and `chunk_ablation_experiment.py` at once with
one edit and no per-script boilerplate. Pointing `load_dotenv` at an absolute
`_PROJECT_ROOT/.env` makes it robust to the current working directory.

**Regression 3.9 preserved.** `python-dotenv`'s `load_dotenv(override=False)` (the default) does
**not** overwrite variables already present in the environment. Any key exported in the shell
keeps its shell value even if the same key also appears in `.env`; only keys absent from the
environment are filled from `.env`. Absence of a `.env` file is a silent no-op (returns
`False`), so nothing breaks when the file is not present.

**Regression 3.10 preserved.** No change to version control: `.gitignore` already ignores
`.env`, `.env.local`, and `.env.*.local` while tracking `.env.example`. This fix adds only a
runtime load call and commits no secrets.

---

## Data Models

No schemas change. Two existing shapes are relevant to the fixes:

**Configuration values (BUG 1, 6)** — resolved in `configs/config.py` after `load_dotenv`:

| Key | Source | Default after fix | Role |
|-----|--------|-------------------|------|
| `GROQ_MODEL` | env / `.env` | `llama-3.1-8b-instant` | generator (unchanged) |
| `JUDGE_PROVIDER` | env / `.env` | `groq` | judge provider |
| `JUDGE_MODEL` | env / `.env` | `openai/gpt-oss-20b` | judge model (changed) |
| `GROQ_API_KEY` / `MISTRAL_API_KEY` | env / `.env` | `""` | enable generator + judge |

**CSV schemas (BUG 5a)** — the paradox fix depends on knowing which file has which columns:

- `outputs/raw_eval_results.csv` (per-sample) — has `architecture`, `precision_omega`,
  `query_type`, `noise_ratio`, etc. **Correct input** for paradox analysis.
- `outputs/eval_results.csv` (summary from `compare_architectures`) — aggregate columns only;
  **no** `hit_rate`, **no** `query_type`. Must not be used for paradox pivots.

---

## Error Handling

- **Silent judge failures (BUG 1).** `RAGASEvaluator._score` intentionally returns `None` on any
  judge error so one bad sample does not abort a run. The trade-off is that a wholesale
  misconfiguration (retired model) also fails silently — mitigated by the recommended one-call
  judge health check before long runs. Behavior for a correctly configured judge is unchanged.
- **Missing API keys (BUG 3).** The `GROQ_API_KEY or MISTRAL_API_KEY` guard prevents constructing
  `RAGASEvaluator` (which would raise) when no key exists; the pipeline proceeds with token-only
  metrics (3.2).
- **Missing `.env` (BUG 6).** `load_dotenv` on an absent path is a no-op; configuration falls
  back to shell env and defaults exactly as before.
- **Missing dataset / CSV (BUG 5).** `chunk_ablation_experiment.py` keeps its explicit
  existence check and clean `sys.exit(1)` with a helpful message; `paradox_analysis.py` keeps its
  defensive column checks and heatmap `try/except`.
- **Corpus mismatch (BUG 4).** If the persisted store predates the MS MARCO default, retrieval
  quality would look wrong; the fix documents the rebuild precondition rather than adding runtime
  detection (out of scope).

---

## Testing Strategy

**Runtime constraint.** The framework targets Python 3.12 while the analysis machine runs 3.9.
Tests that import the LangChain/Chroma/LLM stack, and any full-pipeline smoke run, must wait for
the 3.12 runtime (environment setup is a separate, user-owned task). Verification that does not
need that stack (pure config/logic checks) can run earlier. Where a check needs the runtime, it
is marked *(deferred)*.

**Dual approach.** Property-based tests (min. 100 iterations, tagged
`Feature: eval-pipeline-fixes, Property {n}: {text}`) for the universal invariants below; plus
targeted example / smoke / integration checks for configuration and wiring.

Per-fix verification:

- **BUG 1 (2.1, 3.1, 3.5)** — *Example.* Assert `config.JUDGE_MODEL == "openai/gpt-oss-20b"`,
  `config.GROQ_MODEL == "llama-3.1-8b-instant"`, `JUDGE_MODEL != GROQ_MODEL`, and that
  `.env.example` carries `JUDGE_PROVIDER=groq` / `JUDGE_MODEL=openai/gpt-oss-20b`. *(deferred)*
  one-call judge health check confirms the model id is live on Groq.
- **BUG 2 (2.2, 3.4)** — *Property P1.* Pure-function membership test over
  `build_all_architectures` for all `(chunks, kg_builder)` presence combinations. No network.
- **BUG 3 (2.3, 3.2)** — *Property P2.* Monkeypatch key presence and assert the evaluator is
  built iff a key exists; assert `None` + no crash when neither is set.
- **BUG 4 (2.4, 3.8)** — *Integration.* Patch `load_msmarco_documents` / `load_squad_documents`;
  assert `run_eval_mode` calls the MS MARCO loader and routes docs through `chunk_documents`.
  *(deferred for the live-corpus run.)*
- **BUG 5a (2.5, 3.6)** — *Smoke.* Run `paradox_analysis.py` against a small
  `raw_eval_results.csv`; expect exit 0, a paradox rate in `[0, 100]`, and
  `outputs/plots/paradox_heatmap.png`; static check that the summary re-read and duplicate
  imports are gone. *(deferred for the run.)*
- **BUG 5b (2.6, 3.7)** — *Smoke.* With a dataset at `SYNTHETIC_DATA_PATH`, assert the script
  locates it and proceeds; *(deferred)* confirm the divergence CSV + `divergence_proof.png`.
- **BUG 6 (2.7, 3.9)** — *Property P3.* Temp `.env` + `os.environ`, `load_dotenv(override=False)`,
  assert shell values win and `.env`-only keys resolve from `.env`. No network.
- **Regression 3.3** — *Property P4.* Token-overlap metric bounds; pure, no LLM.
- **Regression 3.10** — *Smoke.* Assert `.gitignore` ignores `.env` variants and `.env.example`
  stays tracked.

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of
a system — essentially, a formal statement about what the system should do. Properties serve as
the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Only invariants with a meaningful universal ("for all") statement are listed. The remaining
criteria (2.1, 2.4, 2.5, 2.6, 3.1, 3.5, 3.6, 3.7, 3.8, 3.10) are verified by example, smoke, or
integration tests as described above, per the guidance against property-testing configuration
values, script smoke runs, corpus wiring, and repository-config checks.

### Property 1: Architecture registration depends only on declared inputs

For any invocation of `build_all_architectures`, the returned mapping contains `"naive"`,
`"crag"`, and `"rag_fusion"` unconditionally; it contains `"hybrid"`, `"adaptive"`,
`"grace_omega"`, and `"parent_child"` if and only if a non-empty `chunks` list is provided; and
it contains `"grace_no_gate"`, `"grace_no_kg"`, and `"grace_no_hyde"` if and only if both a
non-empty `chunks` list and a knowledge-graph builder are provided.

**Validates: Requirements 2.2, 3.4**

### Property 2: LLM-judged evaluation is enabled exactly when a key is available

For any combination of `GROQ_API_KEY` and `MISTRAL_API_KEY` environment values, the evaluation
pipeline constructs a RAGAS evaluator if and only if at least one of the two keys is non-empty;
when neither key is set the evaluator is absent and evaluation proceeds without raising.

**Validates: Requirements 2.3, 3.2**

### Property 3: Environment loading fills only missing keys

For any set of key/value pairs, after loading the project `.env` file, every key already present
in the process environment retains its original (shell-provided) value, and every key absent
from the environment but present in `.env` resolves to its `.env` value.

**Validates: Requirements 2.7, 3.9**

### Property 4: Token-overlap retrieval metrics are LLM-free and bounded

For any retrieved-context string and any ground-truth string, `compute_answer_recall`,
`compute_token_iou`, `compute_precision_omega`, and `compute_noise_ratio` each return a value in
the closed interval `[0.0, 1.0]` and complete without invoking any LLM or network call.

**Validates: Requirements 3.3**
