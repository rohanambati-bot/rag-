# Implementation Plan: eval-pipeline-fixes

## Overview

Six independent, minimally-scoped fixes restore the RAG evaluation framework. The plan is
ordered so the enabling fixes land first: loading `.env` (BUG 6) and correcting the judge model
default (BUG 1) make the API keys and judge actually work; then the RAGAS key gate (BUG 3), the
architecture-builder nesting (BUG 2), the eval-mode corpus (BUG 4), and finally the two research
scripts (BUG 5a, BUG 5b). Each fix touches only the file and function named in the design and
cites the requirement clauses it satisfies.

Language: **Python** (existing codebase — LangChain, Chroma, Groq/Mistral SDKs, pandas,
matplotlib/seaborn). Tests use **pytest** (already a dependency); property tests add
**Hypothesis**.

Runtime note: the analysis machine currently runs Python 3.9 while the framework targets 3.12.
Pure config/logic checks and the property tests (P1–P4, no network) can be authored now; any
verification that needs the live LLM/corpus or the full LangChain/Chroma stack is marked with `*`
and annotated **(deferred)** until the 3.12 runtime is available (a separate, user-owned task).

## Tasks

- [x] 1. BUG 6 — Load `.env` before configuration is read (`configs/config.py`)
  - [x] 1.1 Load the project-root `.env` at the top of `configs/config.py`
    - Add `from dotenv import load_dotenv` and, before any `os.getenv(...)` call, compute
      `_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` and call
      `load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))`
    - Keep the default `override=False` so shell-exported variables win over `.env`; an absent
      `.env` file is a silent no-op
    - `python-dotenv==1.0.1` is already in `requirements.txt` — no dependency change needed
    - _Requirements: 2.7, 3.9, 3.10_
  - [x] 1.2 Write property test for `.env` fill-only-missing precedence
    - **Feature: eval-pipeline-fixes, Property 3: Environment loading fills only missing keys**
    - Add `hypothesis` to the "Testing & lint" section of `requirements.txt`; create
      `tests/test_env_loading.py`
    - Generate key/value pairs; seed a subset into `os.environ`, write the rest to a temp `.env`,
      call `load_dotenv(override=False)`, then assert shell-present keys keep their original value
      and `.env`-only keys resolve to their `.env` value (min. 100 iterations, no network)
    - **Validates: Requirements 2.7, 3.9**
  - [x] 1.3 Smoke-check version-control hygiene for env files (regression)
    - In a new `tests/test_gitignore_env.py`, assert `.gitignore` ignores `.env` and local
      `.env.*` variants while `.env.example` stays tracked
    - _Requirements: 3.10_

- [x] 2. BUG 1 — Correct the judge model default (`configs/config.py`, `.env.example`)
  - [x] 2.1 Change the judge default to a live, cross-family Groq model
    - In `configs/config.py`, change `JUDGE_MODEL` default from `"llama-3.1-70b-versatile"` to
      `"openai/gpt-oss-20b"`; leave `JUDGE_PROVIDER` default `"groq"` and `GROQ_MODEL`
      (`llama-3.1-8b-instant`) unchanged so the judge stays a different family from the generator
    - Set the same default in `.env.example` (`JUDGE_PROVIDER=groq`,
      `JUDGE_MODEL=openai/gpt-oss-20b`), keeping the `GROQ_MODEL` generator line as-is
    - No change to `RAGASEvaluator` — its Groq path already forwards `self.model`; init precedence
      is untouched
    - _Requirements: 2.1, 3.1, 3.5_
  - [x] 2.2 Write example test for the corrected defaults
    - In `tests/test_config_judge.py`, assert `config.JUDGE_MODEL == "openai/gpt-oss-20b"`,
      `config.GROQ_MODEL == "llama-3.1-8b-instant"`, and `JUDGE_MODEL != GROQ_MODEL`; assert
      `.env.example` contains `JUDGE_PROVIDER=groq` and `JUDGE_MODEL=openai/gpt-oss-20b`
    - _Requirements: 2.1, 3.1_

- [x] 3. Checkpoint — configuration foundation
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. BUG 3 — Construct RAGAS on a Groq **or** Mistral key (`main.py._run_evaluation`)
  - [x] 4.1 Widen the evaluator construction guard
    - Import `GROQ_API_KEY` alongside `MISTRAL_API_KEY` from `configs.config` in `main.py`
    - In `_run_evaluation()`, change
      `ragas_evaluator = RAGASEvaluator() if MISTRAL_API_KEY else None` to
      `ragas_evaluator = RAGASEvaluator() if (GROQ_API_KEY or MISTRAL_API_KEY) else None`
    - Leave `EvaluationPipeline(ragas_evaluator=ragas_evaluator)` and provider/model precedence
      unchanged (this only widens *when* the evaluator is built)
    - _Requirements: 2.3, 3.2, 3.5_
  - [x] 4.2 Write property test for the RAGAS construction predicate
    - **Feature: eval-pipeline-fixes, Property 2: LLM-judged evaluation is enabled exactly when a key is available**
    - In `tests/test_ragas_gate.py`, monkeypatch every combination of `GROQ_API_KEY` /
      `MISTRAL_API_KEY` presence; assert an evaluator is built iff at least one key is non-empty,
      and that neither-set yields `None` with no exception raised (min. 100 iterations, no network)
    - **Validates: Requirements 2.3, 3.2**
  - [x] 4.3 Write property test that token-overlap metrics stay LLM-free and bounded (regression)
    - **Feature: eval-pipeline-fixes, Property 4: Token-overlap retrieval metrics are LLM-free and bounded**
    - In `tests/test_metric_properties.py`, generate arbitrary retrieved-context and ground-truth
      strings; assert `compute_answer_recall`, `compute_token_iou`, `compute_precision_omega`, and
      `compute_noise_ratio` (in `module3/evaluation.py`) each return a value in `[0.0, 1.0]` and
      complete with no LLM/network call (min. 100 iterations)
    - **Validates: Requirements 3.3**

- [x] 5. BUG 2 — Un-gate Parent-Child in `build_all_architectures` (`module2/rag_architectures.py`)
  - [x] 5.1 Register `parent_child` on chunks, keep ablations on the KG builder
    - Move the `architectures["parent_child"] = ParentChildRAG(...)` registration out of the
      `if chunks and kg_builder:` block up to the `if chunks:` level (it depends only on chunks)
    - Keep `grace_no_gate`, `grace_no_kg`, `grace_no_hyde` gated under `if kg_builder:`; leave
      `grace_omega` registered at the `if chunks:` level (unchanged)
    - Result: full mode registers 10 architectures, eval mode (no KG) registers 7 including
      `parent_child`
    - _Requirements: 2.2, 3.4_
  - [x] 5.2 Write property test for input-driven architecture registration
    - **Feature: eval-pipeline-fixes, Property 1: Architecture registration depends only on declared inputs**
    - In `tests/test_architecture_registration.py`, call `build_all_architectures` over all
      `(chunks present?, kg_builder present?)` combinations with lightweight stub/dummy inputs;
      assert `naive`/`crag`/`rag_fusion` always present; `hybrid`/`adaptive`/`grace_omega`/
      `parent_child` present iff non-empty `chunks`; `grace_no_gate`/`grace_no_kg`/`grace_no_hyde`
      present iff both `chunks` and `kg_builder` (min. 100 iterations, no network)
    - **Validates: Requirements 2.2, 3.4**

- [x] 6. BUG 4 — Load the MS MARCO corpus in `main.py.run_eval_mode`
  - [x] 6.1 Build eval-mode chunks from MS MARCO to match the vectorstore
    - Add `load_msmarco_documents` to the `module1.data_preprocessing` import in `run_eval_mode()`
    - Replace `docs = load_squad_documents()` with `docs = load_msmarco_documents()`, keeping the
      subsequent `chunks = chunk_documents(docs)` and
      `build_all_architectures(vectorstore=vectorstore, chunks=chunks)` wiring
    - _Requirements: 2.4, 3.8_
  - [x] 6.2 Write integration test for the eval-mode corpus wiring
    - In `tests/test_eval_mode_corpus.py`, patch `load_msmarco_documents`, `load_squad_documents`,
      `load_vector_store`, `get_embedding_model`, `chunk_documents`, and `_run_evaluation`; call
      `run_eval_mode` and assert the MS MARCO loader is called (and SQuAD is not) and the returned
      docs are routed through `chunk_documents`
    - _Requirements: 2.4, 3.8_
  - [x] 6.3 Confirm single-corpus retrieval on the live runtime
    - On the Python 3.12 runtime with an MS MARCO-built vectorstore, run eval mode and confirm
      dense, sparse (BM25), and graph retrievers all operate over the one MS MARCO corpus
    - VERIFIED on the 3.12 `.venv`: `data/chroma_db_full` holds 11,510 vectors;
      `load_msmarco_documents()` → `chunk_documents()` yields 11,510 chunks, all
      `source=ms_marco_v2.1`; dense self-search overlap = 1.00 (the vectorstore and the chunk
      corpus are identical); BM25 built over the chunks returns MS MARCO hits; the graph
      (spaCy NER, 1413 entities / 3326 edges) resolves real entity queries to MS MARCO chunks.
      Single MS MARCO corpus confirmed for dense + sparse + graph.
    - _Requirements: 2.4, 3.8_

- [x] 7. BUG 5a — Remove the dead summary re-read in `paradox_analysis.py`
  - [x] 7.1 Delete the stale bottom block and duplicated imports
    - Remove everything from the second `df = pd.read_csv("outputs/eval_results.csv")` to end of
      file (the block that references the absent `hit_rate` / `query_type` columns on the summary
      schema) and collapse the imports to a single `pandas`/`seaborn`/`matplotlib` header
    - Keep the defensive top half (raw-results load with `OLD_PATH` fallback, column-existence
      guards, `hit_rate` proxy, `query_type` default, paradox flagging, rate print, heatmap write)
    - _Requirements: 2.5, 3.6_
  - [x] 7.2 Write static + smoke test for the trimmed script
    - In `tests/test_paradox_analysis.py`, statically assert the summary re-read and duplicate
      imports are gone; run the script against a tiny `raw_eval_results.csv` fixture and expect
      exit 0 with a paradox rate in `[0, 100]` (the heatmap PNG-producing run is **(deferred)** to
      the 3.12 runtime)
    - _Requirements: 2.5, 3.6_

- [x] 8. BUG 5b — Read `SYNTHETIC_DATA_PATH` in `chunk_ablation_experiment.py`
  - [x] 8.1 Source the dataset path from config instead of the hard-coded filename
    - Add `from configs.config import SYNTHETIC_DATA_PATH`; replace the hard-coded
      `SYNTHETIC_PATH = "outputs/synthetic_data.json"` usage so the existence check and the
      `open(...)` both use `SYNTHETIC_DATA_PATH` (the existing `sys.path.insert(0, ".")` already
      allows importing `configs.config`)
    - _Requirements: 2.6, 3.7_
  - [x] 8.2 Write smoke test for dataset discovery
    - In `tests/test_chunk_ablation.py`, place a small dataset at `SYNTHETIC_DATA_PATH` (patched to
      a temp path) and assert the script locates it and proceeds past the existence check; the
      divergence CSV + `divergence_proof.png` producing run is **(deferred)** to the 3.12 runtime
    - _Requirements: 2.6, 3.7_

- [x] 9. Checkpoint — all fixes and offline tests
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Optional hardening and deferred live verification (needs Python 3.12 runtime)
  - [x] 10.1 Add a one-call judge health-check guard
    - Add a small helper (e.g. in `module3/evaluation.py`) that issues a single trivial judge
      prompt and confirms a numeric score returns; call it before long evaluation runs so a
      mis-set/retired judge model fails loudly instead of silently (per the design's operational
      note). Leaves the no-key fallback (3.2) intact
    - _Requirements: 2.1, 3.5_
  - [x] 10.2 Run the full pipeline end-to-end on the live runtime
    - On Python 3.12 with a Groq key, run the full pipeline and confirm faithfulness, context
      precision, context recall, and answer relevance are recorded (BUG 1 + BUG 3), the expected
      architecture set is present (BUG 2), and the judge health check passes
    - VERIFIED on the 3.12 `.venv` via a bounded live run (kept small to respect the Groq free
      tier): judge health check passes (probe score 1.000); all four RAGAS metrics are recorded
      in [0,1] and written to `outputs/eval_results.csv` (faithfulness/context_precision/
      context_recall/answer_relevance); all 10 architectures register, including `parent_child`.
      NOTE: the runtime `.env` runs generator `openai/gpt-oss-20b` + judge `qwen/qwen3.8-27b`
      (both live on Groq and cross-family) — a valid override of the spec defaults; the config
      default judge is still `openai/gpt-oss-20b`.
    - _Requirements: 2.1, 2.2, 2.3_

## Notes

- Tasks marked with `*` are optional test/hardening sub-tasks and can be skipped for a faster MVP;
  core implementation tasks (1.1, 2.1, 4.1, 5.1, 6.1, 7.1, 8.1) are never optional.
- Items annotated **(deferred)** require the Python 3.12 runtime or a live LLM/corpus; author them
  now if useful, but they only execute once the runtime is provisioned (a separate, user-owned
  environment task — out of scope for these code fixes).
- Each fix is scoped to the single file/function named in the design; no refactoring or new
  features are introduced, and every task cites the requirement clauses it satisfies.
- Property tests P1–P4 use Hypothesis (min. 100 iterations) and are tagged
  `Feature: eval-pipeline-fixes, Property N` per the design's Testing Strategy.
- Checkpoints (tasks 3 and 9) provide incremental validation after the config foundation and after
  all fixes.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.3", "5.1", "7.1", "8.1"] },
    { "id": 1, "tasks": ["2.1", "4.1", "1.2", "5.2", "7.2", "8.2"] },
    { "id": 2, "tasks": ["6.1", "2.2", "4.2", "4.3"] },
    { "id": 3, "tasks": ["6.2", "6.3", "10.1"] },
    { "id": 4, "tasks": ["10.2"] }
  ]
}
```
