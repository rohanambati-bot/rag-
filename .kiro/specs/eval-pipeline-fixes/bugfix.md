# Bugfix Requirements Document

## Introduction

The RAG evaluation framework in this project benchmarks multiple RAG architectures using both token-overlap retrieval metrics and LLM-judged (RAGAS-style) quality metrics. Six independent defects currently combine to prevent the framework from producing valid benchmark results:

1. **Deprecated judge model** — the RAGAS judge is configured with a Groq model that has been retired, so every judge call fails silently and all LLM-judged metrics are dropped.
2. **Architectures wrongly dropped** — the architecture builder gates Parent-Child RAG (and the GRACE ablation variants) behind the knowledge-graph builder, so evaluation runs omit architectures that should be present.
3. **RAGAS gated on the wrong key** — the RAGAS evaluator is constructed only when a Mistral key is present, so users who follow the documented free-tier (Groq) setup get no LLM-judged metrics.
4. **Mixed corpora in eval mode** — evaluation mode searches one corpus with dense retrieval and a different corpus with sparse/graph retrieval, making hybrid and graph results meaningless.
5. **Broken research scripts** — two standalone research scripts crash or exit early because they reference stale file paths and non-existent columns.
6. **Environment file never loaded** — the project ships a committed `.env.example` and lists `python-dotenv` as a dependency, but no startup code loads a `.env` file, so keys and judge settings placed in `.env` never reach configuration and the framework falls back as though nothing were set (this also nullifies the corrected judge-model default from defect 1).

This document captures, for each defect, the current defective behavior, the required corrected behavior, and the existing behavior that must be preserved. Each bug condition is expressed as the WHEN clause of its numbered requirement, and every Current Behavior clause has a matching Expected Behavior clause.

Out of scope: provisioning the required Python runtime (the analysis machine currently runs Python 3.9 while the project targets 3.12). This is an environment setup task handled separately and requires no code change in these fixes.

## Bug Analysis

### Current Behavior (Defect)

The following describes what the framework does today when each bug condition is met.

1.1 WHEN the evaluation pipeline runs with a Groq API key configured and the judge uses the default judge model, THEN the system issues judge (RAGAS) LLM calls against a retired model, every call raises an error that is silently caught, each judged metric returns None, and faithfulness, context precision, context recall, and answer relevance are dropped from the results — producing output indistinguishable from having no API key.

1.2 WHEN the architecture builder is invoked with chunks but without a knowledge-graph builder (as happens in eval mode), THEN the system does not register Parent-Child RAG nor the three GRACE ablation variants, returning only six architectures even though Parent-Child RAG depends only on chunks.

1.3 WHEN the evaluation pipeline runs with only a Groq API key set (the documented free-tier setup) and no Mistral key, THEN the system does not construct the RAGAS evaluator, so all LLM-judged metrics are absent and the quality gate fails on missing data.

1.4 WHEN eval mode runs, THEN the system performs dense retrieval over the persisted MS MARCO corpus while building the sparse/graph (BM25, hybrid, GRACE) indexes from SQuAD documents, so different retrievers search different corpora and hybrid and graph results are meaningless.

1.5 WHEN paradox_analysis.py is executed, THEN the system runs its defensive top section correctly but then unconditionally re-reads the summary results CSV and accesses columns that do not exist in it, raising a KeyError and crashing (the script also contains duplicated imports).

1.6 WHEN chunk_ablation_experiment.py is executed, THEN the system looks for the synthetic dataset at a path and filename the pipeline never writes, reports the file missing, and exits immediately without running the ablation.

1.7 WHEN the framework is started from any entry point (a CLI mode in main.py or a standalone research script) with the required settings present only in a `.env` file and not exported in the shell, THEN the system reads every configuration value through `os.getenv` without any prior call to load the `.env` file, so `GROQ_API_KEY`, `MISTRAL_API_KEY`, `JUDGE_PROVIDER`, `JUDGE_MODEL`, and the remaining `.env` values are absent when `configs/config.py` is imported, and both the generator and the judge silently fall back as though no keys were set — which also defeats the corrected judge-model default described in 2.1.

### Expected Behavior (Correct)

The following describes what the framework should do instead for each bug condition.

2.1 WHEN the evaluation pipeline runs with a Groq API key configured and the judge uses the default judge model, THEN the system SHALL issue judge calls against a live Groq model that belongs to a different model family than the generator — the generator remaining the Llama-family instant model (`llama-3.1-8b-instant`) and the judge defaulting to the GPT-OSS-family model (`openai/gpt-oss-20b`) — with this default set consistently in both `configs/config.py` and the committed `.env.example`, so judge calls succeed and faithfulness, context precision, context recall, and answer relevance are recorded in the results.

2.2 WHEN the architecture builder is invoked with chunks but without a knowledge-graph builder, THEN the system SHALL register Parent-Child RAG (which depends only on chunks); and WHEN a knowledge-graph builder is also provided, THEN the system SHALL additionally register the three GRACE ablation variants, so that full mode registers all intended architectures.

2.3 WHEN the evaluation pipeline runs with either a Groq API key or a Mistral API key set, THEN the system SHALL construct the RAGAS evaluator so that LLM-judged metrics are produced and made available to the quality gate.

2.4 WHEN eval mode runs, THEN the system SHALL build the chunk set from the same MS MARCO documents that back the persisted vectorstore, so dense, sparse, and graph retrievers all operate over a single shared corpus.

2.5 WHEN paradox_analysis.py is executed, THEN the system SHALL complete using only its defensive, self-contained analysis (reading the raw results with column existence checks) without re-reading the summary CSV or accessing absent columns, and SHALL run without duplicated imports.

2.6 WHEN chunk_ablation_experiment.py is executed, THEN the system SHALL locate the synthetic dataset at the path configured for the pipeline (`SYNTHETIC_DATA_PATH`) and proceed to run the ablation.

2.7 WHEN the framework is started from any entry point, THEN the system SHALL load environment variables from a project-root `.env` file, when such a file is present, before `configs/config.py` reads them, so that `GROQ_API_KEY`, `MISTRAL_API_KEY`, `JUDGE_PROVIDER`, `JUDGE_MODEL`, and related settings placed in `.env` take effect for both answer generation and judging.

### Unchanged Behavior (Regression Prevention)

The following existing behavior must be preserved by these fixes.

3.1 WHEN the evaluation pipeline runs, THEN the system SHALL CONTINUE TO use the existing generator model for answer generation, keeping the judge decoupled from the generator by using a different model family.

3.2 WHEN no LLM API key is set at all, THEN the system SHALL CONTINUE TO fall back to extractive answering and to omit LLM-judged metrics without crashing.

3.3 WHEN retrieval metrics are computed, THEN the system SHALL CONTINUE TO produce the token-overlap metrics (answer recall, token IoU, precision omega, noise ratio) without requiring any LLM.

3.4 WHEN full mode runs with both chunks and a knowledge-graph builder, THEN the system SHALL CONTINUE TO register the previously working architectures (naive, CRAG, RAG-Fusion, hybrid, adaptive, GRACE-Ω).

3.5 WHEN both a Groq key and a Mistral key are configured, THEN the system SHALL CONTINUE TO select the judge provider and model according to the existing judge-provider configuration and precedence.

3.6 WHEN paradox_analysis.py runs against valid raw results, THEN the system SHALL CONTINUE TO compute the paradox rate and produce the paradox heatmap output.

3.7 WHEN chunk_ablation_experiment.py finds the synthetic dataset, THEN the system SHALL CONTINUE TO run the chunk-size ablation and produce its divergence results and figure.

3.8 WHEN full mode runs, THEN the system SHALL CONTINUE TO build all retrievers over the single MS MARCO corpus as it does today.

3.9 WHEN required settings are exported in the shell environment, THEN the system SHALL CONTINUE TO honor those shell-exported values, and any variable already present in the shell environment SHALL retain its shell-provided value even when the same key also appears in a `.env` file.

3.10 THE system SHALL CONTINUE TO exclude real environment files (`.env` and local `.env` variants) from version control while tracking only the `.env.example` template, so that no secrets are committed.
