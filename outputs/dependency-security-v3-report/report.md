# Dependency security v3 — offline advisory triage

56 scanner matches; 56 alias-deduplicated advisories; 56 package/advisory rows in 9 packages. All 56 rows remain open.

## Boundaries

- Offline transformation of a supplied receipt, not a fresh advisory lookup or security sign-off.
- Severity and full affected ranges are unknown; fix_versions are scanner-supplied, not verified safe targets.
- Aliased records are unioned transitively; shared IDs across packages retain separate exposure rows.
- Dependency paths are shortest paths per direct root from active installed Requires-Dist metadata, not call graphs.
- Source imports are static evidence only; dynamic imports, model artifacts, native operations and exploitability are unverified.
- URL requirements are inventoried but artifact bytes are not independently verified; skipped scanner packages remain coverage gaps.
- Environment marker evaluation describes this interpreter/platform, not clean Linux resolution.
- No requirement changes or advisory suppressions are made by this report.

Installed/lock/audit consistency issues: 0.

## Package decisions

### chromadb 1.5.9 — 4 open rows

Embedded vector-store use is present. The follow-up identifies server/model-repository endpoints; source imports do not prove those endpoints exposed or unreachable.

No fix supplied in this receipt; obtain upstream guidance for embedded versus server use. Do not substitute a network Chroma server without reviewing its boundary.

Metadata paths: chromadb; langchain-chroma → chromadb

Static import evidence: none in scanned scope

### datasets 3.1.0 — 1 open rows

Dataset loading is used. Folder-builder exploitation has not been tested; trusted dataset names do not establish safety of arbitrary metadata.

Migrate datasets and fsspec together. Installed datasets 3.1.0 requires fsspec<=2024.9.0; the audit supplies datasets 5.0.1 and fsspec 2026.6.0, not a tested compatible pair. The earlier 5.1.0 attempt conflicts with the retained Hugging Face Hub <1 stack.

Metadata paths: datasets

Static import evidence: `module1/data_preprocessing.py:34`, `preprocess_full.py:222`, `preprocess_train.py:87`, `scripts/run_retrieval_ablation_local.py:131`, `scripts/validate_research.py:95`

### fsspec 2024.9.0 — 1 open rows

Installed through datasets/langchain-community. No direct fsspec import is not proof that reference-filesystem template handling cannot be invoked transitively.

datasets 3.1.0 caps fsspec<=2024.9.0, excluding the supplied fix 2026.6.0. Migrate the loader stack in an isolated environment and test real project behavior.

Metadata paths: chromadb → tokenizers → huggingface-hub → fsspec; datasets → fsspec; langchain-chroma → chromadb → tokenizers → huggingface-hub → fsspec; langchain-huggingface → huggingface-hub → fsspec; sentence-transformers → huggingface-hub → fsspec; torch → fsspec; torchvision → torch → fsspec; transformers → huggingface-hub → fsspec

Static import evidence: none in scanned scope

### langchain 0.3.30 — 1 open rows

No direct package import alone does not justify removal: langchain-community declares langchain as a runtime dependency. Named vulnerable helper calls remain unproven.

Supplied fix 1.3.9 is outside direct <0.4. Removing the direct requirement leaves the transitive dependency; a coordinated integration migration needs tests.

Metadata paths: langchain; langchain-community → langchain

Static import evidence: none in scanned scope

### langchain-core 0.3.86 — 2 open rows

Document/embedding interfaces are imported. Prompt-file loading/image URL token counting reachability is not established by this source inventory.

Supplied 1.x fixes conflict with direct <1.0 and installed langchain-huggingface 0.2.0's langchain-core<1.0.0 constraint. Migrate the integration together.

Metadata paths: langchain → langchain-core; langchain-chroma → langchain-core; langchain-community → langchain-core; langchain-core; langchain-huggingface → langchain-core; langchain-text-splitters → langchain-core

Static import evidence: `module1/data_preprocessing.py:35`, `module2/rag_architectures.py:32`, `module5/document_upload.py:29`, `preprocess_full.py:230`, `preprocess_train.py:201`, `train_in_shards.py:40`

### langchain-text-splitters 0.3.11 — 1 open rows

RecursiveCharacterTextSplitter is imported. This does not prove the advisory's HTML/URL loader is exposed or unreachable through other code.

Supplied fix 1.1.2 is outside direct <0.4. Verify a compatible LangChain integration set and ingestion behavior before changing the constraints.

Metadata paths: langchain → langchain-text-splitters; langchain-community → langchain → langchain-text-splitters; langchain-text-splitters

Static import evidence: `module1/data_preprocessing.py:36`, `module5/document_upload.py:30`, `preprocess_full.py:231`, `preprocess_train.py:202`, `train_in_shards.py:41`

### sentence-transformers 3.3.1 — 1 open rows

SentenceTransformer/CrossEncoder model loading is present. Model-import/trust paths are security relevant; use trusted pinned model artifacts while migration is open.

Supplied fix 5.6.0 changes the retained 3.3.1 model stack. Test model loading and retrieval/reranking contracts in an isolated candidate environment.

Metadata paths: langchain-huggingface → sentence-transformers; sentence-transformers

Static import evidence: `module2/rag_architectures.py:748`, `module2/rag_architectures.py:1010`, `scripts/optimize_retrieval.py:229`, `scripts/retrieval_ablation.py:604`

### torch 2.5.1 — 19 open rows

Torch device/model stack is used. Checkpoint and native-operation advisories require individual feature review; absence of torch.load in project code is not an exemption.

torchvision 0.20.1 requires torch==2.5.1. Supplied fixes span later versions and some are empty; no single patch-only clean solution follows from this receipt. Coordinate Torch/vision (or verify removal) and run native/model checks.

Metadata paths: langchain-huggingface → sentence-transformers → torch; sentence-transformers → torch; torch; torchvision → torch

Static import evidence: `module2/rag_architectures.py:732`, `preprocess_train.py:49`, `scripts/optimize_retrieval.py:228`, `scripts/run_retrieval_ablation_local.py:77`

### transformers 4.46.3 — 26 open rows

Model loading is reachable through Sentence Transformers/Hugging Face embeddings; individual processor/conversion advisory paths remain unverified.

A 4.x update could only partially remediate this receipt: supplied fixes include 5.0.0, 5.3.0 and 5.10.0; sentence-transformers 3.3.1 requires transformers<5.0.0. A coupled model-stack migration and tokenizers update require isolated tests.

Metadata paths: langchain-huggingface → transformers; sentence-transformers → transformers; transformers

Static import evidence: none in scanned scope

## Advisory matrix

Severity and affected range are **unknown for every row**. Fix versions below are supplied by the scanner; an empty list means no supplied fix, not no possible fix.

| Package | Advisory (aliases in JSON/CSV) | Supplied fixes | Disposition |
|---|---|---|---|
| chromadb | CVE-2026-45829 | none supplied | open |
| chromadb | CVE-2026-45830 | none supplied | open |
| chromadb | CVE-2026-45831 | none supplied | open |
| chromadb | CVE-2026-45833 | none supplied | open |
| datasets | CVE-2026-66007 | 5.0.1 | open |
| fsspec | CVE-2026-104851 | 2026.6.0 | open |
| langchain | CVE-2026-55443 | 1.3.9 | open |
| langchain-core | CVE-2026-26013 | 1.2.11 | open |
| langchain-core | CVE-2026-34070 | 1.2.22 | open |
| langchain-text-splitters | CVE-2026-41481 | 1.1.2 | open |
| sentence-transformers | CVE-2026-68770 | 5.6.0 | open |
| torch | CVE-2025-2148 | none supplied | open |
| torch | CVE-2025-2149 | none supplied | open |
| torch | CVE-2025-2953 | none supplied | open |
| torch | CVE-2025-2998 | none supplied | open |
| torch | CVE-2025-2999 | 2.9.1 | open |
| torch | CVE-2025-3000 | 2.13.0 | open |
| torch | CVE-2025-3001 | 2.10.0 | open |
| torch | CVE-2025-32434 | 2.6.0 | open |
| torch | CVE-2025-3730 | 2.8.0 | open |
| torch | CVE-2025-46148 | 2.7.0 | open |
| torch | CVE-2025-55551 | 2.9.0 | open |
| torch | CVE-2025-55552 | 2.9.0 | open |
| torch | CVE-2025-55553 | 2.7.1 | open |
| torch | CVE-2025-55554 | 2.9.0 | open |
| torch | CVE-2025-55557 | 2.7.1 | open |
| torch | CVE-2025-55558 | 2.7.1 | open |
| torch | CVE-2025-55560 | 2.7.1 | open |
| torch | CVE-2026-24747 | 2.10.0 | open |
| torch | CVE-2026-4538 | none supplied | open |
| transformers | CVE-2024-11392 | 4.48.0 | open |
| transformers | CVE-2024-11393 | 4.48.0 | open |
| transformers | CVE-2024-11394 | 4.48.0 | open |
| transformers | CVE-2024-12720 | 4.48.0 | open |
| transformers | CVE-2025-1194 | 4.50.0 | open |
| transformers | CVE-2025-14920 | none supplied | open |
| transformers | CVE-2025-14921 | none supplied | open |
| transformers | CVE-2025-14924 | none supplied | open |
| transformers | CVE-2025-14926 | none supplied | open |
| transformers | CVE-2025-14927 | none supplied | open |
| transformers | CVE-2025-14928 | none supplied | open |
| transformers | CVE-2025-14929 | none supplied | open |
| transformers | CVE-2025-14930 | none supplied | open |
| transformers | CVE-2025-2099 | 4.49.0 | open |
| transformers | CVE-2025-3263 | 4.51.0 | open |
| transformers | CVE-2025-3264 | 4.51.0 | open |
| transformers | CVE-2025-3777 | 4.52.1 | open |
| transformers | CVE-2025-3933 | 4.52.1 | open |
| transformers | CVE-2025-5197 | 4.53.0 | open |
| transformers | CVE-2025-6051 | 4.53.0 | open |
| transformers | CVE-2025-6638 | 4.53.0 | open |
| transformers | CVE-2025-6921 | 4.53.0 | open |
| transformers | CVE-2026-1839 | 5.0.0 | open |
| transformers | CVE-2026-4372 | 5.3.0 | open |
| transformers | CVE-2026-5241 | none supplied | open |
| transformers | CVE-2026-9856 | 5.10.0 | open |
