# A1 — strict generation and v2 evidence

## Implemented source

- `module2/answer_validation.py`: strict `extractive-answer/v2` JSON protocol; no additional fields, duplicate keys, non-finite JSON constants, empty answers or free-form prose. At most four claims, 600 characters per quote, 2,500 rendered answer characters, 12,000 raw-output/evidence characters and 4,000 question characters.
- `module2/rag_architectures.py`: generation invokes the structured answer seam, with a distinct trusted system message and JSON-encoded question/evidence data. Accepted output is rendered only from exact whole sentence/line spans in the actual prompt-visible slices. Original retrieval indices survive blank documents and truncation. Each selected span has inline `[context_index]` and `answer_citations` offsets. The existing `citations` field remains retrieval metadata provenance. Both `run()` and `generate()` use the gate without shared last-answer state. The old keyless string fallback also uses the gate.
- Answer transports retain configured model choices and make one answer attempt, no SDK retries, with 20-second SDK network timeouts. Non-stop finish reasons, exceptions and invalid outputs abstain. Rate waits above two seconds abstain. No fallback to unvalidated provider prose after an answer failure.
- Conservative screens reject no-keyword-overlap evidence, recognizable directives/exfiltration requests and simple differing same-subject assertions. These are lexical/surface checks, not semantic judgments.
- `scripts/evaluate_answer_quality_v2.py` and `tests/fixtures/answer_quality_v2.json`: fixed, hash-pinned v2 dataset/schema and create-only deterministic runner. 64 cases, eight per category: supported, unsupported, conflict, malicious, citation failures, malformed outputs, evidence scoping and semantic unknowns. 44 use the real extractive selector and 20 use scripted invocation outputs. All traverse the actual production gate; only invocation is faked.
- `tests/test_answer_validation.py`, `tests/test_generation_boundaries.py`, `tests/test_reranking_and_fidelity.py`, `tests/test_answer_boundaries_eval.py`: production failure/span/role/timeout checks and v2 evaluator mutation checks. Historical v1 fixture/schema/script/receipt hashes remain checked, including its previously unsafe echoes and passthrough. The frozen v1 runner/data/receipts were not edited or regenerated.

## Actual checks

All commands were run from the project root. No provider/judge calls, model downloads, private dotenv reads, commits, pushes or deployments were performed.

```bash
PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GROQ_API_KEY= MISTRAL_API_KEY= OPENROUTER_API_KEY= OPENAI_API_KEY= .venv/bin/python -m pytest -q tests/test_answer_validation.py tests/test_generation_boundaries.py tests/test_reranking_and_fidelity.py tests/test_answer_boundaries_eval.py
```

Final focused result: **211 passed in 1.11 s**. Earlier working-step runs passed 58, 105 and 205 tests as coverage was added.

```bash
make test
```

Whole-workspace check at the preceding implementation state: **659 passed, 3 skipped, 17 warnings in 64.99 s**. Other agents were working concurrently; this is not an exclusive A1 test count. Six focused tests were added afterwards and the final 211-test focused check passed. No A1 production source changed after that whole-workspace run. Warnings were existing PyMuPDF SWIG/Chroma deprecations and SciPy precision-loss warnings.

```bash
PYTHON_DOTENV_DISABLED=1 .venv/bin/python -m ruff check --select E,F,W --ignore E501 module2/rag_architectures.py module2/answer_validation.py scripts/evaluate_answer_quality_v2.py tests/test_answer_validation.py tests/test_answer_boundaries_eval.py tests/test_generation_boundaries.py tests/test_reranking_and_fidelity.py
git diff --check -- module2/rag_architectures.py module2/answer_validation.py scripts/evaluate_answer_quality_v2.py tests/test_answer_validation.py tests/test_answer_boundaries_eval.py tests/test_generation_boundaries.py tests/test_reranking_and_fidelity.py tests/fixtures/answer_quality_v2.json
```

Both passed.

```bash
PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GROQ_API_KEY= MISTRAL_API_KEY= OPENROUTER_API_KEY= OPENAI_API_KEY= .venv/bin/python scripts/evaluate_answer_quality_v2.py --output-dir outputs/security-v4-answer-quality-v2-20261019
PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GROQ_API_KEY= MISTRAL_API_KEY= OPENROUTER_API_KEY= OPENAI_API_KEY= .venv/bin/python scripts/evaluate_answer_quality_v2.py --output-dir outputs/security-v4-answer-quality-v2-20261019-replay
diff -r outputs/security-v4-answer-quality-v2-20261019 outputs/security-v4-answer-quality-v2-20261019-replay
```

Each run: **64 cases, 960 deterministic validity/support/protocol assertions, zero failures**. Replay is byte-identical. Status observations are 17 answers / 47 abstentions, not quality rates. There were 38 invocation-seam calls and zero provider/judge calls. All seven semantic dimensions and Cohen's kappa are null. Each hash-ordered review record has two blank reviewer slots; the key is separate.

Dataset SHA-256: `df37fd2b675168e97f165bcb3ac24adfb4054f2a75deff8e530193e257ccbe83`.
Schema SHA-256: `d192dd94447a544ee680d4d8fc4abd7dca04aa3f716f38ab464515e1cadd2601`.
Each run manifest pins the runner, production architecture and validation-module sources plus all emitted artifact hashes and Python version (3.12.14).

## Candid limits and integration handoff

- Exact substring/whole-span support proves provenance, not entailment, relevance, truth or completeness. A population question can receive a capital fact sharing its keywords; false source text can be quoted. Those cases are deliberately retained as semantic unknowns, never relabeled correct.
- The pattern screen misses paraphrased/obfuscated instructions. The conflict screen misses paraphrases and can reject compatible descriptions. No general injection-resistance guarantee is claimed. Public synthetic canaries are not actual secret-exfiltration tests.
- The answer mode is deliberately extractive: one to four whole quotes, with conservative rejection of long/partial spans. Retrieval rankings, candidate pools and retrieval models were not changed.
- SDK timeout/retry arguments are verified with fake clients, not live providers. SDK network timeouts are not hard whole-process wall-clock deadlines. No human reviewers or judges have assessed semantic quality.
- F1/main should include `module2/answer_validation.py` in serving cache implementation fingerprints. A1 did not edit serving/cache/API/UI code. Inline citations are present in answer strings; extra `answer_citations`/`answer_validation` may be omitted by existing API/UI serializers unless explicitly adopted.
- I1/main: the A1-owned module has no direct persistent Chroma opens; RAPTOR uses two ephemeral `Chroma.from_documents` paths without `persist_directory`. No conflicting index/cache edits were made here.
- Existing v1 output expectations are historical evidence. Running that old diagnostic script against new production is not a valid v2 quality assessment; its unsafe-passthrough expectations were not preserved as current requirements.
