# Roadmap / Improvements

Prioritised by impact-per-effort. Each item states what it buys and, where relevant,
which quality dimension it raises. Grounded in what the current code actually does, so
none of this is aspirational hand-waving.

Legend: **Impact** and **Effort** are H/M/L.

---

## Tier 1 — do these next (highest return)

### 1. Clean load test at scale · Impact H · Effort L
The one number I could not produce honestly tonight. The cold-path load figure in
`outputs/load_test.json` is contaminated: the 4M-passage embedding job was saturating
the CPU during the run, so cold retrieval showed ~9 s/query. On an idle machine that
should be tens of milliseconds.
- **Do:** once `preprocess_train.py` has finished, re-run
  `python load_test.py --url ... --key ... --arch naive --concurrency 16 --requests 500`
  and also `--arch rerank`. Sweep concurrency (1, 4, 8, 16, 32) to find the knee.
- **Why:** turns "designed for load" into "measured under load" — the single biggest
  credibility gain left on the serving side.

### 2. Complete the LLM-judged evaluation (Run 2) · Impact H · Effort L (cost $)
`docs/paper.md` §5.3 (generation quality) and §5.4 (GRACE-Ω ablations) are the only
`[TBD]` sections left. They need the 8 LLM-dependent architectures, which cost
OpenRouter credits — which is why this was not auto-run.
- **Do:** `make eval-full` (or the command in §"Run 2" below). Start with a modest
  `--samples 50` to bound cost, inspect, then scale.
- **Why:** finishes the results section end-to-end and lets the ablations (does the
  PΩ gate / KG verification / HyDE actually help?) be answered with data.

### 3. Validate the judge against human labels · Impact H · Effort M
Every LLM-judged number currently rests on an unvalidated judge.
- **Do:** hand-label ~50 (question, answer, context) triples for faithfulness, then
  compute Cohen's kappa vs the LLM judge (κ helper already cited in references.bib).
  Report κ, not raw agreement (raw agreement overstates a judge because it ignores
  chance).
- **Why:** converts "the judge said 0.8" into "the judge agrees with humans at κ=…",
  which is the difference between a defensible and an indefensible generation metric.

---

## Tier 2 — strengthens the system

### 4. Distributed cache index · Impact M · Effort M
The rewritten cache keeps its vector index per-process; multiple workers share Redis
persistence but each warms its own index, so hit rate is per-worker until warm.
- **Do:** push the ANN index into a shared service (Redis Stack vector search, or
  Qdrant/Weaviate) so all workers share one index.
- **Why:** required before running more than one API replica in earnest.

### 5. Horizontal scaling story · Impact M · Effort M
Chroma here is a local single-node store.
- **Do:** document (and ideally demo) running behind a load balancer with a shared
  vector DB; the `/readyz` gating already supports rolling deploys.
- **Why:** the honest gap called out in the README's "Honest scope".

### 6. GRACE-Ω latency · Impact M · Effort M
~130 s to construct (LLM-built RAPTOR tree) and ~8.5 s/query.
- **Do:** persist the RAPTOR tree to disk and reload it (like the child vectorstore
  already is), and cache HyDE expansions. Consider dropping RAPTOR entirely given the
  reranker outperforms it.
- **Why:** currently unusable for interactive serving; this is also a paper talking
  point about complexity-vs-benefit.

### 7. Streaming responses · Impact M · Effort M
`/query` returns the whole answer at once.
- **Do:** add an SSE/streaming variant for the generator path.
- **Why:** perceived latency; expected of any modern LLM API.

---

## Tier 3 — polish and research depth

### 8. BEIR-style multi-domain evaluation · Impact M · Effort M
Everything is MS MARCO (web-search QA). Generality is untested.
- **Do:** run the retrieval-only harness against 2–3 BEIR datasets.
- **Why:** shows the reranking win isn't MS-MARCO-specific.

### 9. Reranker depth sweep · Impact M · Effort L
`RERANK_CANDIDATES=50` is arbitrary.
- **Do:** sweep candidate depth (20/50/100/200) and plot nDCG@10 vs latency.
- **Why:** gives the accuracy/latency trade curve — a strong figure.

### 10. Multi-hop / Self-RAG architecture · Impact M · Effort H
The "MULTIHOP" route is a label, not real iterative retrieval.
- **Do:** implement a genuine second-hop retrieval conditioned on first-hop results
  (Self-RAG style, already cited).
- **Why:** a real new architecture rather than a composition of existing ones.

### 11. CI: run the retrieval-only eval as a gate · Impact L · Effort L
- **Do:** add the LLM-free n≥200 retrieval eval to CI and fail on nDCG@10 regression
  vs a stored baseline.
- **Why:** catches retrieval regressions automatically; extends the existing gate.

### 12. Package the spaCy model · Impact L · Effort L
`KnowledgeGraphBuilder` needs `en_core_web_sm`; its auto-download shells out to
`python -m spacy download`, which fails when `python` isn't on PATH (only `python3`).
- **Do:** add `en_core_web_sm` to `requirements.txt` (as the pinned wheel URL) or a
  post-install step, and fix the download command to use `sys.executable`.
- **Why:** on a fresh machine the KG silently falls back to regex NER, degrading
  graph retrieval quality.

---

## Run 2 — the exact command (Tier 1 #2)

Left un-run tonight on purpose: it spends OpenRouter credits, and I don't auto-spend
an unattended budget. Trigger it yourself:

```bash
# Bounded first (control cost), all architectures, LLM judge on:
python main.py --mode eval --samples 50 --store full
python main.py --mode visualize      # regenerate figures incl. RAGAS metrics
python analyze_results.py            # refresh PΩ / miss-rate analysis

# Then, if the bounded run looks right, scale up (more cost):
python main.py --mode eval --samples 200 --store full
```

Watch OpenRouter usage while it runs. The generator is `openai/gpt-oss-20b` and the
judge is `qwen/qwen3.8-27b`; both are paid models. A 50-sample × 12-architecture run
is on the order of a few thousand LLM calls.

Once it completes, fill `docs/paper.md` §5.3 and §5.4 from the refreshed
`outputs/eval_results.csv` and `outputs/significance_tests.csv`.
