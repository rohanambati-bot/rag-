# Evaluating Retrieval-Augmented Generation Architectures on MS MARCO: A Reproducible Framework with Rank-Aware and LLM-Judged Metrics

> **Status: draft scaffold.** Section 5 (Results) is intentionally unpopulated.
> Numbers are filled in from `outputs/eval_results.csv` and
> `outputs/significance_tests.csv` after the large-n runs complete. Placeholders are
> marked `[TBD]`. Do not quote any figure from this file until the corresponding run
> has been executed and the CSV committed.
>
> Citation keys refer to `docs/references.bib`.

---

## Abstract

Retrieval-Augmented Generation architectures are commonly compared using small
samples, bespoke metrics, and uncorrected significance tests, which makes published
gains hard to trust or reproduce. We present a reproducible evaluation framework for
12 RAG architectures on MS MARCO v2.1 that grades retrieval against the dataset's
existing human relevance judgements (`is_selected`), yielding standard rank-aware
metrics (nDCG@10, MRR@10, Recall@k) at zero API cost, alongside RAGAS-style
LLM-judged generation metrics using a judge decoupled from the generator. All
comparisons are question-paired and Benjamini-Hochberg corrected.

On 500 queries we find that a two-stage cascade — wide first-stage retrieval followed
by a cross-encoder reranker — significantly outperforms a dense baseline
(nDCG@10 0.632 → 0.699, +0.067, corrected p = 3.0×10⁻⁶), reducing the rate of queries
where a known-relevant passage is missed entirely from 11.1% to 5.0%, at a cost of
129× retrieval latency. We also report a negative result on our own previously
proposed metric: PΩ correlates *negatively* with ground-truth ranking quality across
architectures (ρ = −0.200), ranking the weakest system first, because it rewards
returning fewer tokens even when relevant passages are lost. We conclude that
qrels-graded evaluation is essential, that reranking is the highest-leverage
component we tested, and that a new metric must be validated against ground truth
before it is used to select architectures. Code, prompts, seeds and per-sample
outputs are released.

---

## 1. Introduction

**Problem.** Retrieval-Augmented Generation [@lewis2020rag] now has many competing
architectures — hybrid sparse/dense fusion, hierarchical summarisation
[@sarthi2024raptor], query expansion [@gao2022hyde], complexity routing
[@jeong2024adaptive], corrective loops [@yan2024crag], self-reflection
[@asai2023selfrag]. Practitioners choosing between them face a literature where
each method is evaluated on different corpora with different metrics, so relative
performance is hard to establish.

**Gap.** Two recurring weaknesses in applied RAG comparisons:

1. *Metric substitution.* Retrieval quality is often reported with bespoke
   token-overlap scores rather than the rank-aware measures the IR community uses
   (nDCG@10, MRR@10, Recall@k) [@jarvelin2002ndcg; @thakur2021beir], making results
   incomparable to published baselines.
2. *Statistical under-powering.* Small sample counts and uncorrected multiple
   comparisons, despite evidence that leaderboard-style single-metric gains are
   often not significant [@lin2021significant; @urbano2025reliable].

**Contributions.**

- **C1.** A reproducible framework evaluating N RAG architectures under one corpus,
  one embedding model, one generator and one decoupled judge, with fixed seeds.
- **C2.** Dual-track evaluation: rank-aware IR metrics graded against MS MARCO's
  human `is_selected` judgements, plus RAGAS-style LLM-judged generation metrics
  [@es2024ragas]. The retrieval track requires no LLM, so it scales to large query
  counts at zero API cost.
- **C3.** A quantified cross-encoder reranking ablation [@nogueira2019passage],
  isolating the second stage from all other architectural differences.
- **C4.** Question-paired significance testing with Benjamini-Hochberg correction
  [@benjamini1995fdr; @urbano2025reliable], including explicit detection of variants
  that are behaviourally identical to the baseline.
- **C5.** An analysis of PΩ, a proposed precision-style metric, tested against
  ground-truth relevance rather than asserted.

---

## 2. Related Work

**Architectures.** Hybrid retrieval fuses BM25 [@robertson2009bm25] with dense
bi-encoder search [@reimers2019sbert], typically via Reciprocal Rank Fusion
[@cormack2009rrf]. RAPTOR [@sarthi2024raptor] builds a summarisation tree for
multi-hop questions. HyDE [@gao2022hyde] embeds a hypothetical answer.
Adaptive-RAG [@jeong2024adaptive] routes by question complexity. CRAG
[@yan2024crag] adds a retrieval evaluator that triggers corrective action, and
Self-RAG [@asai2023selfrag] critiques retrieved passages with reflection tokens.
Graph-structured retrieval [@edge2024graphrag] links entities across passages.

**Reranking.** Cross-encoders jointly encode query and passage
[@nogueira2019passage], with sequence-to-sequence [@nogueira2020monot5] and
late-interaction [@khattab2020colbert] variants. We use `bge-reranker-base`
[@xiao2024bge].

**Evaluation.** RAGAS [@es2024ragas] and ARES [@saadfalcon2024ares] automate RAG
assessment. LLM judges approach human agreement levels but exhibit position and
verbosity biases [@zheng2023judging], which motivates decoupling the judge from the
generator and reporting chance-corrected agreement [@cohen1960kappa].

**Positioning.** Relative to the above, this work does not propose a new
architecture as its primary claim. It contributes a *measurement* framework and an
honest comparison, including negative results.

---

## 3. Method

### 3.1 Corpus and relevance judgements

MS MARCO v2.1 [@nguyen2016msmarco]. Passages are extracted per query, deduplicated
on a 200-character prefix, and chunked (400 tokens, 50 overlap).

The key enabling detail: each MS MARCO passage carries `is_selected`, a human
judgement that the passage contains the answer. This label is preserved through
loading and chunking, so **qrels are recovered for free** — every chunk inherits its
parent passage's judgement (`build_qrels`, `module3/evaluation.py`). No manual
annotation and no LLM are required to grade retrieval.

| Split | Queries | Passages | Chunks | Store |
|---|---|---|---|---|
| validation | 1,000 | 9,858 | 11,510 | `data/chroma_db_full` |
| train | 808,731 | 4,137,589 | `[TBD]` | `data/chroma_db_train` |

Labelled QA pairs available for evaluation: **503,370** (train split).

*Limitation.* Global deduplication means a passage relevant to query A but first
seen under query B appears once, so Recall is marginally conservative.

### 3.2 Architectures evaluated

| Key | Name | Retrieval strategy |
|---|---|---|
| `naive` | Naive RAG | Dense top-k |
| `crag` | CRAG | Dense + LLM relevance grading + corrective rewrite-retry |
| `rag_fusion` | RAG-Fusion | 3 LLM query variants + RRF |
| `hybrid` | Hybrid Multi-Query | LLM rewrites × (dense + BM25), RRF |
| `adaptive` | Adaptive RAG | Complexity routing (narrow / wide / hybrid) |
| `parent_child` | Parent-Child | Child-chunk search, parent-chunk return |
| `rerank` | Rerank RAG | Wide dense+BM25 → cross-encoder → top-k |
| `grace_omega` | GRACE-Ω | HyDE + routing + RAPTOR/KG + PΩ gate + KG verification |
| `grace_rerank` | GRACE-Ω + Rerank | GRACE-Ω with a cross-encoder second stage |
| `grace_no_gate` | ablation A | PΩ gate removed |
| `grace_no_kg` | ablation B | Cross-structure verification removed |
| `grace_no_hyde` | ablation C | HyDE removed (KG retained) |

Ablations A-C differ from GRACE-Ω in exactly one component. Ablation C previously
also dropped the knowledge graph, confounding two variables; this is corrected and
regression-tested (`tests/test_architecture_registration.py`).

### 3.3 Metrics

**Rank-aware (LLM-free, graded against qrels).** Recall@k, Precision@k, Hit Rate,
MRR@10 (MS MARCO's official passage-ranking metric), nDCG@10 [@jarvelin2002ndcg].

**Token-overlap.** Answer-Recall, Token-IoU, Noise Ratio, and PΩ = p² where
p = |GT ∩ Retrieved| / |Retrieved| (a quadratic penalty on retrieved tokens absent
from the gold answer).

**LLM-judged** [@es2024ragas]. Faithfulness, Context Precision, Context Recall,
Answer Relevance. The judge is a different model family from the generator
(generator `openai/gpt-oss-20b`, judge `qwen/qwen3.8-27b`, both via OpenRouter) to
reduce self-preference bias [@zheng2023judging]. A one-call health check runs before
every evaluation so a mis-set judge fails loudly rather than silently returning no
scores.

### 3.4 Statistical procedure

Per-architecture means with 95% bootstrap confidence intervals (1,000 resamples,
seed 42) [@efron1979bootstrap]. Each architecture is compared against Naive RAG on
the **same questions** (joined on question id, not zipped positionally), using a
paired t-test and Wilcoxon signed-rank test, with Cohen's d_z as effect size.
p-values are corrected across the family of comparisons with Benjamini-Hochberg
[@benjamini1995fdr], following recommendations for multi-system IR comparison
[@urbano2025reliable]. Variants whose per-question scores are identical to the
baseline are reported as such rather than as undefined test statistics.

### 3.5 Reproducibility

Fixed seed (42) for sampling and bootstrap; pinned dependencies; all prompts in
source; per-sample outputs in `outputs/raw_eval_results.csv`; 164 automated tests
including property-based tests for metric bounds.

---

## 4. Experimental Setup

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, normalised)
- **Vector store:** Chroma; **sparse:** BM25 Okapi; **reranker:**
  `BAAI/bge-reranker-base`, 50 candidates → top-5
- **Generator:** `openai/gpt-oss-20b`; **Judge:** `qwen/qwen3.8-27b` (OpenRouter)
- **Retrieval depth:** k=5; **chunking:** 400/50
- **Hardware:** Apple Silicon, MPS acceleration, 16 GB RAM
- **Runs:** (a) retrieval-only, n=`[TBD]` queries, LLM-free;
  (b) full generation+judge, n=`[TBD]`

---

## 5. Results

**Run 1 (complete): LLM-free retrieval evaluation.** n=500 MS MARCO validation
queries, 496 with paired relevance judgements, over the 11,510-chunk validation
index. Four architectures whose retrieval requires no LLM call. Zero API cost.
Source: `outputs/eval_results.csv`, `outputs/raw_eval_results.csv`,
`outputs/significance_tests.csv`.

**Run 2 (pending): full generation + LLM-judged evaluation.** §5.3-5.4 remain
`[TBD]`; those architectures invoke the LLM during retrieval (HyDE, query rewriting,
routing) and so were excluded from the zero-cost sweep.

### 5.1 Retrieval effectiveness (qrels-graded, n=500)

| Architecture | nDCG@10 | MRR@10 | Recall@k | Hit Rate | Latency (ms) |
|---|---|---|---|---|---|
| **Rerank RAG** | **0.699** [0.674, 0.725] | **0.689** [0.660, 0.719] | **0.857** [0.833, 0.881] | **0.950** | 1538.3 |
| Naive RAG | 0.632 [0.604, 0.662] | 0.611 [0.580, 0.644] | 0.800 [0.769, 0.831] | 0.889 | 11.9 |
| Parent-Child RAG | 0.632 [0.603, 0.662] | 0.611 [0.580, 0.644] | 0.801 [0.771, 0.831] | 0.889 | 7.8 |
| CRAG (lexical) | 0.575 [0.545, 0.610] | 0.570 [0.538, 0.606] | 0.702 [0.667, 0.739] | 0.786 | 11.2 |

Brackets are 95% bootstrap CIs (1,000 resamples, seed 42).

### 5.2 Effect of cross-encoder reranking

Reranking improves every rank-aware metric, and every improvement survives
Benjamini-Hochberg correction across the comparison family:

| Metric | Naive | Rerank | Δ | BH-corrected p | d_z |
|---|---|---|---|---|---|
| nDCG@10 | 0.632 | 0.699 | **+0.067** | 3.0×10⁻⁶ | 0.218 |
| MRR@10 | 0.611 | 0.689 | **+0.077** | 2.1×10⁻⁵ | 0.210 |
| Recall@k | 0.800 | 0.857 | **+0.057** | 4.0×10⁻⁵ | 0.191 |
| Hit Rate | 0.889 | 0.950 | **+0.060** | 6.8×10⁻⁵ | 0.186 |

**These are the first statistically significant architecture differences in this
project.** Previous runs (n=12, uncorrected, positionally paired) found none.

**The retrieval paradox, quantified.** Defining a *miss* as a query for which a
relevant passage exists in the corpus but no relevant passage appears in the
returned top-k:

| Architecture | Miss rate |
|---|---|
| CRAG (lexical) | 21.4% |
| Naive RAG | 11.1% |
| Parent-Child RAG | 11.1% |
| **Rerank RAG** | **5.0%** |

Reranking cuts the miss rate from 11.1% to 5.0% — it recovers **55% of the
baseline's misses**. This is direct evidence for the project's original hypothesis:
the answer-bearing passage is frequently retrieved by the first stage but ranked
below the cut-off, and a cross-encoder is what promotes it.

**Cost.** The gain is not free: +0.067 nDCG@10 costs **129× latency** (11.9 ms →
1538.3 ms) with 50 candidates on contended MPS hardware. Reranking is therefore a
quality/latency trade rather than a strict improvement, and candidate depth is the
obvious tuning knob.

### 5.3 Generation quality (LLM-judged)

`[TBD — Run 2.]`

### 5.4 GRACE-Ω ablations

`[TBD — Run 2.]`

### 5.5 Is PΩ a useful metric? (No — reported as a negative result)

PΩ = p² was proposed in this project as a retrieval-precision metric. With human
relevance judgements available it can now be validated rather than assumed. It
fails three ways.

**1. It disagrees with ground truth about which architecture is best.**
Between-architecture Spearman correlation with nDCG@10 is **ρ = −0.200** (negative):

| Ranking by | Best → worst |
|---|---|
| PΩ | CRAG, Rerank, Naive, Parent-Child |
| nDCG@10 | Rerank, Naive, Parent-Child, **CRAG** |

PΩ ranks CRAG **first**; ground-truth nDCG@10 ranks it **last**. CRAG's lexical
filter discards documents, which shrinks the retrieved-token denominator and inflates
answer-token concentration — while simultaneously losing relevant passages (hit rate
0.786 vs 0.889, miss rate 21.4%). PΩ rewards exactly the behaviour that damages
retrieval.

**2. Per-query correlation is weak.** Within-architecture Spearman(PΩ, nDCG@10)
ranges 0.049-0.298; for the best architecture (Rerank RAG) it is 0.049 and not
significant (p=0.28). Pooled ρ = 0.149.

**3. It has almost no dynamic range.** Architecture means span 0.0152-0.0233 on a
[0,1] scale. With MS MARCO's short gold answers the bag-of-words denominator
dominates, so PΩ largely measures retrieved-context length.

Notably, PΩ *did* reach significance for CRAG (+0.008, p_bh < 10⁻⁶) — a
statistically significant improvement on a metric that is anti-correlated with
retrieval quality. This is a useful cautionary illustration: significance on an
unvalidated metric is not evidence of system quality.

**Recommendation.** PΩ should not be used to select architectures. It may retain
diagnostic value as a *context-density* indicator when reported next to a rank-aware
metric, but the p² formulation and its use as a headline score are not supported.

### 5.6 Secondary observations

- **Parent-Child RAG is behaviourally identical to Naive RAG** in this
  configuration (Δ = 0.000 on MRR@10, hit rate and answer-recall; flagged
  automatically as `identical_to_baseline`). Without a separately built child index
  it falls back to the same dense search. Its only real difference is lower latency
  (7.8 ms vs 11.9 ms). Earlier reports treated it as a distinct architecture.
- **CRAG with lexical grading is significantly worse than the baseline on every
  rank-aware metric** (nDCG −0.057, Recall −0.098, Hit Rate −0.103; all p_bh < 10⁻⁴).
  Keyword-overlap filtering actively removes relevant passages. The faithful
  LLM-graded CRAG path is evaluated in Run 2.
- **Token-overlap and rank-aware metrics broadly agree on Rerank RAG**
  (answer-recall +0.037, IoU +0.010, both significant), so the PΩ failure in §5.5 is
  specific to the p² concentration formulation, not to token-overlap metrics
  generally.

---

## 6. Discussion

**Reranking is the highest-leverage component tested.** A conceptually simple
cascade — retrieve wide, score jointly, keep few — produced the only large,
significant retrieval gains observed, consistent with the cross-encoder literature
[@nogueira2019passage]. It required no LLM, no knowledge graph and no routing logic.
For practitioners this suggests spending the first engineering increment on a second
retrieval stage rather than on architectural elaboration of the first.

**Filtering is dangerous without a good relevance signal.** CRAG's lexical grader was
*significantly worse* than doing nothing on every rank-aware metric. A filter's value
is bounded by the quality of its relevance estimator: with a weak estimator, discarding
candidates removes more signal than noise. This also explains why the faithful
LLM-graded CRAG of [@yan2024crag] is not interchangeable with a keyword approximation
of it.

**Unvalidated metrics can invert conclusions.** PΩ was significant, and significantly
*wrong*: it selected the worst-ranking system. The failure mode is structural — a
metric whose denominator is retrieved-token count rewards returning less, which is
only desirable if what is dropped is irrelevant. Any new RAG metric should be checked
against relevance judgements before being used for selection, and reporting a raw
correlation with a rank-aware metric is a cheap way to do so.

**Measurement infrastructure changed the conclusions, not just the error bars.**
Under the previous setup (n=12, no rank-aware metrics, positional pairing) nothing was
significant and the recorded conclusion favoured the baseline on faithfulness. With
qrels and n=500, clear and correctly-corrected differences appear. The binding
constraint was the evaluation design, not the architectures.

**Latency must be reported alongside quality.** At 129× the baseline latency,
reranking is unsuitable for latency-critical serving at this candidate depth. The
accuracy/latency curve as a function of candidate count is the natural next
experiment.

### 6.1 Threats to validity

### 6.1 Threats to validity

- **Judge validity.** Not yet validated against human labels; Cohen's kappa on a
  labelled subset is the required check [@cohen1960kappa; @zheng2023judging].
- **Single embedding model / single generator.** Results may not transfer.
- **Qrels sparsity.** MS MARCO labels few passages per query, so Recall is
  conservative and unjudged-but-relevant passages count as non-relevant.
- **Deduplication.** See §3.1.
- **Single corpus.** MS MARCO is web-search QA; BEIR-style multi-domain evaluation
  [@thakur2021beir] would test generality.
- **Partial architecture coverage.** Run 1 covers only the four LLM-free
  architectures. The eight LLM-dependent ones (including GRACE-Ω and its ablations)
  await Run 2, so no claim is made here about them.
- **Reranker latency is confounded.** Measurements were taken while a large embedding
  job shared the MPS device, so absolute latency is pessimistic; the 129× ratio should
  be re-measured on an idle machine.
- **Effect sizes are modest.** d_z ≈ 0.19-0.22 is a small-to-moderate paired effect.
  It is reliably detectable at n≈500 but would be invisible at n=12 — which is
  precisely why the earlier runs found nothing.

---

## 7. Conclusion

Three findings, in order of confidence.

**1. Cross-encoder reranking significantly improves retrieval on MS MARCO.**
nDCG@10 rises from 0.632 to 0.699 (+0.067, corrected p = 3.0×10⁻⁶, d_z = 0.218), and
the rate of queries where a known-relevant passage is missed entirely falls from 11.1%
to 5.0%. The cost is 129× retrieval latency, making this a deliberate trade rather
than a free win.

**2. PΩ, as formulated, should not be used to select architectures.** It correlates
negatively with ground-truth ranking quality across architectures (ρ = −0.200) and
ranks the weakest system first, because its denominator rewards returning fewer tokens
regardless of whether relevant passages survive. We report this as a negative result
about our own proposal.

**3. Evaluation design was the limiting factor.** Reusing MS MARCO's existing human
relevance labels made rank-aware, LLM-free evaluation possible at 500 queries and
turned a previously null comparison into clearly significant differences. The labels
were already in the data; only the wiring was missing.

The framework, corrected statistics, and per-sample outputs are released so these
claims can be checked and extended.

---

## Appendix A — Reproduction

```bash
# Retrieval-only, LLM-free (no API key needed), large n
python main.py --mode eval --retrieval-only --samples 500 --store full

# Full pipeline with generation + LLM judge
python main.py --mode eval --samples 200 --store full

# Evaluate against the train-split store
python main.py --mode eval --retrieval-only --samples 1000 --store train

# Tests
python -m pytest -q
```

## Appendix B — Artefacts

| File | Contents |
|---|---|
| `outputs/eval_results.csv` | Per-architecture means + bootstrap CIs |
| `outputs/raw_eval_results.csv` | Per-sample rows (every metric, every query) |
| `outputs/significance_tests.csv` | Paired, BH-corrected comparisons |
| `outputs/plots/ir_metrics.png` | **Headline figure**: rank-aware metrics with 95% CIs |
| `outputs/plots/` | Other generated figures |
| `analyze_results.py` | PΩ validity, miss-rate and latency-trade analysis (§5.2, §5.5) |
| `docs/references.bib` | Bibliography |

## Appendix C — Honest disclosure

Earlier committed results in this repository were produced with n=12 per
architecture, no rank-aware metrics, and positionally-paired significance tests. No
comparison reached significance. Those numbers should not be cited. The framework
changes described in §3.3-3.4 exist specifically to correct that, and any result
reported in §5 must come from a run using the corrected pipeline.
