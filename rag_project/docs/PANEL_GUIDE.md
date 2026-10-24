# Panel Presentation Guide

Everything you need to present this project confidently. Read this once before you
walk in. Numbers here come from the committed artefacts (`outputs/eval_results.csv`,
`outputs/significance_tests.csv`, `outputs/load_test.json`) — you can show them live.

---

## 1. The one-sentence pitch

> "I built a reproducible framework that benchmarks 12 RAG architectures on MS MARCO
> using the metrics the IR field actually uses, and it produced statistically
> significant findings — including a result that overturned my own initial
> assumption — wrapped in a production-grade serving layer."

If you only get 30 seconds, say that, then show the results table.

---

## 2. The narrative arc (how to tell the story)

Panels reward a *story with a turn*, not a feature list. Yours has a genuine one.

1. **Problem.** RAG has many competing architectures. Teams pick between them using
   small samples, home-grown metrics, and no significance testing — so nobody can
   trust the comparisons.
2. **What I built.** A framework that (a) reuses MS MARCO's *human relevance labels*
   to grade retrieval with standard metrics (nDCG@10, MRR@10, Recall@k) at zero LLM
   cost, (b) adds LLM-judged generation metrics with a judge decoupled from the
   generator, and (c) does question-paired, multiple-comparison-corrected statistics.
3. **The turn.** When I ran it properly (n=500, corrected stats), two things happened
   that a weaker project would hide: a simple cross-encoder reranker beat my own
   elaborate architecture, and my own proposed metric (PΩ) turned out to be
   *negatively* correlated with retrieval quality. I report both as findings.
4. **Engineering.** I then hardened the serving layer to production standard:
   vector-indexed cache, fail-closed auth, observability, readiness probes.
5. **Honesty.** I know exactly what is and isn't production-proven, and I say so.

The turn in step 3 is your strongest asset. Lead with it. Examiners trust a student
who reports a negative result about their own idea far more than one who only shows
wins.

---

## 3. Talking points by topic (with the real numbers)

### 3a. The core insight — you already had the labels

"MS MARCO ships a human `is_selected` flag on every passage. I preserved it through
loading and chunking, so I could reconstruct **qrels** (relevance judgements) for
free — no manual labelling, no LLM. That let me compute the metrics MS MARCO is
actually scored on, over **500 queries**, at **zero API cost**."

Why it matters: it's the difference between "my custom score says X" and "the
standard, human-graded metric says X."

### 3b. The headline result — reranking

From `outputs/eval_results.csv` and `significance_tests.csv` (n=500, 496 paired):

| Architecture | nDCG@10 | MRR@10 | Recall@k | Hit Rate |
|---|---|---|---|---|
| **Rerank RAG** | **0.699** | **0.689** | **0.857** | **0.950** |
| Naive RAG | 0.632 | 0.611 | 0.800 | 0.889 |
| Parent-Child RAG | 0.632 | 0.611 | 0.801 | 0.889 |
| CRAG (lexical) | 0.575 | 0.570 | 0.702 | 0.786 |

"Adding a cross-encoder reranker improved nDCG@10 from 0.632 to 0.699 — a **+0.067**
gain, **p = 3×10⁻⁶** after Benjamini-Hochberg correction, effect size d_z = 0.22.
These are the **first statistically significant** architecture differences in the
project; the earlier n=12 run found none."

The concrete mechanism (this is the memorable line):

> "The answer-bearing passage was often retrieved but ranked below the cut-off. The
> reranker cut the rate of queries that miss a known-relevant passage entirely from
> **11.1% to 5.0% — a net 55% miss reduction, with 42 rescues and 12 lost hits.**"

This is a full-system comparison (wider dense+BM25 pool plus cross-encoder), not
an identical-pool ablation proving that the reranker alone caused the gain.

### 3c. The negative result — PΩ (be proud of this)

"I had proposed a metric, PΩ. With ground-truth labels I could finally validate it —
and it failed. Across architectures it correlates **negatively** with nDCG@10
(Spearman ρ = −0.20): it ranked the weakest system (CRAG) as best. The reason is
structural — its denominator rewards returning *fewer* tokens, so it favours a system
that discards passages even when those passages were relevant. I report this as a
negative result and recommend against using PΩ for architecture selection."

Say this proactively. It demonstrates scientific integrity, which is what a capstone
panel is actually grading.

### 3d. Statistical rigor

"Every comparison is question-paired — the same query under two systems — not zipped
positionally, which was a bug in the original code. I run a paired t-test and a
Wilcoxon signed-rank test, report Cohen's d_z, and correct for multiple comparisons
with Benjamini-Hochberg across the comparisons actually run for each metric
(three non-baseline architectures in the four-system retrieval experiment).
Systems that return identical context to the baseline are flagged rather than
producing meaningless NaN statistics."

### 3e. The engineering (production layer)

"The serving layer is built to be operated. Concretely:

- **Semantic cache** rewritten from an O(n) design that ran Redis `KEYS` on every
  lookup to an in-process vector index — benchmarked **217× faster at 20k entries**
  (183 ms → 0.8 ms per lookup).
- **Auth fails closed** — no key means 503, not open access.
- **Observability** — JSON logs with per-request correlation IDs, Prometheus metrics,
  and split liveness (`/healthz`) vs readiness (`/readyz`) probes.
- **One retrieval stack** — the API, UI and CLI all evaluate the same MS MARCO corpus;
  previously the API silently served a different (SQuAD) corpus."

### 3f. Load behaviour (real, measured)

From `outputs/load_test.json`:

"Under concurrent load — and while a 4-million-passage embedding job was saturating
the CPU — the cache-hit path sustained **~60 requests/second with zero errors**
(p50 182 ms, p99 1.3 s) on a single worker. The system sheds load with 503/504 rather
than falling over."

**Caveat to state honestly if asked:** the *cold* retrieval path was slow in that
measurement (seconds per query) purely because the training job was starving the CPU;
a clean re-run on an idle machine is a documented next step (`docs/ROADMAP.md`).

### 3g. Scale

"The framework runs on the full MS MARCO train split — **808,731 queries, ~4.1M
passages** embedded into a vector store — via a checkpoint-resumable pipeline that
survives interruption."

---

## 4. Live demo script (safe, ~4 minutes)

Have these ready in a terminal. All are quick and don't need the LLM (no cost, no
flakiness in front of the panel).

```bash
# 1. Tests are green (credibility in one command)
python -m pytest -q

# 2. The real, significant results
column -s, -t outputs/eval_results.csv | head -6
cat outputs/significance_tests.csv | column -s, -t | head -8

# 3. The PΩ negative result + reranking miss-rate analysis
python analyze_results.py                # prints Spearman, miss rates, latency

# 4. The headline figure
open outputs/plots/ir_metrics.png

# 5. The service (start it before the panel; it warms in ~1 min)
API_KEY=$(openssl rand -hex 24) python api.py &
curl localhost:8000/healthz              # liveness (instant)
curl localhost:8000/readyz               # readiness (503 until warm, then 200)
curl localhost:8000/metrics | grep rag_  # Prometheus metrics
# show fail-closed auth:
curl -X POST localhost:8000/query -d '{"question":"x","architecture":"naive"}' # 401
```

Rehearse #5 once beforehand and leave the server running so `/readyz` is green when
you present. If the venue Wi-Fi is bad, skip anything that touches OpenRouter and
lean on the retrieval-only results, which are all local.

---

## 5. Anticipated panel questions (with answers)

**Q: Your novel architecture (GRACE-Ω) didn't win. Isn't that a failure?**
A: "It's the finding. I built the measurement framework to answer exactly this
question honestly, and it showed a simpler, cheaper technique wins on retrieval. That
is more useful to a practitioner than an unverified claim that my architecture is
best. GRACE-Ω also composes established techniques (RAPTOR, HyDE, KG, routing) — the
contribution was always the framework and the honest comparison."

**Q: Why should I trust an LLM judge?**
A: "For retrieval I don't rely on it at all — those metrics are graded against human
labels. For generation metrics I decouple the judge model from the generator to
reduce self-preference bias, and I run a health check so a broken judge fails loudly.
I'm also explicit that judge–human agreement (Cohen's kappa) is not yet validated —
that's a stated limitation, not a hidden one."

**Q: n=500 — is that enough?**
A: "For retrieval metrics, yes — IR practice expects ~50+ topics and I have 500 with
paired judgements. The effects I claim (d_z ≈ 0.2) are reliably detectable at that n,
which is exactly why the earlier n=12 run found nothing significant."

**Q: What's actually novel here?**
A: "Three things: reusing MS MARCO's own labels to make retrieval evaluation free and
standard; a corrected statistical protocol most applied RAG comparisons skip; and a
validated-then-rejected metric, which is a genuine negative result. The engineering
breadth — 12 architectures, a production API, and a deterministic regression suite —
supports those claims."

**Q: Is this production-ready?**
A: "The serving layer is production-*grade* in design — fail-closed auth,
observability, readiness gating, a vector-indexed cache I benchmarked at 217× the old
one. It is not production-*proven*: no load test at scale on dedicated hardware, no
staging, single-node vector store. I've documented exactly that gap. I'd call it a
research framework with a deployable prototype."

**Q: Why is GRACE-Ω so slow (8.5 s/query)?**
A: "It builds an LLM-generated summarisation tree and runs multiple LLM steps per
query. That cost is why I made the API construct it lazily and off the event loop, and
it's part of why the plain reranker is the better practical choice — which the data
shows."

**Q: What would you do with more time?** → point to `docs/ROADMAP.md` and name the top
three: clean load test at scale, kappa judge validation, completing the LLM-judged
evaluation (Run 2).

---

## 6. Things to say proactively (pre-empt the tough questions)

- "My earlier committed results were n=12 and not significant — those numbers should
  not be cited, and I documented that." (Shows you police your own work.)
- "PΩ failed validation; I kept it in the paper as a negative result."
- "The cold-path load number is contaminated by a concurrent job; the cache-path
  number is the clean one."

Saying these *first* converts potential 'gotcha' moments into evidence of maturity.

---

## 7. What genuinely makes this stand out (your closing)

1. **Scientific honesty** — negative results reported, not buried.
2. **Standard metrics, not home-grown ones** — comparable to the literature.
3. **Corrected statistics** — most student (and many industry) RAG comparisons skip
   this entirely.
4. **Engineering breadth** — data pipeline at 4M-passage scale, 12 architectures, a
   real API with auth/observability/caching, and a deterministic regression suite.
5. **You can defend every number** — it's all reproducible from committed artefacts.

Close with: *"The point of the project isn't that one architecture won — it's that I
built something that can tell you which one wins, honestly, and it did."*

---

## 8. Artefact index (what to have open)

| File | Shows |
|---|---|
| `outputs/eval_results.csv` | Per-architecture metrics, n=500 |
| `outputs/significance_tests.csv` | Paired, BH-corrected comparisons |
| `outputs/plots/ir_metrics.png` | Headline retrieval figure |
| `outputs/load_test.json` | Throughput / latency numbers |
| `docs/paper.md` | Full write-up (§5 results are real; §5.3/5.4 pending Run 2) |
| `docs/references.bib` | 27 citations |
| `docs/ROADMAP.md` | What's next and why |
