# RAG System Evaluation Report

> _Status: Preliminary results — full benchmark in progress._
>
> The numbers below are an honest report of the most recent evaluation run on a
> small SQuAD subset (n=50 per architecture). They are NOT yet representative of
> the system's final performance because:
>
> 1. **Sample size is too small** for statistically meaningful comparison
>    (95% CI half-width on a mean of 0.3 with n=50 is ≈ ±0.13, which is wider
>    than the gap between any two architectures here).
> 2. **The judge LLM was the same family as the generator LLM**, which biases
>    faithfulness scores. We have since decoupled the judge (see
>    `module3/evaluation.py`).
> 3. **The retrieval metrics (IoU, noise ratio) had a definitional flaw** — they
>    compared the entire retrieved context against a 1-3 token gold answer, so
>    noise ratio was structurally near-1.0. The metrics are now bounded
>    correctly (see `compute_token_iou`, `compute_noise_ratio`).
>
> Re-run targets after the fixes land: `n=200` per architecture, 95% bootstrap
> CIs, separate judge LLM, MS MARCO test split.

## Architecture Comparison (preliminary, n=50, SQuAD subset)

| Architecture       | Faithfulness | Context Precision | Answer Recall | Noise Ratio | Latency (ms) |
|:-------------------|-------------:|------------------:|--------------:|------------:|-------------:|
| Naive RAG          |        0.330 |             0.234 |         0.158 |       0.965 |          469 |
| Hybrid Multi-Query |        0.210 |             0.110 |         0.154 |       0.990 |        1 109 |
| Parent-Child RAG   |        0.272 |             0.372 |         0.120 |       0.991 |          489 |
| CRAG               |        0.372 |             0.276 |         0.026 |       0.963 |          959 |

## Quality Gate Status (current thresholds)

| Metric             | Threshold | Naive | Hybrid | P-C   | CRAG  |
|:-------------------|----------:|------:|-------:|------:|------:|
| Faithfulness ≥     |     0.75  |   ✗   |   ✗    |   ✗   |   ✗   |
| Context Precision ≥|     0.70  |   ✗   |   ✗    |   ✗   |   ✗   |
| Answer Relevance ≥ |     0.80  |   ✗   |   ✗    |   ✗   |   ✗   |
| Noise Ratio ≤      |     0.25  |   ✗   |   ✗    |   ✗   |   ✗   |

All four architectures currently fail the quality gate. This report exists
to document the baseline before fixes; it should be regenerated after the
metric repair, judge decoupling, and larger sample run.

## Reproduction

```bash
python main.py --mode eval --samples 200
python main.py --mode visualize
```

The command above will overwrite this file with up-to-date results from
`outputs/eval_results.csv` via `module4.visualization.generate_summary_report`.
