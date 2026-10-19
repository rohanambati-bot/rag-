# module4/visualization.py
"""
Module 4 — Visualization & Reporting
──────────────────────────────────────
Plots:
  1. IoU comparison bar chart across architectures
  2. Chunk size ablation analysis
  3. Metrics heatmap
  4. Architecture performance radar/bar comparison
  5. Latency distribution box plot
  6. RAGAS metrics grouped bar chart
  7. Noise ratio trend

All plots saved as PNG to outputs/plots/.
"""

import os
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PLOT_DIR = os.path.join(OUTPUTS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ─── Colour Palette ────────────────────────────────────────────────────────────
PALETTE = {
    # Baselines / classic architectures
    "Naive RAG":              "#e05c5c",  # red
    "Hybrid Multi-Query":     "#f0a500",  # amber
    "Parent-Child RAG":       "#4ea8de",  # blue
    "CRAG":                   "#6bcb77",  # green
    "RAG-Fusion":             "#c44dff",  # violet
    "Adaptive RAG":           "#00b8a9",  # teal
    "Prod CRAG + Cache":      "#845ec2",  # purple
    # Two-stage retrieval
    "Rerank RAG":             "#ff7043",  # deep orange
    "GRACE-Ω + Rerank":       "#8d6e63",  # brown
    # GRACE-Ω family
    "GRACE-Ω":                "#1f4e8c",  # navy
    "GRACE-Ω (no PΩ gate)":   "#5c86c4",  # light navy
    "GRACE-Ω (no KG verify)": "#9db8de",  # paler navy
    "GRACE-Ω (no HyDE)":      "#c9d7ec",  # palest navy
}
# Distinct from PALETTE so an unregistered architecture can never be drawn in the
# same colour as a registered one (previously Rerank RAG collided with CRAG).
FALLBACK_COLORS = [
    "#7f7f7f", "#bcbd22", "#17becf", "#e377c2", "#2ca02c", "#d62728",
]

sns.set_theme(style="whitegrid", font_scale=1.1)


def _arch_color(arch: str, i: int = 0) -> str:
    return PALETTE.get(arch, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])


def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(PLOT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)
    return path


# ─── 1. IoU Comparison ────────────────────────────────────────────────────────

def plot_iou_comparison(summary_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    archs = summary_df["architecture"].tolist()
    iou_vals = summary_df["iou"].tolist()
    colors = [_arch_color(a, i) for i, a in enumerate(archs)]

    bars = ax.bar(archs, iou_vals, color=colors, edgecolor="white", linewidth=1.2, width=0.55)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_ylabel("Mean IoU")
    ax.set_title("IoU Comparison Across RAG Architectures", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    return _save(fig, "iou_comparison.png")


# ─── 2. Chunk Size Ablation ───────────────────────────────────────────────────

def plot_chunk_ablation(
    chunk_sizes: List[int],
    faithfulness_scores: List[float],
    iou_scores: List[float],
    noise_scores: List[float],
) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(chunk_sizes, faithfulness_scores, "o-", color="#845ec2", label="Faithfulness", lw=2)
    ax.plot(chunk_sizes, iou_scores, "s-", color="#4ea8de", label="IoU", lw=2)
    ax.plot(chunk_sizes, noise_scores, "^-", color="#e05c5c", label="Noise Ratio", lw=2)
    ax.set_xlabel("Chunk Size (tokens)")
    ax.set_ylabel("Score")
    ax.set_title("Chunk Size Ablation Analysis", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    return _save(fig, "chunk_ablation.png")


def plot_chunk_ablation_default() -> str:
    """
    ILLUSTRATIVE ONLY — hardcoded example numbers, not measured results.

    Kept for slide/demo use. Do not include the output in any results section:
    real chunk-size ablation is produced by ``chunk_ablation_experiment.py``.
    """
    chunk_sizes = [100, 200, 300, 400, 600, 800]
    faithfulness = [0.61, 0.68, 0.74, 0.83, 0.80, 0.77]
    iou = [0.48, 0.54, 0.61, 0.69, 0.66, 0.62]
    noise = [0.40, 0.35, 0.28, 0.18, 0.22, 0.27]
    return plot_chunk_ablation(chunk_sizes, faithfulness, iou, noise)


# ─── 3. Metrics Heatmap ───────────────────────────────────────────────────────

def plot_ir_metrics(summary_df: pd.DataFrame) -> str:
    """
    Grouped bar chart of the rank-aware IR metrics, with 95% CI error bars.

    These are the headline retrieval numbers: unlike the token-overlap metrics they
    are graded against MS MARCO's human relevance judgements, so they are directly
    comparable to published retrieval results. Error bars come from the bootstrap
    CIs already stored in the summary, so a difference that is not visually separated
    should not be described as an improvement.
    """
    metrics = [c for c in ("ndcg_at_10", "mrr_at_10", "recall_at_k", "hit_rate")
               if c in summary_df.columns and summary_df[c].notna().any()]
    if not metrics:
        logger.info("No rank-aware IR columns present — skipping IR metrics plot.")
        return ""

    labels = {
        "ndcg_at_10": "nDCG@10", "mrr_at_10": "MRR@10",
        "recall_at_k": "Recall@k", "hit_rate": "Hit Rate",
    }
    archs = summary_df["architecture"].tolist()
    x = np.arange(len(metrics))
    width = min(0.8 / max(len(archs), 1), 0.22)

    fig, ax = plt.subplots(figsize=(max(9, len(metrics) * 2.4), 5.5))
    for i, arch in enumerate(archs):
        row = summary_df[summary_df["architecture"] == arch].iloc[0]
        values, lo_err, hi_err = [], [], []
        for m in metrics:
            v = float(row.get(m, np.nan))
            values.append(v)
            lo = row.get(f"{m}_ci_lo", np.nan)
            hi = row.get(f"{m}_ci_hi", np.nan)
            lo_err.append(max(0.0, v - float(lo)) if pd.notna(lo) else 0.0)
            hi_err.append(max(0.0, float(hi) - v) if pd.notna(hi) else 0.0)

        offset = (i - (len(archs) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=arch,
               color=_arch_color(arch, i), edgecolor="white", linewidth=0.8)
        ax.errorbar(x + offset, values, yerr=[lo_err, hi_err],
                    fmt="none", ecolor="#333", elinewidth=1.1, capsize=3)
        # Place labels above the upper CI cap so they never overlap the error bars.
        if len(archs) <= 6:
            for xi, v, hi in zip(x + offset, values, hi_err):
                ax.text(xi, v + hi + 0.022, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in metrics])
    ax.set_ylim(0, 1.18)  # headroom for the rotated value labels
    ax.set_ylabel("Score")
    ax.set_title(
        "Rank-aware retrieval metrics (graded against human relevance judgements)\n"
        "Error bars: 95% bootstrap CI",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, ncol=min(len(archs), 3))
    return _save(fig, "ir_metrics.png")


def plot_metrics_heatmap(summary_df: pd.DataFrame) -> str:
    metric_cols = [
        c for c in [
            "ndcg_at_10", "mrr_at_10", "recall_at_k", "hit_rate",
            "faithfulness", "context_precision", "context_recall",
            "answer_relevance", "iou", "precision_omega",
        ]
        if c in summary_df.columns and summary_df[c].notna().any()
    ]
    if not metric_cols:
        logger.warning("No metric columns found for heatmap.")
        return ""

    heatmap_df = summary_df.set_index("architecture")[metric_cols]
    fig, ax = plt.subplots(figsize=(len(metric_cols) * 1.6 + 2, len(heatmap_df) + 1))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Architecture Metric Heatmap", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, "metrics_heatmap.png")


# ─── 4. Architecture Performance Comparison ───────────────────────────────────

def plot_architecture_comparison(summary_df: pd.DataFrame) -> str:
    ragas_cols = [
        c for c in ["faithfulness", "context_precision", "answer_relevance"]
        if c in summary_df.columns
    ]
    if not ragas_cols:
        logger.warning("No RAGAS columns for architecture comparison.")
        return ""

    archs = summary_df["architecture"].tolist()
    x = np.arange(len(archs))
    num_cols = len(ragas_cols)
    width = 0.8 / max(num_cols + 0.5, 1)

    fig, ax = plt.subplots(figsize=(max(13, len(archs) * 1.55), 6.5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]  # Distinct Blue, Orange, Green matching standard palette

    for j, col in enumerate(ragas_cols):
        vals = summary_df[col].tolist()
        offset = (j - (num_cols - 1) / 2) * width
        c = colors[j % len(colors)]
        ax.bar(x + offset, vals, width, label=col.replace("_", " ").title(),
               color=c, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(archs, rotation=18, ha="right", fontsize=10.5, fontweight="semibold")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Architecture Performance: RAGAS Metrics", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

    # Add threshold lines
    ax.axhline(0.75, color="#e05c5c", lw=1.2, ls="--", alpha=0.75, label="Faithfulness gate (75%)")
    ax.axhline(0.80, color="#f0a500", lw=1.2, ls=":", alpha=0.75, label="Ans Rel gate (80%)")
    return _save(fig, "architecture_comparison.png")


# ─── 5. Latency Distribution ──────────────────────────────────────────────────

def plot_latency_distribution(results_df: pd.DataFrame) -> str:
    if "latency_ms" not in results_df.columns or "architecture" not in results_df.columns:
        logger.warning("Missing latency or architecture columns.")
        return ""

    fig, ax = plt.subplots(figsize=(10, 5))
    archs = results_df["architecture"].unique().tolist()
    data = [results_df[results_df["architecture"] == a]["latency_ms"].dropna().tolist()
            for a in archs]
    colors = [_arch_color(a, i) for i, a in enumerate(archs)]

    bp = ax.boxplot(data, labels=archs, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel("Latency (ms)")
    ax.set_title("End-to-End Latency Distribution per Architecture", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    return _save(fig, "latency_distribution.png")


# ─── 6. RAGAS Grouped Bar Chart ───────────────────────────────────────────────

def plot_ragas_metrics(summary_df: pd.DataFrame) -> str:
    """Dedicated grouped bar chart for all four RAGAS metrics."""
    ragas_cols = [
        c for c in [
            "faithfulness", "context_precision", "context_recall", "answer_relevance"
        ] if c in summary_df.columns
    ]
    if not ragas_cols:
        return ""

    archs = summary_df["architecture"].tolist()
    x = np.arange(len(ragas_cols))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, arch in enumerate(archs):
        row = summary_df[summary_df["architecture"] == arch].iloc[0]
        vals = [float(row.get(c, 0)) for c in ragas_cols]
        offset = (i - len(archs) / 2) * width + width / 2
        ax.bar(
            x + offset, vals, width,
            label=arch,
            color=_arch_color(arch, i),
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n").title() for c in ragas_cols], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("RAGAS Metric Comparison Across All Architectures", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, "ragas_metrics.png")


# ─── 7. Noise Ratio Trend ─────────────────────────────────────────────────────

def plot_noise_ratio(summary_df: pd.DataFrame) -> str:
    if "noise_ratio" not in summary_df.columns:
        return ""

    archs = summary_df["architecture"].tolist()
    noise_vals = summary_df["noise_ratio"].tolist()
    colors = [_arch_color(a, i) for i, a in enumerate(archs)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(archs, noise_vals, color=colors, edgecolor="white", height=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
    ax.set_xlim(0, 0.5)
    ax.axvline(0.25, color="#e05c5c", lw=1.5, ls="--", label="Max allowed (0.25)")
    ax.set_xlabel("Noise Ratio (lower is better)")
    ax.set_title("Noise Ratio Across Architectures", fontsize=14, fontweight="bold")
    ax.legend()
    return _save(fig, "noise_ratio.png")


# ─── 8. Full Report ───────────────────────────────────────────────────────────

def plot_query_type_breakdown(raw_df: pd.DataFrame, output_dir: str):
    """PΩ by architecture × query type heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if "query_type" not in raw_df.columns:
        return None
    if "precision_omega" not in raw_df.columns:
        return None

    pivot = raw_df.pivot_table(
        values="precision_omega",
        index="architecture",
        columns="query_type",
        aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f",
                cmap="RdYlGn", ax=ax,
                vmin=0, vmax=1)
    ax.set_title(
        "Precision Omega (PΩ) by Architecture × Query Type\n"
        "Green = high information density, Red = high noise",
        fontsize=13
    )
    ax.set_xlabel("Query Type")
    ax.set_ylabel("RAG Architecture")
    plt.tight_layout()
    path = os.path.join(output_dir, "pomega_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def generate_all_plots(
    summary_df: pd.DataFrame,
    raw_results_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    """Generate all visualizations and return list of saved paths."""
    saved = []

    # Headline figure first: rank-aware metrics graded against human judgements.
    saved.append(plot_ir_metrics(summary_df))
    saved.append(plot_iou_comparison(summary_df))
    # NOTE: plot_chunk_ablation_default() is deliberately NOT called here. It draws
    # hardcoded illustrative numbers, and emitting it during a real run put a
    # synthetic figure next to measured ones. Real chunk-ablation data comes from
    # chunk_ablation_experiment.py, which writes its own figure.
    saved.append(plot_metrics_heatmap(summary_df))
    saved.append(plot_architecture_comparison(summary_df))
    saved.append(plot_ragas_metrics(summary_df))
    saved.append(plot_noise_ratio(summary_df))

    if raw_results_df is not None and "latency_ms" in raw_results_df.columns:
        saved.append(plot_latency_distribution(raw_results_df))

    if raw_results_df is not None:
        saved.append(plot_query_type_breakdown(raw_results_df, PLOT_DIR))

    saved = [p for p in saved if p]
    logger.info("Generated %d plots in %s", len(saved), PLOT_DIR)
    return saved


def generate_summary_report(summary_df: pd.DataFrame, output_path: Optional[str] = None) -> str:
    """
    Generate a Markdown summary report of evaluation results.
    """
    lines = [
        "# RAG System Evaluation Report",
        "",
        "## Architecture Comparison",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Best Architecture",
        "",
    ]

    # Prefer a rank-aware retrieval metric graded against human relevance labels;
    # fall back to faithfulness only when qrels were unavailable. Reporting a
    # "winner" on a single uncorrected mean is not evidence, so the caveat is
    # emitted alongside it and readers are pointed at the significance table.
    ranked_by = next(
        (m for m in ("ndcg_at_10", "mrr_at_10", "recall_at_k") if m in summary_df.columns),
        None,
    )
    headline = ranked_by or ("faithfulness" if "faithfulness" in summary_df.columns else None)

    if headline:
        best = summary_df.loc[summary_df[headline].idxmax()]
        lines.append(
            f"Highest mean **{headline}**: **{best['architecture']}** "
            f"({best[headline]:.4f}), n={int(best['n']) if 'n' in best else 'n/a'}."
        )
        lines.append("")
        lines.append(
            "> Ranking by mean alone does not establish a difference. See "
            "`significance_tests.csv` for question-paired, Benjamini-Hochberg "
            "corrected comparisons against the Naive baseline."
        )

    report = "\n".join(lines)

    path = output_path or os.path.join(OUTPUTS_DIR, "evaluation_report.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(report)
    logger.info("Report saved to %s", path)
    return report


# ─── Demo with fixed benchmark data (matches slide 7 values) ──────────────────

def demo_plots() -> None:
    """Generate demo plots using the benchmark data covering all 9 architectures."""
    data = {
        "architecture": [
            "Naive RAG",
            "Hybrid Multi-Query",
            "Parent-Child RAG",
            "CRAG",
            "RAG-Fusion",
            "Adaptive RAG",
            "Rerank RAG",
            "GRACE-Ω",
            "GRACE-Ω + Rerank",
        ],
        "n":                  [50, 50, 50, 50, 50, 50, 50, 50, 50],
        "faithfulness":       [0.61, 0.74, 0.79, 0.83, 0.78, 0.81, 0.89, 0.82, 0.87],
        "context_precision":  [0.57, 0.70, 0.76, 0.79, 0.74, 0.78, 0.83, 0.77, 0.82],
        "context_recall":     [0.55, 0.68, 0.73, 0.77, 0.71, 0.76, 0.85, 0.79, 0.84],
        "answer_relevance":   [0.63, 0.75, 0.80, 0.82, 0.79, 0.81, 0.86, 0.83, 0.85],
        "noise_ratio":        [0.31, 0.18, 0.14, 0.11, 0.15, 0.12, 0.08, 0.10, 0.09],
        "iou":                [0.0826, 0.0703, 0.0819, 0.0840, 0.0696, 0.0819, 0.1240, 0.0682, 0.0837],
        "precision_omega":    [0.0041, 0.0073, 0.0039, 0.0053, 0.0040, 0.0039, 0.0125, 0.0039, 0.0045],
        "latency_ms":         [126.3, 270.5, 103.8, 687.8, 197.8, 342.3, 1356.3, 536.8, 1065.0],
        "ndcg_at_10":         [0.632, 0.665, 0.632, 0.575, 0.640, 0.658, 0.699, 0.645, 0.682],
        "mrr_at_10":          [0.611, 0.642, 0.611, 0.570, 0.620, 0.635, 0.689, 0.628, 0.669],
        "recall_at_k":        [0.800, 0.825, 0.801, 0.702, 0.810, 0.820, 0.857, 0.815, 0.845],
        "hit_rate":           [0.889, 0.910, 0.889, 0.786, 0.900, 0.915, 0.950, 0.895, 0.935],
    }
    summary_df = pd.DataFrame(data)

    # For latency box plot we need per-sample data — simulate with noise
    rows = []
    for _, row in summary_df.iterrows():
        for _ in range(30):
            rows.append({
                "architecture": row["architecture"],
                "latency_ms": max(10, row["latency_ms"] + np.random.normal(0, row["latency_ms"] * 0.15)),
            })
    raw_df = pd.DataFrame(rows)

    paths = generate_all_plots(summary_df, raw_df)
    print(f"\nGenerated {len(paths)} plots:")
    for p in paths:
        print(" ", p)

    generate_summary_report(summary_df)


if __name__ == "__main__":
    demo_plots()
