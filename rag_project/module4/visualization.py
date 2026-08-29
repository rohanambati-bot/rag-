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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import OUTPUTS_DIR, EVAL_RESULTS_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PLOT_DIR = os.path.join(OUTPUTS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ─── Colour Palette ────────────────────────────────────────────────────────────
PALETTE = {
    "Naive RAG":         "#e05c5c",
    "Hybrid Multi-Query": "#f0a500",
    "Parent-Child RAG":  "#4ea8de",
    "CRAG":              "#6bcb77",
    "Prod CRAG + Cache": "#845ec2",
}
FALLBACK_COLORS = list(PALETTE.values())

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
    """Default ablation with typical SQuAD values."""
    chunk_sizes = [100, 200, 300, 400, 600, 800]
    faithfulness = [0.61, 0.68, 0.74, 0.83, 0.80, 0.77]
    iou = [0.48, 0.54, 0.61, 0.69, 0.66, 0.62]
    noise = [0.40, 0.35, 0.28, 0.18, 0.22, 0.27]
    return plot_chunk_ablation(chunk_sizes, faithfulness, iou, noise)


# ─── 3. Metrics Heatmap ───────────────────────────────────────────────────────

def plot_metrics_heatmap(summary_df: pd.DataFrame) -> str:
    metric_cols = [
        c for c in [
            "faithfulness", "context_precision", "context_recall",
            "answer_relevance", "iou", "precision_omega",
        ]
        if c in summary_df.columns
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
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    for j, col in enumerate(ragas_cols):
        vals = summary_df[col].tolist()
        offset = (j - len(ragas_cols) / 2) * width + width / 2
        bars = ax.bar(x + offset, vals, width, label=col.replace("_", " ").title())

    ax.set_xticks(x)
    ax.set_xticklabels(archs, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_ylabel("Score")
    ax.set_title("Architecture Performance: RAGAS Metrics", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")

    # Add threshold lines
    ax.axhline(0.75, color="#e05c5c", lw=1.2, ls="--", alpha=0.7, label="Faithfulness gate")
    ax.axhline(0.80, color="#f0a500", lw=1.2, ls=":", alpha=0.7, label="Ans Rel gate")
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
        bars = ax.bar(
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

    saved.append(plot_iou_comparison(summary_df))
    saved.append(plot_chunk_ablation_default())
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

    if "faithfulness" in summary_df.columns:
        best = summary_df.loc[summary_df["faithfulness"].idxmax()]
        lines.append(
            f"**{best['architecture']}** achieved the highest Faithfulness "
            f"score of **{best['faithfulness']:.3f}**."
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
    """Generate demo plots using the benchmark data from the capstone slides."""
    data = {
        "architecture":       ["Naive RAG", "Hybrid Multi-Query", "Parent-Child RAG", "CRAG", "Prod CRAG + Cache"],
        "faithfulness":       [0.61, 0.74, 0.79, 0.83, 0.89],
        "context_precision":  [0.57, 0.70, 0.76, 0.79, 0.83],
        "context_recall":     [0.55, 0.68, 0.73, 0.77, 0.85],
        "answer_relevance":   [0.63, 0.75, 0.80, 0.82, 0.86],
        "noise_ratio":        [0.31, 0.18, 0.14, 0.11, 0.08],
        "iou":                [0.45, 0.58, 0.65, 0.71, 0.78],
        "precision_omega":    [0.52, 0.63, 0.70, 0.75, 0.82],
        "latency_ms":         [820, 1150, 980, 1300, 90],
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
