#!/usr/bin/env python3
"""
Post-hoc analysis of the retrieval evaluation.

Answers two questions the summary table cannot:

1. **Does PΩ track real retrieval quality?** PΩ is this project's proposed
   precision-style metric. With human relevance judgements now available we can
   correlate it against nDCG@10 instead of assuming it is meaningful.
2. **How often does the "retrieval paradox" occur, and does reranking fix it?**
   Defined here concretely: a relevant passage exists in the corpus for the query,
   but the architecture's top-k contains none of it.

Usage: python analyze_results.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, ".")
from configs.config import OUTPUTS_DIR  # noqa: E402

RAW = os.path.join(OUTPUTS_DIR, "raw_eval_results.csv")


def main() -> None:
    if not os.path.exists(RAW):
        print(f"No raw results at {RAW}. Run: python main.py --mode eval ...")
        sys.exit(1)

    df = pd.read_csv(RAW)
    print("=" * 78)
    print("POST-HOC ANALYSIS")
    print("=" * 78)
    print(f"rows={len(df):,}  architectures={df['architecture'].nunique()}  "
          f"questions={df['question'].nunique():,}")

    has_ndcg = "ndcg_at_10" in df.columns and df["ndcg_at_10"].notna().any()

    # ── 1. Is PΩ a useful metric? ───────────────────────────────────────────────
    if has_ndcg and "precision_omega" in df.columns:
        print("\n" + "-" * 78)
        print("1. DOES PΩ TRACK GROUND-TRUTH RETRIEVAL QUALITY?")
        print("-" * 78)
        print(f"{'architecture':<20} {'PΩ':>8} {'nDCG@10':>9} {'spearman':>10} {'p':>10}")
        for arch, grp in df.groupby("architecture"):
            sub = grp[["precision_omega", "ndcg_at_10"]].dropna()
            if len(sub) < 10:
                continue
            rho, p = stats.spearmanr(sub["precision_omega"], sub["ndcg_at_10"])
            print(f"{arch:<20} {sub['precision_omega'].mean():>8.4f} "
                  f"{sub['ndcg_at_10'].mean():>9.4f} {rho:>10.3f} {p:>10.4f}")

        pooled = df[["precision_omega", "ndcg_at_10"]].dropna()
        rho, p = stats.spearmanr(pooled["precision_omega"], pooled["ndcg_at_10"])
        print(f"\nPooled Spearman(PΩ, nDCG@10) = {rho:.3f} (p={p:.2e}, n={len(pooled)})")

        # Between-architecture agreement: do the two metrics rank systems the same?
        means = df.groupby("architecture")[["precision_omega", "ndcg_at_10"]].mean()
        if len(means) >= 3:
            rho_b, p_b = stats.spearmanr(means["precision_omega"], means["ndcg_at_10"])
            print(f"Between-architecture Spearman = {rho_b:.3f} (p={p_b:.4f})")
            print("\nArchitecture ranking by each metric (best first):")
            print("  by PΩ      :", list(means.sort_values("precision_omega", ascending=False).index))
            print("  by nDCG@10 :", list(means.sort_values("ndcg_at_10", ascending=False).index))
            if rho_b < 0:
                print("\n  => The two metrics DISAGREE on which architecture is best.")
                print("     PΩ rewards returning fewer tokens, which raises answer-token")
                print("     concentration even when genuinely relevant passages are lost.")

        print(f"\nPΩ observed range across architectures: "
              f"{means['precision_omega'].min():.4f} - {means['precision_omega'].max():.4f}")
        print("  (a narrow band near zero indicates PΩ is dominated by context length,")
        print("   not retrieval quality)")

    # ── 2. Retrieval paradox rate ───────────────────────────────────────────────
    if has_ndcg and "hit_rate" in df.columns:
        print("\n" + "-" * 78)
        print("2. RETRIEVAL PARADOX: relevant passage exists but top-k misses it")
        print("-" * 78)
        labelled = df[df.get("num_relevant_known", 0) > 0] if "num_relevant_known" in df.columns else df
        print(f"{'architecture':<20} {'n':>6} {'miss rate':>11} {'hit rate':>10}")
        rates = {}
        for arch, grp in labelled.groupby("architecture"):
            sub = grp[grp["hit_rate"].notna()]
            if sub.empty:
                continue
            miss = float((sub["hit_rate"] == 0).mean())
            rates[arch] = miss
            print(f"{arch:<20} {len(sub):>6} {miss:>10.1%} {sub['hit_rate'].mean():>10.3f}")

        baseline = next((a for a in rates if "Naive" in a), None)
        rerank = next((a for a in rates if "Rerank" in a), None)
        if baseline and rerank:
            closed = rates[baseline] - rates[rerank]
            share = closed / rates[baseline] if rates[baseline] else 0.0
            print(f"\nReranking reduces the miss rate from {rates[baseline]:.1%} to "
                  f"{rates[rerank]:.1%}")
            print(f"  => closes {closed:.1%} of queries in absolute terms "
                  f"({share:.0%} of the baseline's misses)")

    # ── 3. Cost of the accuracy gain ────────────────────────────────────────────
    if "latency_ms" in df.columns:
        print("\n" + "-" * 78)
        print("3. LATENCY COST")
        print("-" * 78)
        lat = df.groupby("architecture")["latency_ms"].agg(["mean", "median"])
        for arch, row in lat.iterrows():
            print(f"{arch:<20} mean={row['mean']:>10.1f}ms  median={row['median']:>10.1f}ms")
        if has_ndcg:
            base = next((a for a in lat.index if "Naive" in a), None)
            rer = next((a for a in lat.index if "Rerank" in a), None)
            if base and rer:
                nd = df.groupby("architecture")["ndcg_at_10"].mean()
                gain = nd[rer] - nd[base]
                slow = lat.loc[rer, "mean"] / max(lat.loc[base, "mean"], 1e-9)
                print(f"\nReranking: +{gain:.4f} nDCG@10 for {slow:.0f}x latency")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
