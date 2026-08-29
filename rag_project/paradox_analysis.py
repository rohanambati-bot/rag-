# paradox_analysis.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# paradox_analysis.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Use raw results which has query_type column
RAW_PATH = "outputs/raw_eval_results.csv"
OLD_PATH = "outputs/eval_results.csv"

if os.path.exists(RAW_PATH):
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} rows from raw_eval_results.csv")
elif os.path.exists(OLD_PATH):
    df = pd.read_csv(OLD_PATH)
    print(f"Loaded from eval_results.csv (re-run pipeline for full results)")
else:
    print("ERROR: No results file found. Run main.py --mode full first.")
    exit(1)

# Check required columns exist
required = ["precision_omega", "architecture"]
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"ERROR: Missing columns: {missing}")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

# Add hit_rate proxy if not present
if "hit_rate" not in df.columns:
    df["hit_rate"] = (df["precision_omega"] > 0).astype(float)

# Add query_type if not present
if "query_type" not in df.columns:
    df["query_type"] = "unknown"

# Flag paradoxes
df["paradox"] = (
    (df["hit_rate"] >= 1.0) &
    (df["precision_omega"] < 0.4)
).astype(float)

# Overall stats
overall = df["paradox"].mean() * 100
print(f"\nOverall paradox rate: {overall:.1f}%")
print(f"Total queries: {len(df)}")
print(f"Paradox queries: {int(df['paradox'].sum())}")

# Per architecture
print("\nParadox rate by architecture:")
print(df.groupby("architecture")["paradox"].mean().mul(100).round(1))

# Pivot heatmap
try:
    pivot = df.pivot_table(
        values="paradox",
        index="architecture",
        columns="query_type",
        aggfunc="mean"
    ) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f",
                cmap="RdYlGn_r", ax=ax, vmin=0, vmax=20)
    ax.set_title(
        f"Retrieval Paradox Rate (%) by Architecture × Query Type\n"
        f"Overall paradox rate: {overall:.1f}%  "
        f"Red = high, Green = low",
        fontsize=12
    )
    plt.tight_layout()
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/paradox_heatmap.png", dpi=150)
    print("\nParadox heatmap saved to outputs/plots/paradox_heatmap.png")
except Exception as e:
    print(f"Heatmap failed: {e}")
    print("Run with full results (including query_type column) for heatmap.")

df = pd.read_csv("outputs/eval_results.csv")

# Flag paradoxes
df["paradox"] = (df["hit_rate"] >= 1.0) & (df["precision_omega"] < 0.4)

# Pivot: architecture × query_type → paradox rate
pivot = df.pivot_table(
    values="paradox",
    index="architecture",
    columns="query_type",
    aggfunc="mean"
) * 100  # convert to percentage

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt=".1f",
            cmap="RdYlGn_r", ax=ax,
            vmin=0, vmax=20)
ax.set_title(
    "Retrieval Paradox Rate (%) by Architecture × Query Type\n"
    "Red = high paradox rate, Green = low",
    fontsize=13
)
plt.tight_layout()
plt.savefig("outputs/plots/paradox_heatmap.png", dpi=150)

# Print the overall paradox rate
overall = df["paradox"].mean() * 100
print(f"Overall paradox rate: {overall:.1f}%")