#!/usr/bin/env python3
"""
Chunk Ablation Experiment — Paper Figure 1
Proves that Precision@K inflates as PΩ drops as chunk size grows.
This is the core empirical evidence for our metric contribution.
Run: python chunk_ablation_experiment.py
"""

import sys
import os
import json
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from configs.config import SYNTHETIC_DATA_PATH
from module1.data_preprocessing import (
    chunk_documents, build_vector_store, get_embedding_model, load_msmarco_documents
)
from module2.rag_architectures import NaiveRAG
from module3.evaluation import (
    compute_precision_omega
)

# Load synthetic data
if not os.path.exists(SYNTHETIC_DATA_PATH):
    print(f"ERROR: {SYNTHETIC_DATA_PATH} not found. Run main.py --mode full first.")
    sys.exit(1)

with open(SYNTHETIC_DATA_PATH) as f:
    data = json.load(f)

# Use 50 samples for speed
samples = [s for s in data if s.get("ground_truth", "").strip()][:50]
print(f"Running ablation on {len(samples)} samples...")

chunk_sizes = [100, 200, 400, 600, 800, 1000]
results = []

# Load the same MS MARCO corpus family as the synthetic questions and production
# validation store. Mixing MS MARCO queries with SQuAD contexts would make the
# chunk-size comparison a corpus mismatch rather than an ablation.
docs = load_msmarco_documents(max_samples=1000)
embedding_model = get_embedding_model()

for size in chunk_sizes:
    print(f"\nChunk size: {size} characters...")

    # Re-chunk with this size
    chunks = chunk_documents(docs, chunk_size=size, chunk_overlap=min(50, size//4))

    # Each variant owns a unique temporary path. The shared builder guards its
    # first native open and pins it for this process; paths are never reused or
    # published. Only this owned TemporaryDirectory is cleaned afterwards.
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = build_vector_store(chunks, embedding_model, persist_dir=tmpdir)
        rag = NaiveRAG(vectorstore=vectorstore, top_k=5)

        pk_scores = []
        pom_scores = []
        iou_scores = []

        for sample in samples:
            query = sample.get("question", "")
            ground_truth = sample.get("ground_truth", "")
            if not query or not ground_truth:
                continue

            try:
                result = rag.run(query)
                retrieved_text = " ".join(result.get("contexts", []))

                # Precision@K — did any retrieved chunk contain answer tokens?
                gt_tokens = set(ground_truth.lower().split())
                retrieved_chunks = result.get("contexts", [])
                hits = sum(
                    1 for chunk in retrieved_chunks
                    if any(tok in chunk.lower() for tok in gt_tokens if len(tok) > 3)
                )
                pk = hits / max(len(retrieved_chunks), 1)

                # PΩ
                pom = compute_precision_omega(retrieved_text, ground_truth)

                pk_scores.append(pk)
                pom_scores.append(pom)

            except Exception as e:
                print(f"  Sample failed: {e}")
                continue

        if pk_scores:
            results.append({
                "chunk_size": size,
                "precision_k": np.mean(pk_scores),
                "pomega": np.mean(pom_scores),
                "n_samples": len(pk_scores),
            })
            print(f"  Precision@K={np.mean(pk_scores):.4f}  PΩ={np.mean(pom_scores):.4f}  n={len(pk_scores)}")

# Save results
df = pd.DataFrame(results)
os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/chunk_ablation_results.csv", index=False)
print("\nResults saved to outputs/chunk_ablation_results.csv")

# Generate THE KEY GRAPH
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df.chunk_size, df.precision_k,
        'b-o', label='Precision@K', linewidth=2.5, markersize=8)
ax.plot(df.chunk_size, df.pomega,
        'r-s', label='Precision Omega (PΩ)', linewidth=2.5, markersize=8)
ax.fill_between(
    df.chunk_size, df.precision_k, df.pomega,
    alpha=0.15, color='red',
    label='Misleading zone (Precision@K overestimates quality)'
)
ax.axvline(x=400, color='green', linestyle='--', linewidth=1.5,
           label='Reference chunk size (400 characters)')
ax.set_xlabel('Chunk Size (characters)', fontsize=13)
ax.set_ylabel('Score (0–1)', fontsize=13)
ax.set_title(
    'Precision@K Inflates as PΩ Drops with Increasing Chunk Size\n'
    'Standard metrics actively mislead RAG architecture selection',
    fontsize=13
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
path = "outputs/plots/divergence_proof.png"
os.makedirs("outputs/plots", exist_ok=True)
fig.savefig(path, dpi=150)
print(f"KEY GRAPH saved to {path}")
plt.show()
