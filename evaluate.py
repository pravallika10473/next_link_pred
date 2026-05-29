"""
Evaluate the trained link prediction bi-encoder.

Two modes:
  1. Retrieval test  — embed a corpus of docs, query it, see top-k results
  2. Recall@K        — measure how often the correct doc is in top-K

Usage:
  python evaluate.py                         # interactive query mode
  python evaluate.py --recall                # compute Recall@K on val set
  python evaluate.py --model model/link-predictor/epoch-2
"""

import argparse
import torch
import numpy as np
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="model/link-predictor", help="Path to trained model")
parser.add_argument("--recall", action="store_true", help="Compute Recall@K on val set")
parser.add_argument("--topk",   type=int, default=5)
args = parser.parse_args()

# ── Load model ────────────────────────────────────────────────────────────────
device = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Device: {device}")
print(f"Loading model from: {args.model}")
model = SentenceTransformer(args.model, device=device)

# ── Load val data ─────────────────────────────────────────────────────────────
dataset = load_from_disk("data/cc_news_pairs")
split    = dataset.train_test_split(test_size=0.01, seed=42)
val_data = split["test"]

queries   = [row["query"]    for row in val_data]
positives = [row["positive"] for row in val_data]

print(f"Val pairs: {len(val_data)}")

# ── Embed the corpus (all positive docs) ─────────────────────────────────────
print("\nEmbedding corpus...")
doc_embeddings = model.encode(
    positives,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)  # shape: [N, 768]

# ── Recall@K evaluation ───────────────────────────────────────────────────────
if args.recall:
    print("\nEmbedding queries...")
    query_embeddings = model.encode(
        queries,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )  # shape: [N, 768]

    # similarity matrix: [N_queries, N_docs]
    scores = query_embeddings @ doc_embeddings.T

    for k in [1, 3, 5, 10]:
        # For each query i, check if correct doc i is in top-k results
        top_k_indices = np.argsort(-scores, axis=1)[:, :k]
        correct = sum(i in top_k_indices[i] for i in range(len(queries)))
        recall = correct / len(queries) * 100
        print(f"Recall@{k:2d}: {recall:.1f}%  ({correct}/{len(queries)})")

# ── Interactive query mode ────────────────────────────────────────────────────
else:
    print("\nEntering interactive query mode.")
    print("Type a headline or description and see which articles it retrieves.")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Query: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        # Embed the query and compute similarity against all docs
        query_emb = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )  # shape: [1, 768]

        scores = (query_emb @ doc_embeddings.T)[0]  # shape: [N]
        top_k  = np.argsort(-scores)[:args.topk]

        print(f"\nTop {args.topk} results:")
        for rank, idx in enumerate(top_k):
            print(f"\n  [{rank+1}] score={scores[idx]:.3f}")
            print(f"  title:   {queries[idx]}")
            print(f"  article: {positives[idx][:200]}...")
        print()
