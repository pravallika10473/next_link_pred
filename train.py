import os
import torch
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

# ── Device ────────────────────────────────────────────────────────────────────
# Use CUDA on CHPC, MPS on Mac, CPU as fallback
device = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Device: {device}")

# ── Model ─────────────────────────────────────────────────────────────────────
# all-mpnet-base-v2 is a BERT-based bi-encoder
# Both query and document go through the SAME encoder (shared weights)
# Output: 768-dim L2-normalized embedding per input
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME, device=device)

print(f"Model: {MODEL_NAME}")
print(f"Embedding dim: {model.get_sentence_embedding_dimension()}")
print(f"Max tokens:    {model.max_seq_length}")

# ── Data ──────────────────────────────────────────────────────────────────────
from sentence_transformers import InputExample
from torch.utils.data import DataLoader

dataset = load_from_disk("data/cc_news_pairs")

# 99% train, 1% validation
split      = dataset.train_test_split(test_size=0.01, seed=42)
train_data = split["train"]
val_data   = split["test"]

# InputExample is just a container for a pair of texts
# texts[0] = query, texts[1] = positive document
train_examples = [
    InputExample(texts=[row["query"], row["positive"]])
    for row in train_data
]

# 128 on A100 80GB — all-mpnet needs ~600MB per sequence at 384 tokens
BATCH_SIZE = 128

train_loader = DataLoader(
    train_examples,
    batch_size=BATCH_SIZE,
    shuffle=True,            # shuffle so every epoch sees different in-batch negatives
    collate_fn=lambda x: x, # return list as-is — we handle tokenization manually
)

print(f"\nTrain pairs:   {len(train_data):,}")
print(f"Val pairs:     {len(val_data):,}")
print(f"Batch size:    {BATCH_SIZE}")
print(f"Batches/epoch: {len(train_loader):,}")

# ── Loss ──────────────────────────────────────────────────────────────────────
from sentence_transformers import losses

# MultipleNegativesRankingLoss does exactly what we described:
#   1. encode queries  → query_emb  [B, 768]
#   2. encode positives → doc_emb   [B, 768]
#   3. scores = query_emb @ doc_emb.T   [B, B]
#   4. loss = CrossEntropy(scores / temperature, [0,1,2,...,B-1])
#
# temperature=0.05 (default) sharpens the distribution —
# the model is punished more harshly for giving high scores to wrong docs
train_loss = losses.MultipleNegativesRankingLoss(model)

print(f"\nLoss: MultipleNegativesRankingLoss")
print(f"Temperature: 0.05 (default)")
print(f"Negatives per query per batch: {BATCH_SIZE - 1}")

# ── Optimizer & Scheduler ─────────────────────────────────────────────────────
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

EPOCHS      = 3
LR          = 2e-5   # standard fine-tuning LR for BERT-based models

total_steps  = len(train_loader) * EPOCHS   # total number of weight updates
warmup_steps = int(total_steps * 0.1)       # first 10% of steps = warmup

# AdamW: Adam optimizer with weight decay
# weight_decay penalizes large weights → prevents overfitting
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# Linear warmup then linear decay:
#
#  LR
#  ^
#  |      /\
#  |     /  \
#  |    /    \________
#  |   /
#  +--+--+-----------> steps
#     ^  ^
#     |  warmup ends (step 232)
#     training starts
#
# Warmup: LR ramps from 0 → 2e-5 over first 232 steps
# Then:   LR decays linearly from 2e-5 → 0 over remaining steps
# Why warmup? The model starts with pretrained weights — jumping straight
# to full LR can destroy them. Warming up lets the model adjust gently.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

print(f"\nOptimizer:    AdamW  (lr={LR}, weight_decay=0.01)")
print(f"Epochs:       {EPOCHS}")
print(f"Total steps:  {total_steps:,}")
print(f"Warmup steps: {warmup_steps:,}  (first 10%)")

# ── Training Loop ─────────────────────────────────────────────────────────────
OUTPUT_DIR = "model/link-predictor"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nStarting training...")

# model.fit() is the modern sentence-transformers API — handles tokenization,
# device placement, forward pass, backward pass, and checkpointing internally.
# We pass our optimizer and scheduler so our warmup/decay schedule is respected.
model.fit(
    train_objectives=[(train_loader, train_loss)],
    epochs=EPOCHS,
    optimizer_class=AdamW,
    optimizer_params={"lr": LR, "weight_decay": 0.01},
    scheduler="WarmupLinear",
    warmup_steps=warmup_steps,
    output_path=OUTPUT_DIR,
    checkpoint_path=os.path.join(OUTPUT_DIR, "checkpoints"),
    checkpoint_save_steps=len(train_loader),  # save once per epoch
    show_progress_bar=True,
)

print("\nTraining complete.")
print(f"Model saved to {OUTPUT_DIR}")
