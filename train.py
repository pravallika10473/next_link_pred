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

# On CHPC A100 you can push this to 256 — more negatives per batch = better signal
BATCH_SIZE = 256

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
from tqdm import tqdm

OUTPUT_DIR = "model/link-predictor"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nStarting training...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in loop:
        # Step A — pull queries and docs out of the batch
        queries   = [ex.texts[0] for ex in batch]
        docs      = [ex.texts[1] for ex in batch]

        # Step B — tokenize: text → token IDs + attention masks
        query_features = model.tokenize(queries)
        doc_features   = model.tokenize(docs)

        # move only tensors to device (newer sentence-transformers includes non-tensor values)
        query_features = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in query_features.items()}
        doc_features   = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in doc_features.items()}

        # Step C — forward pass through loss
        # internally: encode both → build [B,B] similarity matrix → cross-entropy
        loss_value = train_loss([query_features, doc_features], labels=None)

        # Step D — backward pass: compute gradients for all 110M weights
        loss_value.backward()

        # Step E — gradient clipping: cap gradient norm at 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Step F — optimizer step: update weights using gradients
        optimizer.step()

        # Step G — scheduler step: adjust learning rate
        scheduler.step()

        # Step H — zero gradients so they don't accumulate into next batch
        optimizer.zero_grad()

        total_loss += loss_value.item()
        loop.set_postfix(
            loss=f"{loss_value.item():.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}"
        )

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # Save checkpoint after each epoch
    ckpt_path = os.path.join(OUTPUT_DIR, f"epoch-{epoch+1}")
    model.save(ckpt_path)
    print(f"Checkpoint saved → {ckpt_path}")

print("\nTraining complete.")
