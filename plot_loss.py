"""
Run after training to plot the loss curve from loss_log.csv.
Usage: python plot_loss.py
"""

import csv
import matplotlib.pyplot as plt
import os

LOG_FILE = "model/link-predictor/loss_log.csv"

epochs, steps, losses = [], [], []

with open(LOG_FILE) as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        epochs.append(int(row["epoch"]))
        steps.append(i)
        losses.append(float(row["loss"]))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(steps, losses, linewidth=1, color="#2563eb", alpha=0.8)

# Mark epoch boundaries
batches_per_epoch = max(steps) // max(epochs) if max(epochs) > 0 else 1
for e in range(1, max(epochs)):
    ax.axvline(x=e * batches_per_epoch, color="gray", linestyle="--", alpha=0.5, label=f"Epoch {e}" if e == 1 else "")

ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("Training Loss — Link Prediction Bi-Encoder")
ax.legend(["Loss", "Epoch boundary"])
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = "model/link-predictor/loss_curve.png"
plt.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")
plt.show()
