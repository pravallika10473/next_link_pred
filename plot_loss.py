"""
Parse loss from SLURM .out log and plot the curve.

Usage:
  python plot_loss.py --log logs/<job_id>/<job_id>_train.out
"""

import re
import ast
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--log", required=True, help="Path to SLURM .out log file")
parser.add_argument("--out", default="model/link-predictor/loss_curve.png")
args = parser.parse_args()

# Each logged step looks like:
# {'loss': '0.1778', 'grad_norm': '1.563', 'learning_rate': '1.265e-05', 'epoch': '1.292'}
pattern = re.compile(r"\{'loss':.*?'epoch':.*?\}")

losses = []
epochs = []

with open(args.log) as f:
    for line in f:
        match = pattern.search(line)
        if match:
            try:
                d = ast.literal_eval(match.group())
                losses.append(float(d["loss"]))
                epochs.append(float(d["epoch"]))
            except Exception:
                continue

if not losses:
    print("No loss entries found in log. Check the log path.")
    exit(1)

print(f"Found {len(losses)} loss entries")
print(f"  First loss: {losses[0]:.4f}  (epoch {epochs[0]:.2f})")
print(f"  Final loss: {losses[-1]:.4f}  (epoch {epochs[-1]:.2f})")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(epochs, losses, linewidth=1.2, color="#2563eb", alpha=0.85)

# Mark epoch boundaries
for e in range(1, int(max(epochs)) + 1):
    ax.axvline(x=e, color="gray", linestyle="--", alpha=0.4)
    ax.text(e + 0.02, max(losses) * 0.95, f"Epoch {e}", fontsize=8, color="gray")

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Loss — Link Prediction Bi-Encoder")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"Saved → {args.out}")
