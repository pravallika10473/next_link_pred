#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --job-name=link_pred_train
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/general/vast/u1475870/next_link_pred/logs/%j/%j_train.out
#SBATCH --error=/scratch/general/vast/u1475870/next_link_pred/logs/%j/%j_train.err
#SBATCH --mail-user=pravallikaslurm@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --requeue
#SBATCH --open-mode=append

echo "Job started on $(date)"
echo "Running on node: $SLURMD_NODENAME"

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH_DIR="/scratch/general/vast/u1475870/next_link_pred"
LOG_DIR="$SCRATCH_DIR/logs/$SLURM_JOB_ID"
mkdir -p "$LOG_DIR"

# Code lives directly in scratch — no copy needed
cd "$SCRATCH_DIR"
echo "Working directory: $(pwd)"
echo "Contents:"
ls -l

# ── Cache ─────────────────────────────────────────────────────────────────────
export HF_HOME="$SCRATCH_DIR/hf_cache"
mkdir -p "$HF_HOME"

# ── Modules ───────────────────────────────────────────────────────────────────
module purge
module load cuda/12.5.0
module load cudnn

# ── Environment ───────────────────────────────────────────────────────────────
source /scratch/general/vast/u1475870/next_link_pred/venv/bin/activate

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# ensure torchvision is installed
pip install -q torchvision --index-url https://download.pytorch.org/whl/cu121

# ── GPU info ──────────────────────────────────────────────────────────────────
nvidia-smi > "$LOG_DIR/gpu_info.txt" 2>&1
echo "GPU info saved to $LOG_DIR/gpu_info.txt"

# ── Train ─────────────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Starting link prediction training..."
PYTHONPATH=. python train.py 2>&1 | tee "$LOG_DIR/train_output.txt"

if [ $? -eq 0 ]; then
    echo "Training completed successfully"
    echo "Plotting loss curve..."
    python plot_loss.py --log "$LOG_DIR/${SLURM_JOB_ID}_train.out" --out model/link-predictor/loss_curve.png
else
    echo "Training failed — check $LOG_DIR/train_output.txt"
fi

deactivate
echo "Job ended on $(date)"
