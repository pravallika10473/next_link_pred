#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --job-name=link_pred_eval
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/general/vast/u1475870/next_link_pred/logs/%j/%j_eval.out
#SBATCH --error=/scratch/general/vast/u1475870/next_link_pred/logs/%j/%j_eval.err
#SBATCH --mail-user=pravallikaslurm@gmail.com
#SBATCH --mail-type=END,FAIL

echo "Eval job started on $(date)"
echo "Running on node: $SLURMD_NODENAME"

SCRATCH_DIR="/scratch/general/vast/u1475870/next_link_pred"
LOG_DIR="$SCRATCH_DIR/logs/$SLURM_JOB_ID"
mkdir -p "$LOG_DIR"

cd "$SCRATCH_DIR"

export HF_HOME="$SCRATCH_DIR/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module purge
module load cuda/12.5.0
module load cudnn

# deactivate conda base so it doesn't conflict with venv
conda deactivate 2>/dev/null || true

source "$SCRATCH_DIR/venv/bin/activate"

echo "Python: $(which python)"
nvidia-smi > "$LOG_DIR/gpu_info.txt" 2>&1

# pin transformers to avoid torchvision.io dependency added in 4.47+
pip install -q "transformers==4.44.2"

# ── Eval on each saved checkpoint ─────────────────────────────────────────────
for CKPT in model/link-predictor/checkpoint-387 model/link-predictor/checkpoint-774 model/link-predictor; do
    if [ -d "$CKPT" ]; then
        echo ""
        echo "========================================="
        echo "Evaluating: $CKPT"
        echo "========================================="
        python evaluate.py --recall --model "$CKPT" --topk 10
    fi
done

deactivate
echo "Eval job ended on $(date)"
