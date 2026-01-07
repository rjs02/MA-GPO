#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_4090:1
#SBATCH --gres=gpumem:23872m
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=eval_gpm
#SBATCH --mail-type=BEGIN,END

# --- Environment Setup ---
module load stack/2024-06 python_cuda/3.11.6 eth_proxy

# use OpenNLHF env!
source /cluster/home/rosieber/OpenNLHF/.venv/bin/activate

# --- Directory Setup ---
SCRATCH_DIR="/cluster/scratch/rosieber/MA"
mkdir -p $SCRATCH_DIR/logs

export HF_HOME="$SCRATCH_DIR/cache/huggingface"
export TRITON_CACHE_DIR="${SCRATCH_DIR}/.triton/autotune"
mkdir -p "$TRITON_CACHE_DIR"

# --- Configuration ---

# Base model (must match training)
BASE_MODEL="Qwen/Qwen3-0.6B"

# Trained checkpoint path
# Update this to your actual checkpoint path after training
# MODEL_PATH="$SCRATCH_DIR/experiments/qwen3-0.6b-gpm-dim6-ufb"
MODEL_PATH="${1:-$SCRATCH_DIR/experiments/qwen3-0.6b-gpm-dim6-ufb}"

# Dataset to evaluate on
DATASET_PATH="${LASDIR}/data/ufb/pref_val"
# Or use training data for comparison:
# DATASET_PATH="${LASDIR}/data/ufb/pref_train"

# GPM settings (must match training!)
VALUE_HEAD_DIM=6
TAU=0.1

# Eval settings
MAX_SAMPLES=5000
MAX_LEN=2048

# --- Run Evaluation ---
echo "=============================================="
echo "GPM Full Evaluation"
echo "=============================================="
echo "Model Path: $MODEL_PATH"
echo "Base Model: $BASE_MODEL"
echo "Dataset: $DATASET_PATH"
echo "Value Head Dim: $VALUE_HEAD_DIM"
echo "Tau: $TAU"
echo "=============================================="

python scripts/evaluate_gpm_full.py \
    --model_path $MODEL_PATH \
    --base_model $BASE_MODEL \
    --dataset $DATASET_PATH \
    --value_head_dim $VALUE_HEAD_DIM \
    --tau $TAU \
    --max_samples $MAX_SAMPLES \
    --max_len $MAX_LEN \
    --bf16

echo "Evaluation finished."
