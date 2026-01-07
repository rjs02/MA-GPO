#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_4090:2
#SBATCH --gres=gpumem:23872m
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=train_gpm_grouped
#SBATCH --mail-type=BEGIN,END

# =============================================================================
# GPM Training with Grouped Data Format (Linear Complexity)
# =============================================================================
#
# This script uses the optimized grouped data format where all K responses
# per prompt are processed in ONE forward pass. This reduces computational
# complexity from O(K²) to O(K) per prompt.
#
# EFFICIENCY COMPARISON:
# - Old format: 4 responses × 4 dimensions = up to 48 pairs = 96 forward passes
# - New format: 4 responses = 4 forward passes (24x more efficient!)
#
# REQUIREMENTS:
# 1. First build grouped data:
#    python context/build_ufb_data_grouped.py --output_dir $LASDIR/data/ufb_grouped
#
# 2. Then run this training script

# --- Environment Setup ---
module load stack/2024-06 python_cuda/3.11.6 eth_proxy

# Use OpenNLHF env
source /cluster/home/rosieber/OpenNLHF/.venv/bin/activate

# --- Directory Setup ---
SCRATCH_DIR="/cluster/scratch/rosieber/MA"
mkdir -p $SCRATCH_DIR/logs
mkdir -p $SCRATCH_DIR/wandb
mkdir -p $SCRATCH_DIR/experiments
mkdir -p $SCRATCH_DIR/cache/huggingface

# Set environment variables
export HF_HOME="$SCRATCH_DIR/cache/huggingface"
export WANDB_DIR="$SCRATCH_DIR/wandb"
export WANDB_CACHE_DIR="$SCRATCH_DIR/wandb/cache"

# --- Model & Data Config ---
MODEL="Qwen/Qwen3-0.6B"

# IMPORTANT: Use grouped data format from build_ufb_data_grouped.py
DATASET_PATH="${LASDIR}/data/ufb_grouped/pref_grouped_train"
EVAL_DATASET_PATH="${LASDIR}/data/ufb_grouped/pref_grouped_val"

# GPM-specific settings
VALUE_HEAD_DIM=6        # Higher dims (6, 8) capture more complex intransitive preferences
TAU=0.1                 # Temperature for preference scaling
LR=2e-5
EPOCHS=2

# For grouped format, batch_size=1 at entry level is typical
# since each entry already contains multiple responses (e.g., 4)
# Effective batch size comes from gradient accumulation
MICRO_BATCH_SIZE=1
ACCUMULATED_GRADIENT=64  # Adjust based on memory

export EXP_NAME="qwen3-0.6b-gpm-grouped-dim${VALUE_HEAD_DIM}-ufb"
export SAVE_PATH="$SCRATCH_DIR/experiments/$EXP_NAME"

export TRITON_CACHE_DIR="${SCRATCH_DIR}/.triton/autotune"
mkdir -p "$TRITON_CACHE_DIR"

# --- Run Training ---
echo "========================================"
echo "Starting GPM Training with Grouped Data"
echo "========================================"
echo "Model: $MODEL"
echo "Value Head Dim: $VALUE_HEAD_DIM"
echo "Train Dataset: $DATASET_PATH"
echo "Eval Dataset: $EVAL_DATASET_PATH"
echo "Output Dir: $SAVE_PATH"
echo ""
echo "EFFICIENCY: Using O(K) forward passes per prompt instead of O(K²)"
echo "========================================"

# Find a free port
export MASTER_PORT=$(python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Using MASTER_PORT=$MASTER_PORT"

deepspeed --master_port $MASTER_PORT --num_gpus=2 scripts/train_rm_grouped.py \
    --pretrain $MODEL \
    --dataset $DATASET_PATH \
    --eval_dataset $EVAL_DATASET_PATH \
    --save_path $SAVE_PATH \
    --max_epochs $EPOCHS \
    --micro_train_batch_size $MICRO_BATCH_SIZE \
    --accumulated_gradient $ACCUMULATED_GRADIENT \
    --max_samples 50000 \
    --learning_rate $LR \
    --max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --flash_attn \
    --is_general_preference \
    --value_head_dim $VALUE_HEAD_DIM \
    --general_preference_tau $TAU \
    --gradient_checkpointing \
    --margin_loss \
    --use_wandb "rjs02-eth-z-rich" \
    --wandb_project "GPM-Grouped" \
    --wandb_org "rjs02-eth-z-rich" \
    --wandb_run_name "${EXP_NAME}_$(date +%Y%m%d_%H%M%S)" \
    --eval_steps 100 \
    --save_steps 500 \
    --logging_steps 10 \
    --save_best_model 3

echo "Training finished."
