#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_4090:2
#SBATCH --gres=gpumem:23872m
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=train_gpm
#SBATCH --mail-type=BEGIN,END

# TODO for scaling up: no max samples, longer max len!, larger model


# --- Environment Setup ---
module load stack/2024-06 python_cuda/3.11.6 eth_proxy

# use OpenNLHF env!
source /cluster/home/rosieber/OpenNLHF/.venv/bin/activate

# --- Directory Setup ---
SCRATCH_DIR="/cluster/scratch/rosieber/MA"
mkdir -p $SCRATCH_DIR/logs
mkdir -p $SCRATCH_DIR/wandb
mkdir -p $SCRATCH_DIR/experiments
mkdir -p $SCRATCH_DIR/cache/huggingface

# Set environment variables for cache and output
export HF_HOME="$SCRATCH_DIR/cache/huggingface"
# export HF_TOKEN=$(cat $HOME/.cache/huggingface/token)
export WANDB_DIR="$SCRATCH_DIR/wandb"
export WANDB_CACHE_DIR="$SCRATCH_DIR/wandb/cache"

# --- Model & Data Config ---
MODEL="Qwen/Qwen3-0.6B"
# Dataset should be in HuggingFace datasets format or JSONL
# With fields: prompt, chosen, rejected, margin (optional)
# DATASET_PATH="/cluster/home/rosieber/OpenNLHF/data/cyclic_triplets_v2/Cyclic_1"
# DATASET_PATH="${LASDIR}/data/ufb/pref_averaged_train"
# EVAL_DATASET_PATH="${LASDIR}/data/ufb/pref_averaged_val"
DATASET_PATH="${LASDIR}/data/rlhflow_preference"
EVAL_DATASET_PATH="${LASDIR}/data/rlhflow_preference"


# GPM-specific settings
VALUE_HEAD_DIM=6        # Higher dims (6, 8) capture more complex intransitive preferences
TAU=0.1                 # Temperature for preference scaling
LR=1e-6
EPOCHS=1
MICRO_BATCH_SIZE=4      # Per-GPU batch size
ACCUMULATED_GRADIENT=8  # Gradient accumulation steps
# Effective batch size = MICRO_BATCH_SIZE * ACCUMULATED_GRADIENT * num_gpus = 8 * 8 * 1 = 64

DATE=$(date +%Y%m%d_%H%M%S)

export EXP_NAME="qwen3-0.6b-gpm-dim${VALUE_HEAD_DIM}-rlhflow-${DATE}"
export SAVE_PATH="$SCRATCH_DIR/experiments/$EXP_NAME"

export TRITON_CACHE_DIR="${SCRATCH_DIR}/.triton/autotune"
mkdir -p "$TRITON_CACHE_DIR"

# --- Run Training ---
echo "Starting GPM training..."
echo "Model: $MODEL"
echo "Value Head Dim: $VALUE_HEAD_DIM"
echo "Output Dir: $SAVE_PATH"

# Find a free port
export MASTER_PORT=$(python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Using MASTER_PORT=$MASTER_PORT"

deepspeed --master_port $MASTER_PORT --num_gpus=2 scripts/train_rm_general_preference.py \
    --pretrain $MODEL \
    --dataset $DATASET_PATH \
    --eval_dataset $EVAL_DATASET_PATH \
    --max_eval_samples 128 \
    --save_path $SAVE_PATH \
    --max_epochs $EPOCHS \
    --micro_train_batch_size $MICRO_BATCH_SIZE \
    --accumulated_gradient $ACCUMULATED_GRADIENT \
    --learning_rate $LR \
    --max_len 2048 \
    --is_general_preference \
    --zero_stage 2 \
    --bf16 \
    --flash_attn \
    --value_head_dim $VALUE_HEAD_DIM \
    --general_preference_tau $TAU \
    --use_separate_prompt \
    --gradient_checkpointing \
    --use_wandb "rjs02-eth-z-rich" \
    --wandb_project "GPO PM" \
    --wandb_org "rjs02-eth-z-rich" \
    --wandb_run_name "${EXP_NAME}" \
    --eval_steps 50 \
    --save_steps 500 \
    --logging_steps 10

echo "Training finished."
