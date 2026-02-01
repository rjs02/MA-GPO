#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_4090:4
#SBATCH --gres=gpumem:23872m
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16384
#SBATCH --job-name=train_gpm_hh_4b_4gpu_offload
#SBATCH --mail-type=BEGIN,END

# Training script for GPM on Anthropic/hh-rlhf dataset
# Using 4x RTX 4090 GPUs with DeepSpeed ZeRO-3 + OPTIMIZER OFFLOADING
# Effective batch size: 128
# Optimizer offloading: Reduces GPU memory by ~8GB per GPU, adds ~15-20% training time


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
MODEL="Qwen/Qwen3-4B"
# Preprocessed hh-rlhf dataset (stored as message lists)
# Chat template is applied at runtime by the model's tokenizer
# IMPORTANT: Point to the specific split directories, not the parent!
DATASET_PATH="$LASDIR/data/hh_rlhf/train"
EVAL_DATASET_PATH="$LASDIR/data/hh_rlhf/test"

# GPM-specific settings
VALUE_HEAD_DIM=${VALUE_HEAD_DIM:-6}        # Higher dims (6, 8) capture more complex intransitive preferences
TAU=${TAU:-0.1}                 # Temperature for preference scaling
LR=${LR:-1e-5}
EPOCHS=${EPOCHS:-2}
NUM_GPUS=${NUM_GPUS:-4}              # Number of GPUs
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}      # Per-GPU batch size
ACCUMULATED_GRADIENT=${ACCUMULATED_GRADIENT:-2} # Gradient accumulation steps
# Effective batch size = MICRO_BATCH_SIZE * ACCUMULATED_GRADIENT * NUM_GPUS = 2 * 16 * 4 = 128

DATE=$(date +%Y%m%d_%H%M%S)

EXP_NAME=${EXP_NAME:-"qwen3-4b-gpm-dim${VALUE_HEAD_DIM}-hh-rlhf-4gpu-offload-${DATE}"}
export SAVE_PATH="$SCRATCH_DIR/experiments/$EXP_NAME"

export TRITON_CACHE_DIR="${SCRATCH_DIR}/.triton/autotune"
mkdir -p "$TRITON_CACHE_DIR"

# --- Run Training ---
echo "Starting GPM training on hh-rlhf dataset..."
echo "Model: $MODEL"
echo "Value Head Dim: $VALUE_HEAD_DIM"
echo "Dataset: $DATASET_PATH"
echo "Output Dir: $SAVE_PATH"
echo "Number of GPUs: $NUM_GPUS"
echo "Micro Batch Size: $MICRO_BATCH_SIZE"
echo "Gradient Accumulation: $ACCUMULATED_GRADIENT"
echo "Effective Batch Size: $(($MICRO_BATCH_SIZE * $ACCUMULATED_GRADIENT * $NUM_GPUS))"

# Find a free port
export MASTER_PORT=$(python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Using MASTER_PORT=$MASTER_PORT"

deepspeed --master_port $MASTER_PORT --num_gpus=$NUM_GPUS scripts/train_rm_general_preference.py \
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
    --zero_stage 3 \
    --bf16 \
    --flash_attn \
    --adam_offload \
    --value_head_dim $VALUE_HEAD_DIM \
    --general_preference_tau $TAU \
    --use_separate_prompt \
    --gradient_checkpointing \
    --use_wandb "rjs02-eth-z-rich" \
    --wandb_project "GPO PM" \
    --wandb_org "rjs02-eth-z-rich" \
    --wandb_run_name "${EXP_NAME}" \
    --eval_steps 25 \
    --save_steps 500 \
    --logging_steps 5

echo "Training finished."
