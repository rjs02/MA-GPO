#!/bin/bash
# Borda Inflation Experiment: Train RM + PM on Llama-3-8B with UltraFeedback
#
# Trains two models sequentially:
#   1. RM (Bradley-Terry, dim=1, tau=1.0) — Borda proxy
#   2. PM (GPM, dim=8, tau=0.1) — pairwise preference model
#
# Uses UltraFeedback multidimensional dataset.
# Runs on Clariden cluster with 4x GH200 (97GB each).

set -euxo pipefail

# === Configuration ===

# Base model
BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"

# Data paths — UltraFeedback multidimensional (from build_ufb_data.py)
DATA_DIR="${DATA_DIR:-${MA_SCRATCH_IOPS}/data/ufb_multidim}"
TRAIN_DATA="${DATA_DIR}/pref_train"
VAL_DATA="${DATA_DIR}/pref_val"

# Output
DATE=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="${MA_SCRATCH_CAP:-/capstor/scratch/cscs/rosieber/MA}/runs/borda_inflation"

# Which model(s) to train: "rm", "pm", or "both"
TRAIN_MODE="${TRAIN_MODE:-both}"

# Batch config (4x GH200)
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
ACCUMULATED_GRADIENT="${ACCUMULATED_GRADIENT:-2}"
EFFECTIVE_BATCH_SIZE=$((MICRO_BATCH_SIZE * ACCUMULATED_GRADIENT * 4))

# Training
MAX_EPOCHS="${MAX_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_LEN="${MAX_LEN:-3072}"

# Logging
SAVE_STEPS="${SAVE_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
EVAL_STEPS="${EVAL_STEPS:-50}"
WANDB_PROJECT="${WANDB_PROJECT:-Borda-Inflation}"

# === Setup ===
echo "=== Borda Inflation: Llama-3-8B Training ==="
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""
echo "Base model: ${BASE_MODEL}"
echo "Data: ${DATA_DIR}"
echo "Train mode: ${TRAIN_MODE}"
echo ""

# Verify data exists
if [ ! -d "${TRAIN_DATA}" ]; then
    echo "ERROR: Training data not found at ${TRAIN_DATA}"
    echo "Run build_ufb_data.py first."
    exit 1
fi

nvidia-smi
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL settings for GH200 on Alps
export NCCL_PROTO=^LL128
export NCCL_IB_DISABLE=1
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PHB
export FI_CXI_DISABLE_HOST_REGISTER=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "NCCL: Configured for GH200/Alps (LL128 disabled, Libfabric enabled)"

# Project root
cd "${MA_HOME:-/users/rosieber/MA}/MA-GPO"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# === Train function ===
train_model() {
    local dim=$1
    local tau=$2
    local is_gpo=$3
    local model_name=$4

    local OUTPUT_DIR="${BASE_OUTPUT_DIR}/llama3-8b-${model_name}-${DATE}_${SLURM_JOB_ID:-0}"
    mkdir -p "${OUTPUT_DIR}"/{checkpoints,logs}

    local TRITON_CACHE_DIR="${OUTPUT_DIR}/triton_cache"
    mkdir -p "$TRITON_CACHE_DIR"
    export TRITON_CACHE_DIR

    local WANDB_RUN_NAME="${model_name}_dim${dim}_lr${LEARNING_RATE}_bs${EFFECTIVE_BATCH_SIZE}_${DATE}"

    echo ""
    echo "========================================"
    echo "Training ${model_name} (dim=${dim}, tau=${tau})"
    echo "Output: ${OUTPUT_DIR}"
    echo "========================================"
    echo ""

    local GPO_FLAGS=""
    if [ "${is_gpo}" = "true" ]; then
        GPO_FLAGS="--is_general_preference --return_prompt_length"
    fi

    deepspeed --num_gpus 4 scripts/train_rm_general_preference.py \
        --pretrain "${BASE_MODEL}" \
        \
        --dataset "${TRAIN_DATA}" \
        --eval_dataset "${VAL_DATA}" \
        --use_separate_prompt \
        --max_len ${MAX_LEN} \
        \
        ${GPO_FLAGS} \
        --general_preference_tau ${tau} \
        --value_head_dim ${dim} \
        \
        --micro_train_batch_size ${MICRO_BATCH_SIZE} \
        --accumulated_gradient ${ACCUMULATED_GRADIENT} \
        \
        --max_epochs ${MAX_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        \
        --zero_stage 2 \
        --bf16 \
        --flash_attn \
        --gradient_checkpointing \
        \
        --save_path "${OUTPUT_DIR}/model_exports" \
        --ckpt_path "${OUTPUT_DIR}/checkpoints" \
        --save_steps ${SAVE_STEPS} \
        --eval_steps ${EVAL_STEPS} \
        \
        --logging_steps ${LOGGING_STEPS} \
        --use_wandb True \
        --wandb_org rjs02-eth-z-rich \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_run_name "${WANDB_RUN_NAME}" \
        2>&1 | tee "${OUTPUT_DIR}/logs/training.log"

    echo "Finished training ${model_name}: ${OUTPUT_DIR}"
    # Save the output path for downstream scripts
    echo "${OUTPUT_DIR}" >> "${BASE_OUTPUT_DIR}/trained_models.txt"
}

# === Run training ===
mkdir -p "${BASE_OUTPUT_DIR}"

if [ "${TRAIN_MODE}" = "rm" ] || [ "${TRAIN_MODE}" = "both" ]; then
    train_model 1 1.0 "false" "rm"
fi

if [ "${TRAIN_MODE}" = "pm" ] || [ "${TRAIN_MODE}" = "both" ]; then
    train_model 8 0.1 "true" "pm"
fi

echo ""
echo "=== All Training Complete ==="
echo "End time: $(date)"
echo "Trained models saved to: ${BASE_OUTPUT_DIR}/trained_models.txt"
