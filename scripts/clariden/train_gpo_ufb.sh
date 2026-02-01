#!/bin/bash
# GPO Training Script for UltraFeedback with Anti-Length Augmentation
#
# Runs on Clariden cluster with 4x GH200 (97GB each)
# Uses DeepSpeed ZeRO-2 for fast weight synchronization
#
# Prerequisites:
#   Run prepare_ultrafeedback.py first to create the augmented dataset:
#   python scripts/dataset/prepare_ultrafeedback.py \
#       --output_dir $MA_SCRATCH_IOPS/argilla_ufb_pref/noise_0.500000_antilen_0.500000 \
#       --noise_ratio 0.5 --anti_length_frac 0.5

set -euxo pipefail

# === Configuration ===

# Model - expand glob pattern if needed
# SFT_MODEL_PATTERN="${SFT_MODEL:-/capstor/scratch/cscs/rosieber/MA/runs/sft/qwen3-8b-*/checkpoints}"
# # Expand glob and take the first match (or keep as-is if no glob)
# if [[ "$SFT_MODEL_PATTERN" == *"*"* ]]; then
#     SFT_MODEL=$(ls -d $SFT_MODEL_PATTERN 2>/dev/null | head -1)
#     if [ -z "$SFT_MODEL" ]; then
#         echo "ERROR: No model found matching pattern: $SFT_MODEL_PATTERN"
#         exit 1
#     fi
# else
#     SFT_MODEL="$SFT_MODEL_PATTERN"
# fi
SFT_MODEL="/capstor/scratch/cscs/rosieber/MA/runs/sft/qwen3-8b-20260119_192554_1396836/checkpoints"

# Data paths (experiment-aware directory structure)
# These should match the parameters used in prepare_ultrafeedback.py
NOISE_RATIO="${NOISE_RATIO:-1.000000}"
ANTI_LENGTH_FRAC="${ANTI_LENGTH_FRAC:-0.500000}"
DATA_DIR="${MA_SCRATCH_IOPS}/data/argilla_ufb_pref/noise_${NOISE_RATIO}_antilen_${ANTI_LENGTH_FRAC}"
TRAIN_DATA="${DATA_DIR}/train.jsonl"
VAL_DATA="${DATA_DIR}/val.jsonl"
TEST_DATA="${DATA_DIR}/test.jsonl"

# Output
DATE=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${MA_SCRATCH_CAP:-/capstor/scratch/cscs/rosieber/MA}/runs/gpo/qwen3-8b-ufb-${DATE}_${SLURM_JOB_ID:-0}"
mkdir -p "${OUTPUT_DIR}"/{checkpoints,logs}

# Batch config (4x GH200)
# Effective batch: micro * accumulated * world_size = 4 * 4 * 4 = 64
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
ACCUMULATED_GRADIENT="${ACCUMULATED_GRADIENT:-2}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-${MICRO_BATCH_SIZE} * ${ACCUMULATED_GRADIENT} * 4}"

# GPO settings
VALUE_HEAD_DIM="${VALUE_HEAD_DIM:-8}"      # 8-dim for richer preference representation
TAU="${TAU:-0.1}"                          # General preference temperature

# Training
MAX_EPOCHS="${MAX_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_LEN="${MAX_LEN:-3072}"

# Logging
SAVE_STEPS="${SAVE_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
EVAL_STEPS="${EVAL_STEPS:-50}"
WANDB_PROJECT="${WANDB_PROJECT:-GPO-UltraFeedback}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-dim${VALUE_HEAD_DIM}_lr${LEARNING_RATE}_bs${EFFECTIVE_BATCH_SIZE}_gpo_noise${NOISE_RATIO}_antilen${ANTI_LENGTH_FRAC}_${DATE}}"

# === Setup ===
echo "=== GPO Training: UltraFeedback with Anti-Length Augmentation ==="
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""
echo "Model: ${SFT_MODEL}"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Augmentation (baked into data): noise_ratio=${NOISE_RATIO}, anti_length_frac=${ANTI_LENGTH_FRAC}"

# Check data exists
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "ERROR: Training data not found: ${TRAIN_DATA}"
    echo "Run prepare_ultrafeedback.py first to create the data:"
    echo "  python scripts/dataset/prepare_ultrafeedback.py \\"
    echo "      --output_dir ${DATA_DIR} \\"
    echo "      --noise_ratio ${NOISE_RATIO} --anti_length_frac ${ANTI_LENGTH_FRAC}"
    exit 1
fi

nvidia-smi
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Memory management
export TRITON_CACHE_DIR="${OUTPUT_DIR}/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL settings for GH200 on Alps (fixes hangs)
# 1. Disable LL128 protocol - causes deadlocks on non-NVSwitch systems
export NCCL_PROTO=^LL128

# 2. Slingshot is Ethernet, not InfiniBand
export NCCL_IB_DISABLE=1
export NCCL_NET="AWS Libfabric"

# 3. GPU Direct RDMA level for Grace-Hopper
export NCCL_NET_GDR_LEVEL=PHB

# 4. Libfabric/CXI tuning
export FI_CXI_DISABLE_HOST_REGISTER=1

# 5. Debugging and error handling
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "NCCL: Configured for GH200/Alps (LL128 disabled, Libfabric enabled)"

# === Training ===
cd "${MA_HOME:-/users/rosieber/MA}/MA-GPO"

# Add project root to PYTHONPATH so general_preference module can be found
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

deepspeed --num_gpus 4 scripts/train_rm_general_preference.py \
    --pretrain "${SFT_MODEL}" \
    \
    `# === Dataset (pre-augmented UltraFeedback) ===` \
    --dataset "${TRAIN_DATA}" \
    --eval_dataset "${VAL_DATA}" \
    --use_separate_prompt \
    --max_len ${MAX_LEN} \
    \
    `# === GPO Model ===` \
    --is_general_preference \
    --general_preference_tau ${TAU} \
    --value_head_dim ${VALUE_HEAD_DIM} \
    --return_prompt_length \
    \
    `# === Batch sizes ===` \
    --micro_train_batch_size ${MICRO_BATCH_SIZE} \
    --accumulated_gradient ${ACCUMULATED_GRADIENT} \
    \
    `# === Training ===` \
    --max_epochs ${MAX_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    \
    `# === DeepSpeed ZeRO-2 ===` \
    --zero_stage 2 \
    --bf16 \
    --flash_attn \
    --gradient_checkpointing \
    \
    `# === Checkpointing ===` \
    --save_path "${OUTPUT_DIR}/checkpoints" \
    --ckpt_path "${OUTPUT_DIR}/checkpoints" \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    \
    `# === Logging ===` \
    --logging_steps ${LOGGING_STEPS} \
    --use_wandb True \
    --wandb_org rjs02-eth-z-rich \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    2>&1 | tee "${OUTPUT_DIR}/logs/training.log"

echo "=== Training Complete ==="
echo "End time: $(date)"
echo "Output: ${OUTPUT_DIR}"
