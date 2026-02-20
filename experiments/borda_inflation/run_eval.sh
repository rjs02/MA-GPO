#!/bin/bash
# Borda Inflation Experiment: Evaluation Pipeline
# Called by submit_eval.sh via srun --environment=openrlhf.
#
# Pipeline:
#   1. prepare_eval_data.py  — one-time, creates fixed eval splits (CPU)
#   2. run_inference.py      — GPU inference on eval_seen and eval_unseen
#   3. analyze_inflation.py  — social choice metrics + figures (CPU)
#
# Can also run interactively:
#   bash experiments/borda_inflation/run_eval.sh

set -euxo pipefail

# === Configuration ===
BASE_DIR="${MA_SCRATCH_CAP:-/capstor/scratch/cscs/rosieber/MA}/runs/borda_inflation"

# Model checkpoints
if [ -z "${RM_CHECKPOINT:-}" ] || [ -z "${PM_CHECKPOINT:-}" ]; then
    if [ -f "${BASE_DIR}/rm_checkpoint.txt" ] && [ -f "${BASE_DIR}/pm_checkpoint.txt" ]; then
        echo "Reading model paths from checkpoint files"
        RM_DIR=$(cat "${BASE_DIR}/rm_checkpoint.txt")
        PM_DIR=$(cat "${BASE_DIR}/pm_checkpoint.txt")
        RM_CHECKPOINT=$(ls -d "${RM_DIR}"/model_exports/global_step_* 2>/dev/null | tail -1)
        PM_CHECKPOINT=$(ls -d "${PM_DIR}"/model_exports/global_step_* 2>/dev/null | tail -1)
    else
        echo "ERROR: No trained models found."
        echo "Set RM_CHECKPOINT and PM_CHECKPOINT env vars, or run training first."
        exit 1
    fi
fi

echo "RM checkpoint: ${RM_CHECKPOINT}"
echo "PM checkpoint: ${PM_CHECKPOINT}"

DATA_DIR="${DATA_DIR:-${MA_SCRATCH_IOPS}/data/ufb_multidim}"
RESULTS_DIR="${BASE_DIR}/eval_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
EVAL_DATA_DIR="${RESULTS_DIR}/eval_data"
N_EVAL_SEEN="${N_EVAL_SEEN:-1024}"
N_EVAL_UNSEEN="${N_EVAL_UNSEEN:-1024}"

mkdir -p "${RESULTS_DIR}"

# === Setup ===
echo "=== Borda Inflation Evaluation Pipeline ==="
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Results: ${RESULTS_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_PROTO=^LL128
export NCCL_IB_DISABLE=1
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PHB
export FI_CXI_DISABLE_HOST_REGISTER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "${MA_HOME:-/users/rosieber/MA}/MA-GPO"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
pip install matplotlib scipy
nvidia-smi

# ============================================================
# Step 1: Prepare eval data (one-time, CPU-only)
# ============================================================
echo ""
echo "========================================="
echo "STEP 1: Preparing eval data"
echo "========================================="
python experiments/borda_inflation/prepare_eval_data.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${EVAL_DATA_DIR}" \
    --n_eval_seen "${N_EVAL_SEEN}" \
    --n_eval_unseen "${N_EVAL_UNSEEN}" \
    2>&1 | tee "${RESULTS_DIR}/step1_prepare.log"

# ============================================================
# Step 2: GPU inference on eval_seen
# ============================================================
echo ""
echo "========================================="
echo "STEP 2a: GPU inference on eval_seen"
echo "========================================="
python experiments/borda_inflation/run_inference.py \
    --rm_checkpoint "${RM_CHECKPOINT}" \
    --pm_checkpoint "${PM_CHECKPOINT}" \
    --eval_data "${EVAL_DATA_DIR}/eval_seen.json" \
    --output_path "${RESULTS_DIR}/inference_seen.pkl" \
    2>&1 | tee "${RESULTS_DIR}/step2a_inference_seen.log"

echo ""
echo "========================================="
echo "STEP 2b: GPU inference on eval_unseen"
echo "========================================="
python experiments/borda_inflation/run_inference.py \
    --rm_checkpoint "${RM_CHECKPOINT}" \
    --pm_checkpoint "${PM_CHECKPOINT}" \
    --eval_data "${EVAL_DATA_DIR}/eval_unseen.json" \
    --output_path "${RESULTS_DIR}/inference_unseen.pkl" \
    2>&1 | tee "${RESULTS_DIR}/step2b_inference_unseen.log"

# ============================================================
# Step 3: Analysis (CPU, can also be run locally)
# ============================================================
echo ""
echo "========================================="
echo "STEP 3a: Analysis on eval_seen"
echo "========================================="
python experiments/borda_inflation/analyze_inflation.py \
    --eval_data "${EVAL_DATA_DIR}/eval_seen.json" \
    --inference_data "${RESULTS_DIR}/inference_seen.pkl" \
    --output_dir "${RESULTS_DIR}/analysis_seen" \
    2>&1 | tee "${RESULTS_DIR}/step3a_analysis_seen.log"

echo ""
echo "========================================="
echo "STEP 3b: Analysis on eval_unseen"
echo "========================================="
python experiments/borda_inflation/analyze_inflation.py \
    --eval_data "${EVAL_DATA_DIR}/eval_unseen.json" \
    --inference_data "${RESULTS_DIR}/inference_unseen.pkl" \
    --output_dir "${RESULTS_DIR}/analysis_unseen" \
    2>&1 | tee "${RESULTS_DIR}/step3b_analysis_unseen.log"

# ============================================================
# Done
# ============================================================
echo ""
echo "=== Evaluation Pipeline Complete ==="
echo "End time: $(date)"
echo "Results: ${RESULTS_DIR}"
echo ""
echo "Key outputs:"
echo "  - ${EVAL_DATA_DIR}/eval_seen.json"
echo "  - ${EVAL_DATA_DIR}/eval_unseen.json"
echo "  - ${RESULTS_DIR}/inference_seen.pkl (for local re-analysis)"
echo "  - ${RESULTS_DIR}/inference_unseen.pkl (for local re-analysis)"
echo "  - ${RESULTS_DIR}/analysis_seen/summary.json"
echo "  - ${RESULTS_DIR}/analysis_unseen/summary.json"
