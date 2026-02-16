#!/bin/bash
#SBATCH --job-name=borda-analysis
#SBATCH --account=a166
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --output=/users/rosieber/MA/logs/borda-analysis-%j.out
#SBATCH --error=/users/rosieber/MA/logs/borda-analysis-%j.err

# Borda Inflation Experiment: Analysis Pipeline (Steps 1-5)
# Runs after training is complete.
#
# Usage:
#   sbatch experiments/borda_inflation/run_analysis.sh
#
# Requires: RM_CHECKPOINT and PM_CHECKPOINT environment variables pointing to
# trained model directories. These are written to trained_models.txt by
# train_llama_ufb.sh.

set -euxo pipefail

# === Configuration ===
BASE_DIR="${MA_SCRATCH_CAP:-/capstor/scratch/cscs/rosieber/MA}/runs/borda_inflation"

# Model checkpoints — override with env vars or read from trained_models.txt
if [ -z "${RM_CHECKPOINT:-}" ] || [ -z "${PM_CHECKPOINT:-}" ]; then
    if [ -f "${BASE_DIR}/trained_models.txt" ]; then
        echo "Reading model paths from ${BASE_DIR}/trained_models.txt"
        # First line = RM, second line = PM
        RM_CHECKPOINT=$(sed -n '1p' "${BASE_DIR}/trained_models.txt")
        PM_CHECKPOINT=$(sed -n '2p' "${BASE_DIR}/trained_models.txt")
        # Find the exported model checkpoint within each directory
        RM_CHECKPOINT=$(ls -d "${RM_CHECKPOINT}"/checkpoints/global_step_* 2>/dev/null | tail -1)
        PM_CHECKPOINT=$(ls -d "${PM_CHECKPOINT}"/checkpoints/global_step_* 2>/dev/null | tail -1)
    else
        echo "ERROR: No trained models found. Set RM_CHECKPOINT and PM_CHECKPOINT or run training first."
        exit 1
    fi
fi

echo "RM checkpoint: ${RM_CHECKPOINT}"
echo "PM checkpoint: ${PM_CHECKPOINT}"

# Data and output
DATA_DIR="${DATA_DIR:-${MA_SCRATCH_IOPS}/data/ufb_multidim}"
RESULTS_DIR="${BASE_DIR}/results_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RESULTS_DIR}"

# SAE config
SAE_LAYER="${SAE_LAYER:-16}"
SAE_WIDTH="${SAE_WIDTH:-32k}"

# === Setup ===
echo "=== Borda Inflation Analysis Pipeline ==="
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Results: ${RESULTS_DIR}"

# Memory and NCCL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_PROTO=^LL128
export NCCL_IB_DISABLE=1
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PHB
export FI_CXI_DISABLE_HOST_REGISTER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "${MA_HOME:-/users/rosieber/MA}/MA-GPO"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

nvidia-smi

# ============================================================
# Step 1: Compute inflation scores
# ============================================================
echo ""
echo "========================================="
echo "STEP 1: Computing Borda inflation scores"
echo "========================================="
python experiments/borda_inflation/compute_inflation.py \
    --rm_checkpoint "${RM_CHECKPOINT}" \
    --pm_checkpoint "${PM_CHECKPOINT}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${RESULTS_DIR}" \
    2>&1 | tee "${RESULTS_DIR}/step1_compute.log"

# ============================================================
# Step 2: Dimension analysis (Track A)
# ============================================================
echo ""
echo "========================================="
echo "STEP 2: Dimension analysis (Track A)"
echo "========================================="
python experiments/borda_inflation/analyze_dimensions.py \
    --results_dir "${RESULTS_DIR}" \
    2>&1 | tee "${RESULTS_DIR}/step2_dimensions.log"

# ============================================================
# Step 3: SAE analysis (Track B)
# ============================================================
echo ""
echo "========================================="
echo "STEP 3: SAE analysis (Track B)"
echo "========================================="
python experiments/borda_inflation/analyze_sae.py \
    --rm_checkpoint "${RM_CHECKPOINT}" \
    --results_dir "${RESULTS_DIR}" \
    --layer ${SAE_LAYER} \
    --sae_width ${SAE_WIDTH} \
    2>&1 | tee "${RESULTS_DIR}/step3_sae.log"

# ============================================================
# Step 4: Visualization
# ============================================================
echo ""
echo "========================================="
echo "STEP 4: Generating visualizations"
echo "========================================="
python experiments/borda_inflation/visualize.py \
    --results_dir "${RESULTS_DIR}" \
    2>&1 | tee "${RESULTS_DIR}/step4_visualize.log"

# ============================================================
# Done
# ============================================================
echo ""
echo "=== Analysis Pipeline Complete ==="
echo "End time: $(date)"
echo "Results: ${RESULTS_DIR}"
echo ""
echo "Key outputs:"
echo "  - ${RESULTS_DIR}/inflation_summary.json"
echo "  - ${RESULTS_DIR}/analysis/dimension_analysis.json"
echo "  - ${RESULTS_DIR}/sae_analysis/sae_summary.json"
echo "  - ${RESULTS_DIR}/figures/"
