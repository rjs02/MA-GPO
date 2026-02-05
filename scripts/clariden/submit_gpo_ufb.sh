#!/bin/bash
#SBATCH --job-name=gpo-ufb
#SBATCH --account=a166
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --output=/users/rosieber/MA/logs/gpo-ufb-%j.out
#SBATCH --error=/users/rosieber/MA/logs/gpo-ufb-%j.err

# GPO Training on UltraFeedback with Anti-Length Augmentation
# Submit with: sbatch scripts/clariden/submit_gpo_ufb.sh
#
# Override defaults with environment variables:
#   NOISE_RATIO=0.3 ANTI_LENGTH_FRAC=0.7 sbatch scripts/clariden/submit_gpo_ufb.sh

# Disable core dumps
ulimit -c 0

echo "=== GPO UltraFeedback Training Submission ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"

# Create log directories
mkdir -p "${MA_LOGS:-/users/rosieber/MA/logs}"
mkdir -p "${MA_SCRATCH_CAP:-/capstor/scratch/cscs/rosieber/MA}/runs/gpo"

# Run training
srun -ul --environment=openrlhf \
    bash "${MA_HOME:-/users/rosieber/MA}/MA-GPO/scripts/clariden/train_gpo_ufb-multidim.sh"

echo "End time: $(date)"
