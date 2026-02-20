#!/bin/bash
#SBATCH --job-name=borda-analysis
#SBATCH --account=a166
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --output=/users/rosieber/MA/logs/borda-analysis-%j.out
#SBATCH --error=/users/rosieber/MA/logs/borda-analysis-%j.err

# Borda Inflation Experiment: Analysis Pipeline (Steps 1-4)
# Runs after training is complete.
#
# Usage:
#   sbatch experiments/borda_inflation/submit_analysis.sh
#
# Override checkpoints:
#   RM_CHECKPOINT=/path/to/rm PM_CHECKPOINT=/path/to/pm sbatch experiments/borda_inflation/submit_analysis.sh

ulimit -c 0

echo "=== Borda Inflation Analysis Submission ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"

mkdir -p "${MA_LOGS:-/users/rosieber/MA/logs}"

srun -ul --environment=openrlhf \
    bash "${MA_HOME:-/users/rosieber/MA}/MA-GPO/experiments/borda_inflation/run_analysis.sh"

echo "End time: $(date)"
