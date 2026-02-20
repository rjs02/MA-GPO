#!/bin/bash
#SBATCH --job-name=borda-eval
#SBATCH --account=a166
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --output=/users/rosieber/MA/logs/borda-eval-%j.out
#SBATCH --error=/users/rosieber/MA/logs/borda-eval-%j.err

# Borda Inflation Evaluation: prepare data, run inference, analyze
# Submit with: sbatch experiments/borda_inflation/submit_eval.sh
#
# Override checkpoints:
#   RM_CHECKPOINT=/path/to/rm PM_CHECKPOINT=/path/to/pm sbatch ...
#
# Override eval size:
#   N_EVAL_SEEN=512 N_EVAL_UNSEEN=512 sbatch ...

ulimit -c 0

echo "=== Borda Inflation Evaluation Submission ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"

mkdir -p "${MA_LOGS:-/users/rosieber/MA/logs}"
mkdir -p "${MA_SCRATCH_CAP:-/capstor/scratch/cscs/rosieber/MA}/runs/borda_inflation"

srun -ul --environment=openrlhf \
    bash "${MA_HOME:-/users/rosieber/MA}/MA-GPO/experiments/borda_inflation/run_eval.sh"

echo "End time: $(date)"
