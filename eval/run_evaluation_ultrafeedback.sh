#!/bin/bash
#SBATCH --job-name=eval-gpo-rm
#SBATCH --account=a166
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --output=/users/rosieber/MA/logs/eval-gpo-rm-%j.out
#SBATCH --error=/users/rosieber/MA/logs/eval-gpo-rm-%j.err

# Evaluation: GPO vs RM on UltraFeedback
# Submit with: sbatch eval/run_evaluation_ultrafeedback.sh

# Disable core dumps
ulimit -c 0

echo "=== GPO vs RM Evaluation Submission ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"

# Configuration
RM_CHECKPOINT="/capstor/scratch/cscs/rosieber/MA/runs/gpo/qwen3-8b-ufbmultidim-20260203_141509_1482607/model_exports/global_step_11400"
GPO_CHECKPOINT="/capstor/scratch/cscs/rosieber/MA/runs/gpo/qwen3-8b-ufbmultidim-20260203_131559_1482403/model_exports/global_step_11500"
DATASET="/capstor/scratch/cscs/rosieber/MA/data/ultrafeedback_multidim/test"  # Adjust path
OUTPUT_DIR="/capstor/scratch/cscs/rosieber/MA/eval_results/$(date +%Y%m%d_%H%M%S)"

# Create directories
mkdir -p "${MA_LOGS:-/users/rosieber/MA/logs}"
mkdir -p "$(dirname $OUTPUT_DIR)"

# Run evaluation
echo "======================================================================"
echo "Evaluating GPO vs RM on UltraFeedback"
echo "======================================================================"
echo "RM checkpoint:  $RM_CHECKPOINT"
echo "GPO checkpoint: $GPO_CHECKPOINT"
echo "Dataset:        $DATASET"
echo "Output:         $OUTPUT_DIR"
echo ""

cd "${MA_HOME:-/users/rosieber/MA}/MA-GPO"

srun -ul python eval/evaluate_trained_models_ultrafeedback.py \
    --rm_checkpoint "$RM_CHECKPOINT" \
    --gpo_checkpoint "$GPO_CHECKPOINT" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --split test \
    --max_prompts 500 \
    --max_responses 50

echo ""
echo "======================================================================"
echo "Evaluation complete! Results saved to:"
echo "  $OUTPUT_DIR"
echo ""
echo "Quick summary:"
python eval/inspect_results.py "$OUTPUT_DIR/evaluation_results.json"
echo "======================================================================"
echo "End time: $(date)"
