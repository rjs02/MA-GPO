#!/bin/bash
set -euxo pipefail

cd "${MA_HOME:-/users/rosieber/MA}/MA-GPO"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

RM_CHECKPOINT=/capstor/scratch/cscs/rosieber/MA/runs/borda_inflation/llama3-8b-rm-20260217_112115_1534864/model_exports/global_step_14000
RESULTS_DIR=/capstor/scratch/cscs/rosieber/MA/runs/borda_inflation/results_1539235

# Step 3: SAE analysis
python experiments/borda_inflation/analyze_sae.py \
    --rm_checkpoint "${RM_CHECKPOINT}" \
    --results_dir "${RESULTS_DIR}" \
    --layer 16 \
    --sae_width 32k

# Step 4: Visualization
python experiments/borda_inflation/visualize.py \
    --results_dir "${RESULTS_DIR}"
