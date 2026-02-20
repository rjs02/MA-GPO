#!/bin/bash
set -euxo pipefail

cd "${MA_HOME:-/users/rosieber/MA}/MA-GPO"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
pip install matplotlib pandas scipy statsmodels

RESULTS_DIR="${MA_SCRATCH_CAP:-/capstor/scratch/cscs/rosieber/MA}/runs/borda_inflation/tabular_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

python experiments/borda_inflation/tabular_inflation.py \
    --output_dir "${RESULTS_DIR}" \
    --n_workers 64 \
    2>&1 | tee "${RESULTS_DIR}/tabular.log"

echo "Results: ${RESULTS_DIR}"
