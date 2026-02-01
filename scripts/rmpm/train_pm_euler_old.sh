#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:1
#SBATCH --gres=gpumem:23000m
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=8192
#SBATCH --job-name=train_gpo
#SBATCH --mail-type=BEGIN,END

set -x

# --- Environment Setup ---
module load stack/2024-06 python_cuda/3.11.6 eth_proxy
# Prevent Euler proxy from intercepting local Ray traffic
export no_proxy="localhost,127.0.0.1,$(hostname -i)"
unset http_proxy https_proxy

if [ ! -d ".venv" ]; then
    uv venv
    uv pip install -e .
fi

# use OpenNLHF env!
source /cluster/home/rosieber/OpenNLHF/.venv/bin/activate

nvidia-smi

# --- Paths ---
SCRATCH_DIR="/cluster/scratch/rosieber/MA"
LAS_DIR="/cluster/project/infk/krause/rosieber"

POLICY_MODEL="qwen3-1.7b-sft-20251228_111716"
# POLICY_MODEL="Qwen/Qwen3-1.7B"
# PREF_MODEL="PM_UFB_TRAIN_20251223_101131"
PREF_MODEL="llm-blender/PairRM"
# PREF_MODEL="Skywork/Skywork-Reward-V2-Qwen3-0.6B"

# Sanitize model names for use in paths (replace / with _)
POLICY_MODEL_PATH="${POLICY_MODEL//\//_}"
PREF_MODEL_PATH="${PREF_MODEL//\//_}"

PROMPT_DATA="${LAS_DIR}/data/ufb/sft_train_256"

DATE=$(date +%Y%m%d_%H%M%S)

# Configs for Qwen3-1.7B
# thinking: T=0.6, Top-P=0.95, Top-K=20, Min-P=0
# no thinking: T=0.7, Top-P=0.8, Top-K=20, Min-P=0
# NashMP: T=1.0 for higher diversity for REINFORCE gradient estimator


# --- Training ---
  deepspeed --num_gpus 1 scripts/train_rm_general_preference.py \
      --pretrain Qwen/Qwen3-0.6B \
      --dataset ${LAS_DIR}/data/ufb/pref_train \
      --save_path ./checkpoints/gpm-qwen3-0.6b \
      --value_head_dim 6 \
      --is_general_preference \
      --general_preference_tau 0.1 \
      --use_separate_prompt \
      --bf16 \
      --max_len 2048
