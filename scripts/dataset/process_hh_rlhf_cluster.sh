#!/bin/bash
# Script to preprocess the Anthropic/hh-rlhf dataset on the cluster
# This creates training data for GPM models with Qwen3-4B tokenizer

# Use the OpenRLHF environment
source /cluster/home/rosieber/OpenNLHF/.venv/bin/activate

# Set cache directories
export HF_HOME="/cluster/scratch/rosieber/MA/cache/huggingface"
export WANDB_DIR="/cluster/scratch/rosieber/MA/wandb"

# Output directory
OUTPUT_DIR="/cluster/scratch/rosieber/MA/data/hh_rlhf"

echo "Processing Anthropic/hh-rlhf dataset"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Data will be stored as message lists (role/content dicts)."
echo "The model's tokenizer will apply the chat template at training time."
echo ""

# Run preprocessing
python scripts/dataset/preprocess_hh_rlhf.py \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Preprocessing complete!"
echo "Dataset saved to: $OUTPUT_DIR"
