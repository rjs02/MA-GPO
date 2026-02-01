# Anthropic/hh-rlhf Dataset Preprocessing for GPM Training

## Overview

This directory contains scripts to preprocess the [Anthropic/hh-rlhf dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) for training General Preference Models (GPM) with Qwen3-4B as the base model.

## Dataset Structure

### Multi-turn Conversations ✓

The preprocessing script **correctly handles multi-turn conversations**. Analysis of 1,000 examples shows:

- **29.4%** single-turn (1 user message → 1 assistant response)
- **25.8%** two-turn (2 exchanges between user and assistant)
- **24.0%** three-turn conversations
- **~20%** four or more turns

**Key property**: In all examples, the `chosen` and `rejected` versions share the **exact same conversation history** up to the final assistant response. Only the last assistant message differs.

### Available Subsets

The dataset contains several subsets (via `data_dir` parameter):

| Subset | Description | Use Case |
|--------|-------------|----------|
| **helpful-base** | Helpfulness preferences from base models (52B context-distilled) | General helpfulness training |
| **helpful-rejection-sampled** | Generated via rejection sampling (best-of-16) | Higher quality responses |
| **helpful-online** | Sampled during iterated online training | Advanced training stage |
| **harmless-base** | Harmlessness preferences from base models | Safety/alignment training |
| **red-team-attempts** | Adversarial red teaming dialogues | Robustness testing |
| **default** (no data_dir) | All helpful + harmless subsets combined | Comprehensive training |

### Dataset Sizes

When loading all subsets (default):
- **Train split**: 160,800 examples
- **Test split**: 8,552 examples

### Which Subsets to Use?

For GPM training, we recommend:

1. **Start with `default`** (all subsets): Best for general-purpose preference models
2. **helpful-rejection-sampled**: Use if you want higher-quality helpful responses
3. **harmless-base**: Add if you want to emphasize safety/harmlessness

You **do NOT need to process all subsets separately**. The default setting combines helpful and harmless data, which is ideal for most use cases.

## Preprocessing

### What the Script Does

1. **Loads** the raw hh-rlhf dataset from HuggingFace
2. **Parses** conversation format (`\n\nHuman:` and `\n\nAssistant:` markers)
3. **Converts** to message lists with `role` and `content` keys (same format as ultrafeedback)
4. **Creates** four fields per example:
   - `prompt`: List of messages (conversation history up to final response)
   - `chosen`: List with one message (the preferred final assistant response)
   - `rejected`: List with one message (the less-preferred final assistant response)
   - `margin`: Score margin (always 0 for hh-rlhf, as no explicit margins are provided)
5. **Saves** in HuggingFace datasets format for efficient loading during training

**Important**: The chat template is **NOT** applied during preprocessing. It's applied at **training time** by the model's tokenizer, allowing the same preprocessed data to work with different models.

### Usage

#### On Your Laptop (Small Sample)

```bash
# Test with 10 examples
/home/robin/repos/OpenRLHF/.venv/bin/python scripts/dataset/preprocess_hh_rlhf.py \
    --output_dir "./data/hh_rlhf_test" \
    --max_samples 10
```

#### On the Cluster (Full Dataset)

```bash
# Process all ~160k training examples
bash scripts/dataset/process_hh_rlhf_cluster.sh
```

Or run preprocessing directly:

```bash
source /cluster/home/rosieber/OpenNLHF/.venv/bin/activate

python scripts/dataset/preprocess_hh_rlhf.py \
    --output_dir ${LASDIR}/data/hh_rlhf
```

#### Process Specific Subset

```bash
# Process only helpful-rejection-sampled subset
python scripts/dataset/preprocess_hh_rlhf.py \
    --output_dir "./data/hh_rlhf_helpful_rs" \
    --subset "helpful-rejection-sampled"
```

### Output Format Example

```json
{
  "prompt": [
    {
      "role": "user",
      "content": "What are some cuss words in english?"
    },
    {
      "role": "assistant",
      "content": "Here's an incomplete list.\n\nAss, dick, bugger, crap, fuck, shit, ..."
    },
    {
      "role": "user",
      "content": "What's your favorite one?"
    }
  ],
  "chosen": [
    {
      "role": "assistant",
      "content": "I haven't even thought about it."
    }
  ],
  "rejected": [
    {
      "role": "assistant",
      "content": "Ass."
    }
  ],
  "margin": 0
}
```

Notice how:
- Data is stored as **message lists** with `role` and `content` keys
- The conversation history is identical in prompt
- Only the final assistant response differs (chosen is safer/more helpful)
- Same format as ultrafeedback dataset
- The chat template will be applied at training time by the tokenizer

## Training

Use the preprocessed data with your GPM training script:

```bash
# Submit training job on cluster
sbatch scripts/rmpm/train_pm_hh_rlhf.sh
```

The training script uses the `--use_separate_prompt` flag to indicate that the data has separate `prompt`, `chosen`, and `rejected` fields (as opposed to the combined format).

## Scripts

- **`preprocess_hh_rlhf.py`**: Main preprocessing script
- **`process_hh_rlhf_cluster.sh`**: Cluster job for preprocessing full dataset
- **`train_pm_hh_rlhf.sh`**: Training script for GPM on preprocessed hh-rlhf data
- **`explore_hh_rlhf.py`**: Quick exploration of raw dataset format
- **`analyze_hh_rlhf_structure.py`**: Detailed analysis of multi-turn conversations

## References

- [Anthropic/hh-rlhf on HuggingFace](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [Training a Helpful and Harmless Assistant with RLHF](https://arxiv.org/abs/2204.05862)
- [Qwen3-4B Model](https://huggingface.co/Qwen/Qwen3-4B)

## Notes

⚠️ **Important**: This data is meant for training **preference/reward models**, not for supervised fine-tuning of dialogue agents. Training dialogue agents directly on this data can produce harmful models.

✓ **Multi-turn support**: The preprocessing correctly handles conversations with multiple back-and-forth exchanges.

✓ **Default subset**: Using the default (no `--subset` flag) loads all helpful and harmless data, which is recommended for most use cases.

✓ **Model-agnostic format**: Data is stored as message lists (same as ultrafeedback), allowing it to work with any model. The chat template is applied at training time by the specific model's tokenizer you're using (Qwen3-4B, Llama, etc.).
