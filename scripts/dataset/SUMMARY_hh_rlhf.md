# Summary: hh-rlhf Preprocessing for GPM Training

## Key Points

### ✅ Fixed: Correct Data Format

**Original issue**: The first version applied the chat template during preprocessing, which was incompatible with the training script's expectations.

**Solution**: The data is now stored in **message list format** (same as ultrafeedback):
```json
{
  "prompt": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}],
  "margin": 0
}
```

The training script applies the chat template **at runtime** using the model's tokenizer (lines 88-98 in `general_preference/datasets/reward_dataset.py`).

### ✅ Multi-turn Support

The preprocessing correctly handles multi-turn conversations:
- **29.4%** single-turn exchanges
- **70.6%** multi-turn exchanges (up to 10 user turns)
- In all cases, only the final assistant response differs between chosen/rejected

### ✅ Dataset Subsets

| Subset | Size (train) | Use Case |
|--------|--------------|----------|
| **default** (recommended) | 160,800 | All helpful + harmless data |
| helpful-base | ~84,000 | Base helpfulness preferences |
| helpful-rejection-sampled | ~44,000 | Higher quality via best-of-16 |
| helpful-online | ~32,000 | From online training iteration |
| harmless-base | ~42,000 | Safety/harmlessness preferences |
| red-team-attempts | Various | Adversarial testing |

**Recommendation**: Use default (processes all subsets together).

## Files Created

1. **`scripts/dataset/preprocess_hh_rlhf.py`** - Main preprocessing script
   - Parses hh-rlhf conversation format
   - Converts to message lists (role/content dicts)
   - No longer requires `--model_name` parameter

2. **`scripts/dataset/process_hh_rlhf_cluster.sh`** - Cluster preprocessing job
   - Processes full dataset (~161k examples)
   - Output: `/cluster/scratch/rosieber/MA/data/hh_rlhf`

3. **`scripts/rmpm/train_pm_hh_rlhf.sh`** - Training script
   - Uses Qwen3-4B as base model
   - Configured for GPM training with `--use_separate_prompt`
   - Chat template applied at runtime

4. **`scripts/dataset/README_hh_rlhf.md`** - Complete documentation

5. Analysis scripts:
   - `explore_hh_rlhf.py` - Quick dataset exploration
   - `analyze_hh_rlhf_structure.py` - Multi-turn conversation analysis

## Usage

### Preprocess on Cluster
```bash
bash scripts/dataset/process_hh_rlhf_cluster.sh
```

### Train GPM
```bash
sbatch scripts/rmpm/train_pm_hh_rlhf.sh
```

### Test Locally (10 samples)
```bash
/home/robin/repos/OpenRLHF/.venv/bin/python scripts/dataset/preprocess_hh_rlhf.py \
    --output_dir "./data/hh_rlhf_test" \
    --max_samples 10
```

## Comparison: hh-rlhf vs ultrafeedback

| Aspect | hh-rlhf | ultrafeedback |
|--------|---------|---------------|
| **Format** | Message lists ✓ | Message lists ✓ |
| **Size** | 160,800 train examples | ~61,000 train examples |
| **Margins** | Always 0 (no scores) | Score-based margins |
| **Dimensions** | Combined helpful/harmless | 4 separate dimensions |
| **Multi-turn** | Yes (up to 10 turns) | Yes (variable) |
| **Prompt field** | Full conversation history | User instruction only |
| **Source** | Anthropic RLHF data | Community aggregated |

Both datasets now use **identical preprocessing format** and work with the same training script via `--use_separate_prompt`.

## Training Script Compatibility

The training script (`train_rm_general_preference.py`) with `--use_separate_prompt` flag:

1. **Loads** data with prompt/chosen/rejected as message lists
2. **Combines** prompt + chosen/rejected (list concatenation)
3. **Applies** chat template at runtime using model's tokenizer
4. **Tokenizes** and creates batches

This works identically for both hh-rlhf and ultrafeedback data!
