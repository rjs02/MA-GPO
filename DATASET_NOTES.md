# Dataset Notes for Intransitivity Analysis

## The Problem You Just Encountered

When you ran:
```bash
python scripts/experiments/analyze_ultrafeedback.py \
    --dataset $MA_SCRATCH_IOPS/data/argilla_ufb_pref/noise_1.000000_antilen_0.500000/train.jsonl
```

**Results:**
- 10,000 preferences
- 9,339 unique prompts
- **0 conflicts, 0 cycles**

### Why This Happened

The dataset `argilla/ultrafeedback-binarized-preferences-cleaned` is **BINARIZED**:
- Each prompt has exactly **1 pair**: (chosen, rejected)
- No multiple responses to compare
- **No opportunity for conflicts or cycles!**

Example:
```json
{
  "prompt": "Write a poem",
  "chosen": "Response A",
  "rejected": "Response B"
}
```

To detect intransitivity, you need **at least 3 responses per prompt** so cycles can form:
- A > B
- B > C  
- C > A ← Creates a cycle!

## Solutions

### Option 1: Use Multidimensional UltraFeedback

The **full UltraFeedback dataset** (`openbmb/UltraFeedback`) has 4 responses per prompt rated on 4 dimensions.

You already have a script that processes this! Use `build_ufb_data.py`:

```bash
# This creates multidimensional pairs (NOT averaged)
python scripts/dataset/build_ufb_data.py \
    --output_dir $MA_SCRATCH_IOPS/data/ufb_multidim \
    --cleaned_dataset_name argilla/ultrafeedback-binarized-preferences-cleaned \
    --full_dataset_name openbmb/UltraFeedback
    # DON'T use --averaged flag!

# This creates ~100k+ pairs with multiple responses per prompt
```

Then analyze:
```bash
python scripts/experiments/analyze_ultrafeedback.py \
    --dataset $MA_SCRATCH_IOPS/data/ufb_multidim/pref_train \
    --output_dir results/ufb_multidim_analysis \
    --max_samples 10000
```

### Option 2: Use Your Cyclic Datasets

You already have scripts that explicitly find cycles! Use `ufb_cyclic_grouped.py`:

```bash
python scripts/dataset/ufb_cyclic_grouped.py \
    --output_path $MA_SCRATCH_IOPS/data/ufb_cyclic \
    --min_margin 1

# This finds natural cycles like:
# - honesty > truthfulness > helpfulness > honesty
```

Then analyze:
```bash
python scripts/experiments/analyze_ultrafeedback.py \
    --dataset $MA_SCRATCH_IOPS/data/ufb_cyclic/Cyclic_1 \
    --output_dir results/cyclic_analysis
```

### Option 3: Analyze Dimension-Level Conflicts

Even in binarized data, if you have the `dimension` field, you can analyze conflicts **across dimensions**:

```python
# Custom analysis
from datasets import load_from_disk
dataset = load_from_disk("path/to/dataset")

# Group by prompt and dimension
conflicts = 0
for prompt in unique_prompts:
    samples = [s for s in dataset if s['prompt'] == prompt]
    
    # Check if different dimensions have different preferences
    if has_dimensional_conflicts(samples):
        conflicts += 1
```

## What Your Current Dataset Is Good For

The **binarized cleaned dataset** you're using is perfect for:

✅ **Training preference models** (BT and GPM)
- Clean, deduplicated pairs
- Good for learning general preferences

✅ **Evaluating accuracy**
- Test if model predicts chosen > rejected

❌ **NOT good for:**
- Intransitivity analysis (no cycles possible)
- Measuring loop ratio (need multiple comparisons per prompt)

## Recommended Workflow

### Step 1: Process Multidimensional Data

```bash
# Generate multidimensional dataset (if not done already)
python scripts/dataset/build_ufb_data.py \
    --output_dir $MA_SCRATCH_IOPS/data/ufb_multidim \
    --full_dataset_name openbmb/UltraFeedback
```

This creates:
- `pref_train/` - With dimension field and multiple pairs per prompt
- Each prompt has pairs from all 4 dimensions
- **This is where cycles naturally appear!**

### Step 2: Analyze Transitivity

```bash
python scripts/experiments/analyze_ultrafeedback.py \
    --dataset $MA_SCRATCH_IOPS/data/ufb_multidim/pref_train \
    --output_dir results/ufb_multidim_analysis \
    --max_samples 20000  # Use more samples since there are more pairs
```

Expected results:
- Conflict rate: 0.20-0.30 (20-30%)
- Triangle cycles: Thousands
- Loop ratio: 0.15-0.25

### Step 3: Evaluate Your Models

Use your **existing trained models** on the test set:

```bash
# Your models were trained on binarized data, which is fine!
# Now evaluate on multidimensional test data to see performance on cycles

python scripts/eval/compare_bt_vs_gpm.py \
    --bt_model_path $MA_SCRATCH_CAP/runs/gpo/bt_model/checkpoints \
    --gpm_model_path $MA_SCRATCH_CAP/runs/gpo/gpm_model/checkpoints \
    --dataset $MA_SCRATCH_IOPS/data/ufb_multidim/pref_test \
    --output_dir results/comparison_on_multidim
```

This will show if GPM handles dimensional conflicts better than BT.

## Understanding Your Training Data

Your current training data structure:
```
argilla_ufb_pref/noise_1.000000_antilen_0.500000/
├── train.jsonl    # 109k binarized pairs
├── val.jsonl      # ~6k pairs
└── test.jsonl     # ~6k pairs
```

**What the augmentation means:**
- `noise_1.000000` = 100% augmented data added (anti-length + RRM pairs)
- `antilen_0.500000` = 50% of augmented samples are anti-length pairs
- These augmentations add **relevance-based** pairs, not dimensional conflicts

**For intransitivity analysis, you need:**
```
ufb_multidim/
├── pref_train/    # Multiple pairs per prompt from different dimensions
├── pref_val/
└── pref_test/
```

## Quick Test

Run this to see your dataset structure:

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('json', data_files='$MA_SCRATCH_IOPS/data/argilla_ufb_pref/noise_1.000000_antilen_0.500000/train.jsonl', split='train')
print('Fields:', ds.column_names)
print('Sample:', ds[0])

# Count prompts with multiple pairs
from collections import defaultdict
prompt_counts = defaultdict(int)
for sample in ds.select(range(10000)):
    prompt = str(sample['prompt'][0]['content'] if isinstance(sample['prompt'], list) else sample['prompt'])
    prompt_counts[prompt] += 1

multi = sum(1 for count in prompt_counts.values() if count > 1)
print(f'\nPrompts with >1 pair: {multi}/{len(prompt_counts)}')
"
```

If output shows "Prompts with >1 pair: 0/9339" → You need multidimensional data!

## Summary

| Dataset Type | Pairs per Prompt | Good For | Has Cycles? |
|--------------|------------------|----------|-------------|
| **Binarized** (what you used) | 1 | Training, basic eval | ❌ No |
| **Multidimensional** (what you need) | 10-20 | Transitivity analysis | ✅ Yes |
| **Cyclic filtered** | 3+ | Extreme intransitivity | ✅ Yes (intentional) |

**Action:** Run `build_ufb_data.py` without `--averaged` to get the right dataset structure!
