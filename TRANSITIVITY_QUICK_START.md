# Quick Start Guide: Analyzing Your Models

This guide shows the most common workflows for analyzing intransitivity in your preference datasets and evaluating your trained models.

## Setup

```bash
cd /home/robin/repos/Zhang-GPO
# Ensure you have the dependencies
pip install scipy scikit-learn matplotlib seaborn
```

## Workflow 1: Analyze Your UltraFeedback Dataset

**Goal:** Understand how much intransitivity exists in your training data.

```bash
# Quick analysis (10k sample for speed)
python scripts/experiments/analyze_ultrafeedback.py \
    --dataset $MA_SCRATCH_IOPS/data/argilla_ufb_pref/noise_1.000000_antilen_0.500000/train.jsonl \
    --output_dir results/ufb_quick_analysis \
    --max_samples 10000 \
    --generate_plots

# Full analysis (all 110k samples - takes ~20 minutes)
python scripts/experiments/analyze_ultrafeedback.py \
    --dataset $MA_SCRATCH_IOPS/data/argilla_ufb_pref/noise_1.000000_antilen_0.500000/train.jsonl \
    --output_dir results/ufb_full_analysis \
    --generate_plots
```

**Output Files:**
- `transitivity_metrics.json` - All metrics
- `analysis_report.txt` - Human-readable summary
- `preference_heatmap.png` - Visualization

**Key Questions Answered:**
- What % of pairs have conflicts? (conflict_rate)
- How many cycles exist? (triangle_cycles)
- How cyclic is the dataset? (mean_loop_ratio_hodge: 0=transitive, 1=cyclic)

## Workflow 2: Evaluate Your Trained Models

**Goal:** Compare how your 1-dim vs 8-dim models perform on test data.

### Step 1: Evaluate Each Model Individually

```bash
# Evaluate 1-dim Bradley-Terry model
python scripts/eval/evaluate_transitivity_aware.py \
    --model_path $MA_SCRATCH_CAP/runs/gpo/qwen3-8b-ufb-DATE_JOBID/checkpoints \
    --base_model /capstor/scratch/cscs/rosieber/MA/runs/sft/qwen3-8b-DATE_JOBID/checkpoints \
    --dataset $MA_SCRATCH_IOPS/data/argilla_ufb_pref/noise_1.000000_antilen_0.500000/test.jsonl \
    --output results/eval_bt_1dim.json \
    --max_samples 5000  # Optional: sample for speed

# Evaluate 8-dim GPM model
python scripts/eval/evaluate_transitivity_aware.py \
    --model_path $MA_SCRATCH_CAP/runs/gpo/qwen3-8b-ufb-DATE_JOBID/checkpoints \
    --base_model /capstor/scratch/cscs/rosieber/MA/runs/sft/qwen3-8b-DATE_JOBID/checkpoints \
    --is_gpm \
    --value_head_dim 8 \
    --dataset $MA_SCRATCH_IOPS/data/argilla_ufb_pref/noise_1.000000_antilen_0.500000/test.jsonl \
    --output results/eval_gpm_8dim.json \
    --max_samples 5000
```

### Step 2: Side-by-Side Comparison

```bash
python scripts/eval/compare_bt_vs_gpm.py \
    --bt_model_path $MA_SCRATCH_CAP/runs/gpo/bt_model/checkpoints \
    --gpm_model_path $MA_SCRATCH_CAP/runs/gpo/gpm_model/checkpoints \
    --base_model /capstor/scratch/cscs/rosieber/MA/runs/sft/qwen3-8b-DATE/checkpoints \
    --value_head_dim 8 \
    --dataset $MA_SCRATCH_IOPS/data/argilla_ufb_pref/noise_1.000000_antilen_0.500000/test.jsonl \
    --output_dir results/comparison_bt_vs_gpm \
    --max_samples 5000 \
    --generate_plots
```

**Output Files:**
- `bt_results.json`, `gpm_results.json` - Detailed metrics
- `comparison_report.txt` - **READ THIS FIRST** - Recommendations
- `bt_vs_gpm_comparison.png` - Visual comparison

**Key Questions Answered:**
- Which model is more accurate?
- Which handles intransitive pairs better?
- Is the 8-dim GPM worth the added complexity?

## Workflow 3: Generate Synthetic Test Cases

**Goal:** Create controlled experiments where ground truth is known.

```bash
# Fully transitive (baseline - both models should get 100%)
python scripts/dataset/generate_synthetic_1_100.py \
    --output_dir data/synthetic/transitive \
    --n_samples 10000 \
    --poison_strategy none

# 30% local cycles (BT should struggle, GPM should handle)
python scripts/dataset/generate_synthetic_1_100.py \
    --output_dir data/synthetic/local_p0.3 \
    --n_samples 10000 \
    --poison_strategy local_cycles \
    --poison_ratio 0.3

# Now evaluate both models on these
python scripts/eval/compare_bt_vs_gpm.py \
    --bt_model_path ... \
    --gpm_model_path ... \
    --dataset data/synthetic/local_p0.3/test \
    --output_dir results/synthetic_comparison
```

## Workflow 4: Create Poisoned Datasets for Future Training

**Goal:** Generate datasets with controlled intransitivity levels for systematic experiments.

```bash
# Create 5 versions with increasing intransitivity
for poison_ratio in 0.1 0.2 0.3 0.4 0.5; do
    python scripts/dataset/poison_ultrafeedback.py \
        --input_dataset data/ufb/train \
        --output_dir data/ufb/poisoned/label_flip_${poison_ratio} \
        --strategy label_flip \
        --flip_ratio $poison_ratio
done

# Analyze each to verify loop ratio
for poison_ratio in 0.1 0.2 0.3 0.4 0.5; do
    python scripts/experiments/analyze_ultrafeedback.py \
        --dataset data/ufb/poisoned/label_flip_${poison_ratio} \
        --output_dir results/poisoned_analysis_${poison_ratio} \
        --max_samples 5000
done
```

**Then train models on each and plot performance vs loop ratio.**

## Understanding the Output

### From `analysis_report.txt`:

```
Conflict Rate: 0.25
→ 25% of pairs have bidirectional preferences (A>B and B>A both exist)
→ HIGH intransitivity - GPM recommended

Triangle Cycles: 15,234
→ Many A>B>C>A cycles exist
→ Bradley-Terry cannot model these

Loop Ratio (Hodge): 0.18
→ 18% of preference energy is cyclic
→ Moderate intransitivity
```

### From `comparison_report.txt`:

```
Overall Accuracy:
  Bradley-Terry: 0.7234
  GPM:           0.7856
  Difference:    +0.0622 (+8.6%)
  → GPM shows SIGNIFICANT improvement

Accuracy on fully inconsistent pairs:
  Bradley-Terry: 0.52 (near random)
  GPM:           0.68
  → GPM can model intransitive preferences that BT cannot
```

## Common Issues

### "Checkpoint not found"
- Ensure the path points to directory with `config.json`
- Check that model was saved with `strategy.save_model()`

### "CUDA out of memory"
- Reduce `--max_samples` (e.g., 1000 for quick tests)
- Reduce `--max_len` (e.g., 1024 instead of 2048)

### "Takes too long"
- Use `--max_samples 5000` for quick analysis
- Skip Hodge/spectral for dataset analysis (just count cycles)

### "Results look weird"
- Check that `--is_gpm` flag matches model type
- Verify `--value_head_dim` matches training (1 for BT, 8 for GPM)
- Ensure test data is from same distribution as training

## Next Steps

1. **Analyze your dataset** → Understand baseline intransitivity
2. **Evaluate your existing models** → See which performs better
3. **Generate poisoned datasets** → Create controlled experiments
4. **Train new models** → Test on varying intransitivity levels
5. **Write paper** → Use generated tables and figures

## Questions?

- Check `scripts/README_TRANSITIVITY.md` for detailed documentation
- Look at example outputs in `results/` directories
- Review the code comments in each script

## Useful Commands

```bash
# Check if dataset has dimension field
python -c "from datasets import load_from_disk; ds = load_from_disk('path'); print(ds.column_names)"

# Quick metric calculation (no plots)
python scripts/metrics/transitivity_metrics.py \
    --dataset path/to/data.jsonl \
    --output metrics.json

# Visualize results from JSON
python -c "import json; print(json.dumps(json.load(open('results.json')), indent=2))"
```
