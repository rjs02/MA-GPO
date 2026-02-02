# Transitivity Analysis Framework

Comprehensive toolkit for quantifying and evaluating intransitivity in preference datasets for LLM alignment.

## Overview

This framework implements rigorous mathematical methods to:
1. **Quantify dataset intransitivity** using Hodge decomposition, spectral analysis, and cycle counting
2. **Evaluate model performance** on datasets with varying degrees of cyclic preferences
3. **Compare Bradley-Terry vs GPM** models systematically
4. **Generate controlled experiments** with synthetic and poisoned datasets

## Architecture

```
scripts/
├── metrics/
│   └── transitivity_metrics.py      # Core algorithms (Hodge, MFAS, spectral)
├── eval/
│   ├── evaluate_transitivity_aware.py  # Enhanced model evaluation
│   ├── evaluate_gpm_full.py            # (existing) Full GPM evaluation
│   └── compare_bt_vs_gpm.py            # Side-by-side comparison
├── dataset/
│   ├── generate_synthetic_1_100.py     # Synthetic 1-100 experiments
│   ├── poison_ultrafeedback.py         # Controlled poisoning
│   └── prepare_ultrafeedback.py        # (existing) Data preparation
├── experiments/
│   └── analyze_ultrafeedback.py        # Full dataset analysis
└── visualization/
    └── plot_transitivity.py            # All plotting utilities
```

## Installation

Dependencies are in the main `requirements.txt`. Additional packages needed:

```bash
pip install scipy scikit-learn matplotlib seaborn
```

## Quick Start

### 1. Analyze Your Dataset

Compute transitivity metrics on your UltraFeedback dataset:

```bash
python scripts/experiments/analyze_ultrafeedback.py \
    --dataset $MA_SCRATCH_IOPS/data/argilla_ufb_pref/noise_1.000000_antilen_0.500000/train.jsonl \
    --output_dir results/ufb_analysis \
    --max_samples 10000  # Optional: sample for speed
```

**Output:**
- `transitivity_metrics.json` - All computed metrics
- `analysis_report.txt` - Human-readable summary
- `preference_heatmap.png` - Visualization

**Key Metrics:**
- **Conflict Rate**: % of pairs with A>B and B>A
- **Triangle Cycles**: Number of A>B>C>A cycles
- **Loop Ratio (Hodge)**: Fraction of cyclic energy (0=transitive, 1=fully cyclic)
- **MFAS Score**: Fraction of edges to remove for acyclicity

### 2. Evaluate Your Models

Evaluate a single model with transitivity-aware metrics:

```bash
# Bradley-Terry (1-dim)
python scripts/eval/evaluate_transitivity_aware.py \
    --model_path /path/to/bt_model/checkpoints \
    --base_model Qwen/Qwen2.5-0.5B \
    --dataset /path/to/test.jsonl \
    --output evaluation_bt.json

# GPM (8-dim)
python scripts/eval/evaluate_transitivity_aware.py \
    --model_path /path/to/gpm_model/checkpoints \
    --base_model Qwen/Qwen2.5-0.5B \
    --is_gpm \
    --value_head_dim 8 \
    --dataset /path/to/test.jsonl \
    --output evaluation_gpm.json
```

**Output Metrics:**
- Accuracy stratified by empirical consistency
- Brier score, KL divergence, ECE, NLL
- Reward separability (BT) or embedding stats (GPM)
- Calibration analysis

### 3. Compare BT vs GPM

Side-by-side comparison of your trained models:

```bash
python scripts/eval/compare_bt_vs_gpm.py \
    --bt_model_path /path/to/bt_model/checkpoints \
    --gpm_model_path /path/to/gpm_model/checkpoints \
    --base_model Qwen/Qwen2.5-0.5B \
    --value_head_dim 8 \
    --dataset /path/to/test.jsonl \
    --output_dir results/comparison \
    --generate_plots
```

**Output:**
- `bt_results.json`, `gpm_results.json` - Individual results
- `comparison_report.txt` - Detailed comparison
- `bt_vs_gpm_comparison.png` - Visual comparison
- `comparison_summary.txt` - Recommendations

## Detailed Usage

### Dataset Transitivity Analysis

The core `TransitivityAnalyzer` class can be used programmatically:

```python
from scripts.metrics.transitivity_metrics import (
    TransitivityAnalyzer,
    load_preferences_from_dataset
)
from datasets import load_from_disk

# Load dataset
dataset = load_from_disk("path/to/dataset")
preferences = load_preferences_from_dataset(dataset)

# Analyze
analyzer = TransitivityAnalyzer(verbose=True)
results = analyzer.analyze_dataset(
    preferences,
    compute_hodge=True,
    compute_spectral=True,
    compute_mfas=True,
)

print(f"Conflict rate: {results['conflict_rate']:.2%}")
print(f"Loop ratio: {results['mean_loop_ratio_hodge']:.4f}")
print(f"Triangle cycles: {results['triangle_cycles']:,}")
```

**Computational Complexity:**
- Per-prompt analysis: O(n³) for n responses per prompt
- For 110k samples with ~2-4 responses each: ~10-30 minutes
- Uses sparse matrices and sampling for scalability

### Synthetic 1-100 Experiments

Generate controlled synthetic datasets for unit testing:

```bash
# Fully transitive baseline
python scripts/dataset/generate_synthetic_1_100.py \
    --output_dir data/synthetic/1-100_transitive \
    --n_samples 10000 \
    --poison_strategy none

# Local cycles (20% poisoned)
python scripts/dataset/generate_synthetic_1_100.py \
    --output_dir data/synthetic/1-100_local_p0.2 \
    --n_samples 10000 \
    --poison_strategy local_cycles \
    --poison_ratio 0.2

# Global inversion cycles
python scripts/dataset/generate_synthetic_1_100.py \
    --output_dir data/synthetic/1-100_global_p0.5 \
    --n_samples 10000 \
    --poison_strategy global_inversion \
    --poison_ratio 0.5

# Dimensional trade-offs (multi-objective)
python scripts/dataset/generate_synthetic_1_100.py \
    --output_dir data/synthetic/1-100_dimensional_p0.3 \
    --n_samples 10000 \
    --poison_strategy dimensional \
    --poison_ratio 0.3
```

**Poisoning Strategies:**
1. **Local Cycles**: Dense triangles (i > i+1 > i+2 > i) → high curl energy
2. **Global Inversion**: Long-range inconsistencies (mult. of 10 lose to 1-5) → harmonic flow
3. **Dimensional**: Multi-objective (magnitude, reverse, centrality) → realistic

**Use Case:** Train BT and GPM on each, measure performance decay vs poisoning ratio.

### Poisoning UltraFeedback

Create controlled intransitivity levels for training experiments:

```bash
# Label flipping (simple)
python scripts/dataset/poison_ultrafeedback.py \
    --input_dataset data/ultrafeedback/pref_train \
    --output_dir data/ultrafeedback/poisoned/label_flip_0.2 \
    --strategy label_flip \
    --flip_ratio 0.2

# Dimensional filtering (realistic)
python scripts/dataset/poison_ultrafeedback.py \
    --input_dataset data/ultrafeedback/pref_train \
    --output_dir data/ultrafeedback/poisoned/dimensional_mixed \
    --strategy dimensional \
    --dimensions "honesty,truthfulness,helpfulness" \
    --dimension_weights "0.4,0.3,0.3"

# Cycle oversampling
python scripts/dataset/poison_ultrafeedback.py \
    --input_dataset data/ultrafeedback/pref_train \
    --output_dir data/ultrafeedback/poisoned/cycle_2x \
    --strategy cycle_oversample \
    --oversample_factor 2.0
```

**Recommended Workflow:**
1. Generate datasets with loop ratios: [0.05, 0.1, 0.2, 0.3, 0.4]
2. Train both BT (1-dim) and GPM (8-dim) on each
3. Compare performance vs loop ratio
4. Identify crossover point where GPM becomes necessary

## Visualization

All visualization utilities are in `plot_transitivity.py`:

```python
from scripts.visualization.plot_transitivity import (
    plot_preference_heatmap,
    plot_calibration_curve,
    plot_accuracy_by_consistency,
    plot_comparison_bt_vs_gpm,
)

# Preference heatmap
plot_preference_heatmap(
    pref_matrix,  # n×n matrix of P(i > j)
    response_labels=["resp1", "resp2", ...],
    output_path="heatmap.png"
)

# Calibration curve
plot_calibration_curve(
    model_probs,      # Model predicted P(chosen > rejected)
    empirical_probs,  # Dataset win rates
    model_name="GPM",
    output_path="calibration.png"
)

# Accuracy by consistency
plot_accuracy_by_consistency(
    accuracy_by_consistency_dict,
    model_name="Bradley-Terry",
    output_path="accuracy_stratified.png"
)
```

## Metrics Interpretation

### Dataset Metrics

| Metric | Range | Interpretation |
|--------|-------|----------------|
| **Conflict Rate** | [0, 1] | % of pairs with bidirectional preferences. >0.2 = high intransitivity |
| **Loop Ratio (Hodge)** | [0, 1] | Fraction of cyclic energy. 0 = fully transitive, 1 = fully cyclic |
| **Triangle Cycles** | [0, ∞) | Number of A>B>C>A cycles. Direct evidence of intransitivity |
| **MFAS Score** | [0, 1] | % edges to remove for acyclicity. Measure of global inconsistency |
| **Spectral Gap** | [0, ∞) | Eigenvalue gap. Large = transitive, small = cyclic |

### Model Metrics

| Metric | Range | Interpretation | Lower is Better? |
|--------|-------|----------------|------------------|
| **Accuracy** | [0, 1] | Overall correct predictions | No (higher better) |
| **Brier Score** | [0, 1] | MSE of probabilities | Yes |
| **KL Divergence** | [0, ∞) | Distribution mismatch | Yes |
| **ECE** | [0, 1] | Calibration error | Yes |
| **NLL** | [0, ∞) | Negative log likelihood | Yes |

**For BT models:**
- **Reward Separability** (σ of all rewards): Should be >0.5. If <0.1, model has collapsed.

**For GPM models:**
- **Embedding Variance**: Should be >1e-4. Lower indicates collapse.
- **Mean Distance**: Distance between chosen/rejected embeddings. Should be positive.

## Expected Results

### UltraFeedback (110k samples)

Based on your observation of ~25% conflicting responses:

- **Conflict Rate**: 0.20-0.30
- **Loop Ratio**: 0.15-0.25 (moderate intransitivity)
- **Triangle Cycles**: Thousands (especially in "helpfulness" dimension)
- **MFAS**: 0.10-0.15 (10-15% edges need removal)

**Recommendation:** GPM should outperform BT, especially on high-intransitivity prompts.

### Synthetic 1-100

| Poison Ratio | BT Accuracy | GPM Accuracy | BT Reward σ | GPM Embedding σ |
|--------------|-------------|--------------|-------------|-----------------|
| 0.0 (clean) | 1.00 | 1.00 | High | High |
| 0.2 | 0.85 | 0.95 | Moderate | High |
| 0.5 | 0.65 | 0.85 | Low | Moderate |
| 0.8 | 0.55 | 0.75 | Very low | Moderate |

**Key Observation:** BT reward σ collapses as intransitivity increases; GPM maintains separation.

## Training Models

To train models on different poisoning levels, modify your training script:

```bash
# Bradley-Terry (1-dim)
bash scripts/clariden/train_gpo_ufb.sh \
    --pretrain $SFT_MODEL \
    --dataset data/ultrafeedback/poisoned/label_flip_0.2/train \
    --value_head_dim 1 \
    --is_general_preference False

# GPM (8-dim)
bash scripts/clariden/train_gpo_ufb.sh \
    --pretrain $SFT_MODEL \
    --dataset data/ultrafeedback/poisoned/label_flip_0.2/train \
    --value_head_dim 8 \
    --is_general_preference True
```

**Systematic Sweep:**
```bash
for poison_ratio in 0.0 0.1 0.2 0.3 0.4; do
    # Generate poisoned data
    python scripts/dataset/poison_ultrafeedback.py \
        --input_dataset data/ufb/train \
        --output_dir data/ufb/poisoned/p${poison_ratio} \
        --strategy label_flip \
        --flip_ratio $poison_ratio
    
    # Train BT
    bash train_bt.sh --dataset data/ufb/poisoned/p${poison_ratio}/train
    
    # Train GPM
    bash train_gpm.sh --dataset data/ufb/poisoned/p${poison_ratio}/train
done
```

## Paper-Ready Outputs

All scripts generate publication-quality outputs:

### Tables (LaTeX format)

From `comparison_report.txt`:
```
\begin{table}
\caption{Bradley-Terry vs GPM Performance}
\begin{tabular}{lcc}
Metric & BT & GPM \\
\hline
Accuracy & 0.7234 & 0.7856 \\
Brier Score & 0.1845 & 0.1423 \\
...
\end{tabular}
\end{table}
```

### Figures (PNG, 300 DPI)

- Preference heatmaps
- Calibration curves
- Accuracy by consistency
- Performance decay plots

### Statistical Tests

Add to your evaluation:
```python
from scipy.stats import ttest_rel

# Paired t-test on per-sample accuracies
t_stat, p_value = ttest_rel(bt_correct, gpm_correct)
print(f"GPM improvement p-value: {p_value:.4f}")
```

## Troubleshooting

### Memory Issues

For large datasets (>100k):
- Use `--max_samples 10000` to sample
- Hodge decomposition works per-prompt (no memory explosion)
- Spectral analysis only on prompts with <100 responses

### Long Computation Times

Optimization strategies:
- Skip `--compute_hodge` and `--compute_spectral` for quick analysis
- Use conflict rate and cycle count (fast) as primary metrics
- Parallelize per-prompt analysis (future enhancement)

### Model Loading Errors

Ensure:
- Checkpoint has `config.json` (saved with `strategy.save_model()`)
- `base_model` matches training
- `value_head_dim` matches training (1 for BT, 8 for GPM)
- `is_general_preference` flag is correct

## Citation

If you use this framework, please cite:

```bibtex
@article{your-paper-2026,
  title={Quantifying Intransitivity in Preference-Based Reward Modeling},
  author={Your Name},
  year={2026}
}
```

## Contributing

To add new metrics:
1. Add to `TransitivityAnalyzer` in `transitivity_metrics.py`
2. Update evaluation in `evaluate_transitivity_aware.py`
3. Add visualization in `plot_transitivity.py`
4. Document in this README

## References

- **Hodge Decomposition**: Jiang et al., "Statistical Ranking and Combinatorial Hodge Theory"
- **General Preference Models**: Zhang et al., "General Preference Modeling with Preference Representations"
- **Nash Learning**: Munos et al., "Nash Learning from Human Feedback"
- **Pluralistic Alignment**: Sorensen et al., "Value Kaleidoscope"

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
