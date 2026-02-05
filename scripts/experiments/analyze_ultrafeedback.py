#!/usr/bin/env python3
"""
Analyze UltraFeedback dataset for intransitivity.

Computes all transitivity metrics and generates visualizations.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.metrics.transitivity_metrics import (
    TransitivityAnalyzer,
    load_preferences_from_dataset,
    load_preferences_from_jsonl,
)
from scripts.visualization.plot_transitivity import (
    plot_preference_heatmap,
)
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm


def main(args):
    print("="*70)
    print("UltraFeedback Transitivity Analysis")
    print("="*70)
    
    # Load dataset
    print(f"\nLoading dataset from {args.dataset}...")
    if args.dataset.endswith('.jsonl'):
        from datasets import load_dataset
        dataset = load_dataset('json', data_files=args.dataset, split='train')
    else:
        dataset = load_from_disk(args.dataset)
    
    print(f"Loaded {len(dataset):,} samples")
    
    # Convert to preferences format
    print("\nConverting to preferences format...")
    preferences = load_preferences_from_dataset(
        dataset,
        prompt_key=args.prompt_key,
        chosen_key=args.chosen_key,
        rejected_key=args.rejected_key,
        max_samples=args.max_samples,
    )
    
    print(f"Converted {len(preferences):,} preferences")
    
    # Analyze transitivity
    print("\nAnalyzing transitivity...")
    analyzer = TransitivityAnalyzer(verbose=True)
    
    results = analyzer.analyze_dataset(
        preferences,
        max_samples=args.max_samples,
        compute_hodge=args.compute_hodge,
        compute_spectral=args.compute_spectral,
        compute_mfas=args.compute_mfas,
    )
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\nDataset Statistics:")
    print(f"  Total pairs: {results['total_pairs']:,}")
    print(f"  Unique prompts: {results['unique_prompts']:,}")
    print(f"  Conflict rate: {results['conflict_rate']:.4f}")
    print(f"  Mean conflict rate per prompt: {results['mean_conflict_rate']:.4f}")
    
    print(f"\nCycle Statistics:")
    # Backward compatibility: handle both old and new key names
    if 'triangle_cycles_majority' in results:
        print(f"  Total triangle cycles (majority-vote): {results['triangle_cycles_majority']:,}")
        print(f"  Total triangle cycles (instance-level): {results['triangle_cycles_instance']:,}")
        print(f"  Mean cycles per prompt (majority): {results['mean_cycles_per_prompt_majority']:.2f}")
        print(f"  Mean cycles per prompt (instance): {results['mean_cycles_per_prompt_instance']:.2f}")
    else:
        # Old format
        print(f"  Total triangle cycles: {results.get('triangle_cycles', 0):,}")
        print(f"  Mean cycles per prompt: {results.get('mean_cycles_per_prompt', 0):.2f}")
    
    if args.compute_hodge:
        print(f"\nHodge Decomposition:")
        print(f"  Mean loop ratio (raw): {results['mean_loop_ratio_hodge']:.4f}")
        print(f"  Mean baseline loop ratio: {results['mean_baseline_loop_ratio']:.4f}")
        print(f"  Mean adjusted loop ratio: {results['mean_adjusted_loop_ratio']:.4f}")
        if 'n_hodge_computed' in results:
            print(f"  Computed for {results['n_hodge_computed']:,} / {results['unique_prompts']:,} prompts")
    
    if args.compute_spectral:
        print(f"\nSpectral Analysis:")
        print(f"  Mean spectral gap: {results['mean_spectral_gap']:.4f}")
        if 'n_spectral_computed' in results:
            print(f"  Computed for {results['n_spectral_computed']:,} / {results['unique_prompts']:,} prompts")
    
    if args.compute_mfas:
        print(f"\nMFAS Score:")
        print(f"  Mean MFAS score: {results['mean_mfas_score']:.4f}")
        if 'n_mfas_computed' in results:
            print(f"  Computed for {results['n_mfas_computed']:,} / {results['unique_prompts']:,} prompts")
    
    print(f"\nConflict Rate Distribution:")
    for k, v in results['conflict_rate_distribution'].items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\nDataset Type:")
    print(f"  Prompts with multiple responses (≥3): {results.get('prompts_with_multiple_responses', 0):,}")
    print(f"  Prompts with conflicts: {results.get('prompts_with_conflicts', 0):,}")
    
    # Show response count distribution
    if 'response_count_distribution' in results:
        print(f"\nResponse Count Distribution:")
        for k, v in results['response_count_distribution'].items():
            print(f"  {k}: {v}")
    
    if results.get('prompts_with_multiple_responses', 0) < 100:
        print("\n⚠️  WARNING: Very few prompts have multiple responses!")
        print("   This dataset appears to be BINARIZED (only chosen vs rejected per prompt).")
        print("   For intransitivity analysis, you need the MULTIDIMENSIONAL dataset.")
        print("\n   Suggested alternatives:")
        print("   1. Use openbmb/UltraFeedback (not binarized)")
        print("   2. Use build_ufb_data.py with --averaged=False to get multidimensional pairs")
        print("   3. Or analyze at the dimension level if your data has 'dimension' field")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    results_path = os.path.join(args.output_dir, "transitivity_metrics.json")
    with open(results_path, 'w') as f:
        # Convert any remaining numpy types
        import numpy as np
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_clean = convert_numpy(results)
        json.dump(results_clean, f, indent=2)
    print(f"\n✓ Saved metrics to {results_path}")
    
    # Generate visualizations if requested
    if args.generate_plots:
        print("\nGenerating visualizations...")
        
        # For visualization, we need to build a global preference matrix
        # Use a sample of prompts to avoid memory issues
        if len(preferences) > 10000:
            print("  Sampling 5000 pairs for visualization...")
            import random
            vis_prefs = random.sample(preferences, 5000)
        else:
            vis_prefs = preferences
        
        # Build preference matrix for a single prompt (example)
        prompt_groups = {}
        for prompt, chosen, rejected in vis_prefs:
            if prompt not in prompt_groups:
                prompt_groups[prompt] = []
            prompt_groups[prompt].append((chosen, rejected))
        
        # Find prompt with most comparisons
        largest_prompt = max(prompt_groups.items(), key=lambda x: len(x[1]))
        prompt, pairs = largest_prompt
        
        # Extract responses
        responses = set()
        for chosen, rejected in pairs:
            responses.add(chosen)
            responses.add(rejected)
        responses = sorted(list(responses))
        
        print(f"  Creating heatmap for prompt with {len(responses)} responses...")
        
        # Build preference matrix
        resp_to_idx = {r: i for i, r in enumerate(responses)}
        pref_matrix = np.zeros((len(responses), len(responses)))
        
        for chosen, rejected in pairs:
            i = resp_to_idx[chosen]
            j = resp_to_idx[rejected]
            pref_matrix[i, j] += 1
        
        # Convert to probabilities
        total = pref_matrix + pref_matrix.T
        with np.errstate(divide='ignore', invalid='ignore'):
            pref_matrix = np.where(total > 0, pref_matrix / total, 0.5)
        
        # Plot heatmap
        heatmap_path = os.path.join(args.output_dir, "preference_heatmap.png")
        plot_preference_heatmap(
            pref_matrix,
            response_labels=[r[:30] for r in responses],  # Truncate labels
            title=f"Preference Matrix (n={len(responses)} responses)",
            output_path=heatmap_path,
        )
        
        print(f"  ✓ Saved heatmap to {heatmap_path}")
    
    # Generate summary report
    print("\nGenerating summary report...")
    report_path = os.path.join(args.output_dir, "analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("UltraFeedback Transitivity Analysis Report\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Samples analyzed: {len(preferences):,}\n\n")
        
        f.write("KEY FINDINGS:\n\n")
        
        f.write(f"1. Conflict Rate: {results['conflict_rate']:.2%}\n")
        f.write(f"   Interpretation: {results['conflict_rate']:.2%} of preference pairs have\n")
        f.write(f"   contradictory labels (A>B and B>A both present).\n\n")
        
        f.write(f"2. Triangle Cycles:\n")
        if 'triangle_cycles_majority' in results:
            f.write(f"   Majority-vote: {results['triangle_cycles_majority']:,}\n")
            f.write(f"   Instance-level: {results['triangle_cycles_instance']:,}\n")
            f.write(f"   Interpretation: Cycles in the aggregated preference graph vs. raw pairs.\n")
            f.write(f"   Majority-vote counts cycles where A beats B more often than B beats A.\n")
            f.write(f"   Instance-level counts every individual triplet forming A>B, B>C, C>A.\n")
        else:
            f.write(f"   Total: {results.get('triangle_cycles', 0):,}\n")
            f.write(f"   Interpretation: {results.get('triangle_cycles', 0):,} instances of A>B>C>A cycles.\n")
        f.write(f"   These cannot be captured by Bradley-Terry models.\n\n")
        
        if args.compute_hodge:
            f.write(f"3. Loop Ratio (Hodge):\n")
            f.write(f"   Raw loop ratio: {results['mean_loop_ratio_hodge']:.4f}\n")
            f.write(f"   Baseline (structural floor): {results['mean_baseline_loop_ratio']:.4f}\n")
            f.write(f"   Adjusted loop ratio: {results['mean_adjusted_loop_ratio']:.4f}\n")
            f.write(f"   \n")
            f.write(f"   Interpretation: The raw loop ratio shows {results['mean_loop_ratio_hodge']:.1%}\n")
            f.write(f"   of preference energy is in cyclic components.\n")
            f.write(f"   The adjusted loop ratio ({results['mean_adjusted_loop_ratio']:.1%}) corrects for\n")
            f.write(f"   structural baseline intransitivity due to observation topology.\n")
            f.write(f"   0 = fully transitive, 1 = fully cyclic.\n\n")
        
        if args.compute_mfas:
            f.write(f"4. MFAS Score: {results['mean_mfas_score']:.4f}\n")
            f.write(f"   Interpretation: {results['mean_mfas_score']:.1%} of edges would need\n")
            f.write(f"   to be removed to make the preference graph acyclic.\n\n")
        
        f.write("RECOMMENDATIONS:\n\n")
        
        # Use adjusted loop ratio if available, otherwise fall back to conflict rate
        if 'mean_adjusted_loop_ratio' in results and not np.isnan(results['mean_adjusted_loop_ratio']):
            adj_loop = results['mean_adjusted_loop_ratio']
            if adj_loop > 0.15:
                f.write("- HIGH intransitivity detected (adjusted loop ratio > 0.15)\n")
                f.write("  → General Preference Models (GPM) strongly recommended\n")
                f.write("  → Bradley-Terry models may exhibit reward collapse\n\n")
            elif adj_loop > 0.05:
                f.write("- MODERATE intransitivity detected (adjusted loop ratio 0.05-0.15)\n")
                f.write("  → GPM may provide benefits over BT on high-conflict prompts\n")
                f.write("  → Consider analyzing per-dimension conflict rates\n\n")
            else:
                f.write("- LOW intransitivity detected (adjusted loop ratio < 0.05)\n")
                f.write("  → Bradley-Terry models should work well\n")
                f.write("  → GPM benefits may be marginal\n\n")
        elif results['conflict_rate'] > 0.2:
            f.write("- HIGH intransitivity detected (>20% conflicts)\n")
            f.write("  → General Preference Models (GPM) recommended\n")
            f.write("  → Bradley-Terry models may exhibit reward collapse\n\n")
        elif results['conflict_rate'] > 0.1:
            f.write("- MODERATE intransitivity detected (10-20% conflicts)\n")
            f.write("  → GPM may provide benefits over BT on high-conflict prompts\n")
            f.write("  → Consider analyzing per-dimension conflict rates\n\n")
        else:
            f.write("- LOW intransitivity detected (<10% conflicts)\n")
            f.write("  → Bradley-Terry models should work well\n")
            f.write("  → GPM benefits may be marginal\n\n")
        
        f.write("="*70 + "\n")
    
    print(f"✓ Saved report to {report_path}")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze UltraFeedback dataset for intransitivity"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset (JSONL or HF Dataset directory)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit analysis to N samples (for speed)"
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="prompt",
        help="Key for prompt field"
    )
    parser.add_argument(
        "--chosen_key",
        type=str,
        default="chosen",
        help="Key for chosen response"
    )
    parser.add_argument(
        "--rejected_key",
        type=str,
        default="rejected",
        help="Key for rejected response"
    )
    parser.add_argument(
        "--compute_hodge",
        action="store_true",
        default=True,
        help="Compute Hodge decomposition"
    )
    parser.add_argument(
        "--compute_spectral",
        action="store_true",
        default=True,
        help="Compute spectral metrics"
    )
    parser.add_argument(
        "--compute_mfas",
        action="store_true",
        default=True,
        help="Compute MFAS score"
    )
    parser.add_argument(
        "--generate_plots",
        action="store_true",
        default=True,
        help="Generate visualization plots"
    )
    
    args = parser.parse_args()
    main(args)


"""
Example usage:

# Analyze full dataset
python scripts/experiments/analyze_ultrafeedback.py \
    --dataset $MA_SCRATCH_IOPS/data/argilla_ufb_pref/noise_1.000000_antilen_0.500000/train.jsonl \
    --output_dir results/ufb_analysis \
    --generate_plots

# Quick analysis (sample)
python scripts/experiments/analyze_ultrafeedback.py \
    --dataset $MA_SCRATCH_IOPS/data/argilla_ufb_pref/noise_1.000000_antilen_0.500000/train.jsonl \
    --output_dir results/ufb_analysis_sample \
    --max_samples 10000 \
    --generate_plots

python scripts/experiments/analyze_ultrafeedback.py \
    --dataset $MA_SCRATCH_IOPS/data/ufb-multidim-nc/pref_train \
    --output_dir results/ufb_analysis_multidim_nc \
    --generate_plots
"""

