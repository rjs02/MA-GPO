#!/usr/bin/env python3
"""
Side-by-side comparison of Bradley-Terry vs GPM models.

Evaluates both models on the same dataset and generates comparative reports.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.eval.evaluate_transitivity_aware import (
    load_model,
    evaluate_model,
    print_results,
)
from scripts.visualization.plot_transitivity import (
    plot_comparison_bt_vs_gpm,
)
from transformers import AutoTokenizer
from datasets import load_from_disk


def main(args):
    print("="*70)
    print("Bradley-Terry vs GPM Comparison")
    print("="*70)
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print(f"\nLoading dataset from {args.dataset}...")
    if args.dataset.endswith('.jsonl'):
        from datasets import load_dataset
        dataset = load_dataset('json', data_files=args.dataset, split='train')
    else:
        dataset = load_from_disk(args.dataset)
    
    if args.max_samples and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
    
    print(f"  Using {len(dataset):,} samples")
    
    # Evaluate Bradley-Terry model
    print("\n" + "="*70)
    print("Evaluating Bradley-Terry Model")
    print("="*70)
    
    print(f"\nLoading BT model from {args.bt_model_path}...")
    bt_model = load_model(
        args.bt_model_path,
        args.base_model,
        is_gpm=False,
        value_head_dim=1,
        bf16=args.bf16,
    )
    
    bt_results = evaluate_model(
        bt_model, dataset, tokenizer,
        is_gpm=False,
        tau=args.tau,
        value_head_dim=1,
        max_len=args.max_len,
        prompt_key=args.prompt_key,
        chosen_key=args.chosen_key,
        rejected_key=args.rejected_key,
    )
    
    print_results(bt_results, "Bradley-Terry")
    
    # Evaluate GPM model
    print("\n" + "="*70)
    print("Evaluating GPM Model")
    print("="*70)
    
    print(f"\nLoading GPM model from {args.gpm_model_path}...")
    gpm_model = load_model(
        args.gpm_model_path,
        args.base_model,
        is_gpm=True,
        value_head_dim=args.value_head_dim,
        bf16=args.bf16,
    )
    
    gpm_results = evaluate_model(
        gpm_model, dataset, tokenizer,
        is_gpm=True,
        tau=args.tau,
        value_head_dim=args.value_head_dim,
        max_len=args.max_len,
        prompt_key=args.prompt_key,
        chosen_key=args.chosen_key,
        rejected_key=args.rejected_key,
    )
    
    print_results(gpm_results, "GPM")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save individual results
    bt_path = os.path.join(args.output_dir, "bt_results.json")
    with open(bt_path, 'w') as f:
        json.dump(bt_results, f, indent=2)
    print(f"\n✓ Saved BT results to {bt_path}")
    
    gpm_path = os.path.join(args.output_dir, "gpm_results.json")
    with open(gpm_path, 'w') as f:
        json.dump(gpm_results, f, indent=2)
    print(f"✓ Saved GPM results to {gpm_path}")
    
    # Generate comparison visualizations
    if args.generate_plots:
        print("\nGenerating comparison plots...")
        plot_comparison_bt_vs_gpm(bt_results, gpm_results, args.output_dir)
    
    # Generate comparative summary
    print("\nGenerating comparative summary...")
    summary_path = os.path.join(args.output_dir, "comparison_report.txt")
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BRADLEY-TERRY vs GPM COMPARISON REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Samples: {len(dataset):,}\n")
        f.write(f"BT Model: {args.bt_model_path}\n")
        f.write(f"GPM Model: {args.gpm_model_path}\n")
        f.write(f"GPM Dimensions: {args.value_head_dim}\n\n")
        
        f.write("="*70 + "\n")
        f.write("OVERALL PERFORMANCE\n")
        f.write("="*70 + "\n\n")
        
        bt_acc = bt_results['overall_accuracy']
        gpm_acc = gpm_results['overall_accuracy']
        diff_acc = gpm_acc - bt_acc
        
        f.write(f"Accuracy:\n")
        f.write(f"  Bradley-Terry: {bt_acc:.4f}\n")
        f.write(f"  GPM:           {gpm_acc:.4f}\n")
        f.write(f"  Difference:    {diff_acc:+.4f} ({diff_acc/bt_acc*100:+.1f}%)\n")
        
        if diff_acc > 0.05:
            f.write(f"  → GPM shows SIGNIFICANT improvement\n")
        elif diff_acc > 0.02:
            f.write(f"  → GPM shows MODERATE improvement\n")
        elif diff_acc > 0:
            f.write(f"  → GPM shows SLIGHT improvement\n")
        else:
            f.write(f"  → BT performs comparably or better\n")
        
        f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("PROBABILISTIC METRICS\n")
        f.write("="*70 + "\n\n")
        
        metrics = ['brier_score', 'kl_forward', 'ece', 'nll']
        metric_names = {
            'brier_score': 'Brier Score',
            'kl_forward': 'KL Divergence (forward)',
            'ece': 'Expected Calibration Error',
            'nll': 'Negative Log Likelihood',
        }
        
        for metric in metrics:
            bt_val = bt_results['probabilistic_metrics'][metric]
            gpm_val = gpm_results['probabilistic_metrics'][metric]
            diff = gpm_val - bt_val
            
            f.write(f"{metric_names[metric]}:\n")
            f.write(f"  Bradley-Terry: {bt_val:.4f}\n")
            f.write(f"  GPM:           {gpm_val:.4f}\n")
            f.write(f"  Difference:    {diff:+.4f}")
            
            # Lower is better for these metrics
            if diff < -0.01:
                f.write(f" (GPM better)\n")
            elif diff > 0.01:
                f.write(f" (BT better)\n")
            else:
                f.write(f" (comparable)\n")
            f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("ACCURACY BY CONSISTENCY LEVEL\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"{'Level':<30} {'BT':>10} {'GPM':>10} {'Diff':>10}\n")
        f.write("-"*70 + "\n")
        
        for level in bt_results['accuracy_by_consistency'].keys():
            bt_acc_level = bt_results['accuracy_by_consistency'][level]['accuracy']
            gpm_acc_level = gpm_results['accuracy_by_consistency'][level]['accuracy']
            bt_count = bt_results['accuracy_by_consistency'][level]['count']
            
            if bt_count > 0:
                diff_level = gpm_acc_level - bt_acc_level
                f.write(f"{level:<30} {bt_acc_level:>10.4f} {gpm_acc_level:>10.4f} {diff_level:>+10.4f}\n")
        
        f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("="*70 + "\n\n")
        
        # Analyze where GPM excels
        fully_inconsistent_bt = bt_results['accuracy_by_consistency'].get('fully_inconsistent', {}).get('accuracy', 0)
        fully_inconsistent_gpm = gpm_results['accuracy_by_consistency'].get('fully_inconsistent', {}).get('accuracy', 0)
        
        if fully_inconsistent_gpm > fully_inconsistent_bt + 0.05:
            f.write("1. GPM significantly outperforms BT on fully inconsistent pairs\n")
            f.write("   → GPM can model intransitive preferences that BT cannot\n\n")
        
        # Check for reward collapse
        if 'reward_separability' in bt_results:
            bt_collapse = bt_results['reward_separability']['collapse_warning']
            if bt_collapse:
                f.write("2. Bradley-Terry model shows signs of reward collapse\n")
                f.write(f"   → Reward std: {bt_results['reward_separability']['all_rewards_std']:.4f}\n")
                f.write("   → Model may be struggling with intransitive preferences\n\n")
        
        # Check embedding quality
        if 'embedding_stats' in gpm_results:
            gpm_collapse = gpm_results['embedding_stats']['collapse_warning']
            if not gpm_collapse:
                f.write("3. GPM maintains diverse embeddings\n")
                f.write(f"   → Avg variance: {gpm_results['embedding_stats']['avg_variance_chosen']:.6f}\n")
                f.write("   → Model is learning meaningful distinctions\n\n")
        
        # Dataset intransitivity
        inconsistency_rate = bt_results['consistency_stats']['inconsistency_rate']
        f.write(f"4. Dataset intransitivity: {inconsistency_rate:.2%}\n")
        if inconsistency_rate > 0.2:
            f.write("   → HIGH intransitivity - GPM recommended\n")
        elif inconsistency_rate > 0.1:
            f.write("   → MODERATE intransitivity - GPM may help\n")
        else:
            f.write("   → LOW intransitivity - BT should suffice\n")
        f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("="*70 + "\n\n")
        
        if gpm_acc > bt_acc + 0.05:
            f.write("✓ STRONGLY RECOMMEND GPM\n")
            f.write("  GPM shows significant accuracy improvements and better calibration.\n")
        elif gpm_acc > bt_acc + 0.02:
            f.write("✓ RECOMMEND GPM\n")
            f.write("  GPM shows moderate improvements, especially on intransitive pairs.\n")
        elif gpm_acc > bt_acc:
            f.write("→ CONSIDER GPM\n")
            f.write("  GPM shows slight improvements. Benefits may justify added complexity.\n")
        else:
            f.write("→ BRADLEY-TERRY SUFFICIENT\n")
            f.write("  Dataset appears sufficiently transitive for Bradley-Terry modeling.\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ Saved comparison report to {summary_path}")
    
    print("\n" + "="*70)
    print("Comparison complete!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Bradley-Terry vs GPM models"
    )
    
    # Model paths
    parser.add_argument("--bt_model_path", type=str, required=True,
                       help="Path to Bradley-Terry model checkpoint")
    parser.add_argument("--gpm_model_path", type=str, required=True,
                       help="Path to GPM model checkpoint")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B",
                       help="Base model name")
    parser.add_argument("--value_head_dim", type=int, default=8,
                       help="GPM value head dimension")
    parser.add_argument("--tau", type=float, default=0.1,
                       help="GPM temperature")
    parser.add_argument("--bf16", action="store_true", default=True)
    
    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to evaluation dataset")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--generate_plots", action="store_true", default=True)
    
    args = parser.parse_args()
    main(args)


"""
Example usage:

python scripts/eval/compare_bt_vs_gpm.py \
    --bt_model_path /path/to/bt_model/checkpoints \
    --gpm_model_path /path/to/gpm_model/checkpoints \
    --base_model Qwen/Qwen2.5-0.5B \
    --value_head_dim 8 \
    --dataset /path/to/test_dataset \
    --output_dir results/bt_vs_gpm_comparison \
    --generate_plots
"""
