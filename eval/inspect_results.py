#!/usr/bin/env python3
"""Quick inspection tool for evaluation results."""

import argparse
import json
from pathlib import Path
import numpy as np


def inspect_results(results_path):
    """Load and print summary of evaluation results."""
    results_path = Path(results_path)
    
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        return
    
    with open(results_path) as f:
        results = json.load(f)
    
    print("="*70)
    print("EVALUATION RESULTS SUMMARY")
    print("="*70)
    
    # Summary statistics
    summary = results.get('summary', {})
    
    print("\n" + "-"*70)
    print("ACCURACY METRICS")
    print("-"*70)
    print(f"RM Mean Accuracy:   {summary.get('rm_mean_accuracy', 0):.4f}")
    print(f"GPO Mean Accuracy:  {summary.get('gpo_mean_accuracy', 0):.4f}")
    print(f"GPO Advantage:      {summary.get('accuracy_gap', 0):+.4f}")
    
    print("\n" + "-"*70)
    print("PROBABILISTIC METRICS (lower is better)")
    print("-"*70)
    print(f"KL Divergence:")
    print(f"  RM:  {summary.get('rm_mean_kl_div', 0):.4f}")
    print(f"  GPO: {summary.get('gpo_mean_kl_div', 0):.4f}")
    print(f"  Gap: {summary.get('kl_div_gap', 0):+.4f} (positive = GPO better)")
    
    print(f"\nNLL:")
    print(f"  RM:  {summary.get('rm_mean_nll', 0):.4f}")
    print(f"  GPO: {summary.get('gpo_mean_nll', 0):.4f}")
    print(f"  Gap: {summary.get('nll_gap', 0):+.4f} (positive = GPO better)")
    
    print(f"\nBrier Score:")
    print(f"  RM:  {summary.get('rm_mean_brier', 0):.4f}")
    print(f"  GPO: {summary.get('gpo_mean_brier', 0):.4f}")
    print(f"  Gap: {summary.get('brier_gap', 0):+.4f} (positive = GPO better)")
    
    print(f"\nExpected Calibration Error:")
    print(f"  RM:  {summary.get('rm_mean_ece', 0):.4f}")
    print(f"  GPO: {summary.get('gpo_mean_ece', 0):.4f}")
    print(f"  Gap: {summary.get('ece_gap', 0):+.4f} (positive = GPO better)")
    
    print("\n" + "-"*70)
    print("CONFIDENCE & DIVERSITY")
    print("-"*70)
    print(f"Avg Confidence:")
    print(f"  RM:  {summary.get('rm_mean_avg_confidence', 0):.4f}")
    print(f"  GPO: {summary.get('gpo_mean_avg_confidence', 0):.4f}")
    
    print(f"\nEntropy (higher = less collapsed):")
    print(f"  RM:  {summary.get('rm_mean_entropy', 0):.4f}")
    print(f"  GPO: {summary.get('gpo_mean_entropy', 0):.4f}")
    print(f"  Gap: {summary.get('entropy_gap', 0):+.4f} (positive = GPO less collapsed)")
    
    print("\n" + "-"*70)
    print("INTRANSITIVITY CAPTURE")
    print("-"*70)
    print(f"Pred Loop Ratio:")
    print(f"  RM:  {summary.get('rm_mean_pred_loop_ratio', 0):.4f}")
    print(f"  GPO: {summary.get('gpo_mean_pred_loop_ratio', 0):.4f}")
    print(f"  Gap: {summary.get('pred_loop_ratio_gap', 0):+.4f} (GPO captures more curl)")
    
    if 'mean_data_loop_ratio' in summary:
        print(f"\nData Loop Ratio: {summary.get('mean_data_loop_ratio', 0):.4f}")
    
    # Data statistics
    rm_results = results.get('rm', {})
    gpo_results = results.get('gpo', {})
    data_metrics = results.get('data_metrics', {})
    
    n_prompts = rm_results.get('n_prompts_evaluated', 0)
    print("\n" + "-"*70)
    print("DATASET STATISTICS")
    print("-"*70)
    print(f"Prompts Evaluated: {n_prompts}")
    
    if data_metrics:
        loop_ratios = [m['loop_ratio'] for m in data_metrics.values()]
        print(f"Mean Data Loop Ratio: {np.mean(loop_ratios):.4f} ± {np.std(loop_ratios):.4f}")
        print(f"Range: [{np.min(loop_ratios):.4f}, {np.max(loop_ratios):.4f}]")
    
    # Per-prompt analysis
    print("\n" + "-"*70)
    print("TOP 5 PROMPTS BY GPO ADVANTAGE")
    print("-"*70)
    
    rm_prompt_results = rm_results.get('prompt_results', {})
    gpo_prompt_results = gpo_results.get('prompt_results', {})
    
    gaps = []
    for prompt in rm_prompt_results.keys():
        if prompt in gpo_prompt_results:
            gap = gpo_prompt_results[prompt]['accuracy'] - rm_prompt_results[prompt]['accuracy']
            gaps.append((prompt, gap, 
                        rm_prompt_results[prompt]['accuracy'],
                        gpo_prompt_results[prompt]['accuracy']))
    
    gaps.sort(key=lambda x: x[1], reverse=True)
    
    for i, (prompt, gap, rm_acc, gpo_acc) in enumerate(gaps[:5], 1):
        print(f"{i}. Gap={gap:+.3f}  RM={rm_acc:.3f}  GPO={gpo_acc:.3f}")
        print(f"   {prompt[:80]}...")
        print()
    
    print("-"*70)
    print("TOP 5 PROMPTS BY DATA INTRANSITIVITY")
    print("-"*70)
    
    intrans_prompts = []
    for prompt in rm_prompt_results.keys():
        if prompt in data_metrics:
            loop_ratio = data_metrics[prompt]['loop_ratio']
            intrans_prompts.append((
                prompt, loop_ratio,
                rm_prompt_results[prompt]['accuracy'],
                gpo_prompt_results[prompt]['accuracy']
            ))
    
    intrans_prompts.sort(key=lambda x: x[1], reverse=True)
    
    for i, (prompt, loop, rm_acc, gpo_acc) in enumerate(intrans_prompts[:5], 1):
        gap = gpo_acc - rm_acc
        print(f"{i}. Loop={loop:.3f}  Gap={gap:+.3f}  RM={rm_acc:.3f}  GPO={gpo_acc:.3f}")
        print(f"   {prompt[:80]}...")
        print()
    
    # Hypothesis test
    print("="*70)
    print("CAPTURE HYPOTHESIS TEST")
    print("="*70)
    
    # Correlation: data loop ratio vs pred loop ratio
    if data_metrics and rm_prompt_results and gpo_prompt_results:
        common_prompts = set(data_metrics.keys()) & set(rm_prompt_results.keys()) & set(gpo_prompt_results.keys())
        
        data_loops = [data_metrics[p]['loop_ratio'] for p in common_prompts]
        rm_pred_loops = [rm_prompt_results[p]['pred_loop_ratio'] for p in common_prompts]
        gpo_pred_loops = [gpo_prompt_results[p]['pred_loop_ratio'] for p in common_prompts]
        
        # Pearson correlation
        r_rm = np.corrcoef(data_loops, rm_pred_loops)[0, 1]
        r_gpo = np.corrcoef(data_loops, gpo_pred_loops)[0, 1]
        
        print(f"Correlation: data loop ratio vs predicted loop ratio")
        print(f"  RM:  r = {r_rm:.4f}")
        print(f"  GPO: r = {r_gpo:.4f}")
        print()
        
        if abs(r_gpo) > abs(r_rm):
            print("✓ GPO captures more intransitivity structure than RM")
        else:
            print("✗ RM captures similar or more structure (unexpected)")
    
    print("\n" + "="*70)
    print(f"Full results: {results_path}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect evaluation results")
    parser.add_argument(
        "results_path",
        type=str,
        help="Path to evaluation_results.json"
    )
    args = parser.parse_args()
    
    inspect_results(args.results_path)
