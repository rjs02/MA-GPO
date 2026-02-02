#!/usr/bin/env python3
"""
Visualization utilities for intransitivity analysis.

Includes:
- Preference heatmaps (sorted by Borda count)
- Calibration curves
- Performance decay plots (for synthetic experiments)
- Per-dimension analysis plots
- Hodge potential landscapes
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path


# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def plot_preference_heatmap(
    preference_matrix: np.ndarray,
    response_labels: Optional[List[str]] = None,
    title: str = "Preference Matrix",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Plot preference heatmap sorted by Borda count.
    
    Transitive preferences appear as upper-triangular matrix.
    Cycles appear as off-diagonal blocks.
    
    Args:
        preference_matrix: P[i,j] = empirical probability that i > j
        response_labels: Optional labels for responses
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
    """
    n = preference_matrix.shape[0]
    
    # Compute Borda count (total wins) for sorting
    borda_counts = np.sum(preference_matrix, axis=1)
    sorted_indices = np.argsort(borda_counts)[::-1]  # Descending order
    
    # Reorder matrix
    sorted_matrix = preference_matrix[sorted_indices, :][:, sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(sorted_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('P(row > col)', rotation=270, labelpad=20)
    
    # Labels
    if response_labels is not None:
        sorted_labels = [response_labels[i] for i in sorted_indices]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(sorted_labels, rotation=90)
        ax.set_yticklabels(sorted_labels)
    
    ax.set_xlabel('Response (sorted by Borda count)')
    ax.set_ylabel('Response (sorted by Borda count)')
    ax.set_title(title)
    
    # Add diagonal line (perfect transitivity would be upper-triangular)
    ax.plot([0, n-1], [0, n-1], 'b--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved preference heatmap to {output_path}")
    
    return fig


def plot_calibration_curve(
    model_probs: np.ndarray,
    empirical_probs: np.ndarray,
    model_name: str = "Model",
    n_bins: int = 10,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
):
    """
    Plot calibration curve: model probability vs empirical win rate.
    
    Perfect calibration lies on the diagonal.
    
    Args:
        model_probs: Model predicted probabilities
        empirical_probs: Empirical win rates from dataset
        model_name: Name for legend
        n_bins: Number of bins for grouping
        output_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Bin predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    mean_predicted = []
    mean_empirical = []
    counts = []
    
    for i in range(n_bins):
        mask = (model_probs >= bin_edges[i]) & (model_probs < bin_edges[i+1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (model_probs >= bin_edges[i]) & (model_probs <= bin_edges[i+1])
        
        if np.any(mask):
            mean_predicted.append(np.mean(model_probs[mask]))
            mean_empirical.append(np.mean(empirical_probs[mask]))
            counts.append(np.sum(mask))
        else:
            mean_predicted.append(bin_centers[i])
            mean_empirical.append(bin_centers[i])
            counts.append(0)
    
    mean_predicted = np.array(mean_predicted)
    mean_empirical = np.array(mean_empirical)
    counts = np.array(counts)
    
    # Plot calibration curve
    ax.plot(mean_predicted, mean_empirical, 'o-', label=model_name, markersize=6, linewidth=2)
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    
    # Add count annotations
    for i, (x, y, c) in enumerate(zip(mean_predicted, mean_empirical, counts)):
        if c > 0:
            ax.annotate(f'{c}', (x, y), textcoords="offset points", 
                       xytext=(0, 5), ha='center', fontsize=7, alpha=0.7)
    
    ax.set_xlabel('Model Probability')
    ax.set_ylabel('Empirical Win Rate')
    ax.set_title('Calibration Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved calibration curve to {output_path}")
    
    return fig


def plot_accuracy_by_consistency(
    accuracy_by_consistency: Dict,
    model_name: str = "Model",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot accuracy stratified by empirical consistency level.
    
    Shows how model performance degrades with increasing intransitivity.
    
    Args:
        accuracy_by_consistency: Dict from evaluate_transitivity_aware.py
        model_name: Name for title
        output_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    bucket_names = []
    accuracies = []
    counts = []
    
    # Define order for buckets
    order = [
        "fully_consistent_1",
        "highly_consistent",
        "moderately_consistent",
        "low_consistency",
        "fully_inconsistent",
        "reversed",
        "fully_consistent_0",
    ]
    
    for name in order:
        if name in accuracy_by_consistency:
            stats = accuracy_by_consistency[name]
            if stats['count'] > 0:
                bucket_names.append(name.replace('_', ' ').title())
                accuracies.append(stats['accuracy'])
                counts.append(stats['count'])
    
    # Plot bars
    x = np.arange(len(bucket_names))
    bars = ax.bar(x, accuracies, alpha=0.7, edgecolor='black')
    
    # Color bars by accuracy
    for bar, acc in zip(bars, accuracies):
        if acc > 0.8:
            bar.set_color('green')
        elif acc > 0.6:
            bar.set_color('yellow')
        elif acc > 0.4:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Add count labels
    for i, (acc, count) in enumerate(zip(accuracies, counts)):
        ax.text(i, acc + 0.02, f'n={count}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Consistency Level')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Accuracy by Consistency Level - {model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved accuracy by consistency to {output_path}")
    
    return fig


def plot_performance_decay(
    results_by_poison_ratio: Dict[float, Dict],
    metrics: List[str] = ['overall_accuracy', 'reward_separability.all_rewards_std'],
    model_labels: Optional[Dict[str, str]] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
):
    """
    Plot performance decay as function of poisoning ratio.
    
    For synthetic experiments comparing BT vs GPM.
    
    Args:
        results_by_poison_ratio: Dict[poison_ratio] -> Dict[model_name] -> results
        metrics: List of metric paths to plot (dot notation for nested)
        model_labels: Optional pretty names for models
        output_path: Path to save figure
        figsize: Figure size
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        for model_name in results_by_poison_ratio[list(results_by_poison_ratio.keys())[0]].keys():
            poison_ratios = []
            metric_values = []
            
            for p_ratio in sorted(results_by_poison_ratio.keys()):
                if model_name in results_by_poison_ratio[p_ratio]:
                    poison_ratios.append(p_ratio)
                    
                    # Navigate nested dict
                    value = results_by_poison_ratio[p_ratio][model_name]
                    for key in metric.split('.'):
                        value = value[key]
                    metric_values.append(value)
            
            label = model_labels.get(model_name, model_name) if model_labels else model_name
            ax.plot(poison_ratios, metric_values, 'o-', label=label, linewidth=2, markersize=6)
        
        ax.set_xlabel('Poisoning Ratio')
        ax.set_ylabel(metric.split('.')[-1].replace('_', ' ').title())
        ax.set_title(metric.split('.')[-1].replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved performance decay plot to {output_path}")
    
    return fig


def plot_comparison_bt_vs_gpm(
    bt_results: Dict,
    gpm_results: Dict,
    output_dir: str,
):
    """
    Create comprehensive comparison plots for BT vs GPM.
    
    Generates multiple plots:
    1. Calibration curves (both models)
    2. Accuracy by consistency (side-by-side)
    3. Metric comparison bar chart
    
    Args:
        bt_results: Results dict from evaluate_transitivity_aware.py
        gpm_results: Results dict from evaluate_transitivity_aware.py
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Calibration curves (both on same plot)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # We need the raw probabilities, which aren't in the results dict
    # So we'll just show a comparison of key metrics instead
    
    # 2. Accuracy by consistency comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # BT accuracy
    bt_acc = bt_results['accuracy_by_consistency']
    order = ["fully_consistent_1", "highly_consistent", "moderately_consistent", 
             "low_consistency", "fully_inconsistent", "reversed"]
    
    bt_names = []
    bt_accs = []
    gpm_accs = []
    
    for name in order:
        if name in bt_acc and bt_acc[name]['count'] > 0:
            bt_names.append(name.replace('_', ' ').title())
            bt_accs.append(bt_acc[name]['accuracy'])
            gpm_accs.append(gpm_results['accuracy_by_consistency'][name]['accuracy'])
    
    x = np.arange(len(bt_names))
    width = 0.35
    
    ax1.bar(x - width/2, bt_accs, width, label='Bradley-Terry', alpha=0.7)
    ax1.bar(x + width/2, gpm_accs, width, label='GPM', alpha=0.7)
    ax1.set_xlabel('Consistency Level')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy by Consistency Level')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bt_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    
    # Metric comparison
    metric_names = ['Overall\nAccuracy', 'Brier\nScore', 'KL Div\n(forward)', 'ECE', 'NLL']
    bt_values = [
        bt_results['overall_accuracy'],
        bt_results['probabilistic_metrics']['brier_score'],
        bt_results['probabilistic_metrics']['kl_forward'],
        bt_results['probabilistic_metrics']['ece'],
        bt_results['probabilistic_metrics']['nll'],
    ]
    gpm_values = [
        gpm_results['overall_accuracy'],
        gpm_results['probabilistic_metrics']['brier_score'],
        gpm_results['probabilistic_metrics']['kl_forward'],
        gpm_results['probabilistic_metrics']['ece'],
        gpm_results['probabilistic_metrics']['nll'],
    ]
    
    x2 = np.arange(len(metric_names))
    ax2.bar(x2 - width/2, bt_values, width, label='Bradley-Terry', alpha=0.7)
    ax2.bar(x2 + width/2, gpm_values, width, label='GPM', alpha=0.7)
    ax2.set_xlabel('Metric')
    ax2.set_ylabel('Value')
    ax2.set_title('Metric Comparison')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metric_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'bt_vs_gpm_comparison.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    
    # 3. Create summary table
    summary_path = Path(output_dir) / 'comparison_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Bradley-Terry vs GPM Comparison\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Overall Accuracy:\n")
        f.write(f"  Bradley-Terry: {bt_results['overall_accuracy']:.4f}\n")
        f.write(f"  GPM:           {gpm_results['overall_accuracy']:.4f}\n")
        f.write(f"  Difference:    {gpm_results['overall_accuracy'] - bt_results['overall_accuracy']:+.4f}\n\n")
        
        f.write(f"Probabilistic Metrics:\n")
        for metric in ['brier_score', 'kl_forward', 'ece', 'nll']:
            bt_val = bt_results['probabilistic_metrics'][metric]
            gpm_val = gpm_results['probabilistic_metrics'][metric]
            f.write(f"  {metric}:\n")
            f.write(f"    Bradley-Terry: {bt_val:.4f}\n")
            f.write(f"    GPM:           {gpm_val:.4f}\n")
            f.write(f"    Difference:    {gpm_val - bt_val:+.4f}\n\n")
        
        if 'reward_separability' in bt_results:
            f.write(f"Reward Separability (Bradley-Terry):\n")
            f.write(f"  All rewards std: {bt_results['reward_separability']['all_rewards_std']:.4f}\n")
            f.write(f"  Collapse warning: {bt_results['reward_separability']['collapse_warning']}\n\n")
        
        if 'embedding_stats' in gpm_results:
            f.write(f"Embedding Statistics (GPM):\n")
            f.write(f"  Avg variance: {gpm_results['embedding_stats']['avg_variance_chosen']:.6f}\n")
            f.write(f"  Collapse warning: {gpm_results['embedding_stats']['collapse_warning']}\n\n")
    
    print(f"Saved comparison summary to {summary_path}")
    
    return fig


def plot_hodge_landscape_2d(
    responses: List[str],
    potential: np.ndarray,
    curl_vectors: Optional[np.ndarray] = None,
    title: str = "Hodge Potential Landscape",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Plot 2D visualization of Hodge potential with curl vectors.
    
    Uses MDS or t-SNE to embed responses in 2D, then shows potential field.
    
    Args:
        responses: List of response identifiers
        potential: Scalar potential from Hodge decomposition
        curl_vectors: Optional curl flow vectors (cycles)
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
    """
    from sklearn.manifold import MDS
    
    n = len(responses)
    
    # Create distance matrix from potential differences
    dist_matrix = np.abs(potential[:, None] - potential[None, :])
    
    # Embed in 2D using MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(dist_matrix)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points colored by potential
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=potential, 
                        cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    plt.colorbar(scatter, ax=ax, label='Potential')
    
    # Add labels
    for i, resp in enumerate(responses):
        label = resp[:20] + "..." if len(resp) > 20 else resp
        ax.annotate(label, (coords[i, 0], coords[i, 1]), 
                   fontsize=7, alpha=0.7, ha='center')
    
    # Add curl vectors if provided
    if curl_vectors is not None:
        # Plot vector field showing cycles
        ax.quiver(coords[:, 0], coords[:, 1], 
                 curl_vectors[:, 0], curl_vectors[:, 1],
                 alpha=0.5, color='red', width=0.003)
    
    ax.set_xlabel('MDS Dimension 1')
    ax.set_ylabel('MDS Dimension 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved Hodge landscape to {output_path}")
    
    return fig


if __name__ == "__main__":
    # Example usage
    print("Visualization utilities for intransitivity analysis")
    print("\nExample usage:")
    print("""
    from scripts.visualization.plot_transitivity import *
    
    # Plot preference heatmap
    pref_matrix = np.random.rand(10, 10)
    plot_preference_heatmap(pref_matrix, output_path='heatmap.png')
    
    # Plot calibration curve
    model_probs = np.random.rand(1000)
    empirical_probs = np.random.rand(1000)
    plot_calibration_curve(model_probs, empirical_probs, output_path='calibration.png')
    """)
