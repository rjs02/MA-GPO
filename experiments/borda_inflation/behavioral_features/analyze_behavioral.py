#!/usr/bin/env python3
"""Analyze correlations between local behavioral features and Borda inflation.

Reads features.jsonl produced by extract_local_features.py.
Computes correlation matrices, conditional means by quartile, winner bias,
and generates figures.

Usage:
    python experiments/borda_inflation/behavioral_features/analyze_behavioral.py \
        --features experiments/borda_inflation/behavioral_features/features.jsonl \
        --output_dir experiments/borda_inflation/behavioral_features
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")


LOCAL_FEATURES = [
    "word_count", "char_count", "sentence_count", "words_per_sentence",
    "n_bold", "n_bullet_items", "n_numbered_items", "n_headers",
    "n_code_blocks", "n_tables", "n_emojis", "formatting_density",
]

OUTCOME_VARS = [
    "inflation_rm", "inflation_pm", "borda_emp", "rm_reward",
]


def load_features(path):
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def compute_correlations(entries, features, outcomes):
    """Pearson and Spearman correlations between features and outcomes."""
    arrays = {}
    for key in features + outcomes:
        arrays[key] = np.array([e[key] for e in entries], dtype=float)

    pearson = {}
    spearman = {}
    for feat in features:
        for out in outcomes:
            x, y = arrays[feat], arrays[out]
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                pearson[(feat, out)] = (0.0, 1.0)
                spearman[(feat, out)] = (0.0, 1.0)
            else:
                pearson[(feat, out)] = pearsonr(x, y)
                spearman[(feat, out)] = spearmanr(x, y)

    return pearson, spearman, arrays


def quartile_means(values, target, labels=None):
    """Compute mean of target within each quartile of values."""
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    edges = [-np.inf, q25, q50, q75, np.inf]
    if labels is None:
        labels = ["Q1", "Q2", "Q3", "Q4"]

    means = []
    stds = []
    counts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (values > lo) & (values <= hi) if lo != -np.inf else (values <= hi)
        if lo == -np.inf:
            mask = values <= q25
        subset = target[mask]
        means.append(float(np.mean(subset)) if len(subset) > 0 else 0.0)
        stds.append(float(np.std(subset)) if len(subset) > 0 else 0.0)
        counts.append(int(len(subset)))

    return labels, means, stds, counts


def winner_bias(entries, feature_name):
    """Compare mean feature value for winners vs non-winners."""
    rm_winner_vals = [e[feature_name] for e in entries if e["is_rm_winner"]]
    rm_other_vals = [e[feature_name] for e in entries if not e["is_rm_winner"]]
    pm_winner_vals = [e[feature_name] for e in entries if e["is_pm_winner"]]
    pm_other_vals = [e[feature_name] for e in entries if not e["is_pm_winner"]]

    return {
        "rm_winner_mean": float(np.mean(rm_winner_vals)) if rm_winner_vals else 0.0,
        "rm_other_mean": float(np.mean(rm_other_vals)) if rm_other_vals else 0.0,
        "pm_winner_mean": float(np.mean(pm_winner_vals)) if pm_winner_vals else 0.0,
        "pm_other_mean": float(np.mean(pm_other_vals)) if pm_other_vals else 0.0,
        "rm_winner_n": len(rm_winner_vals),
        "pm_winner_n": len(pm_winner_vals),
    }


def fig_correlation_heatmap(pearson, spearman, features, outcomes, output_dir, split_label):
    """Heatmap of Spearman correlations."""
    n_feat = len(features)
    n_out = len(outcomes)
    matrix = np.zeros((n_feat, n_out))
    for i, feat in enumerate(features):
        for j, out in enumerate(outcomes):
            matrix[i, j] = spearman[(feat, out)][0]

    fig, ax = plt.subplots(figsize=(8, max(6, n_feat * 0.5)))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    for i in range(n_feat):
        for j in range(n_out):
            r = matrix[i, j]
            p_val = spearman[(features[i], outcomes[j])][1]
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            ax.text(j, i, f"{r:.3f}{sig}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(n_out))
    ax.set_xticklabels(outcomes, rotation=45, ha="right")
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(features)
    ax.set_title(f"Spearman Correlations ({split_label})")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Spearman r")
    plt.tight_layout()
    fig.savefig(output_dir / f"figure_correlation_{split_label}.png", dpi=150)
    plt.close(fig)


def fig_quartile_bars(arrays, features_to_plot, outcomes_to_plot, output_dir, split_label):
    """Bar charts: mean outcome by feature quartile."""
    n_feats = len(features_to_plot)
    n_outs = len(outcomes_to_plot)
    fig, axes = plt.subplots(n_feats, n_outs, figsize=(4 * n_outs, 3.5 * n_feats),
                             squeeze=False)

    for i, feat in enumerate(features_to_plot):
        for j, out in enumerate(outcomes_to_plot):
            ax = axes[i, j]
            labels, means, stds, counts = quartile_means(arrays[feat], arrays[out])
            bars = ax.bar(labels, means, yerr=stds, capsize=3, alpha=0.7,
                          edgecolor="black", linewidth=0.5)
            for k, (m, c) in enumerate(zip(means, counts)):
                ax.text(k, m, f"n={c}", ha="center", va="bottom", fontsize=7)
            ax.set_ylabel(out)
            ax.set_xlabel(f"{feat} quartile")
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
            if i == 0:
                ax.set_title(out)

    plt.suptitle(f"Mean Outcome by Feature Quartile ({split_label})", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / f"figure_quartiles_{split_label}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_scatter(arrays, x_feat, y_outcomes, output_dir, split_label):
    """Scatter plots of a feature vs outcomes."""
    n = len(y_outcomes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, out in zip(axes, y_outcomes):
        ax.scatter(arrays[x_feat], arrays[out], alpha=0.08, s=8, edgecolor="none")
        r_s, p_s = spearmanr(arrays[x_feat], arrays[out])
        ax.set_xlabel(x_feat)
        ax.set_ylabel(out)
        ax.set_title(f"rho={r_s:.3f} (p={p_s:.2e})")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    plt.suptitle(f"{x_feat} vs Outcomes ({split_label})", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / f"figure_scatter_{x_feat}_{split_label}.png", dpi=150)
    plt.close(fig)


def fig_winner_bias(entries, features, output_dir, split_label):
    """Grouped bar chart: feature means for RM/PM winners vs non-winners."""
    bias_data = {f: winner_bias(entries, f) for f in features}

    feats_to_show = [f for f in features if f != "formatting_density"]
    n = len(feats_to_show)
    x = np.arange(n)
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 5))
    rm_winner = [bias_data[f]["rm_winner_mean"] for f in feats_to_show]
    rm_other = [bias_data[f]["rm_other_mean"] for f in feats_to_show]
    pm_winner = [bias_data[f]["pm_winner_mean"] for f in feats_to_show]
    pm_other = [bias_data[f]["pm_other_mean"] for f in feats_to_show]

    ax.bar(x - 1.5 * width, rm_winner, width, label="RM winner", alpha=0.8)
    ax.bar(x - 0.5 * width, rm_other, width, label="RM non-winner", alpha=0.8)
    ax.bar(x + 0.5 * width, pm_winner, width, label="PM winner", alpha=0.8)
    ax.bar(x + 1.5 * width, pm_other, width, label="PM non-winner", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(feats_to_show, rotation=45, ha="right")
    ax.set_ylabel("Mean feature value")
    ax.set_title(f"Winner Bias: Feature Means ({split_label})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / f"figure_winner_bias_{split_label}.png", dpi=150)
    plt.close(fig)

    return bias_data


def analyze_split(entries, split_label, output_dir):
    """Run full analysis for one split."""
    print(f"\n{'='*60}")
    print(f"Analyzing {split_label} ({len(entries)} responses)")
    print(f"{'='*60}")

    pearson, spearman, arrays = compute_correlations(entries, LOCAL_FEATURES, OUTCOME_VARS)

    print(f"\n  Spearman correlations (feature -> outcome):")
    print(f"  {'feature':<22s}  {'inflation_rm':>14s}  {'inflation_pm':>14s}  {'borda_emp':>14s}  {'rm_reward':>14s}")
    for feat in LOCAL_FEATURES:
        vals = []
        for out in OUTCOME_VARS:
            r, p = spearman[(feat, out)]
            sig = "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "   "
            vals.append(f"{r:+.3f}{sig}")
        print(f"  {feat:<22s}  {'  '.join(f'{v:>14s}' for v in vals)}")

    key_features = ["word_count", "formatting_density"]
    key_outcomes = ["inflation_rm", "inflation_pm"]

    print(f"\n  Quartile analysis:")
    for feat in key_features:
        for out in key_outcomes:
            labels, means, stds, counts = quartile_means(arrays[feat], arrays[out])
            mean_str = "  ".join(f"{l}={m:+.3f}" for l, m in zip(labels, means))
            print(f"    {feat} -> {out}: {mean_str}")

    bias_data = fig_winner_bias(entries, LOCAL_FEATURES, output_dir, split_label)

    print(f"\n  Winner bias (mean feature value):")
    for feat in ["word_count", "char_count", "n_bold", "n_bullet_items",
                 "n_numbered_items", "formatting_density"]:
        b = bias_data[feat]
        print(f"    {feat:<22s}  RM_win={b['rm_winner_mean']:.1f}  RM_oth={b['rm_other_mean']:.1f}  "
              f"PM_win={b['pm_winner_mean']:.1f}  PM_oth={b['pm_other_mean']:.1f}")

    fig_correlation_heatmap(pearson, spearman, LOCAL_FEATURES, OUTCOME_VARS, output_dir, split_label)
    fig_quartile_bars(arrays, key_features, key_outcomes, output_dir, split_label)
    fig_scatter(arrays, "word_count", key_outcomes, output_dir, split_label)
    fig_scatter(arrays, "formatting_density", key_outcomes, output_dir, split_label)

    summary = {
        "split": split_label,
        "n_responses": len(entries),
        "spearman": {
            f"{feat}__{out}": {"r": round(spearman[(feat, out)][0], 6),
                               "p": float(spearman[(feat, out)][1])}
            for feat in LOCAL_FEATURES for out in OUTCOME_VARS
        },
        "pearson": {
            f"{feat}__{out}": {"r": round(pearson[(feat, out)][0], 6),
                               "p": float(pearson[(feat, out)][1])}
            for feat in LOCAL_FEATURES for out in OUTCOME_VARS
        },
        "winner_bias": bias_data,
        "quartile_means": {},
    }
    for feat in key_features:
        for out in key_outcomes:
            labels, means, stds, counts = quartile_means(arrays[feat], arrays[out])
            summary["quartile_means"][f"{feat}__{out}"] = {
                "labels": labels, "means": means, "stds": stds, "counts": counts,
            }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze behavioral features")
    parser.add_argument("--features", type=str, required=True,
                        help="Path to features.jsonl")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save figures and summary")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_features(args.features)
    print(f"Loaded {len(entries)} entries from {args.features}")

    seen = [e for e in entries if e["split"] == "seen"]
    unseen = [e for e in entries if e["split"] == "unseen"]
    print(f"  seen={len(seen)}, unseen={len(unseen)}")

    summaries = {}
    for split_entries, label in [(seen, "seen"), (unseen, "unseen"), (entries, "all")]:
        if len(split_entries) == 0:
            continue
        summaries[label] = analyze_split(split_entries, label, output_dir)

    with open(output_dir / "behavioral_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved behavioral_summary.json")
    print("Done.")


if __name__ == "__main__":
    main()
