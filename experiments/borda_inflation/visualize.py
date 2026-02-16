#!/usr/bin/env python3
"""Visualize Borda inflation analysis results.

Generates all key plots for the experiment:
1. Scatter: r_BT vs r_eff colored by sycophancy proxy
2. Bar chart: prompt classification categories
3. Correlation heatmap: inflation vs features
4. Box plots: dimension scores for BT-top vs PM-top responses
5. Shapira comparison: Delta metrics for BT vs effective reward
6. SAE feature examples (if available)

Usage:
    python experiments/borda_inflation/visualize.py \
        --results_dir experiments/borda_inflation/results \
        --output_dir experiments/borda_inflation/results/figures
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

DIMENSIONS = ["instruction_following", "honesty", "truthfulness", "helpfulness"]

# Style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})


def load_data(results_dir):
    """Load all analysis data."""
    results_dir = Path(results_dir)

    # Flat data
    jsonl_path = results_dir / "inflation_flat.jsonl"
    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)

    # Full results
    pkl_path = results_dir / "inflation_results.pkl"
    with open(pkl_path, "rb") as f:
        full = pickle.load(f)

    # Dimension analysis (if available)
    analysis_path = results_dir / "analysis" / "dimension_analysis.json"
    analysis = None
    if analysis_path.exists():
        with open(analysis_path) as f:
            analysis = json.load(f)

    # SAE analysis (if available)
    sae_path = results_dir / "sae_analysis" / "sae_correlations.json"
    sae_data = None
    if sae_path.exists():
        with open(sae_path) as f:
            sae_data = json.load(f)

    return df, full, analysis, sae_data


def plot_reward_scatter(df, output_dir):
    """Scatter plot: r_BT vs r_eff, colored by sycophancy proxy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: colored by sycophancy proxy
    ax = axes[0]
    has_syc = df["dim_helpfulness"].notna() & df["dim_honesty"].notna() & df["dim_truthfulness"].notna()
    df_syc = df[has_syc].copy()
    df_syc["syc"] = df_syc["dim_helpfulness"] - np.minimum(df_syc["dim_honesty"], df_syc["dim_truthfulness"])

    sc = ax.scatter(
        df_syc["rm_reward"], df_syc["effective_reward"],
        c=df_syc["syc"], cmap="RdYlBu_r", alpha=0.5, s=8,
        vmin=df_syc["syc"].quantile(0.05), vmax=df_syc["syc"].quantile(0.95),
    )
    plt.colorbar(sc, ax=ax, label="Sycophancy proxy")

    # Add diagonal
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.5)

    ax.set_xlabel("BT reward (r_BT)")
    ax.set_ylabel("Effective reward (r_eff)")
    ax.set_title("BT reward vs Effective reward\n(colored by sycophancy proxy)")

    # Right: colored by response length
    ax = axes[1]
    sc = ax.scatter(
        df["rm_reward"], df["effective_reward"],
        c=np.log1p(df["length"]), cmap="viridis", alpha=0.5, s=8,
    )
    plt.colorbar(sc, ax=ax, label="log(length)")
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.5)
    ax.set_xlabel("BT reward (r_BT)")
    ax.set_ylabel("Effective reward (r_eff)")
    ax.set_title("BT reward vs Effective reward\n(colored by response length)")

    plt.tight_layout()
    fig.savefig(output_dir / "reward_scatter.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved reward_scatter.png")


def plot_category_distribution(full, output_dir):
    """Bar chart: prompt classification categories."""
    per_prompt = full["per_prompt"]

    cats = {"Agreement\n(BT=PM)": 0, "BT-inflated\nwinner": 0, "Cycle\n(no winner)": 0}
    for r in per_prompt:
        if r["agree"]:
            cats["Agreement\n(BT=PM)"] += 1
        elif r["condorcet_winner"] is None:
            cats["Cycle\n(no winner)"] += 1
        else:
            cats["BT-inflated\nwinner"] += 1

    total = len(per_prompt)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71", "#e74c3c", "#f39c12"]
    bars = ax.bar(cats.keys(), cats.values(), color=colors, edgecolor="white", linewidth=1.5)

    for bar, count in zip(bars, cats.values()):
        pct = 100 * count / total
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Number of prompts")
    ax.set_title(f"Prompt Classification (n={total})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "category_distribution.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved category_distribution.png")


def plot_correlation_heatmap(df, sae_data, output_dir):
    """Heatmap of feature correlations with inflation."""
    features = {}

    # Dimension features
    for dim in DIMENSIONS:
        col = f"dim_{dim}"
        if col in df.columns:
            valid = df[["inflation", col]].dropna()
            if len(valid) > 10:
                rho, _ = spearmanr(valid["inflation"], valid[col])
                features[dim.replace("_", "\n")] = rho

    # Length
    rho, _ = spearmanr(df["inflation"], df["length"])
    features["length"] = rho

    # Sycophancy proxy
    has_syc = df["dim_helpfulness"].notna() & df["dim_honesty"].notna() & df["dim_truthfulness"].notna()
    df_syc = df[has_syc].copy()
    df_syc["syc"] = df_syc["dim_helpfulness"] - np.minimum(df_syc["dim_honesty"], df_syc["dim_truthfulness"])
    if len(df_syc) > 10:
        rho, _ = spearmanr(df_syc["inflation"], df_syc["syc"])
        features["sycophancy\nproxy"] = rho

    # SAE features (top 5 if available)
    if sae_data:
        for i, feat in enumerate(sae_data[:5]):
            features[f"SAE #{feat['feature_idx']}"] = feat["spearman_rho"]

    if not features:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(features) * 0.8), 4))
    names = list(features.keys())
    values = list(features.values())
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]

    bars = ax.barh(names, values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Spearman correlation with Borda inflation")
    ax.set_title("Feature Correlations with Borda Inflation")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved correlation_heatmap.png")


def plot_winner_comparison(df, output_dir):
    """Box plots: dimension scores for BT-top vs PM-top responses."""
    bt_winners = df[df["is_bt_winner"]].copy()
    pm_winners = df[df["is_pm_winner"]].copy()

    # Only where they disagree
    disagree_prompts = df[~df["prompt_agrees"]]["prompt"].unique()
    bt_w = bt_winners[bt_winners["prompt"].isin(disagree_prompts)]
    pm_w = pm_winners[pm_winners["prompt"].isin(disagree_prompts)]

    if len(bt_w) < 5 or len(pm_w) < 5:
        print("  Not enough disagreeing prompts for winner comparison plot")
        return

    fig, axes = plt.subplots(1, len(DIMENSIONS) + 1, figsize=(16, 5))

    for i, dim in enumerate(DIMENSIONS):
        ax = axes[i]
        col = f"dim_{dim}"
        bt_vals = bt_w[col].dropna()
        pm_vals = pm_w[col].dropna()

        if len(bt_vals) > 0 and len(pm_vals) > 0:
            bp = ax.boxplot([bt_vals, pm_vals], labels=["BT\nwinner", "PM\nwinner"],
                           patch_artist=True, widths=0.6)
            bp["boxes"][0].set_facecolor("#e74c3c")
            bp["boxes"][1].set_facecolor("#3498db")
            bp["boxes"][0].set_alpha(0.7)
            bp["boxes"][1].set_alpha(0.7)

        ax.set_title(dim.replace("_", "\n"), fontsize=9)
        ax.set_ylabel("Score" if i == 0 else "")

    # Length comparison
    ax = axes[-1]
    bt_lens = bt_w["length"].dropna()
    pm_lens = pm_w["length"].dropna()
    if len(bt_lens) > 0 and len(pm_lens) > 0:
        bp = ax.boxplot([bt_lens, pm_lens], labels=["BT\nwinner", "PM\nwinner"],
                       patch_artist=True, widths=0.6)
        bp["boxes"][0].set_facecolor("#e74c3c")
        bp["boxes"][1].set_facecolor("#3498db")
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_alpha(0.7)
    ax.set_title("Length\n(words)", fontsize=9)

    fig.suptitle(f"BT vs PM Winners on Disagreeing Prompts (n={len(disagree_prompts)})",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "winner_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved winner_comparison.png")


def plot_shapira_comparison(analysis, output_dir):
    """Shapira Delta metrics: BT vs effective reward."""
    if analysis is None or "shapira" not in analysis or analysis["shapira"] is None:
        print("  No Shapira analysis data, skipping")
        return

    shapira = analysis["shapira"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Delta_mean
    ax = axes[0]
    names = list(shapira.keys())
    delta_means = [shapira[n]["delta_mean"] for n in names]
    colors = ["#e74c3c", "#3498db"]
    ax.bar(names, delta_means, color=colors[:len(names)], edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Delta_mean")
    ax.set_title("Mean Reward Gap\n(high_syc - low_syc)")
    ax.axhline(0, color="black", linewidth=0.5)

    # Delta_exp
    ax = axes[1]
    delta_exps = [shapira[n]["delta_exp"] for n in names]
    ax.bar(names, delta_exps, color=colors[:len(names)], edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Delta_exp")
    ax.set_title("Exponential Moment Gap\n(high_syc - low_syc)")
    ax.axhline(0, color="black", linewidth=0.5)

    fig.suptitle("Shapira (2026) Amplification Metrics", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "shapira_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved shapira_comparison.png")


def plot_inflation_distribution(df, output_dir):
    """Distribution of inflation scores."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of inflation scores
    ax = axes[0]
    ax.hist(df["inflation"], bins=np.arange(df["inflation"].min() - 0.5, df["inflation"].max() + 1.5),
            edgecolor="white", color="#3498db", alpha=0.8)
    ax.set_xlabel("Inflation score (rank_eff - rank_BT)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Borda Inflation Scores")
    ax.axvline(0, color="red", linewidth=1, linestyle="--", label="No inflation")
    ax.legend()

    # Scatter: BT rank vs effective rank
    ax = axes[1]
    jitter = np.random.RandomState(42).normal(0, 0.1, len(df))
    ax.scatter(df["rm_rank"] + jitter, df["eff_rank"] + jitter, alpha=0.3, s=5, c="#3498db")
    max_rank = max(df["rm_rank"].max(), df["eff_rank"].max())
    ax.plot([0, max_rank], [0, max_rank], "k--", alpha=0.5, linewidth=1)
    ax.set_xlabel("BT rank (0=best)")
    ax.set_ylabel("Effective reward rank (0=best)")
    ax.set_title("BT Rank vs Effective Reward Rank")

    plt.tight_layout()
    fig.savefig(output_dir / "inflation_distribution.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved inflation_distribution.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize Borda inflation results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory with compute_inflation.py outputs")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for figures (default: results_dir/figures)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or Path(args.results_dir) / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, full, analysis, sae_data = load_data(args.results_dir)

    print(f"\nGenerating plots to {output_dir}...")
    plot_reward_scatter(df, output_dir)
    plot_category_distribution(full, output_dir)
    plot_correlation_heatmap(df, sae_data, output_dir)
    plot_winner_comparison(df, output_dir)
    plot_shapira_comparison(analysis, output_dir)
    plot_inflation_distribution(df, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
