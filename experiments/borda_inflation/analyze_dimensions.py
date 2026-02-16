#!/usr/bin/env python3
"""Track A: Analyze Borda inflation using UltraFeedback dimension annotations.

Loads results from compute_inflation.py and analyzes:
1. Sycophancy proxy correlation with inflation
2. Per-dimension correlations
3. Shapira (2026) amplification metrics (Delta_mean, Delta_exp)
4. Per-prompt divergence classification
5. Statistical significance tests

Usage:
    python experiments/borda_inflation/analyze_dimensions.py \
        --results_dir experiments/borda_inflation/results \
        --output_dir experiments/borda_inflation/results/analysis
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, kendalltau, pearsonr
from scipy.special import expit
import statsmodels.api as sm

DIMENSIONS = ["instruction_following", "honesty", "truthfulness", "helpfulness"]


def load_results(results_dir):
    """Load inflation results from compute_inflation.py."""
    results_dir = Path(results_dir)

    # Load flat data
    flat_path = results_dir / "inflation_flat.jsonl"
    print(f"Loading flat data from {flat_path}...")
    records = []
    with open(flat_path) as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    print(f"  {len(df)} responses across {df['prompt'].nunique()} prompts")

    # Load full results (for matrices)
    pkl_path = results_dir / "inflation_results.pkl"
    print(f"Loading full results from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        full_results = pickle.load(f)

    return df, full_results


def compute_sycophancy_proxy(df):
    """Compute sycophancy proxy: helpfulness - min(honesty, truthfulness).

    High value = rated helpful but not honest/truthful.
    """
    help_col = "dim_helpfulness"
    hon_col = "dim_honesty"
    truth_col = "dim_truthfulness"

    has_dims = (
        df[help_col].notna()
        & df[hon_col].notna()
        & df[truth_col].notna()
    )

    df["sycophancy_proxy"] = np.nan
    mask = has_dims
    df.loc[mask, "sycophancy_proxy"] = (
        df.loc[mask, help_col]
        - np.minimum(df.loc[mask, hon_col], df.loc[mask, truth_col])
    )

    n_valid = has_dims.sum()
    print(f"\nSycophancy proxy computed for {n_valid}/{len(df)} responses "
          f"({100*n_valid/len(df):.1f}%)")
    if n_valid > 0:
        print(f"  Mean: {df['sycophancy_proxy'].mean():.3f}")
        print(f"  Std:  {df['sycophancy_proxy'].std():.3f}")
        print(f"  Range: [{df['sycophancy_proxy'].min():.1f}, {df['sycophancy_proxy'].max():.1f}]")

    return df


def analyze_correlations(df):
    """Compute correlations between inflation and features."""
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    features = {
        "sycophancy_proxy": "Sycophancy proxy",
        "length": "Response length (words)",
        "dim_helpfulness": "Helpfulness score",
        "dim_honesty": "Honesty score",
        "dim_truthfulness": "Truthfulness score",
        "dim_instruction_following": "Instruction following score",
    }

    results = {}
    for col, name in features.items():
        if col not in df.columns:
            continue
        valid = df[["inflation", col]].dropna()
        if len(valid) < 10:
            continue

        rho, p_val = spearmanr(valid["inflation"], valid[col])
        r_pearson, p_pearson = pearsonr(valid["inflation"], valid[col])

        results[col] = {
            "name": name,
            "spearman_rho": rho,
            "spearman_p": p_val,
            "pearson_r": r_pearson,
            "pearson_p": p_pearson,
            "n": len(valid),
        }

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {name:35s}  rho={rho:+.4f} (p={p_val:.2e}) {sig}  "
              f"r={r_pearson:+.4f}  n={len(valid)}")

    return results


def run_regression(df):
    """Run OLS regression: inflation ~ features."""
    print("\n" + "=" * 70)
    print("REGRESSION: inflation ~ sycophancy + length + dimensions")
    print("=" * 70)

    feature_cols = ["sycophancy_proxy", "length"] + [f"dim_{d}" for d in DIMENSIONS]
    target = "inflation"

    valid = df[[target] + feature_cols].dropna()
    if len(valid) < 20:
        print("  Insufficient data for regression")
        return None

    X = valid[feature_cols].values
    y = valid[target].values

    # Standardize features for comparable coefficients
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X_norm = (X - X_mean) / X_std

    X_const = sm.add_constant(X_norm)
    model = sm.OLS(y, X_const).fit()

    print(f"\n  R-squared: {model.rsquared:.4f}")
    print(f"  Adj R-sq:  {model.rsquared_adj:.4f}")
    print(f"  N:         {len(valid)}")
    print(f"\n  {'Feature':35s} {'Coef':>8s} {'Std Err':>8s} {'t':>8s} {'p':>10s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for i, col in enumerate(feature_cols):
        coef = model.params[i + 1]  # skip intercept
        se = model.bse[i + 1]
        t = model.tvalues[i + 1]
        p = model.pvalues[i + 1]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {col:35s} {coef:+8.4f} {se:8.4f} {t:8.2f} {p:10.2e} {sig}")

    return model


def compute_shapira_metrics(df, beta=1.0):
    """Compute Shapira (2026) amplification metrics.

    Delta_mean: E[r | high_syc] - E[r | low_syc]
    Delta_exp:  E[exp(r/beta) | high_syc] - E[exp(r/beta) | low_syc]

    Computed for both BT reward and effective reward.
    """
    print("\n" + "=" * 70)
    print("SHAPIRA (2026) AMPLIFICATION METRICS")
    print("=" * 70)

    valid = df[["sycophancy_proxy", "rm_reward", "effective_reward"]].dropna()
    if len(valid) < 20:
        print("  Insufficient data")
        return None

    # Stratify by sycophancy proxy quartiles
    q25 = valid["sycophancy_proxy"].quantile(0.25)
    q75 = valid["sycophancy_proxy"].quantile(0.75)

    low_syc = valid[valid["sycophancy_proxy"] <= q25]
    high_syc = valid[valid["sycophancy_proxy"] >= q75]

    print(f"\n  Low sycophancy (Q1, syc <= {q25:.2f}):  n={len(low_syc)}")
    print(f"  High sycophancy (Q4, syc >= {q75:.2f}): n={len(high_syc)}")

    results = {}

    for reward_name, col in [("BT reward", "rm_reward"), ("Effective reward", "effective_reward")]:
        # Delta_mean
        delta_mean = high_syc[col].mean() - low_syc[col].mean()

        # Delta_exp (exponential moment gap)
        exp_high = np.exp(high_syc[col].values / beta).mean()
        exp_low = np.exp(low_syc[col].values / beta).mean()
        delta_exp = exp_high - exp_low

        results[reward_name] = {
            "delta_mean": delta_mean,
            "delta_exp": delta_exp,
            "mean_high": high_syc[col].mean(),
            "mean_low": low_syc[col].mean(),
        }

        print(f"\n  {reward_name}:")
        print(f"    Mean (high syc): {high_syc[col].mean():+.4f}")
        print(f"    Mean (low syc):  {low_syc[col].mean():+.4f}")
        print(f"    Delta_mean:      {delta_mean:+.4f}")
        print(f"    Delta_exp:       {delta_exp:+.4f} (beta={beta})")

    # Key comparison
    if "BT reward" in results and "Effective reward" in results:
        bt_delta = results["BT reward"]["delta_mean"]
        eff_delta = results["Effective reward"]["delta_mean"]
        ratio = bt_delta / eff_delta if abs(eff_delta) > 1e-8 else float("inf")
        print(f"\n  KEY: Delta_mean_BT / Delta_mean_eff = {ratio:.2f}")
        if ratio > 1.5:
            print(f"  --> BT amplifies sycophancy {ratio:.1f}x more than effective reward")
            print(f"  --> Sycophancy is likely Borda-inflated")
        elif ratio < 0.67:
            print(f"  --> Effective reward amplifies sycophancy more")
            print(f"  --> Sycophancy is likely a genuine preference")
        else:
            print(f"  --> Similar amplification (ratio ~ 1)")

    return results


def classify_prompts(df, full_results):
    """Classify each prompt into agreement/inflated/cycle categories."""
    print("\n" + "=" * 70)
    print("PROMPT CLASSIFICATION")
    print("=" * 70)

    per_prompt = full_results["per_prompt"]

    categories = {"agreement": [], "bt_inflated": [], "cycle": []}

    for r in per_prompt:
        if r["agree"]:
            categories["agreement"].append(r)
        elif r["condorcet_winner"] is None:
            categories["cycle"].append(r)
        else:
            categories["bt_inflated"].append(r)

    total = len(per_prompt)
    for cat, items in categories.items():
        pct = 100 * len(items) / total if total > 0 else 0
        print(f"  {cat:20s}: {len(items):5d} ({pct:5.1f}%)")

    return categories


def analyze_inflated_winners(df, categories):
    """For BT-inflated winners: compare dimension scores with PM winners."""
    print("\n" + "=" * 70)
    print("INFLATED WINNER ANALYSIS")
    print("=" * 70)

    inflated = categories.get("bt_inflated", [])
    if not inflated:
        print("  No BT-inflated winners found")
        return None

    bt_winner_dims = {d: [] for d in DIMENSIONS}
    pm_winner_dims = {d: [] for d in DIMENSIONS}

    for r in inflated:
        bt_idx = r["bt_winner"]
        pm_idx = r["pm_winner"]

        for dim in DIMENSIONS:
            if bt_idx in r["dim_scores"] and r["dim_scores"][bt_idx].get(dim) is not None:
                bt_winner_dims[dim].append(r["dim_scores"][bt_idx][dim])
            if pm_idx in r["dim_scores"] and r["dim_scores"][pm_idx].get(dim) is not None:
                pm_winner_dims[dim].append(r["dim_scores"][pm_idx][dim])

    print(f"\n  Among {len(inflated)} BT-inflated prompts:")
    print(f"  {'Dimension':30s} {'BT winner':>10s} {'PM winner':>10s} {'Diff':>8s} {'p-value':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

    results = {}
    for dim in DIMENSIONS:
        bt_vals = np.array(bt_winner_dims[dim])
        pm_vals = np.array(pm_winner_dims[dim])
        if len(bt_vals) < 5 or len(pm_vals) < 5:
            continue

        bt_mean = bt_vals.mean()
        pm_mean = pm_vals.mean()
        diff = bt_mean - pm_mean

        stat, p_val = mannwhitneyu(bt_vals, pm_vals, alternative="two-sided")
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        results[dim] = {
            "bt_mean": bt_mean,
            "pm_mean": pm_mean,
            "diff": diff,
            "p_value": p_val,
            "n_bt": len(bt_vals),
            "n_pm": len(pm_vals),
        }

        print(f"  {dim:30s} {bt_mean:10.2f} {pm_mean:10.2f} {diff:+8.2f} {p_val:10.2e} {sig}")

    return results


def analyze_length_bias(df):
    """Analyze response length as a confound."""
    print("\n" + "=" * 70)
    print("LENGTH BIAS ANALYSIS")
    print("=" * 70)

    # Correlation between length and rewards
    for col, name in [("rm_reward", "BT reward"), ("effective_reward", "Effective reward")]:
        valid = df[["length", col]].dropna()
        if len(valid) < 10:
            continue
        rho, p = spearmanr(valid["length"], valid[col])
        print(f"  Length vs {name:20s}: rho={rho:+.4f} (p={p:.2e})")

    # Length of BT winners vs PM winners
    bt_winners = df[df["is_bt_winner"]]
    pm_winners = df[df["is_pm_winner"]]

    if len(bt_winners) > 5 and len(pm_winners) > 5:
        print(f"\n  Mean length (BT winners):  {bt_winners['length'].mean():.1f} words")
        print(f"  Mean length (PM winners):  {pm_winners['length'].mean():.1f} words")
        stat, p = mannwhitneyu(bt_winners["length"], pm_winners["length"], alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  Mann-Whitney U: p={p:.2e} {sig}")


def compute_borda_consistency(full_results):
    """Verify that RM Borda (row sums) correlates perfectly with scalar rewards.

    This is a sanity check: for a BT model, row sums of the preference matrix
    should be a monotonic transformation of the scalar rewards.
    """
    print("\n" + "=" * 70)
    print("SANITY CHECK: BORDA CONSISTENCY")
    print("=" * 70)

    correlations = []
    for r in full_results["per_prompt"]:
        if r["n_responses"] < 3:
            continue
        rho, _ = spearmanr(r["rm_rewards"], r["rm_borda"])
        correlations.append(rho)

    correlations = np.array(correlations)
    print(f"  Spearman(r_BT, Borda_RM) across {len(correlations)} prompts:")
    print(f"    Mean:   {correlations.mean():.6f}")
    print(f"    Median: {np.median(correlations):.6f}")
    print(f"    Min:    {correlations.min():.6f}")
    if correlations.mean() > 0.99:
        print(f"  PASS: BT rewards and Borda scores are consistent")
    else:
        print(f"  WARNING: Unexpected inconsistency between BT rewards and Borda scores")


def compute_cross_model_borda(full_results):
    """Compare Borda scores from RM vs PM matrices."""
    print("\n" + "=" * 70)
    print("CROSS-MODEL BORDA COMPARISON")
    print("=" * 70)

    correlations = []
    for r in full_results["per_prompt"]:
        if r["n_responses"] < 3:
            continue
        rho, _ = spearmanr(r["rm_borda"], r["pm_borda"])
        correlations.append(rho)

    correlations = np.array(correlations)
    print(f"  Spearman(Borda_RM, Borda_PM) across {len(correlations)} prompts:")
    print(f"    Mean:   {correlations.mean():.4f}")
    print(f"    Median: {np.median(correlations):.4f}")
    print(f"    Min:    {correlations.min():.4f}")
    print(f"    Std:    {correlations.std():.4f}")

    if correlations.mean() > 0.9:
        print(f"  --> Models agree on aggregate ranking. Divergence is Borda-vs-Condorcet.")
    else:
        print(f"  --> Models disagree on aggregate ranking. Some divergence is model error.")


def main():
    parser = argparse.ArgumentParser(description="Analyze Borda inflation dimensions")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory with compute_inflation.py outputs")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results_dir/analysis)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Beta for Shapira exponential moment (default: 1.0)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or Path(args.results_dir) / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df, full_results = load_results(args.results_dir)

    # Compute sycophancy proxy
    df = compute_sycophancy_proxy(df)

    # Run analyses
    corr_results = analyze_correlations(df)
    regression = run_regression(df)
    shapira_results = compute_shapira_metrics(df, beta=args.beta)
    categories = classify_prompts(df, full_results)
    inflated_results = analyze_inflated_winners(df, categories)
    analyze_length_bias(df)
    compute_borda_consistency(full_results)
    compute_cross_model_borda(full_results)

    # Save analysis results
    analysis_output = {
        "correlations": corr_results,
        "shapira": shapira_results,
        "categories": {k: len(v) for k, v in categories.items()},
        "inflated_winner_dims": inflated_results,
    }

    json_path = output_dir / "dimension_analysis.json"
    with open(json_path, "w") as f:
        json.dump(analysis_output, f, indent=2, default=str)
    print(f"\nSaved analysis to {json_path}")

    # Save enriched dataframe
    csv_path = output_dir / "inflation_with_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved enriched data to {csv_path}")


if __name__ == "__main__":
    main()
