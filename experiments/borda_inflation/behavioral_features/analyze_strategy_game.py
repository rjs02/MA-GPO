"""
Strategy-level game analysis: aggregate per-prompt 4×4 preference matrices
into 7×7 (and 3×3) strategy-level matrices, then compute Borda scores,
logit-transitivity violations, and Condorcet/Nash diagnostics.

Two modes of analysis:
  1. Aggregated: pool all pairwise comparisons across prompts by strategy pair
  2. Prompt-conditioned: compute strategy-level metrics per prompt, then average

Usage:
    python analyze_strategy_game.py \
        --results_dir /path/to/eval_XXXXX \
        --features features_labeled.jsonl \
        --output_dir strategy_analysis
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

# ── Strategy definitions ──

STRATEGY_NAMES = {
    1: "genuinely_helpful",
    2: "sycophantic",
    3: "superficially_polished",
    4: "assertive_confabulation",
    5: "hedging_overcautious",
    6: "evasive_deflective",
    7: "concise_substantive",
}

STRATEGY_SHORT = {
    1: "Helpful",
    2: "Sycophantic",
    3: "Polished",
    4: "Confab.",
    5: "Hedging",
    6: "Evasive",
    7: "Concise",
}

SUBSTANCE_MAP = {
    1: "substance", 2: "surface", 3: "surface", 4: "surface",
    5: "neither", 6: "neither", 7: "substance",
}

SS_NAMES = {0: "substance", 1: "surface", 2: "neither"}
SS_SHORT = {0: "Substance", 1: "Surface", 2: "Neither"}

def strategy_to_ss(sid):
    return {"substance": 0, "surface": 1, "neither": 2}[SUBSTANCE_MAP[sid]]


# ── Core: build aggregated preference matrices ──

def build_aggregated_matrices(prompt_results, prompt_strategies, n_classes, class_fn):
    """
    Build aggregated preference matrices from per-prompt data.

    For each pair of classes (A, B), collect all P[i,j] across prompts where
    response i belongs to class A and response j belongs to class B.

    Args:
        prompt_results: list of dicts with P_emp, P_RM, P_PM (4×4 each)
        prompt_strategies: list of lists (4 strategy labels per prompt)
        n_classes: number of classes (7 or 3)
        class_fn: function mapping strategy_id -> class index

    Returns:
        dict with P_emp, P_RM, P_PM (n_classes × n_classes), plus counts matrix
    """
    # Accumulators: sum of pairwise probs and count of comparisons
    P_emp_sum = np.zeros((n_classes, n_classes))
    P_RM_sum = np.zeros((n_classes, n_classes))
    P_PM_sum = np.zeros((n_classes, n_classes))
    counts = np.zeros((n_classes, n_classes), dtype=int)

    for result, strats in zip(prompt_results, prompt_strategies):
        if strats is None:
            continue
        P_emp = result["P_emp"]
        P_RM = result["P_RM"]
        P_PM = result["P_PM"]
        K = result["K"]

        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                ci = class_fn(strats[i])
                cj = class_fn(strats[j])
                if ci is None or cj is None:
                    continue
                P_emp_sum[ci, cj] += P_emp[i, j]
                P_RM_sum[ci, cj] += P_RM[i, j]
                P_PM_sum[ci, cj] += P_PM[i, j]
                counts[ci, cj] += 1

    # Average
    with np.errstate(divide="ignore", invalid="ignore"):
        P_emp_avg = np.where(counts > 0, P_emp_sum / counts, 0.5)
        P_RM_avg = np.where(counts > 0, P_RM_sum / counts, 0.5)
        P_PM_avg = np.where(counts > 0, P_PM_sum / counts, 0.5)

    # Set diagonal to 0.5
    np.fill_diagonal(P_emp_avg, 0.5)
    np.fill_diagonal(P_RM_avg, 0.5)
    np.fill_diagonal(P_PM_avg, 0.5)

    return {
        "P_emp": P_emp_avg,
        "P_RM": P_RM_avg,
        "P_PM": P_PM_avg,
        "counts": counts,
    }


# ── Social choice metrics on aggregated matrices ──

def borda_scores(P):
    """Row sums minus diagonal (= sum of win rates against all others)."""
    n = P.shape[0]
    return P.sum(axis=1) - 0.5  # subtract diagonal 0.5

def copeland_scores(P):
    """Count of pairwise victories (P[i,j] > 0.5)."""
    n = P.shape[0]
    wins = (P > 0.5).astype(float)
    np.fill_diagonal(wins, 0)
    return wins.sum(axis=1)

def condorcet_winner(P):
    """Return index of Condorcet winner if one exists, else None."""
    n = P.shape[0]
    for i in range(n):
        if all(P[i, j] > 0.5 for j in range(n) if j != i):
            return i
    return None

def logit_transitivity_violations(P):
    """
    For all triples (i,j,k), compute:
        ε = logit(P[i,k]) - logit(P[i,j]) - logit(P[j,k])
    BT requires ε = 0 for all triples.
    Returns list of |ε| values.
    """
    n = P.shape[0]
    eps = 1e-6
    P_clipped = np.clip(P, eps, 1 - eps)
    logit_P = np.log(P_clipped / (1 - P_clipped))

    violations = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i == j or j == k or i == k:
                    continue
                err = logit_P[i, k] - logit_P[i, j] - logit_P[j, k]
                violations.append(abs(err))
    return violations

def bt_mle_from_matrix(P, max_iter=1000, tol=1e-8):
    """Fit BT strengths to a preference matrix via iterative algorithm."""
    n = P.shape[0]
    theta = np.ones(n)
    eps = 1e-6
    P_clipped = np.clip(P, eps, 1 - eps)
    for _ in range(max_iter):
        theta_new = np.zeros(n)
        for i in range(n):
            num = 0
            den = 0
            for j in range(n):
                if i == j:
                    continue
                num += P_clipped[i, j]
                den += (theta[i] + theta[j]) ** (-1) * (P_clipped[i, j] + P_clipped[j, i])
            theta_new[i] = num / (den + eps)
        theta_new = theta_new / theta_new.sum() * n
        if np.max(np.abs(theta_new - theta)) < tol:
            break
        theta = theta_new

    P_bt = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                P_bt[i, j] = 0.5
            else:
                P_bt[i, j] = theta[i] / (theta[i] + theta[j])
    return theta, P_bt

def bt_distortion(P):
    """MAE between P and BT-MLE fit of P."""
    _, P_bt = bt_mle_from_matrix(P)
    mask = ~np.eye(P.shape[0], dtype=bool)
    return float(np.mean(np.abs(P[mask] - P_bt[mask])))


# ── Prompt-conditioned analysis ──

def prompt_conditioned_stats(prompt_results, prompt_strategies, n_classes, class_fn):
    """
    Per-prompt: for each pair of strategies present, record the pairwise probs.
    Then average across prompts to get prompt-conditioned means.

    Also compute per-prompt Borda "inflation" at the strategy level.
    """
    # Per-class: collect inflation values per prompt
    # inflation = borda_RM[class] - borda_emp[class] at the prompt level
    class_inflation_rm = defaultdict(list)
    class_inflation_pm = defaultdict(list)
    class_borda_emp = defaultdict(list)
    class_borda_rm = defaultdict(list)
    class_borda_pm = defaultdict(list)

    for result, strats in zip(prompt_results, prompt_strategies):
        if strats is None:
            continue
        K = result["K"]
        P_emp = result["P_emp"]
        P_RM = result["P_RM"]
        P_PM = result["P_PM"]

        # Per-response borda scores
        borda_emp = P_emp.sum(axis=1) - 0.5
        borda_rm = P_RM.sum(axis=1) - 0.5
        borda_pm = P_PM.sum(axis=1) - 0.5

        for i in range(K):
            ci = class_fn(strats[i])
            if ci is None:
                continue
            class_borda_emp[ci].append(borda_emp[i])
            class_borda_rm[ci].append(borda_rm[i])
            class_borda_pm[ci].append(borda_pm[i])
            class_inflation_rm[ci].append(borda_rm[i] - borda_emp[i])
            class_inflation_pm[ci].append(borda_pm[i] - borda_emp[i])

    result = {}
    for c in range(n_classes):
        result[c] = {
            "n": len(class_borda_emp.get(c, [])),
            "borda_emp_mean": float(np.mean(class_borda_emp[c])) if class_borda_emp[c] else None,
            "borda_rm_mean": float(np.mean(class_borda_rm[c])) if class_borda_rm[c] else None,
            "borda_pm_mean": float(np.mean(class_borda_pm[c])) if class_borda_pm[c] else None,
            "inflation_rm_mean": float(np.mean(class_inflation_rm[c])) if class_inflation_rm[c] else None,
            "inflation_rm_std": float(np.std(class_inflation_rm[c])) if class_inflation_rm[c] else None,
            "inflation_pm_mean": float(np.mean(class_inflation_pm[c])) if class_inflation_pm[c] else None,
            "inflation_pm_std": float(np.std(class_inflation_pm[c])) if class_inflation_pm[c] else None,
        }

    return result


# ── Figures ──

def plot_matrix(P, labels, title, path, counts=None):
    """Plot a preference matrix as a heatmap with annotated values."""
    n = P.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), max(5, n * 0.8)))
    im = ax.imshow(P, cmap="RdBu_r", vmin=0, vmax=1, aspect="equal")

    for i in range(n):
        for j in range(n):
            if i == j:
                txt = "-"
            else:
                txt = f"{P[i,j]:.3f}"
                if counts is not None and counts[i, j] > 0:
                    txt += f"\n({counts[i,j]})"
            color = "white" if abs(P[i, j] - 0.5) > 0.25 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Column (j loses)")
    ax.set_ylabel("Row (i wins)")
    plt.colorbar(im, ax=ax, label="P(row > col)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_borda_comparison(borda_emp, borda_rm, borda_pm, labels, title, path):
    """Grouped bar chart comparing Borda scores across 3 matrices."""
    n = len(labels)
    x = np.arange(n)
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 5))
    ax.bar(x - w, borda_emp, w, label="Empirical", color="#4CAF50", alpha=0.8)
    ax.bar(x, borda_rm, w, label="BT RM", color="#F44336", alpha=0.8)
    ax.bar(x + w, borda_pm, w, label="PM", color="#2196F3", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Borda Score")
    ax.set_title(title)
    ax.legend()
    ax.axhline(y=(n - 1) / 2, color="gray", linestyle="--", alpha=0.3, label="neutral")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_inflation_comparison(prompt_cond, labels, class_indices, title, path):
    """Bar chart of prompt-conditioned mean Borda inflation per strategy."""
    n = len(class_indices)
    x = np.arange(n)
    w = 0.3

    rm_means = [prompt_cond[c]["inflation_rm_mean"] or 0 for c in class_indices]
    pm_means = [prompt_cond[c]["inflation_pm_mean"] or 0 for c in class_indices]
    rm_stds = [prompt_cond[c]["inflation_rm_std"] or 0 for c in class_indices]
    pm_stds = [prompt_cond[c]["inflation_pm_std"] or 0 for c in class_indices]

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 5))
    ax.bar(x - w/2, rm_means, w, yerr=rm_stds, label="RM inflation", color="#F44336", alpha=0.7, capsize=3)
    ax.bar(x + w/2, pm_means, w, yerr=pm_stds, label="PM inflation", color="#2196F3", alpha=0.7, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Borda Inflation (model - empirical)")
    ax.set_title(title)
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ── Main ──

def analyze_split(split_name, results_list, strategies_per_prompt, output_dir):
    """Run full analysis for one split."""
    print(f"\n{'='*70}")
    print(f"  {split_name} split ({len(results_list)} prompts)")
    print(f"{'='*70}")

    # Count labeled prompts
    n_labeled = sum(1 for s in strategies_per_prompt if s is not None)
    n_fully_labeled = sum(1 for s in strategies_per_prompt if s is not None and all(x is not None for x in s))
    print(f"Prompts with labels: {n_labeled} / {len(results_list)}")
    print(f"Fully labeled (all 4 responses): {n_fully_labeled}")

    summary = {"n_prompts": len(results_list), "n_labeled": n_labeled}

    # ── 7×7 strategy analysis ──
    print(f"\n--- 7×7 Strategy-Level Game ---")

    agg7 = build_aggregated_matrices(
        results_list, strategies_per_prompt,
        n_classes=7, class_fn=lambda sid: sid - 1 if sid else None,
    )

    labels7 = [STRATEGY_SHORT[i] for i in range(1, 8)]

    # Print counts matrix
    print(f"\nPairwise comparison counts:")
    for i in range(7):
        row = [f"{agg7['counts'][i,j]:5d}" for j in range(7)]
        print(f"  {labels7[i]:>12s}: {' '.join(row)}")

    # Borda scores
    borda_emp7 = borda_scores(agg7["P_emp"])
    borda_rm7 = borda_scores(agg7["P_RM"])
    borda_pm7 = borda_scores(agg7["P_PM"])

    print(f"\nBorda scores (7×7 aggregated):")
    print(f"  {'Strategy':>12s}  {'Empirical':>9s}  {'BT RM':>9s}  {'PM':>9s}  {'RM-Emp':>9s}  {'PM-Emp':>9s}")
    for i in range(7):
        print(f"  {labels7[i]:>12s}  {borda_emp7[i]:>9.3f}  {borda_rm7[i]:>9.3f}  {borda_pm7[i]:>9.3f}  "
              f"{borda_rm7[i]-borda_emp7[i]:>+9.3f}  {borda_pm7[i]-borda_emp7[i]:>+9.3f}")

    # Copeland
    cope_emp7 = copeland_scores(agg7["P_emp"])
    cope_rm7 = copeland_scores(agg7["P_RM"])
    cope_pm7 = copeland_scores(agg7["P_PM"])

    print(f"\nCopeland scores:")
    print(f"  {'Strategy':>12s}  {'Empirical':>9s}  {'BT RM':>9s}  {'PM':>9s}")
    for i in range(7):
        print(f"  {labels7[i]:>12s}  {cope_emp7[i]:>9.0f}  {cope_rm7[i]:>9.0f}  {cope_pm7[i]:>9.0f}")

    # Condorcet winner
    cw_emp = condorcet_winner(agg7["P_emp"])
    cw_rm = condorcet_winner(agg7["P_RM"])
    cw_pm = condorcet_winner(agg7["P_PM"])
    print(f"\nCondorcet winner: Emp={labels7[cw_emp] if cw_emp is not None else 'None (cycle)'}, "
          f"RM={labels7[cw_rm] if cw_rm is not None else 'None (cycle)'}, "
          f"PM={labels7[cw_pm] if cw_pm is not None else 'None (cycle)'}")

    # Logit-transitivity
    lt_emp = logit_transitivity_violations(agg7["P_emp"])
    lt_rm = logit_transitivity_violations(agg7["P_RM"])
    lt_pm = logit_transitivity_violations(agg7["P_PM"])
    print(f"\nLogit-transitivity |ε| (7×7):")
    print(f"  Empirical: mean={np.mean(lt_emp):.4f}, max={np.max(lt_emp):.4f}")
    print(f"  BT RM:     mean={np.mean(lt_rm):.4f}, max={np.max(lt_rm):.4f}")
    print(f"  PM:        mean={np.mean(lt_pm):.4f}, max={np.max(lt_pm):.4f}")

    # BT distortion
    bt_dist_emp = bt_distortion(agg7["P_emp"])
    bt_dist_pm = bt_distortion(agg7["P_PM"])
    print(f"\nBT distortion (MAE of BT-MLE fit):")
    print(f"  Empirical: {bt_dist_emp:.4f}")
    print(f"  PM:        {bt_dist_pm:.4f}")

    # Store 7×7 results
    summary["aggregated_7x7"] = {
        "P_emp": agg7["P_emp"].tolist(),
        "P_RM": agg7["P_RM"].tolist(),
        "P_PM": agg7["P_PM"].tolist(),
        "counts": agg7["counts"].tolist(),
        "borda_emp": borda_emp7.tolist(),
        "borda_rm": borda_rm7.tolist(),
        "borda_pm": borda_pm7.tolist(),
        "copeland_emp": cope_emp7.tolist(),
        "copeland_rm": cope_rm7.tolist(),
        "copeland_pm": cope_pm7.tolist(),
        "condorcet_emp": cw_emp,
        "condorcet_rm": cw_rm,
        "condorcet_pm": cw_pm,
        "lt_emp_mean": float(np.mean(lt_emp)),
        "lt_rm_mean": float(np.mean(lt_rm)),
        "lt_pm_mean": float(np.mean(lt_pm)),
        "bt_distortion_emp": bt_dist_emp,
        "bt_distortion_pm": bt_dist_pm,
    }

    # Figures
    plot_matrix(agg7["P_emp"], labels7, f"P_emp (7×7, {split_name})",
                output_dir / f"figure_P_emp_7x7_{split_name}.png", agg7["counts"])
    plot_matrix(agg7["P_RM"], labels7, f"P_RM (7×7, {split_name})",
                output_dir / f"figure_P_RM_7x7_{split_name}.png", agg7["counts"])
    plot_matrix(agg7["P_PM"], labels7, f"P_PM (7×7, {split_name})",
                output_dir / f"figure_P_PM_7x7_{split_name}.png", agg7["counts"])
    plot_borda_comparison(borda_emp7, borda_rm7, borda_pm7, labels7,
                          f"Borda Scores (7×7, {split_name})",
                          output_dir / f"figure_borda_7x7_{split_name}.png")

    # ── 3×3 substance/surface/neither analysis ──
    print(f"\n--- 3×3 Substance/Surface/Neither Game ---")

    agg3 = build_aggregated_matrices(
        results_list, strategies_per_prompt,
        n_classes=3, class_fn=lambda sid: strategy_to_ss(sid) if sid else None,
    )

    labels3 = [SS_SHORT[i] for i in range(3)]

    borda_emp3 = borda_scores(agg3["P_emp"])
    borda_rm3 = borda_scores(agg3["P_RM"])
    borda_pm3 = borda_scores(agg3["P_PM"])

    print(f"\nBorda scores (3×3 aggregated):")
    print(f"  {'Category':>12s}  {'Empirical':>9s}  {'BT RM':>9s}  {'PM':>9s}  {'RM-Emp':>9s}  {'PM-Emp':>9s}")
    for i in range(3):
        print(f"  {labels3[i]:>12s}  {borda_emp3[i]:>9.3f}  {borda_rm3[i]:>9.3f}  {borda_pm3[i]:>9.3f}  "
              f"{borda_rm3[i]-borda_emp3[i]:>+9.3f}  {borda_pm3[i]-borda_emp3[i]:>+9.3f}")

    print(f"\nPairwise comparison counts (3×3):")
    for i in range(3):
        row = [f"{agg3['counts'][i,j]:6d}" for j in range(3)]
        print(f"  {labels3[i]:>12s}: {' '.join(row)}")

    lt_emp3 = logit_transitivity_violations(agg3["P_emp"])
    lt_rm3 = logit_transitivity_violations(agg3["P_RM"])
    lt_pm3 = logit_transitivity_violations(agg3["P_PM"])
    print(f"\nLogit-transitivity |ε| (3×3):")
    print(f"  Empirical: mean={np.mean(lt_emp3):.4f}")
    print(f"  BT RM:     mean={np.mean(lt_rm3):.4f}")
    print(f"  PM:        mean={np.mean(lt_pm3):.4f}")

    summary["aggregated_3x3"] = {
        "P_emp": agg3["P_emp"].tolist(),
        "P_RM": agg3["P_RM"].tolist(),
        "P_PM": agg3["P_PM"].tolist(),
        "counts": agg3["counts"].tolist(),
        "borda_emp": borda_emp3.tolist(),
        "borda_rm": borda_rm3.tolist(),
        "borda_pm": borda_pm3.tolist(),
        "lt_emp_mean": float(np.mean(lt_emp3)),
        "lt_rm_mean": float(np.mean(lt_rm3)),
        "lt_pm_mean": float(np.mean(lt_pm3)),
    }

    plot_matrix(agg3["P_emp"], labels3, f"P_emp (3×3, {split_name})",
                output_dir / f"figure_P_emp_3x3_{split_name}.png", agg3["counts"])
    plot_matrix(agg3["P_RM"], labels3, f"P_RM (3×3, {split_name})",
                output_dir / f"figure_P_RM_3x3_{split_name}.png", agg3["counts"])
    plot_matrix(agg3["P_PM"], labels3, f"P_PM (3×3, {split_name})",
                output_dir / f"figure_P_PM_3x3_{split_name}.png", agg3["counts"])
    plot_borda_comparison(borda_emp3, borda_rm3, borda_pm3, labels3,
                          f"Borda Scores (3×3, {split_name})",
                          output_dir / f"figure_borda_3x3_{split_name}.png")

    # ── Prompt-conditioned analysis ──
    print(f"\n--- Prompt-Conditioned Analysis ---")

    pc7 = prompt_conditioned_stats(
        results_list, strategies_per_prompt,
        n_classes=7, class_fn=lambda sid: sid - 1 if sid else None,
    )

    print(f"\nPrompt-conditioned mean Borda inflation (7 strategies):")
    print(f"  {'Strategy':>12s}  {'n':>5s}  {'RM infl':>9s}  {'PM infl':>9s}  {'Borda emp':>9s}")
    for i in range(7):
        info = pc7[i]
        if info["n"] > 0:
            print(f"  {labels7[i]:>12s}  {info['n']:>5d}  {info['inflation_rm_mean']:>+9.4f}  "
                  f"{info['inflation_pm_mean']:>+9.4f}  {info['borda_emp_mean']:>9.3f}")
        else:
            print(f"  {labels7[i]:>12s}  {info['n']:>5d}  {'N/A':>9s}  {'N/A':>9s}  {'N/A':>9s}")

    pc3 = prompt_conditioned_stats(
        results_list, strategies_per_prompt,
        n_classes=3, class_fn=lambda sid: strategy_to_ss(sid) if sid else None,
    )

    print(f"\nPrompt-conditioned mean Borda inflation (3 categories):")
    print(f"  {'Category':>12s}  {'n':>5s}  {'RM infl':>9s}  {'PM infl':>9s}  {'Borda emp':>9s}")
    for i in range(3):
        info = pc3[i]
        if info["n"] > 0:
            print(f"  {labels3[i]:>12s}  {info['n']:>5d}  {info['inflation_rm_mean']:>+9.4f}  "
                  f"{info['inflation_pm_mean']:>+9.4f}  {info['borda_emp_mean']:>9.3f}")

    summary["prompt_conditioned_7"] = {
        str(k): v for k, v in pc7.items()
    }
    summary["prompt_conditioned_3"] = {
        str(k): v for k, v in pc3.items()
    }

    plot_inflation_comparison(pc7, labels7, list(range(7)),
                              f"Prompt-Conditioned Borda Inflation (7 strategies, {split_name})",
                              output_dir / f"figure_inflation_prompt_cond_7x7_{split_name}.png")
    plot_inflation_comparison(pc3, labels3, list(range(3)),
                              f"Prompt-Conditioned Borda Inflation (3 categories, {split_name})",
                              output_dir / f"figure_inflation_prompt_cond_3x3_{split_name}.png")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Strategy-Level Game Analysis")
    parser.add_argument("--results_dir", required=True,
                        help="Path to eval results dir (has analysis_seen/, analysis_unseen/)")
    parser.add_argument("--features", required=True,
                        help="Path to features_labeled.jsonl (with strategy_id)")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save figures and summary")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)

    # Load features with labels
    with open(args.features) as f:
        features = [json.loads(line) for line in f]

    # Split into seen/unseen, group by prompt (every 4 entries)
    seen_features = [e for e in features if e["split"] == "seen"]
    unseen_features = [e for e in features if e["split"] == "unseen"]

    def get_strategies_per_prompt(feat_list):
        """Group features by prompt (4 responses each), extract strategy labels."""
        n_prompts = len(feat_list) // 4
        strategies = []
        for p in range(n_prompts):
            group = feat_list[p * 4 : (p + 1) * 4]
            if all("strategy_id" in e for e in group):
                strategies.append([e["strategy_id"] for e in group])
            else:
                strategies.append(None)
        return strategies

    seen_strats = get_strategies_per_prompt(seen_features)
    unseen_strats = get_strategies_per_prompt(unseen_features)

    # Load results.pkl
    full_summary = {}

    for split_name, strats in [("seen", seen_strats), ("unseen", unseen_strats)]:
        pkl_path = results_dir / f"analysis_{split_name}" / "results.pkl"
        if not pkl_path.exists():
            print(f"WARNING: {pkl_path} not found, skipping {split_name}")
            continue

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        results_list = data["results"]

        assert len(results_list) == len(strats), \
            f"Mismatch: {len(results_list)} prompts in results vs {len(strats)} in features"

        full_summary[split_name] = analyze_split(split_name, results_list, strats, output_dir)

    # Combined analysis (merge both splits)
    all_results = []
    all_strats = []
    for split_name, strats in [("seen", seen_strats), ("unseen", unseen_strats)]:
        pkl_path = results_dir / f"analysis_{split_name}" / "results.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            all_results.extend(data["results"])
            all_strats.extend(strats)

    if all_results:
        full_summary["all"] = analyze_split("all", all_results, all_strats, output_dir)

    # Save summary
    with open(output_dir / "strategy_game_summary.json", "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'strategy_game_summary.json'}")
    print(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
