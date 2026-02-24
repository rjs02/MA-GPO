#!/usr/bin/env python3
"""Analyze Borda inflation from preprocessed eval data + inference results.

Runs locally (no GPU needed). Computes social choice metrics, cross-prompt
inflation scores, calibration, logit-transitivity violations, and generates
figures.

Usage:
    python experiments/borda_inflation/analyze_inflation.py \
        --eval_data /path/to/eval_seen.json \
        --inference_data /path/to/inference_seen.pkl \
        --output_dir /path/to/analysis_seen
"""

import argparse
import json
import pickle
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DIMENSIONS = ["instruction_following", "honesty", "truthfulness", "helpfulness"]


# ── Social choice functions (from tabular_inflation.py) ──────────────────────

def borda_scores(P):
    """Borda score = row sum of preference matrix (excluding diagonal 0.5)."""
    return P.sum(axis=1) - 0.5


def condorcet_winner(P):
    """Find Condorcet winner: beats all others pairwise (P[i,j] > 0.5)."""
    n = P.shape[0]
    for i in range(n):
        if all(P[i, j] > 0.5 for j in range(n) if j != i):
            return i
    return None


def copeland_scores(P):
    """Copeland score = number of pairwise victories."""
    n = P.shape[0]
    scores = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j and P[i, j] > 0.5:
                scores[i] += 1
            elif i != j and P[i, j] == 0.5:
                scores[i] += 0.5
    return scores


def nash_equilibrium_lp(P):
    """Nash equilibrium (maximal lottery) via LP + max-entropy tiebreaking."""
    from scipy.optimize import linprog, minimize as sp_minimize

    M = P - 0.5
    n = M.shape[0]

    if n == 1:
        return np.array([1.0]), 0.0

    c = np.zeros(n + 1)
    c[-1] = -1.0

    A_ub = np.zeros((n, n + 1))
    A_ub[:, :n] = -M.T
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(n)

    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0, None)] * n + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if not result.success:
        return np.ones(n) / n, 0.0

    v_star = -result.fun

    constraints = [
        {'type': 'eq', 'fun': lambda p: p.sum() - 1.0},
    ]
    for j in range(n):
        mj = M[:, j].copy()
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, m=mj: m @ p - v_star + 1e-9,
        })

    pi0 = np.maximum(result.x[:n], 1e-6)
    pi0 /= pi0.sum()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp_minimize(
            lambda p: np.sum(p * np.log(np.maximum(p, 1e-30))),
            pi0, method='SLSQP',
            jac=lambda p: np.log(np.maximum(p, 1e-30)) + 1,
            bounds=[(1e-12, 1.0)] * n,
            constraints=constraints,
            options={'maxiter': 500},
        )

    pi = res.x
    pi[pi < 1e-8] = 0
    if pi.sum() > 0:
        pi /= pi.sum()
    else:
        pi = np.ones(n) / n

    return pi, v_star


def bt_mle_from_matrix(P, max_iter=500, tol=1e-10):
    """BT-MLE from a win-rate matrix. Returns strength parameters theta."""
    n = P.shape[0]
    if n == 1:
        return np.array([1.0])

    theta = np.ones(n) / n
    wins = P.sum(axis=1) - 0.5

    for _ in range(max_iter):
        theta_old = theta.copy()
        pairwise_sum = theta[:, None] + theta[None, :]
        inv_sum = 1.0 / pairwise_sum
        np.fill_diagonal(inv_sum, 0)
        denom = inv_sum.sum(axis=1)
        theta = np.where(denom > 0, wins / np.maximum(denom, 1e-10), 1.0 / n)
        theta = np.maximum(theta, 1e-10)
        theta /= theta.sum()
        if np.max(np.abs(theta - theta_old)) < tol:
            break

    return theta


def bt_pref_matrix(theta):
    """Build P from BT strength parameters: P[i,j] = theta_i / (theta_i + theta_j)."""
    n = len(theta)
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                P[i, j] = 0.5
            else:
                P[i, j] = theta[i] / (theta[i] + theta[j])
    return P


def ranking_from_scores(scores):
    """Convert scores to ranks (0 = best)."""
    order = np.argsort(-np.asarray(scores))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    return ranks


def safe_logit(p, eps=0.01):
    """Logit with clipping."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


# ── Analysis functions ───────────────────────────────────────────────────────

def analyze_prompt_data(eval_rec, infer_rec):
    """Compute all per-prompt metrics. Returns a result dict."""
    P_emp = np.array(eval_rec["P_emp"])
    P_RM = infer_rec["P_RM"]
    P_PM = infer_rec["P_PM"]
    rm_rewards = infer_rec["rm_rewards"]
    K = P_emp.shape[0]

    # Empirical social choice
    borda_emp = borda_scores(P_emp)
    cope_emp = copeland_scores(P_emp)
    cw = condorcet_winner(P_emp)
    nash_pi, nash_val = nash_equilibrium_lp(P_emp)

    # BT-MLE from empirical (prompt-local)
    theta_bt = bt_mle_from_matrix(P_emp)
    P_bt = bt_pref_matrix(theta_bt)
    borda_bt = borda_scores(P_bt)
    # Convert theta to log-rewards (centered)
    r_bt = np.log(np.maximum(theta_bt, 1e-12))
    r_bt -= r_bt.mean()

    # RM Borda
    borda_rm = borda_scores(P_RM)

    # PM Borda
    borda_pm = borda_scores(P_PM)

    # Rankings
    rank_emp_borda = ranking_from_scores(borda_emp)
    rank_emp_cope = ranking_from_scores(cope_emp)
    rank_nash = ranking_from_scores(nash_pi)  # highest pi = rank 0
    rank_rm = ranking_from_scores(borda_rm)
    rank_pm = ranking_from_scores(borda_pm)

    # Winners
    winner_emp_borda = int(np.argmax(borda_emp))
    winner_emp_cope = int(np.argmax(cope_emp))
    winner_nash = int(np.argmax(nash_pi))
    winner_rm = int(np.argmax(borda_rm))
    winner_pm = int(np.argmax(borda_pm))

    # Cross-prompt inflation: rank_model - rank_emp_Borda per response
    inflation = rank_rm.astype(int) - rank_emp_borda.astype(int)
    inflation_pm = rank_pm.astype(int) - rank_emp_borda.astype(int)

    # Brier scores (off-diagonal pairs)
    pairs_rm_emp = []
    pairs_pm_emp = []
    pairs_rm_bt = []
    for i in range(K):
        for j in range(K):
            if i != j:
                pairs_rm_emp.append((P_RM[i, j] - P_emp[i, j]) ** 2)
                pairs_pm_emp.append((P_PM[i, j] - P_emp[i, j]) ** 2)
                pairs_rm_bt.append((P_RM[i, j] - P_bt[i, j]) ** 2)

    brier_rm = np.mean(pairs_rm_emp)
    brier_pm = np.mean(pairs_pm_emp)

    # RM vs local BT: P-space and r-space
    p_dist_rm_bt = np.mean(np.abs(P_RM - P_bt))
    r_dist_rm_bt = np.linalg.norm(rm_rewards - r_bt)

    # BT distortion of PM
    theta_pm_fit = bt_mle_from_matrix(P_PM)
    P_pm_bt_fit = bt_pref_matrix(theta_pm_fit)
    bt_distortion_pm = np.mean(np.abs(P_pm_bt_fit - P_PM))

    # Logit-transitivity violations
    lt_pm = []
    lt_emp = []
    for i, j, k in combinations(range(K), 3):
        eps_pm = safe_logit(P_PM[i, k]) - safe_logit(P_PM[i, j]) - safe_logit(P_PM[j, k])
        lt_pm.append(abs(eps_pm))
        eps_emp = safe_logit(P_emp[i, k]) - safe_logit(P_emp[i, j]) - safe_logit(P_emp[j, k])
        lt_emp.append(abs(eps_emp))

    # Kendall tau between all ranking pairs
    def safe_tau(a, b):
        if len(set(a)) == 1 or len(set(b)) == 1:
            return 0.0
        tau, _ = kendalltau(a, b)
        return tau if not np.isnan(tau) else 0.0

    all_ranks = [rank_emp_borda, rank_emp_cope, rank_nash, rank_rm, rank_pm]
    n_methods = len(all_ranks)
    tau_matrix = np.zeros((n_methods, n_methods))
    for a in range(n_methods):
        for b in range(n_methods):
            if a == b:
                tau_matrix[a, b] = 1.0
            elif a < b:
                t = safe_tau(all_ranks[a], all_ranks[b])
                tau_matrix[a, b] = t
                tau_matrix[b, a] = t

    # Response lengths
    lengths = [len(r["text"].split()) for r in eval_rec["responses"]]

    return {
        "K": K,
        "P_emp": P_emp,
        "P_RM": P_RM,
        "P_PM": P_PM,
        "P_bt": P_bt,
        "rm_rewards": rm_rewards,
        "r_bt": r_bt,
        "borda_emp": borda_emp,
        "borda_rm": borda_rm,
        "borda_pm": borda_pm,
        "borda_bt": borda_bt,
        "copeland_emp": cope_emp,
        "nash_pi": nash_pi,
        "nash_val": nash_val,
        "condorcet_winner": cw,
        "has_cycle": cw is None,
        "winner_emp_borda": winner_emp_borda,
        "winner_emp_cope": winner_emp_cope,
        "winner_nash": winner_nash,
        "winner_rm": winner_rm,
        "winner_pm": winner_pm,
        "inflation": inflation,
        "inflation_pm": inflation_pm,
        "brier_rm": brier_rm,
        "brier_pm": brier_pm,
        "p_dist_rm_bt": p_dist_rm_bt,
        "r_dist_rm_bt": r_dist_rm_bt,
        "bt_distortion_pm": bt_distortion_pm,
        "lt_pm_abs": lt_pm,
        "lt_emp_abs": lt_emp,
        "tau_matrix": tau_matrix,
        "lengths": lengths,
    }


def compute_summary(results):
    """Aggregate per-prompt results into summary statistics."""
    n = len(results)

    # Winner agreement matrix (5 methods)
    methods = ["Emp.Borda", "Emp.Copeland", "Nash", "RM.Borda", "PM.Borda"]
    winner_keys = ["winner_emp_borda", "winner_emp_cope", "winner_nash",
                   "winner_rm", "winner_pm"]
    n_methods = len(methods)

    winner_agreement = np.zeros((n_methods, n_methods))
    kendall_tau_avg = np.zeros((n_methods, n_methods))

    for r in results:
        winners = [r[k] for k in winner_keys]
        for a in range(n_methods):
            for b in range(n_methods):
                if winners[a] == winners[b]:
                    winner_agreement[a, b] += 1

    winner_agreement /= n

    # Kendall tau (averaged from per-prompt full matrices)
    tau_matrices = np.array([r["tau_matrix"] for r in results])  # (n_prompts, 5, 5)
    kendall_tau_avg = tau_matrices.mean(axis=0)

    # Aggregate scalars
    brier_rm_vals = [r["brier_rm"] for r in results]
    brier_pm_vals = [r["brier_pm"] for r in results]
    p_dist_vals = [r["p_dist_rm_bt"] for r in results]
    r_dist_vals = [r["r_dist_rm_bt"] for r in results]
    bt_dist_vals = [r["bt_distortion_pm"] for r in results]
    cycle_rate = np.mean([r["has_cycle"] for r in results])

    # Logit-transitivity (aggregate all triples)
    all_lt_pm = []
    all_lt_emp = []
    for r in results:
        all_lt_pm.extend(r["lt_pm_abs"])
        all_lt_emp.extend(r["lt_emp_abs"])

    # Inflation distribution (RM)
    all_inflation = []
    for r in results:
        all_inflation.extend(r["inflation"].tolist())
    infl_counts = {}
    for v in all_inflation:
        infl_counts[v] = infl_counts.get(v, 0) + 1

    # Inflation distribution (PM)
    all_inflation_pm = []
    for r in results:
        all_inflation_pm.extend(r["inflation_pm"].tolist())
    infl_pm_counts = {}
    for v in all_inflation_pm:
        infl_pm_counts[v] = infl_pm_counts.get(v, 0) + 1

    # Per-prompt: RM vs PM disagreement rates with Emp.Borda
    rm_disagree = 0
    pm_disagree = 0
    both_disagree = 0
    rm_only_disagree = 0
    pm_only_disagree = 0
    pm_correct_rm_wrong = 0
    rm_correct_pm_wrong = 0
    for r in results:
        rm_wrong = r["winner_rm"] != r["winner_emp_borda"]
        pm_wrong = r["winner_pm"] != r["winner_emp_borda"]
        rm_disagree += rm_wrong
        pm_disagree += pm_wrong
        both_disagree += (rm_wrong and pm_wrong)
        rm_only_disagree += (rm_wrong and not pm_wrong)
        pm_only_disagree += (not rm_wrong and pm_wrong)
        pm_correct_rm_wrong += (rm_wrong and not pm_wrong)
        rm_correct_pm_wrong += (not rm_wrong and pm_wrong)

    # Pairwise P_RM vs P_emp (for calibration scatter)
    all_p_rm = []
    all_p_emp = []
    all_p_pm = []
    for r in results:
        K = r["K"]
        for i in range(K):
            for j in range(K):
                if i < j:  # upper triangle only (avoid double counting)
                    all_p_rm.append(r["P_RM"][i, j])
                    all_p_emp.append(r["P_emp"][i, j])
                    all_p_pm.append(r["P_PM"][i, j])

    return {
        "n_prompts": n,
        "methods": methods,
        "winner_agreement": winner_agreement.tolist(),
        "kendall_tau_avg": kendall_tau_avg.tolist(),
        "brier_rm_mean": float(np.mean(brier_rm_vals)),
        "brier_rm_std": float(np.std(brier_rm_vals)),
        "brier_pm_mean": float(np.mean(brier_pm_vals)),
        "brier_pm_std": float(np.std(brier_pm_vals)),
        "p_dist_rm_bt_mean": float(np.mean(p_dist_vals)),
        "p_dist_rm_bt_std": float(np.std(p_dist_vals)),
        "r_dist_rm_bt_mean": float(np.mean(r_dist_vals)),
        "r_dist_rm_bt_std": float(np.std(r_dist_vals)),
        "bt_distortion_pm_mean": float(np.mean(bt_dist_vals)),
        "bt_distortion_pm_std": float(np.std(bt_dist_vals)),
        "cycle_rate": float(cycle_rate),
        "lt_pm_abs_mean": float(np.mean(all_lt_pm)),
        "lt_pm_abs_median": float(np.median(all_lt_pm)),
        "lt_emp_abs_mean": float(np.mean(all_lt_emp)),
        "lt_emp_abs_median": float(np.median(all_lt_emp)),
        "inflation_distribution": {str(k): v for k, v in sorted(infl_counts.items())},
        "inflation_pm_distribution": {str(k): v for k, v in sorted(infl_pm_counts.items())},
        "inflation_rm_mean": float(np.mean(all_inflation)),
        "inflation_rm_std": float(np.std(all_inflation)),
        "inflation_rm_abs_mean": float(np.mean(np.abs(all_inflation))),
        "inflation_pm_mean": float(np.mean(all_inflation_pm)),
        "inflation_pm_std": float(np.std(all_inflation_pm)),
        "inflation_pm_abs_mean": float(np.mean(np.abs(all_inflation_pm))),
        "winner_disagree_rm": rm_disagree,
        "winner_disagree_pm": pm_disagree,
        "winner_disagree_both": both_disagree,
        "winner_disagree_rm_only": rm_only_disagree,
        "winner_disagree_pm_only": pm_only_disagree,
        "pairwise_data": {
            "p_rm": all_p_rm,
            "p_emp": all_p_emp,
            "p_pm": all_p_pm,
        },
    }


def make_figures(summary, results, output_dir):
    """Generate analysis figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Winner agreement heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    wa = np.array(summary["winner_agreement"])
    methods = summary["methods"]
    im = ax.imshow(wa, vmin=0, vmax=1, cmap="YlOrRd")
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(j, i, f"{wa[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Winner Agreement Rate")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_dir / "figure_winner_agreement.png", dpi=150)
    plt.close(fig)

    # 2. Calibration: P_RM vs P_emp and P_PM vs P_emp (hexbin)
    p_rm = np.array(summary["pairwise_data"]["p_rm"])
    p_emp = np.array(summary["pairwise_data"]["p_emp"])
    p_pm = np.array(summary["pairwise_data"]["p_pm"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    hb = ax.hexbin(p_rm, p_emp, gridsize=30, cmap="Blues", mincnt=1)
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="perfect calibration")
    ax.set_xlabel("P_RM[i,j]")
    ax.set_ylabel("P_emp[i,j]")
    ax.set_title(f"RM Calibration (Brier={summary['brier_rm_mean']:.4f})")
    ax.legend()
    fig.colorbar(hb, ax=ax, label="count")

    ax = axes[1]
    hb = ax.hexbin(p_pm, p_emp, gridsize=30, cmap="Greens", mincnt=1)
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="perfect calibration")
    ax.set_xlabel("P_PM[i,j]")
    ax.set_ylabel("P_emp[i,j]")
    ax.set_title(f"PM Calibration (Brier={summary['brier_pm_mean']:.4f})")
    ax.legend()
    fig.colorbar(hb, ax=ax, label="count")

    plt.tight_layout()
    fig.savefig(output_dir / "figure_calibration.png", dpi=150)
    plt.close(fig)

    # 3. Cross-prompt inflation histogram (RM vs PM side by side)
    all_inflation_rm = []
    all_inflation_pm = []
    for r in results:
        all_inflation_rm.extend(r["inflation"].tolist())
        all_inflation_pm.extend(r["inflation_pm"].tolist())

    all_vals = all_inflation_rm + all_inflation_pm
    bins = np.arange(min(all_vals) - 0.5, max(all_vals) + 1.5, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    ax = axes[0]
    ax.hist(all_inflation_rm, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("rank_RM - rank_Emp_Borda")
    ax.set_ylabel("Count")
    ax.set_title("RM Borda Inflation")
    mean_rm = np.mean(all_inflation_rm)
    abs_mean_rm = np.mean(np.abs(all_inflation_rm))
    ax.text(0.02, 0.95, f"mean={mean_rm:.3f}\n|mean|={abs_mean_rm:.3f}",
            transform=ax.transAxes, va="top", fontsize=10)

    ax = axes[1]
    ax.hist(all_inflation_pm, bins=bins, edgecolor="black", alpha=0.7, color="forestgreen")
    ax.axvline(0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("rank_PM - rank_Emp_Borda")
    ax.set_title("PM Borda Inflation")
    mean_pm = np.mean(all_inflation_pm)
    abs_mean_pm = np.mean(np.abs(all_inflation_pm))
    ax.text(0.02, 0.95, f"mean={mean_pm:.3f}\n|mean|={abs_mean_pm:.3f}",
            transform=ax.transAxes, va="top", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "figure_inflation_hist.png", dpi=150)
    plt.close(fig)

    # 4. Logit-transitivity violations
    all_lt_pm = []
    all_lt_emp = []
    for r in results:
        all_lt_pm.extend(r["lt_pm_abs"])
        all_lt_emp.extend(r["lt_emp_abs"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(all_lt_emp, bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(all_lt_emp), color="red", linestyle="--",
               label=f"mean={np.mean(all_lt_emp):.3f}")
    ax.set_xlabel("|logit-transitivity residual|")
    ax.set_ylabel("Count")
    ax.set_title("Empirical (P_emp)")
    ax.legend()

    ax = axes[1]
    ax.hist(all_lt_pm, bins=50, alpha=0.7, edgecolor="black", color="green")
    ax.axvline(np.mean(all_lt_pm), color="red", linestyle="--",
               label=f"mean={np.mean(all_lt_pm):.3f}")
    ax.set_xlabel("|logit-transitivity residual|")
    ax.set_ylabel("Count")
    ax.set_title("PM (P_PM)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "figure_logit_transitivity.png", dpi=150)
    plt.close(fig)

    # 5. RM vs local BT distances
    p_dists = [r["p_dist_rm_bt"] for r in results]
    r_dists = [r["r_dist_rm_bt"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(p_dists, bins=40, alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(p_dists), color="red", linestyle="--",
               label=f"mean={np.mean(p_dists):.4f}")
    ax.set_xlabel("mean |P_RM - P_bt_local|")
    ax.set_ylabel("Count")
    ax.set_title("RM vs Local BT (P-space)")
    ax.legend()

    ax = axes[1]
    ax.hist(r_dists, bins=40, alpha=0.7, edgecolor="black", color="orange")
    ax.axvline(np.mean(r_dists), color="red", linestyle="--",
               label=f"mean={np.mean(r_dists):.4f}")
    ax.set_xlabel("||r_RM - r_BT_MLE||_2")
    ax.set_ylabel("Count")
    ax.set_title("RM vs Local BT (Reward-space)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "figure_rm_vs_bt.png", dpi=150)
    plt.close(fig)

    print(f"  Saved 5 figures to {output_dir}")


def build_flat_data(results, eval_records):
    """Build per-response flat data for downstream analysis."""
    flat = []
    for r, rec in zip(results, eval_records):
        for i in range(r["K"]):
            entry = {
                "prompt": rec["prompt"],
                "response_idx": i,
                "rm_reward": float(r["rm_rewards"][i]),
                "r_bt_local": float(r["r_bt"][i]),
                "borda_emp": float(r["borda_emp"][i]),
                "borda_rm": float(r["borda_rm"][i]),
                "borda_pm": float(r["borda_pm"][i]),
                "borda_bt": float(r["borda_bt"][i]),
                "copeland_emp": float(r["copeland_emp"][i]),
                "inflation_rm": int(r["inflation"][i]),
                "inflation_pm": int(r["inflation_pm"][i]),
                "length": int(r["lengths"][i]),
                "is_condorcet_winner": r["condorcet_winner"] == i,
                "is_rm_winner": r["winner_rm"] == i,
                "is_pm_winner": r["winner_pm"] == i,
                "has_cycle": r["has_cycle"],
            }
            # Dimension scores
            scores = rec["responses"][i].get("scores", {})
            for dim in DIMENSIONS:
                entry[f"dim_{dim}"] = scores.get(dim)
            flat.append(entry)
    return flat


def main():
    parser = argparse.ArgumentParser(description="Analyze Borda inflation")
    parser.add_argument("--eval_data", type=str, required=True,
                        help="Path to preprocessed eval JSON")
    parser.add_argument("--inference_data", type=str, required=True,
                        help="Path to inference pickle from run_inference.py")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save analysis results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading eval data from {args.eval_data}...")
    with open(args.eval_data) as f:
        eval_records = json.load(f)

    print(f"Loading inference data from {args.inference_data}...")
    with open(args.inference_data, "rb") as f:
        inference = pickle.load(f)

    infer_prompts = inference["prompts"]
    infer_meta = inference["meta"]
    print(f"  {len(eval_records)} eval prompts, {len(infer_prompts)} inference prompts")
    print(f"  RM checkpoint: {infer_meta['rm_checkpoint']}")
    print(f"  PM checkpoint: {infer_meta['pm_checkpoint']}")

    assert len(eval_records) == len(infer_prompts), \
        f"Mismatch: {len(eval_records)} eval vs {len(infer_prompts)} inference"

    # Verify prompt alignment
    for i in range(len(eval_records)):
        assert eval_records[i]["prompt"] == infer_prompts[i]["prompt"], \
            f"Prompt mismatch at index {i}"

    # Per-prompt analysis
    print("\n" + "=" * 60)
    print("Computing per-prompt metrics")
    print("=" * 60)
    results = []
    for i in tqdm(range(len(eval_records)), desc="Analyzing prompts"):
        r = analyze_prompt_data(eval_records[i], infer_prompts[i])
        results.append(r)

    # Aggregate
    print("\n" + "=" * 60)
    print("Computing aggregate statistics")
    print("=" * 60)
    summary = compute_summary(results)

    # Print key results
    print(f"\n  Prompts:                {summary['n_prompts']}")
    print(f"  Cycle rate:             {summary['cycle_rate']:.3f}")
    print(f"  Brier RM:               {summary['brier_rm_mean']:.4f} +/- {summary['brier_rm_std']:.4f}")
    print(f"  Brier PM:               {summary['brier_pm_mean']:.4f} +/- {summary['brier_pm_std']:.4f}")
    print(f"  |P_RM - P_bt_local|:    {summary['p_dist_rm_bt_mean']:.4f} +/- {summary['p_dist_rm_bt_std']:.4f}")
    print(f"  ||r_RM - r_BT_MLE||:    {summary['r_dist_rm_bt_mean']:.4f} +/- {summary['r_dist_rm_bt_std']:.4f}")
    print(f"  BT distortion of PM:    {summary['bt_distortion_pm_mean']:.4f} +/- {summary['bt_distortion_pm_std']:.4f}")
    print(f"  LT violations PM:       mean={summary['lt_pm_abs_mean']:.4f} median={summary['lt_pm_abs_median']:.4f}")
    print(f"  LT violations emp:      mean={summary['lt_emp_abs_mean']:.4f} median={summary['lt_emp_abs_median']:.4f}")

    print("\n  Winner agreement matrix:")
    wa = np.array(summary["winner_agreement"])
    methods = summary["methods"]
    header = "             " + "  ".join(f"{m:>12s}" for m in methods)
    print(header)
    for i, m in enumerate(methods):
        row = f"  {m:>11s}" + "  ".join(f"{wa[i, j]:>12.3f}" for j in range(len(methods)))
        print(row)

    print("\n  Kendall tau (avg):")
    kt = np.array(summary["kendall_tau_avg"])
    print(header)
    for i, m in enumerate(methods):
        row = f"  {m:>11s}" + "  ".join(f"{kt[i, j]:>12.3f}" for j in range(len(methods)))
        print(row)

    print(f"\n  RM inflation:  mean={summary['inflation_rm_mean']:.3f}  std={summary['inflation_rm_std']:.3f}  |mean|={summary['inflation_rm_abs_mean']:.3f}")
    print(f"  PM inflation:  mean={summary['inflation_pm_mean']:.3f}  std={summary['inflation_pm_std']:.3f}  |mean|={summary['inflation_pm_abs_mean']:.3f}")
    print(f"  RM inflation dist: {summary['inflation_distribution']}")
    print(f"  PM inflation dist: {summary['inflation_pm_distribution']}")
    n = summary['n_prompts']
    print(f"\n  Winner disagreement breakdown (of {n} prompts):")
    print(f"    RM wrong:       {summary['winner_disagree_rm']:>4d} ({summary['winner_disagree_rm']/n:.1%})")
    print(f"    PM wrong:       {summary['winner_disagree_pm']:>4d} ({summary['winner_disagree_pm']/n:.1%})")
    print(f"    Both wrong:     {summary['winner_disagree_both']:>4d} ({summary['winner_disagree_both']/n:.1%})")
    print(f"    RM only wrong:  {summary['winner_disagree_rm_only']:>4d} ({summary['winner_disagree_rm_only']/n:.1%})")
    print(f"    PM only wrong:  {summary['winner_disagree_pm_only']:>4d} ({summary['winner_disagree_pm_only']/n:.1%})")

    # Save
    print("\n" + "=" * 60)
    print("Saving results")
    print("=" * 60)

    # Remove large pairwise arrays from summary before saving to JSON
    summary_json = {k: v for k, v in summary.items() if k != "pairwise_data"}
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"  Saved summary.json")

    # Full results pickle (includes P matrices, per-prompt data)
    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump({"results": results, "summary": summary, "meta": infer_meta}, f)
    print(f"  Saved results.pkl")

    # Flat data
    flat = build_flat_data(results, eval_records)
    with open(output_dir / "flat.jsonl", "w") as f:
        for entry in flat:
            f.write(json.dumps(entry) + "\n")
    print(f"  Saved {len(flat)} entries to flat.jsonl")

    # Figures
    print("\n" + "=" * 60)
    print("Generating figures")
    print("=" * 60)
    make_figures(summary, results, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
