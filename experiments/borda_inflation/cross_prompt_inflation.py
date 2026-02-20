#!/usr/bin/env python3
"""Cross-prompt Borda inflation analysis: parametric BT vs tabular annotations.

Compares three preference sources on ~6K overlapping UltraFeedback test prompts:
  - P_emp: majority-rule from 4 UFB dimension annotations (tabular ground truth)
  - P_BT: sigma(r_i - r_j) from parametric BT reward model (Llama-8B, dim=1)
  - P_PM: from 8-dim preference model (Llama-8B, GPM)

Key question: does the parametric BT model systematically overvalue certain
responses relative to local empirical preferences (cross-prompt Borda inflation)?

Parts:
  0. Data loading and merging (response text matching via openbmb/UltraFeedback)
  1. Pairwise probability comparison (Brier score, calibration curves)
  2. Logit-transitivity violations (PM vs empirical)
  3. Ranking comparisons (5 rankings, winner agreement, Kendall tau)
  4. Cross-prompt inflation characterization (delta_rank vs features)
  5. Effective rewards comparison (all on [0,1] win-rate scale)
  6. RLHF vs NLHF policy simulation (amplification by beta)
  +  Three-way validity check (BT/PM/tabular winner agreement patterns)

Usage:
    python experiments/borda_inflation/cross_prompt_inflation.py \
        --model_results /path/to/inflation_results.pkl \
        --tabular_results /path/to/tabular_results.json \
        --output_dir experiments/borda_inflation/cross_prompt_results
"""

import argparse
import json
import pickle
import sys
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.special import expit, logit as scipy_logit
from scipy.stats import spearmanr, kendalltau

# Import social choice functions from tabular_inflation.py
sys.path.insert(0, str(Path(__file__).parent))
from tabular_inflation import (
    borda_scores as compute_borda,
    copeland_scores as compute_copeland,
    condorcet_winner as find_condorcet,
    nash_equilibrium_lp,
    bt_mle_from_matrix,
)

DIMENSIONS = ["instruction_following", "honesty", "truthfulness", "helpfulness"]


# ── HELPERS ──────────────────────────────────────────────────────────────────


def ranking_from_scores(scores):
    """Convert scores to ranking (0 = best). Ties broken by index."""
    order = np.argsort(-np.asarray(scores, dtype=float))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    return ranks


def safe_logit(p, clip=0.01):
    """Logit with clipping to avoid infinities."""
    return scipy_logit(np.clip(p, clip, 1.0 - clip))


def softmax(x):
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=float)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# ── PART 0: DATA LOADING AND MERGING ────────────────────────────────────────


def load_and_merge(model_pkl_path, tabular_json_path,
                   ufb_dataset="openbmb/UltraFeedback"):
    """Load and align model-based results, tabular results, and UFB annotations.

    Model-based responses are alphabetically sorted (from argilla pairwise
    reconstruction). Tabular results use the openbmb/UltraFeedback completion
    order (0-3). We reload openbmb to build a text-matching permutation.

    Returns list of merged prompt dicts, all in model-based response ordering.
    """
    # 1. Load model-based results
    print("Loading model-based results...")
    with open(model_pkl_path, "rb") as f:
        model_data = pickle.load(f)
    per_prompt = model_data["per_prompt"]
    print(f"  {len(per_prompt)} prompts loaded")

    # 2. Load tabular results
    print("Loading tabular results...")
    with open(tabular_json_path) as f:
        tabular_data = json.load(f)
    print(f"  {len(tabular_data)} prompts loaded")

    # Index tabular by instruction[:200]
    tab_by_key = {}
    for t in tabular_data:
        tab_by_key[t["instruction"][:200]] = t

    # 3. Load openbmb/UltraFeedback for response text matching
    print("Loading openbmb/UltraFeedback for response matching...")
    from datasets import load_dataset
    ufb = load_dataset(ufb_dataset, split="train")

    ufb_by_key = {}
    for row in ufb:
        ufb_by_key[row["instruction"][:200]] = row
    print(f"  {len(ufb_by_key)} UFB prompts indexed")

    # 4. Match and merge
    print("Matching prompts and responses...")
    merged = []
    stats = defaultdict(int)

    for mp in per_prompt:
        stats["total"] += 1
        key = mp["prompt"][:200]

        tab = tab_by_key.get(key)
        if tab is None:
            stats["no_tabular"] += 1
            continue

        ufb_row = ufb_by_key.get(key)
        if ufb_row is None:
            stats["no_ufb"] += 1
            continue

        # Only keep prompts with exactly 4 responses in both sources
        if mp["n_responses"] != 4 or tab["n_responses"] != 4:
            stats["wrong_n"] += 1
            continue

        # Build response permutation: perm[ufb_idx] = model_idx
        ufb_responses = [c["response"] for c in ufb_row["completions"]]
        model_responses = mp["responses"]

        perm = [None] * 4
        used_model = set()

        for ui, ur in enumerate(ufb_responses):
            ur_prefix = ur.strip()[:100]
            best_mi, best_score = None, 0.0

            for mi, mr in enumerate(model_responses):
                if mi in used_model:
                    continue
                mr_prefix = mr.strip()[:100]

                if ur_prefix == mr_prefix:
                    best_mi, best_score = mi, 1.0
                    break

                min_len = min(len(ur_prefix), len(mr_prefix))
                if min_len > 0:
                    overlap = sum(a == b for a, b in zip(ur_prefix, mr_prefix))
                    score = overlap / min_len
                    if score > best_score and score > 0.8:
                        best_mi, best_score = mi, score

            if best_mi is not None:
                perm[ui] = best_mi
                used_model.add(best_mi)

        if None in perm or len(set(perm)) != 4:
            stats["response_mismatch"] += 1
            continue

        # Inverse permutation: inv_perm[model_idx] = ufb_idx
        inv_perm = [0] * 4
        for ui, mi in enumerate(perm):
            inv_perm[mi] = ui

        # ── Align tabular data to model ordering ──

        # Preference matrix
        P_emp_ufb = np.array(tab["P_majority"], dtype=float)
        P_emp = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                P_emp[perm[i], perm[j]] = P_emp_ufb[i, j]

        # Scalar arrays
        tab_borda = np.array([tab["borda_scores"][inv_perm[m]] for m in range(4)])
        tab_copeland = np.array([tab["copeland_scores"][inv_perm[m]] for m in range(4)])
        tab_nash = np.array([tab["nash_pi"][inv_perm[m]] for m in range(4)])
        tab_syc = [tab["syc_proxies"][inv_perm[m]] for m in range(4)]
        tab_avg = [tab["avg_scores"][inv_perm[m]] for m in range(4)]
        tab_inflation = [tab["inflation"][inv_perm[m]] for m in range(4)]

        # Model matrices — ensure diagonals are 0.5
        P_BT = np.array(mp["rm_pref_matrix"], dtype=float)
        P_PM = np.array(mp["pm_pref_matrix"], dtype=float)
        np.fill_diagonal(P_BT, 0.5)
        np.fill_diagonal(P_PM, 0.5)

        # Dimension scores per response (model ordering)
        dim_scores = []
        for m_idx in range(4):
            dim_scores.append(mp["dim_scores"].get(m_idx, {}))

        merged.append({
            "prompt": mp["prompt"],
            "responses": model_responses,
            "n": 4,

            # Three aligned 4x4 preference matrices
            "P_emp": P_emp,
            "P_BT": P_BT,
            "P_PM": P_PM,

            # BT scalar rewards (model ordering)
            "r_BT": np.array(mp["rm_rewards"], dtype=float),

            # Tabular social choice (model ordering)
            "tab_borda": tab_borda,
            "tab_copeland": tab_copeland,
            "tab_nash_pi": tab_nash,
            "tab_has_cycle": tab.get("has_cycle", False),
            "tab_inflation": tab_inflation,

            # Response features (model ordering)
            "lengths": np.array(mp["lengths"], dtype=float),
            "dim_scores": dim_scores,
            "syc_proxies": tab_syc,
            "avg_scores": tab_avg,
        })
        stats["matched"] += 1

    # Sanity: P_emp symmetry
    sym_err = 0
    for m in merged:
        err = np.max(np.abs(m["P_emp"] + m["P_emp"].T - 1.0))
        if err > 0.01:
            sym_err += 1
    if sym_err:
        print(f"  WARNING: {sym_err} prompts have P_emp symmetry violation > 0.01")

    print(f"\nMatch statistics:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    return merged


# ── PART 1: PAIRWISE PROBABILITY COMPARISON ─────────────────────────────────


def part1_pairwise_probs(merged):
    """Compare P_BT, P_PM, P_emp for each pair."""
    print("\n" + "=" * 70)
    print("PART 1: PAIRWISE PROBABILITY COMPARISON")
    print("=" * 70)

    p_emp_list, p_bt_list, p_pm_list = [], [], []

    for m in merged:
        for i in range(4):
            for j in range(i + 1, 4):
                p_emp_list.append(m["P_emp"][i, j])
                p_bt_list.append(m["P_BT"][i, j])
                p_pm_list.append(m["P_PM"][i, j])

    p_emp = np.array(p_emp_list)
    p_bt = np.array(p_bt_list)
    p_pm = np.array(p_pm_list)
    n_pairs = len(p_emp)

    print(f"\n  Total pairs: {n_pairs}")

    # Signed differences
    d_bt = p_bt - p_emp
    d_pm = p_pm - p_emp
    d_bt_pm = p_bt - p_pm

    for name, d in [("P_BT - P_emp", d_bt), ("P_PM - P_emp", d_pm),
                    ("P_BT - P_PM", d_bt_pm)]:
        print(f"  {name:16s}  mean={d.mean():+.4f}  std={d.std():.4f}  "
              f"MAE={np.abs(d).mean():.4f}")

    # Brier scores
    brier_bt = float(np.mean(d_bt ** 2))
    brier_pm = float(np.mean(d_pm ** 2))
    print(f"\n  Brier(BT, emp) = {brier_bt:.6f}")
    print(f"  Brier(PM, emp) = {brier_pm:.6f}")
    print(f"  Ratio BT/PM   = {brier_bt / max(brier_pm, 1e-12):.3f}")

    # Calibration curves (bin by model prediction, report mean P_emp)
    def calibration(p_model, p_true, n_bins=10):
        edges = np.linspace(0, 1, n_bins + 1)
        centers, means, counts = [], [], []
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            mask = (p_model >= lo) & (p_model < hi)
            if b == n_bins - 1:
                mask |= (p_model == hi)
            if mask.sum() > 0:
                centers.append(float((lo + hi) / 2))
                means.append(float(p_true[mask].mean()))
                counts.append(int(mask.sum()))
        return centers, means, counts

    cal_bt = calibration(p_bt, p_emp)
    cal_pm = calibration(p_pm, p_emp)

    print("\n  Calibration (BT): P_BT_bin -> mean P_emp")
    for c, m, n in zip(*cal_bt):
        print(f"    {c:.2f} -> {m:.4f}  (n={n})")
    print("  Calibration (PM): P_PM_bin -> mean P_emp")
    for c, m, n in zip(*cal_pm):
        print(f"    {c:.2f} -> {m:.4f}  (n={n})")

    # Bias by P_emp level
    print("\n  Model bias conditioned on P_emp level:")
    for level in [0.0, 0.25, 0.5, 0.75, 1.0]:
        mask = np.abs(p_emp - level) < 0.01
        n = int(mask.sum())
        if n > 0:
            print(f"    P_emp={level:.2f} (n={n:>6d}): "
                  f"mean P_BT={p_bt[mask].mean():.4f}  "
                  f"mean P_PM={p_pm[mask].mean():.4f}")

    return {
        "n_pairs": n_pairs,
        "brier_bt": brier_bt, "brier_pm": brier_pm,
        "delta_bt_mean": float(d_bt.mean()),
        "delta_pm_mean": float(d_pm.mean()),
        "delta_bt_mae": float(np.abs(d_bt).mean()),
        "delta_pm_mae": float(np.abs(d_pm).mean()),
        "calibration_bt": dict(zip(["centers", "means", "counts"], cal_bt)),
        "calibration_pm": dict(zip(["centers", "means", "counts"], cal_pm)),
        # raw arrays for figures (excluded from JSON via _ prefix)
        "_p_emp": p_emp, "_p_bt": p_bt, "_p_pm": p_pm,
    }


# ── PART 2: LOGIT-TRANSITIVITY VIOLATIONS ───────────────────────────────────


def part2_logit_transitivity(merged):
    """Measure logit-transitivity violations in PM and empirical preferences."""
    print("\n" + "=" * 70)
    print("PART 2: LOGIT-TRANSITIVITY VIOLATIONS")
    print("=" * 70)

    eps_pm, eps_emp, bt_distortion = [], [], []
    n_clipped = 0

    for m in merged:
        P_PM = m["P_PM"]
        P_emp = m["P_emp"]

        # 4 ordered triples from C(4,3)
        for i, j, k in combinations(range(4), 3):
            # PM residual: logit(P(i>k)) - logit(P(i>j)) - logit(P(j>k))
            eps_pm.append(safe_logit(P_PM[i, k]) -
                          safe_logit(P_PM[i, j]) - safe_logit(P_PM[j, k]))
            eps_emp.append(safe_logit(P_emp[i, k]) -
                           safe_logit(P_emp[i, j]) - safe_logit(P_emp[j, k]))

            for p in [P_PM[i, j], P_PM[j, k], P_PM[i, k]]:
                if p < 0.01 or p > 0.99:
                    n_clipped += 1

        # BT-MLE distortion: fit BT to P_PM, measure |P_fit - P_PM|
        theta = bt_mle_from_matrix(P_PM)
        for i in range(4):
            for j in range(i + 1, 4):
                p_fit = theta[i] / (theta[i] + theta[j])
                bt_distortion.append(abs(p_fit - P_PM[i, j]))

    eps_pm = np.array(eps_pm)
    eps_emp = np.array(eps_emp)
    bt_distortion = np.array(bt_distortion)

    print(f"\n  Triples: {len(eps_pm)}  (clipped values: {n_clipped})")

    for name, e in [("PM", eps_pm), ("Empirical", eps_emp)]:
        ae = np.abs(e)
        print(f"\n  {name} |epsilon|:")
        print(f"    mean={ae.mean():.4f}  median={np.median(ae):.4f}  "
              f"max={ae.max():.4f}")
        print(f"    > 0.1: {(ae > 0.1).mean():.1%}   "
              f"> 0.5: {(ae > 0.5).mean():.1%}   "
              f"> 1.0: {(ae > 1.0).mean():.1%}")

    print(f"\n  BT-MLE distortion |P_fit - P_PM| per pair:")
    print(f"    mean={bt_distortion.mean():.4f}  "
          f"median={np.median(bt_distortion):.4f}  "
          f"max={bt_distortion.max():.4f}")

    # Per-prompt average |eps| — correlation between PM and empirical
    n_triples_per = 4  # C(4,3)
    n_prompts = len(merged)
    pm_prompt = [np.abs(eps_pm[i * n_triples_per:(i + 1) * n_triples_per]).mean()
                 for i in range(n_prompts)]
    emp_prompt = [np.abs(eps_emp[i * n_triples_per:(i + 1) * n_triples_per]).mean()
                  for i in range(n_prompts)]

    rho, p = spearmanr(pm_prompt, emp_prompt)
    print(f"\n  Spearman(PM |eps|, emp |eps|) per prompt: rho={rho:.4f}  p={p:.2e}")

    # By cycle presence
    has_cycle = [m["tab_has_cycle"] for m in merged]
    with_c = [pm_prompt[i] for i in range(n_prompts) if has_cycle[i]]
    no_c = [pm_prompt[i] for i in range(n_prompts) if not has_cycle[i]]
    if with_c and no_c:
        print(f"\n  PM |eps| by cycle:  with={np.mean(with_c):.4f} (n={len(with_c)})  "
              f"without={np.mean(no_c):.4f} (n={len(no_c)})")

    return {
        "n_triples": len(eps_pm),
        "pm_abs_mean": float(np.abs(eps_pm).mean()),
        "pm_abs_median": float(np.median(np.abs(eps_pm))),
        "emp_abs_mean": float(np.abs(eps_emp).mean()),
        "bt_distortion_mean": float(bt_distortion.mean()),
        "pm_emp_corr": float(rho),
        "_eps_pm": eps_pm, "_eps_emp": eps_emp,
        "_bt_distortion": bt_distortion,
    }


# ── PART 3: RANKING COMPARISONS ─────────────────────────────────────────────


def part3_rankings(merged):
    """Compare 5 rankings per prompt: winner agreement and Kendall tau."""
    print("\n" + "=" * 70)
    print("PART 3: RANKING COMPARISONS")
    print("=" * 70)

    names = ["Ann.Borda", "Ann.Copeland", "Nash.eff", "Param.BT", "PM.Borda"]
    N = len(names)
    n_prompts = len(merged)

    winner_agree = np.zeros((N, N))
    tau_sum = np.zeros((N, N))

    for m in merged:
        # 1. Annotation Borda (= per-prompt BT-MLE)
        r1 = ranking_from_scores(m["tab_borda"])
        # 2. Annotation Copeland
        r2 = ranking_from_scores(m["tab_copeland"])
        # 3. Nash effective reward = P_emp @ pi_Nash
        nash_eff = m["P_emp"] @ m["tab_nash_pi"]
        r3 = ranking_from_scores(nash_eff)
        # 4. Parametric BT
        r4 = ranking_from_scores(m["r_BT"])
        # 5. PM Borda = row sums of P_PM
        pm_borda = m["P_PM"].sum(axis=1) - 0.5
        r5 = ranking_from_scores(pm_borda)

        rankings = [r1, r2, r3, r4, r5]

        for a in range(N):
            for b in range(N):
                if np.argmin(rankings[a]) == np.argmin(rankings[b]):
                    winner_agree[a, b] += 1
                tau, _ = kendalltau(rankings[a], rankings[b])
                if not np.isnan(tau):
                    tau_sum[a, b] += tau

    winner_agree /= n_prompts
    tau_avg = tau_sum / n_prompts

    print(f"\n  Winner agreement (% of {n_prompts} prompts):")
    header = "  " + " " * 14 + "".join(f"{n:>14s}" for n in names)
    print(header)
    for i in range(N):
        row = f"  {names[i]:14s}" + "".join(f"{100 * winner_agree[i, j]:13.1f}%" for j in range(N))
        print(row)

    print(f"\n  Kendall tau (average):")
    print(header)
    for i in range(N):
        row = f"  {names[i]:14s}" + "".join(f"{tau_avg[i, j]:14.3f}" for j in range(N))
        print(row)

    # Highlight key comparisons
    print(f"\n  Key comparisons (winner agreement):")
    pairs = [(0, 3, "Ann.Borda vs Param.BT (cross-prompt effect)"),
             (3, 1, "Param.BT vs Ann.Copeland (total BT inflation)"),
             (0, 1, "Ann.Borda vs Ann.Copeland (social choice divergence)"),
             (3, 4, "Param.BT vs PM.Borda (BT vs PM model)")]
    for a, b, label in pairs:
        print(f"    {label}: {100 * winner_agree[a, b]:.1f}%  (tau={tau_avg[a, b]:.3f})")

    return {
        "names": names,
        "winner_agreement": winner_agree.tolist(),
        "kendall_tau": tau_avg.tolist(),
    }


# ── PART 4: CROSS-PROMPT INFLATION CHARACTERIZATION ─────────────────────────


def part4_cross_prompt_inflation(merged):
    """Characterize which responses the parametric BT overvalues."""
    print("\n" + "=" * 70)
    print("PART 4: CROSS-PROMPT INFLATION CHARACTERIZATION")
    print("=" * 70)

    delta_ranks, lengths, sycs = [], [], []
    dim_vals = {d: [] for d in DIMENSIONS}
    avg_scores = []

    for m in merged:
        rank_bt = ranking_from_scores(m["r_BT"])
        rank_borda = ranking_from_scores(m["tab_borda"])

        for idx in range(4):
            delta_ranks.append(int(rank_bt[idx]) - int(rank_borda[idx]))
            lengths.append(float(m["lengths"][idx]))
            sycs.append(m["syc_proxies"][idx])
            avg_scores.append(m["avg_scores"][idx])
            for dim in DIMENSIONS:
                dim_vals[dim].append(m["dim_scores"][idx].get(dim))

    dr = np.array(delta_ranks, dtype=float)
    lens = np.array(lengths)
    n_total = len(dr)

    print(f"\n  Responses: {n_total}")
    print(f"  delta_rank = rank_BT_param - rank_Borda_tab  "
          f"(negative = BT overvalues)")
    print(f"\n  Distribution:")
    for v in sorted(set(dr.astype(int))):
        c = int((dr == v).sum())
        print(f"    {int(v):+d}: {c:>6d} ({100 * c / n_total:.1f}%)")

    print(f"\n  BT overvalues (delta < 0): {(dr < 0).sum()} ({100 * (dr < 0).mean():.1f}%)")
    print(f"  Agrees (delta = 0):        {(dr == 0).sum()} ({100 * (dr == 0).mean():.1f}%)")
    print(f"  BT undervalues (delta > 0): {(dr > 0).sum()} ({100 * (dr > 0).mean():.1f}%)")

    # Correlations
    print(f"\n  Spearman correlations (delta_rank vs feature):")
    corrs = {}

    def _corr(name, x_vals, filter_none=False):
        if filter_none:
            valid = [(d, x) for d, x in zip(dr, x_vals) if x is not None]
            if not valid:
                return
            d_v, x_v = zip(*valid)
            d_v, x_v = np.array(d_v), np.array(x_v, dtype=float)
        else:
            d_v, x_v = dr, np.array(x_vals, dtype=float)
        rho, p = spearmanr(d_v, x_v)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {name:28s} rho={rho:+.4f}  p={p:.2e}  n={len(d_v)}  {sig}")
        corrs[name] = {"rho": float(rho), "p": float(p), "n": len(d_v)}

    _corr("length", lens)
    _corr("sycophancy_proxy", sycs, filter_none=True)
    _corr("avg_score", avg_scores, filter_none=True)
    for dim in DIMENSIONS:
        _corr(dim, dim_vals[dim], filter_none=True)

    return {
        "correlations": corrs,
        "delta_distribution": {int(v): int((dr == v).sum())
                               for v in sorted(set(dr.astype(int)))},
        "_delta_ranks": dr, "_lengths": lens,
        "_sycs": sycs, "_dim_vals": dim_vals,
    }


# ── PART 5: EFFECTIVE REWARDS COMPARISON ─────────────────────────────────────


def part5_effective_rewards(merged):
    """Compare win-rate effective rewards on a common [0,1] scale.

    All effective rewards use:  r_eff(i) = sum_{j!=i} w_j * P[i,j] / sum_{j!=i} w_j
    where w_j = 1/(K-1) for Borda/BT/PM, and w_j = pi_Nash[j] for NLHF.
    """
    print("\n" + "=" * 70)
    print("PART 5: EFFECTIVE REWARDS COMPARISON")
    print("=" * 70)

    K = 4
    r_borda, r_bt, r_pm, r_nlhf = [], [], [], []
    lengths, sycs = [], []

    for m in merged:
        P_emp, P_BT, P_PM = m["P_emp"], m["P_BT"], m["P_PM"]
        pi = m["tab_nash_pi"]

        for i in range(K):
            # Mean pairwise win rate excluding self
            r_borda.append((P_emp[i].sum() - 0.5) / (K - 1))
            r_bt.append((P_BT[i].sum() - 0.5) / (K - 1))
            r_pm.append((P_PM[i].sum() - 0.5) / (K - 1))

            # Nash-weighted win rate (opponent drawn from pi_Nash, can be self)
            r_nlhf.append(float(pi @ P_emp[i]))

            lengths.append(float(m["lengths"][i]))
            sycs.append(m["syc_proxies"][i])

    r_borda = np.array(r_borda)
    r_bt = np.array(r_bt)
    r_pm = np.array(r_pm)
    r_nlhf = np.array(r_nlhf)
    lengths = np.array(lengths)

    print(f"\n  Responses: {len(r_borda)}")
    print(f"\n  {'Metric':35s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s}")
    for name, arr in [("r_eff_Borda (annotations)", r_borda),
                      ("r_eff_BT (parametric RM)", r_bt),
                      ("r_eff_PM (preference model)", r_pm),
                      ("r_eff_NLHF (Nash-weighted)", r_nlhf)]:
        print(f"  {name:35s} {arr.mean():8.4f} {arr.std():8.4f} "
              f"{arr.min():8.4f} {arr.max():8.4f}")

    # Cross-prompt inflation score: r_eff_BT - r_eff_Borda
    # (how much does parametric BT overvalue relative to local annotations)
    cross_inflation = r_bt - r_borda
    print(f"\n  Cross-prompt inflation (r_eff_BT - r_eff_Borda):")
    print(f"    mean={cross_inflation.mean():.4f}  std={cross_inflation.std():.4f}")
    print(f"    > 0.05: {(cross_inflation > 0.05).mean():.1%}  "
          f"> 0.1: {(cross_inflation > 0.1).mean():.1%}")
    print(f"    < -0.05: {(cross_inflation < -0.05).mean():.1%}  "
          f"< -0.1: {(cross_inflation < -0.1).mean():.1%}")

    # Total Borda inflation: r_eff_BT - r_eff_NLHF
    total_inflation = r_bt - r_nlhf
    print(f"\n  Total Borda inflation (r_eff_BT - r_eff_NLHF):")
    print(f"    mean={total_inflation.mean():.4f}  std={total_inflation.std():.4f}")

    # Pairwise Spearman correlations
    print(f"\n  Spearman correlations between effective rewards:")
    eff_names = ["Borda", "BT", "PM", "NLHF"]
    eff_arrs = [r_borda, r_bt, r_pm, r_nlhf]
    for i in range(len(eff_names)):
        for j in range(i + 1, len(eff_names)):
            rho, p = spearmanr(eff_arrs[i], eff_arrs[j])
            print(f"    {eff_names[i]:6s} vs {eff_names[j]:6s}: rho={rho:+.4f}")

    # Inflation correlations with features
    print(f"\n  Cross-prompt inflation vs features:")
    rho, p = spearmanr(cross_inflation, lengths)
    print(f"    length:    rho={rho:+.4f}  p={p:.2e}")
    valid = [(ci, s) for ci, s in zip(cross_inflation, sycs) if s is not None]
    if valid:
        ci_v, s_v = zip(*valid)
        rho, p = spearmanr(ci_v, s_v)
        print(f"    sycophancy: rho={rho:+.4f}  p={p:.2e}")

    return {
        "cross_inflation_mean": float(cross_inflation.mean()),
        "cross_inflation_std": float(cross_inflation.std()),
        "total_inflation_mean": float(total_inflation.mean()),
        "_r_borda": r_borda, "_r_bt": r_bt, "_r_pm": r_pm, "_r_nlhf": r_nlhf,
        "_cross_inflation": cross_inflation, "_total_inflation": total_inflation,
        "_lengths": lengths, "_sycs": sycs,
    }


# ── PART 6: RLHF vs NLHF POLICY SIMULATION ─────────────────────────────────


def part6_policy_simulation(merged, betas=None):
    """Simulate RLHF (softmax(r_BT/beta)) vs NLHF (Nash) policies."""
    if betas is None:
        betas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

    print("\n" + "=" * 70)
    print("PART 6: RLHF vs NLHF POLICY SIMULATION")
    print("=" * 70)

    K = 4
    by_beta = {}

    for beta in betas:
        amp_len_rlhf, amp_len_nlhf = [], []
        amp_syc_rlhf, amp_syc_nlhf = [], []
        amp_borda_rlhf, amp_borda_nlhf = [], []
        ent_rlhf, ent_nlhf = [], []

        for m in merged:
            r_BT = m["r_BT"]
            pi_nash = m["tab_nash_pi"]
            P_emp = m["P_emp"]
            lens = m["lengths"]
            syc_vals = np.array([s if s is not None else 0.0 for s in m["syc_proxies"]])
            syc_valid = all(s is not None for s in m["syc_proxies"])

            pi_rlhf = softmax(r_BT / beta)
            pi_unif = np.ones(K) / K

            # Empirical Borda per response (excluding self)
            r_borda = np.array([(P_emp[i].sum() - 0.5) / (K - 1) for i in range(K)])

            # Length amplification
            amp_len_rlhf.append(float(pi_rlhf @ lens - pi_unif @ lens))
            amp_len_nlhf.append(float(pi_nash @ lens - pi_unif @ lens))

            # Sycophancy amplification
            if syc_valid:
                amp_syc_rlhf.append(float(pi_rlhf @ syc_vals - pi_unif @ syc_vals))
                amp_syc_nlhf.append(float(pi_nash @ syc_vals - pi_unif @ syc_vals))

            # Borda win-rate amplification
            amp_borda_rlhf.append(float(pi_rlhf @ r_borda - pi_unif @ r_borda))
            amp_borda_nlhf.append(float(pi_nash @ r_borda - pi_unif @ r_borda))

            # Policy entropy
            ent_rlhf.append(float(-np.sum(pi_rlhf * np.log(np.maximum(pi_rlhf, 1e-30)))))
            ent_nlhf.append(float(-np.sum(pi_nash * np.log(np.maximum(pi_nash, 1e-30)))))

        by_beta[beta] = {
            "amp_length_rlhf": float(np.mean(amp_len_rlhf)),
            "amp_length_nlhf": float(np.mean(amp_len_nlhf)),
            "amp_syc_rlhf": float(np.mean(amp_syc_rlhf)) if amp_syc_rlhf else None,
            "amp_syc_nlhf": float(np.mean(amp_syc_nlhf)) if amp_syc_nlhf else None,
            "amp_borda_rlhf": float(np.mean(amp_borda_rlhf)),
            "amp_borda_nlhf": float(np.mean(amp_borda_nlhf)),
            "entropy_rlhf": float(np.mean(ent_rlhf)),
            "entropy_nlhf": float(np.mean(ent_nlhf)),
        }

        r = by_beta[beta]
        print(f"\n  beta={beta:.2f}:")
        print(f"    Entropy      RLHF={r['entropy_rlhf']:.3f}  "
              f"NLHF={r['entropy_nlhf']:.3f}  "
              f"uniform={np.log(K):.3f}")
        print(f"    Amp(length)  RLHF={r['amp_length_rlhf']:+.1f}  "
              f"NLHF={r['amp_length_nlhf']:+.1f}")
        if r["amp_syc_rlhf"] is not None:
            print(f"    Amp(syc)     RLHF={r['amp_syc_rlhf']:+.4f}  "
                  f"NLHF={r['amp_syc_nlhf']:+.4f}")
        print(f"    Amp(Borda)   RLHF={r['amp_borda_rlhf']:+.4f}  "
              f"NLHF={r['amp_borda_nlhf']:+.4f}")

    return {"betas": betas, "by_beta": {str(b): v for b, v in by_beta.items()}}


# ── THREE-WAY VALIDITY CHECK ────────────────────────────────────────────────


def three_way_check(merged):
    """Classify winner disagreement patterns among BT, PM, and annotations."""
    print("\n" + "=" * 70)
    print("THREE-WAY VALIDITY CHECK")
    print("=" * 70)

    patterns = defaultdict(int)

    for m in merged:
        bt_w = int(np.argmax(m["r_BT"]))
        borda_w = int(np.argmax(m["tab_borda"]))
        pm_borda = m["P_PM"].sum(axis=1) - 0.5
        pm_w = int(np.argmax(pm_borda))

        bt_borda = (bt_w == borda_w)
        bt_pm = (bt_w == pm_w)
        borda_pm = (borda_w == pm_w)

        if bt_borda and bt_pm:
            patterns["all_agree"] += 1
        elif bt_pm and not bt_borda:
            # Both models agree, annotations differ → beneficial generalization
            patterns["bt_pm_agree_borda_diff"] += 1
        elif bt_borda and not bt_pm:
            # BT matches annotations but PM differs → PM captures intransitivity
            patterns["bt_borda_agree_pm_diff"] += 1
        elif borda_pm and not bt_borda:
            # Annotations and PM agree, BT differs → likely BT inflation
            patterns["borda_pm_agree_bt_diff"] += 1
        else:
            patterns["all_disagree"] += 1

    n = len(merged)
    labels = {
        "all_agree": "All agree (no divergence)",
        "bt_pm_agree_borda_diff": "BT=PM, Borda differs (beneficial generalization)",
        "bt_borda_agree_pm_diff": "BT=Borda, PM differs (PM sees more structure)",
        "borda_pm_agree_bt_diff": "Borda=PM, BT differs (potential BT inflation)",
        "all_disagree": "All three disagree (complex divergence)",
    }

    print(f"\n  Winner disagreement patterns ({n} prompts):")
    for pat in ["all_agree", "bt_pm_agree_borda_diff", "bt_borda_agree_pm_diff",
                "borda_pm_agree_bt_diff", "all_disagree"]:
        c = patterns[pat]
        print(f"    {labels[pat]:55s}  {c:>5d} ({100 * c / n:.1f}%)")

    return dict(patterns)


# ── FIGURES ──────────────────────────────────────────────────────────────────


def make_figures(results, output_dir):
    """Generate all 6 analysis figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Pairwise calibration ──
    p1 = results["part1"]
    p_emp, p_bt, p_pm = p1["_p_emp"], p1["_p_bt"], p1["_p_pm"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, pmod, name, brier, cal_key in [
        (axes[0], p_bt, "P_BT (parametric RM)", p1["brier_bt"], "calibration_bt"),
        (axes[1], p_pm, "P_PM (preference model)", p1["brier_pm"], "calibration_pm"),
    ]:
        ax.hexbin(pmod, p_emp, gridsize=30, cmap="Blues", mincnt=1)
        ax.plot([0, 1], [0, 1], "r--", alpha=0.7, label="Perfect calibration")
        cal = p1[cal_key]
        ax.plot(cal["centers"], cal["means"], "ro-", ms=5, label="Binned mean")
        ax.set_xlabel(name)
        ax.set_ylabel("P_emp (annotations)")
        ax.set_title(f"{name}\nBrier = {brier:.4f}")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")

    fig.suptitle("Pairwise Probability Calibration")
    fig.tight_layout()
    fig.savefig(fig_dir / "01_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 01_calibration.png")

    # ── Figure 2: Logit-transitivity ──
    p2 = results["part2"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].hist(np.abs(p2["_eps_pm"]), bins=50, alpha=0.6, color="blue",
                 density=True, label=f"PM (mean={np.abs(p2['_eps_pm']).mean():.3f})")
    axes[0].hist(np.abs(p2["_eps_emp"]), bins=50, alpha=0.6, color="orange",
                 density=True, label=f"Emp (mean={np.abs(p2['_eps_emp']).mean():.3f})")
    axes[0].set_xlabel("|epsilon| (logit-transitivity residual)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Logit-Transitivity Violations")
    axes[0].legend()

    axes[1].hist(p2["_bt_distortion"], bins=50, alpha=0.7, color="green", density=True)
    axes[1].set_xlabel("|P_BT_fit - P_PM|")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"BT-MLE Distortion of PM\nmean={p2['bt_distortion_mean']:.4f}")

    fig.suptitle("BT Misspecification")
    fig.tight_layout()
    fig.savefig(fig_dir / "02_logit_transitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 02_logit_transitivity.png")

    # ── Figure 3: Winner agreement heatmap ──
    p3 = results["part3"]
    agree = np.array(p3["winner_agreement"]) * 100
    names = p3["names"]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(agree, cmap="YlOrRd", vmin=40, vmax=100)
    for i in range(len(names)):
        for j in range(len(names)):
            color = "white" if agree[i, j] > 70 else "black"
            ax.text(j, i, f"{agree[i, j]:.1f}%", ha="center", va="center",
                    fontsize=11, color=color)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_title("Winner Agreement (%)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_winner_agreement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 03_winner_agreement.png")

    # ── Figure 4: Cross-prompt inflation vs features ──
    p4 = results["part4"]
    dr = p4["_delta_ranks"]
    lens4 = p4["_lengths"]
    sycs4 = p4["_sycs"]
    dims4 = p4["_dim_vals"]

    feature_list = [
        ("Length (words)", lens4, False),
        ("Helpfulness", dims4.get("helpfulness", []), True),
        ("Sycophancy proxy", sycs4, True),
        ("Instruction following", dims4.get("instruction_following", []), True),
        ("Honesty", dims4.get("honesty", []), True),
        ("Truthfulness", dims4.get("truthfulness", []), True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for idx, (fname, fvals, has_none) in enumerate(feature_list):
        ax = axes[idx // 3, idx % 3]
        if has_none:
            x = np.array([v if v is not None else np.nan for v in fvals])
            valid = ~np.isnan(x)
        else:
            x = np.asarray(fvals, dtype=float)
            valid = np.ones(len(x), dtype=bool)

        if valid.sum() == 0:
            ax.set_title(fname + " (no data)")
            continue

        ax.scatter(x[valid], dr[valid], alpha=0.02, s=1, color="steelblue")

        # Binned mean trend
        xv = x[valid]
        unique_vals = np.unique(xv)
        if len(unique_vals) > 10:
            edges = np.percentile(xv, np.linspace(0, 100, 11))
        else:
            edges = np.append(unique_vals - 0.5, unique_vals[-1] + 0.5)

        if len(edges) > 1:
            centres, means = [], []
            for b in range(len(edges) - 1):
                mask = valid.copy()
                mask &= (x >= edges[b]) & (x < edges[b + 1])
                if b == len(edges) - 2:
                    mask |= valid & (x == edges[b + 1])
                if mask.sum() > 0:
                    centres.append(float(x[mask].mean()))
                    means.append(float(dr[mask].mean()))
            if centres:
                ax.plot(centres, means, "r-o", ms=5, lw=2, label="Binned mean")
                ax.legend(fontsize=8)

        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel(fname)
        ax.set_ylabel("delta_rank (BT_param − Borda_tab)")
        ax.set_title(fname)

    fig.suptitle("Cross-Prompt Inflation vs Response Features", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / "04_cross_prompt_inflation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 04_cross_prompt_inflation.png")

    # ── Figure 5: Win-rate scatter ──
    p5 = results["part5"]
    r_borda = p5["_r_borda"]
    r_bt_eff = p5["_r_bt"]
    r_nlhf = p5["_r_nlhf"]
    lens5 = p5["_lengths"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc = axes[0].scatter(r_borda, r_bt_eff, c=lens5, cmap="viridis",
                         alpha=0.15, s=2, vmin=0, vmax=500)
    axes[0].plot([0, 1], [0, 1], "r--", alpha=0.7, label="y = x")
    axes[0].set_xlabel("r_eff_Borda (annotations)")
    axes[0].set_ylabel("r_eff_BT (parametric RM)")
    axes[0].set_title("Cross-Prompt Effect: BT vs Annotation Win Rate")
    axes[0].legend()
    fig.colorbar(sc, ax=axes[0], label="Length (words)")

    sc = axes[1].scatter(r_nlhf, r_bt_eff, c=lens5, cmap="viridis",
                         alpha=0.15, s=2, vmin=0, vmax=500)
    axes[1].plot([0, 1], [0, 1], "r--", alpha=0.7, label="y = x")
    axes[1].set_xlabel("r_eff_NLHF (Nash-weighted win rate)")
    axes[1].set_ylabel("r_eff_BT (parametric RM)")
    axes[1].set_title("Total BT Inflation: BT vs NLHF Effective Reward")
    axes[1].legend()
    fig.colorbar(sc, ax=axes[1], label="Length (words)")

    fig.suptitle("Effective Rewards: Parametric BT vs Annotation-Based Estimates")
    fig.tight_layout()
    fig.savefig(fig_dir / "05_effective_rewards.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 05_effective_rewards.png")

    # ── Figure 6: RLHF vs NLHF policy simulation ──
    p6 = results["part6"]
    betas = p6["betas"]
    rbeta = {float(k): v for k, v in p6["by_beta"].items()}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    K = 4

    # Entropy
    ent_rlhf = [rbeta[b]["entropy_rlhf"] for b in betas]
    ent_nlhf_val = rbeta[betas[0]]["entropy_nlhf"]
    axes[0, 0].semilogx(betas, ent_rlhf, "b-o", label="RLHF")
    axes[0, 0].axhline(ent_nlhf_val, color="orange", ls="--",
                        label=f"NLHF ({ent_nlhf_val:.2f})")
    axes[0, 0].axhline(np.log(K), color="gray", ls=":",
                        label=f"Uniform ({np.log(K):.2f})")
    axes[0, 0].set_xlabel("beta"); axes[0, 0].set_ylabel("Entropy")
    axes[0, 0].set_title("Policy Entropy"); axes[0, 0].legend()

    # Length amplification
    amp_len = [rbeta[b]["amp_length_rlhf"] for b in betas]
    nlhf_len = rbeta[betas[0]]["amp_length_nlhf"]
    axes[0, 1].semilogx(betas, amp_len, "b-o", label="RLHF")
    axes[0, 1].axhline(nlhf_len, color="orange", ls="--",
                        label=f"NLHF ({nlhf_len:.1f})")
    axes[0, 1].axhline(0, color="gray", ls=":")
    axes[0, 1].set_xlabel("beta")
    axes[0, 1].set_ylabel("E[length] − E_uniform[length]")
    axes[0, 1].set_title("Length Amplification"); axes[0, 1].legend()

    # Sycophancy amplification
    amp_syc = [rbeta[b].get("amp_syc_rlhf") for b in betas]
    if amp_syc[0] is not None:
        nlhf_syc = rbeta[betas[0]]["amp_syc_nlhf"]
        axes[1, 0].semilogx(betas, amp_syc, "b-o", label="RLHF")
        axes[1, 0].axhline(nlhf_syc, color="orange", ls="--",
                            label=f"NLHF ({nlhf_syc:.4f})")
        axes[1, 0].axhline(0, color="gray", ls=":")
    axes[1, 0].set_xlabel("beta")
    axes[1, 0].set_ylabel("E[sycophancy] − E_uniform[sycophancy]")
    axes[1, 0].set_title("Sycophancy Amplification"); axes[1, 0].legend()

    # Borda win-rate amplification
    amp_borda = [rbeta[b]["amp_borda_rlhf"] for b in betas]
    nlhf_borda = rbeta[betas[0]]["amp_borda_nlhf"]
    axes[1, 1].semilogx(betas, amp_borda, "b-o", label="RLHF")
    axes[1, 1].axhline(nlhf_borda, color="orange", ls="--",
                        label=f"NLHF ({nlhf_borda:.4f})")
    axes[1, 1].axhline(0, color="gray", ls=":")
    axes[1, 1].set_xlabel("beta")
    axes[1, 1].set_ylabel("E[r_eff_Borda] − E_uniform[r_eff_Borda]")
    axes[1, 1].set_title("Win-Rate Amplification"); axes[1, 1].legend()

    fig.suptitle("RLHF vs NLHF: Policy Amplification by beta", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / "06_policy_simulation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 06_policy_simulation.png")


# ── MAIN ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Cross-prompt Borda inflation: parametric BT vs annotations")
    parser.add_argument("--model_results", type=str, required=True,
                        help="Path to inflation_results.pkl")
    parser.add_argument("--tabular_results", type=str, required=True,
                        help="Path to tabular_results.json")
    parser.add_argument("--ufb_dataset", type=str, default="openbmb/UltraFeedback")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/borda_inflation/cross_prompt_results")
    parser.add_argument("--betas", type=str,
                        default="0.01,0.05,0.1,0.5,1.0,2.0,5.0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    betas = [float(b) for b in args.betas.split(",")]

    # Part 0: Load and merge
    merged = load_and_merge(args.model_results, args.tabular_results,
                            args.ufb_dataset)
    print(f"\n{'=' * 70}")
    print(f"MERGED: {len(merged)} prompts with aligned P_emp, P_BT, P_PM")
    print(f"{'=' * 70}")

    if not merged:
        print("ERROR: No prompts matched. Check data paths.")
        sys.exit(1)

    # Run all analyses
    results = {}
    results["part1"] = part1_pairwise_probs(merged)
    results["part2"] = part2_logit_transitivity(merged)
    results["part3"] = part3_rankings(merged)
    results["part4"] = part4_cross_prompt_inflation(merged)
    results["part5"] = part5_effective_rewards(merged)
    results["part6"] = part6_policy_simulation(merged, betas)
    results["validity"] = three_way_check(merged)

    # Figures
    print(f"\n{'=' * 70}")
    print("GENERATING FIGURES")
    print(f"{'=' * 70}")
    make_figures(results, output_dir)

    # Save JSON summary (strip numpy arrays)
    summary = {}
    for key, val in results.items():
        if isinstance(val, dict):
            summary[key] = {k: v for k, v in val.items() if not k.startswith("_")}
        else:
            summary[key] = val

    summary_path = output_dir / "cross_prompt_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary to {summary_path}")

    # Save merged data for further analysis
    merged_path = output_dir / "merged_data.pkl"
    with open(merged_path, "wb") as f:
        pickle.dump(merged, f)
    print(f"Saved merged data to {merged_path}")

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
