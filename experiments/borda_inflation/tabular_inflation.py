#!/usr/bin/env python3
"""Model-free Borda inflation analysis on UltraFeedback annotations.

For each prompt with multiple responses, constructs a pairwise preference matrix
from the 4 UltraFeedback dimension scores using majority rule, then compares:
  - Borda winner (row sums of pairwise matrix)
  - Condorcet winner (beats all others pairwise)
  - Nash equilibrium / maximal lottery (LP with entropy regularization for cycles)

This is a pure social-choice analysis — no learned models involved.

Usage:
    python experiments/borda_inflation/tabular_inflation.py \
        --output_dir experiments/borda_inflation/tabular_results

    # On cluster with 72 cores:
    python experiments/borda_inflation/tabular_inflation.py \
        --output_dir /path/to/results --n_workers 64

        python tabular_inflation.py --output_dir ./tabular_results --n_workers 64
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize as sp_minimize

DIMENSIONS = ["instruction_following", "honesty", "truthfulness", "helpfulness"]


# ── SOCIAL CHOICE FUNCTIONS ──────────────────────────────────────────────────


def majority_pref_matrix(dim_scores):
    """Build pairwise preference matrix using dimension-wise majority rule.

    P[i,j] = fraction of dimensions where response i scores strictly higher
    than response j.  Ties on a dimension count as 0.5 for each side.

    Args:
        dim_scores: dict mapping dimension name -> list of scores (one per response).
                    Scores can be int or None (missing).

    Returns:
        P: (n, n) preference matrix with P[i,j] in [0, 1], diagonal = 0.5.
        valid_dims: number of dimensions with valid scores for each (i,j) pair.
    """
    n = len(next(iter(dim_scores.values())))
    P = np.full((n, n), 0.5)
    valid_dims = np.zeros((n, n), dtype=int)

    for dim_name, scores in dim_scores.items():
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                si, sj = scores[i], scores[j]
                if si is None or sj is None:
                    continue
                valid_dims[i, j] += 1
                if si > sj:
                    P[i, j] += 1.0
                elif si == sj:
                    P[i, j] += 0.5
                # si < sj: add nothing (0)

    # Normalize: divide accumulated wins by number of valid dimensions
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if valid_dims[i, j] > 0:
                # P[i,j] currently has 0.5 (init) + accumulated wins
                # We want: fraction of valid dims where i beats j (with ties as 0.5)
                P[i, j] = (P[i, j] - 0.5) / valid_dims[i, j]
            else:
                P[i, j] = 0.5  # no valid comparison

    return P, valid_dims


def avg_score_pref_matrix(dim_scores):
    """Build pairwise preference matrix using averaged scores.

    P[i,j] = sigmoid(avg_score_i - avg_score_j) using a simple step function:
    1.0 if avg_i > avg_j, 0.5 if equal, 0.0 if avg_i < avg_j.

    This produces a TRANSITIVE matrix (sanity check: Borda = Condorcet always).
    """
    n = len(next(iter(dim_scores.values())))
    avg_scores = []

    for resp_idx in range(n):
        valid = []
        for dim_name in DIMENSIONS:
            s = dim_scores[dim_name][resp_idx]
            if s is not None:
                valid.append(s)
        avg_scores.append(np.mean(valid) if valid else None)

    P = np.full((n, n), 0.5)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if avg_scores[i] is None or avg_scores[j] is None:
                continue
            if avg_scores[i] > avg_scores[j]:
                P[i, j] = 1.0
            elif avg_scores[i] < avg_scores[j]:
                P[i, j] = 0.0
            # equal: stays 0.5

    return P, avg_scores


def borda_scores(P):
    """Borda score = row sum of preference matrix (excluding diagonal)."""
    return P.sum(axis=1) - 0.5  # subtract the 0.5 diagonal


def condorcet_winner(P):
    """Find Condorcet winner: response that beats all others pairwise (P[i,j] > 0.5).

    Returns index or None if no Condorcet winner exists.
    """
    n = P.shape[0]
    for i in range(n):
        if all(P[i, j] > 0.5 for j in range(n) if j != i):
            return i
    return None


def copeland_scores(P):
    """Copeland score = number of pairwise victories (P[i,j] > 0.5)."""
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
    """Nash equilibrium (maximal lottery) via LP + max-entropy tiebreaking.

    Solves the minimax problem for M = P - 0.5.
    Phase 1: LP for game value.
    Phase 2: max-entropy strategy among all optima.

    Returns (pi, value).
    """
    M = P - 0.5
    n = M.shape[0]

    if n == 1:
        return np.array([1.0]), 0.0

    # Phase 1: find game value via LP
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
        # Fallback: uniform
        return np.ones(n) / n, 0.0

    v_star = -result.fun

    # Phase 2: max-entropy among all optima
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


# ── PER-PROMPT ANALYSIS ──────────────────────────────────────────────────────


def analyze_prompt(prompt_data):
    """Analyze a single prompt. Returns a result dict or None on failure.

    Args:
        prompt_data: tuple of (instruction, completions_data) where
                     completions_data is a list of dicts with keys:
                     'response', 'dim_scores' (dict dim->score or None),
                     'avg_score', 'length'
    """
    instruction, completions = prompt_data
    n = len(completions)

    if n < 2:
        return None

    # Extract dimension scores
    dim_scores = {}
    for dim in DIMENSIONS:
        dim_scores[dim] = [c['dim_scores'].get(dim) for c in completions]

    # Check we have at least some valid scores
    total_valid = sum(
        1 for dim in DIMENSIONS
        for s in dim_scores[dim] if s is not None
    )
    if total_valid == 0:
        return None

    # === Majority rule preference matrix ===
    P_maj, valid_dims = majority_pref_matrix(dim_scores)

    # === Average score preference matrix (transitive baseline) ===
    P_avg, avg_scores = avg_score_pref_matrix(dim_scores)

    # === Social choice analysis on majority matrix ===
    borda = borda_scores(P_maj)
    borda_winner = int(np.argmax(borda))
    borda_ranking = np.argsort(-borda)  # descending

    copeland = copeland_scores(P_maj)
    copeland_winner = int(np.argmax(copeland))

    cw = condorcet_winner(P_maj)

    # Nash equilibrium (maximal lottery)
    nash_pi, nash_value = nash_equilibrium_lp(P_maj)
    nash_winner = int(np.argmax(nash_pi))

    # === Also analyze average-score matrix (transitive baseline) ===
    borda_avg = borda_scores(P_avg)
    borda_avg_winner = int(np.argmax(borda_avg))
    cw_avg = condorcet_winner(P_avg)

    # === Check for cycles ===
    # A cycle exists if no Condorcet winner in majority matrix
    has_cycle = cw is None

    # === Borda vs Condorcet disagreement ===
    # Use Copeland winner as Condorcet proxy when no strict Condorcet winner
    pairwise_winner = cw if cw is not None else copeland_winner

    borda_condorcet_agree = (borda_winner == pairwise_winner)

    # === Characterize responses ===
    response_data = []
    for idx, c in enumerate(completions):
        response_data.append({
            'idx': idx,
            'length': c['length'],
            'avg_score': c['avg_score'],
            'dim_scores': c['dim_scores'],
            'borda_score': float(borda[idx]),
            'copeland_score': float(copeland[idx]),
            'nash_weight': float(nash_pi[idx]),
            'is_borda_winner': idx == borda_winner,
            'is_condorcet_winner': idx == cw if cw is not None else False,
            'is_nash_winner': idx == nash_winner,
        })

    # === Inflation per response ===
    borda_ranks = np.empty(n, dtype=int)
    borda_ranks[np.argsort(-borda)] = np.arange(n)

    copeland_ranks = np.empty(n, dtype=int)
    copeland_ranks[np.argsort(-copeland)] = np.arange(n)

    inflation = copeland_ranks - borda_ranks  # positive = Borda overvalues

    # === Sycophancy proxy per response ===
    syc_proxies = []
    for c in completions:
        ds = c['dim_scores']
        h = ds.get('helpfulness')
        o = ds.get('honesty')
        t = ds.get('truthfulness')
        if h is not None and (o is not None or t is not None):
            min_ot = min(x for x in [o, t] if x is not None)
            syc_proxies.append(h - min_ot)
        else:
            syc_proxies.append(None)

    return {
        'instruction': instruction[:200],
        'n_responses': n,
        'has_cycle': has_cycle,
        'condorcet_winner': cw,
        'borda_winner': borda_winner,
        'copeland_winner': copeland_winner,
        'nash_winner': nash_winner,
        'borda_condorcet_agree': borda_condorcet_agree,
        'nash_value': float(nash_value),
        'P_majority': P_maj.tolist(),
        'borda_scores': borda.tolist(),
        'copeland_scores': copeland.tolist(),
        'nash_pi': nash_pi.tolist(),
        'inflation': inflation.tolist(),
        'response_data': response_data,
        'syc_proxies': syc_proxies,
        'avg_scores': [c['avg_score'] for c in completions],
        # Characterize the disagreement
        'borda_winner_length': completions[borda_winner]['length'],
        'pairwise_winner_length': completions[pairwise_winner]['length'],
        'borda_winner_avg': completions[borda_winner]['avg_score'],
        'pairwise_winner_avg': completions[pairwise_winner]['avg_score'],
        'borda_winner_dims': completions[borda_winner]['dim_scores'],
        'pairwise_winner_dims': completions[pairwise_winner]['dim_scores'],
        'borda_winner_syc': syc_proxies[borda_winner],
        'pairwise_winner_syc': syc_proxies[pairwise_winner],
    }


# ── DATA LOADING ─────────────────────────────────────────────────────────────


def load_ultrafeedback(dataset_name="openbmb/UltraFeedback"):
    """Load UltraFeedback and extract per-prompt, per-response dimension scores.

    Returns list of (instruction, completions_data) tuples.
    """
    from datasets import load_dataset

    print(f"Loading {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    print(f"  {len(dataset)} rows")

    prompts = []
    skipped = 0

    for row in dataset:
        instruction = row["instruction"]
        completions_raw = row["completions"]

        if not completions_raw or len(completions_raw) < 2:
            skipped += 1
            continue

        completions = []
        for comp in completions_raw:
            response = comp.get("response", "")
            annotations = comp.get("annotations", {})

            dim_scores = {}
            for dim in DIMENSIONS:
                try:
                    rating = int(annotations[dim]["Rating"])
                    dim_scores[dim] = rating if rating != -1 else None
                except (KeyError, TypeError, ValueError):
                    dim_scores[dim] = None

            valid = [v for v in dim_scores.values() if v is not None]
            avg_score = np.mean(valid) if valid else None
            length = len(response.split())

            completions.append({
                'response': response[:500],  # truncate for memory
                'dim_scores': dim_scores,
                'avg_score': avg_score,
                'length': length,
            })

        # Need at least 2 responses with some valid scores
        has_scores = sum(
            1 for c in completions
            if any(v is not None for v in c['dim_scores'].values())
        )
        if has_scores < 2:
            skipped += 1
            continue

        prompts.append((instruction, completions))

    print(f"  {len(prompts)} prompts with >= 2 scored responses ({skipped} skipped)")

    # Distribution of response counts
    counts = defaultdict(int)
    for _, comps in prompts:
        counts[len(comps)] += 1
    print("  Response count distribution:")
    for k in sorted(counts):
        print(f"    {k} responses: {counts[k]} prompts")

    return prompts


# ── AGGREGATION AND REPORTING ────────────────────────────────────────────────


def aggregate_results(results):
    """Aggregate per-prompt results into summary statistics."""
    n_total = len(results)
    n_cycle = sum(1 for r in results if r['has_cycle'])
    n_agree = sum(1 for r in results if r['borda_condorcet_agree'])
    n_disagree = n_total - n_agree

    # Among disagreeing prompts: characterize Borda vs pairwise winners
    borda_lengths = []
    pairwise_lengths = []
    borda_avgs = []
    pairwise_avgs = []
    borda_sycs = []
    pairwise_sycs = []
    borda_dims = defaultdict(list)
    pairwise_dims = defaultdict(list)

    for r in results:
        if r['borda_condorcet_agree']:
            continue
        borda_lengths.append(r['borda_winner_length'])
        pairwise_lengths.append(r['pairwise_winner_length'])
        if r['borda_winner_avg'] is not None:
            borda_avgs.append(r['borda_winner_avg'])
        if r['pairwise_winner_avg'] is not None:
            pairwise_avgs.append(r['pairwise_winner_avg'])
        if r['borda_winner_syc'] is not None:
            borda_sycs.append(r['borda_winner_syc'])
        if r['pairwise_winner_syc'] is not None:
            pairwise_sycs.append(r['pairwise_winner_syc'])
        for dim in DIMENSIONS:
            bv = r['borda_winner_dims'].get(dim)
            pv = r['pairwise_winner_dims'].get(dim)
            if bv is not None:
                borda_dims[dim].append(bv)
            if pv is not None:
                pairwise_dims[dim].append(pv)

    # Correlation: inflation vs features across ALL responses
    all_inflation = []
    all_lengths = []
    all_sycs = []
    all_dim_vals = {d: [] for d in DIMENSIONS}

    for r in results:
        for idx, rd in enumerate(r['response_data']):
            inf = r['inflation'][idx]
            all_inflation.append(inf)
            all_lengths.append(rd['length'])
            syc = r['syc_proxies'][idx]
            all_sycs.append(syc)
            for dim in DIMENSIONS:
                all_dim_vals[dim].append(rd['dim_scores'].get(dim))

    from scipy.stats import spearmanr, mannwhitneyu

    # Correlations
    correlations = {}
    inf_arr = np.array(all_inflation, dtype=float)

    # Length
    len_arr = np.array(all_lengths, dtype=float)
    rho, p = spearmanr(inf_arr, len_arr)
    correlations['length'] = {'rho': rho, 'p': p, 'n': len(inf_arr)}

    # Sycophancy proxy
    valid = [(i, s) for i, s in zip(all_inflation, all_sycs) if s is not None]
    if valid:
        inf_v, syc_v = zip(*valid)
        rho, p = spearmanr(inf_v, syc_v)
        correlations['sycophancy_proxy'] = {'rho': rho, 'p': p, 'n': len(inf_v)}

    # Per dimension
    for dim in DIMENSIONS:
        valid = [(i, s) for i, s in zip(all_inflation, all_dim_vals[dim])
                 if s is not None]
        if valid:
            inf_v, dim_v = zip(*valid)
            rho, p = spearmanr(inf_v, dim_v)
            correlations[dim] = {'rho': rho, 'p': p, 'n': len(inf_v)}

    # Mann-Whitney U: Borda winners vs pairwise winners on disagreeing prompts
    mw_tests = {}
    if borda_lengths and pairwise_lengths:
        u, p = mannwhitneyu(borda_lengths, pairwise_lengths, alternative='two-sided')
        mw_tests['length'] = {
            'borda_mean': float(np.mean(borda_lengths)),
            'pairwise_mean': float(np.mean(pairwise_lengths)),
            'U': float(u), 'p': float(p)
        }
    if borda_sycs and pairwise_sycs:
        u, p = mannwhitneyu(borda_sycs, pairwise_sycs, alternative='two-sided')
        mw_tests['sycophancy_proxy'] = {
            'borda_mean': float(np.mean(borda_sycs)),
            'pairwise_mean': float(np.mean(pairwise_sycs)),
            'U': float(u), 'p': float(p)
        }
    for dim in DIMENSIONS:
        bv, pv = borda_dims.get(dim, []), pairwise_dims.get(dim, [])
        if bv and pv:
            u, p = mannwhitneyu(bv, pv, alternative='two-sided')
            mw_tests[dim] = {
                'borda_mean': float(np.mean(bv)),
                'pairwise_mean': float(np.mean(pv)),
                'U': float(u), 'p': float(p)
            }

    # Response count breakdown for cycles
    cycle_by_n = defaultdict(int)
    for r in results:
        if r['has_cycle']:
            cycle_by_n[r['n_responses']] += 1

    return {
        'n_prompts': n_total,
        'n_agree': n_agree,
        'n_disagree': n_disagree,
        'n_cycle': n_cycle,
        'pct_agree': 100 * n_agree / n_total if n_total > 0 else 0,
        'pct_disagree': 100 * n_disagree / n_total if n_total > 0 else 0,
        'pct_cycle': 100 * n_cycle / n_total if n_total > 0 else 0,
        'cycle_by_n_responses': dict(cycle_by_n),
        'correlations': correlations,
        'mann_whitney': mw_tests,
    }


def print_report(summary, results):
    """Print human-readable report."""
    print("\n" + "=" * 70)
    print("TABULAR BORDA INFLATION ANALYSIS")
    print("(Model-free, majority-rule preferences from UltraFeedback dimensions)")
    print("=" * 70)

    print(f"\n  Prompts analyzed:  {summary['n_prompts']}")
    print(f"  Borda = Pairwise:  {summary['n_agree']} ({summary['pct_agree']:.1f}%)")
    print(f"  Borda != Pairwise: {summary['n_disagree']} ({summary['pct_disagree']:.1f}%)")
    print(f"  Condorcet cycles:  {summary['n_cycle']} ({summary['pct_cycle']:.1f}%)")

    if summary['cycle_by_n_responses']:
        print("\n  Cycles by # responses:")
        for k in sorted(summary['cycle_by_n_responses']):
            print(f"    {k} responses: {summary['cycle_by_n_responses'][k]}")

    print("\n" + "-" * 70)
    print("CORRELATIONS: inflation (Copeland_rank - Borda_rank) vs features")
    print("-" * 70)
    for name, c in summary['correlations'].items():
        sig = "***" if c['p'] < 0.001 else "**" if c['p'] < 0.01 else "*" if c['p'] < 0.05 else ""
        print(f"  {name:30s}  rho={c['rho']:+.4f}  p={c['p']:.2e}  n={c['n']}  {sig}")

    print("\n" + "-" * 70)
    print("MANN-WHITNEY U: Borda winners vs Pairwise winners (disagreeing prompts)")
    print("-" * 70)
    for name, t in summary['mann_whitney'].items():
        sig = "***" if t['p'] < 0.001 else "**" if t['p'] < 0.01 else "*" if t['p'] < 0.05 else ""
        print(f"  {name:30s}  Borda={t['borda_mean']:.2f}  Pairwise={t['pairwise_mean']:.2f}  "
              f"p={t['p']:.2e}  {sig}")

    # Examples of disagreements
    disagreements = [r for r in results if not r['borda_condorcet_agree']]
    if disagreements:
        print("\n" + "-" * 70)
        print(f"EXAMPLE DISAGREEMENTS (first 10 of {len(disagreements)})")
        print("-" * 70)
        for r in disagreements[:10]:
            bw = r['borda_winner']
            pw = r['copeland_winner'] if r['condorcet_winner'] is None else r['condorcet_winner']
            print(f"\n  Prompt: {r['instruction'][:120]}...")
            print(f"  {r['n_responses']} responses, cycle={r['has_cycle']}")
            print(f"  Borda winner (idx={bw}): score={r['borda_scores'][bw]:.2f}, "
                  f"len={r['response_data'][bw]['length']}, "
                  f"dims={r['response_data'][bw]['dim_scores']}")
            print(f"  Pairwise winner (idx={pw}): score={r['borda_scores'][pw]:.2f}, "
                  f"len={r['response_data'][pw]['length']}, "
                  f"dims={r['response_data'][pw]['dim_scores']}")
            print(f"  P matrix:\n{np.array(r['P_majority'])}")


# ── MAIN ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Model-free Borda inflation analysis on UltraFeedback"
    )
    parser.add_argument("--dataset", type=str, default="openbmb/UltraFeedback",
                        help="HuggingFace dataset name")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/borda_inflation/tabular_results")
    parser.add_argument("--n_workers", type=int, default=0,
                        help="Number of parallel workers (0 = auto: ncpu - 4)")
    parser.add_argument("--chunk_size", type=int, default=200,
                        help="Chunk size for multiprocessing")
    args = parser.parse_args()

    import multiprocessing
    if args.n_workers <= 0:
        args.n_workers = max(1, multiprocessing.cpu_count() - 4)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    prompts = load_ultrafeedback(args.dataset)

    # Process in parallel
    print(f"\nAnalyzing {len(prompts)} prompts with {args.n_workers} workers...")

    results = []
    failed = 0

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {
            executor.submit(analyze_prompt, p): i
            for i, p in enumerate(prompts)
        }

        from tqdm import tqdm
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Analyzing prompts"):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  Error: {e}")

    print(f"\n  Analyzed: {len(results)}, Failed/skipped: {failed}")

    # Aggregate
    summary = aggregate_results(results)

    # Print report
    print_report(summary, results)

    # Save
    summary_path = output_dir / "tabular_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary to {summary_path}")

    # Save full results (for further analysis)
    results_path = output_dir / "tabular_results.json"
    # Only save a subset of fields to keep file size manageable
    compact_results = []
    for r in results:
        compact_results.append({
            'instruction': r['instruction'],
            'n_responses': r['n_responses'],
            'has_cycle': r['has_cycle'],
            'condorcet_winner': r['condorcet_winner'],
            'borda_winner': r['borda_winner'],
            'copeland_winner': r['copeland_winner'],
            'nash_winner': r['nash_winner'],
            'borda_condorcet_agree': r['borda_condorcet_agree'],
            'borda_scores': r['borda_scores'],
            'copeland_scores': r['copeland_scores'],
            'nash_pi': r['nash_pi'],
            'inflation': r['inflation'],
            'avg_scores': r['avg_scores'],
            'syc_proxies': r['syc_proxies'],
            'P_majority': r['P_majority'],
            'borda_winner_length': r['borda_winner_length'],
            'pairwise_winner_length': r['pairwise_winner_length'],
        })

    with open(results_path, "w") as f:
        json.dump(compact_results, f)
    print(f"Saved {len(compact_results)} results to {results_path}")

    # Save disagreement examples
    disagree_path = output_dir / "disagreements.json"
    disagreements = [r for r in compact_results if not r['borda_condorcet_agree']]
    with open(disagree_path, "w") as f:
        json.dump(disagreements[:500], f, indent=2)
    print(f"Saved {min(500, len(disagreements))} disagreement examples to {disagree_path}")


if __name__ == "__main__":
    main()
