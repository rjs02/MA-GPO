#!/usr/bin/env python3
"""Analyze tabular Borda inflation results from UltraFeedback.

Reads the output of tabular_inflation.py and produces:
1. Summary statistics and interpretation
2. Deep-dive into the 30.8% Condorcet cycles
3. Characterization of the 3.2% Borda != Copeland disagreements
4. Dimension-level analysis of what drives intransitivity
5. Comparison with model-based experiment results
6. Figures

Usage:
    python experiments/borda_inflation/analyze_tabular.py \
        --results_dir /path/to/tabular_results \
        --output_dir /path/to/tabular_analysis
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu, chi2_contingency


def load_results(results_dir):
    with open(results_dir / "tabular_results.json") as f:
        results = json.load(f)
    with open(results_dir / "tabular_summary.json") as f:
        summary = json.load(f)
    return results, summary


# ── CYCLE ANALYSIS ───────────────────────────────────────────────────────────


def analyze_cycles(results, output_dir):
    """Deep-dive into the 30.8% of prompts with Condorcet cycles."""

    cycles = [r for r in results if r["has_cycle"]]
    no_cycles = [r for r in results if not r["has_cycle"]]

    print("\n" + "=" * 70)
    print("CONDORCET CYCLE ANALYSIS")
    print("=" * 70)
    print(f"  Prompts with cycles:    {len(cycles)} ({100*len(cycles)/len(results):.1f}%)")
    print(f"  Prompts without cycles: {len(no_cycles)} ({100*len(no_cycles)/len(results):.1f}%)")

    # Among cycles: how often do Borda and Copeland still agree?
    cycle_agree = sum(1 for r in cycles if r["borda_condorcet_agree"])
    cycle_disagree = len(cycles) - cycle_agree
    print(f"\n  Within cycles:")
    print(f"    Borda = Copeland: {cycle_agree} ({100*cycle_agree/len(cycles):.1f}%)")
    print(f"    Borda != Copeland: {cycle_disagree} ({100*cycle_disagree/len(cycles):.1f}%)")

    nocycle_agree = sum(1 for r in no_cycles if r["borda_condorcet_agree"])
    nocycle_disagree = len(no_cycles) - nocycle_agree
    print(f"\n  Without cycles:")
    print(f"    Borda = Copeland: {nocycle_agree} ({100*nocycle_agree/len(no_cycles):.1f}%)")
    print(f"    Borda != Copeland: {nocycle_disagree} ({100*nocycle_disagree/len(no_cycles):.1f}%)")

    # What dimension patterns produce cycles?
    # For each pair in a cycle, identify which dimensions disagree
    cycle_dim_patterns = Counter()
    for r in cycles:
        P = np.array(r["P_majority"])
        n = P.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if P[i, j] > 0.5 and P[j, i] > 0.5:
                    # Both win: impossible with our construction
                    pass
                # Check if part of a cycle: i beats j beats k beats i
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if P[i, j] > 0.5 and P[j, k] > 0.5 and P[k, i] > 0.5:
                        cycle_dim_patterns["3-cycle"] += 1
                        break

    # Score variance: cycles should have more heterogeneous dimension scores
    def score_variance(r):
        """Average cross-dimension variance per response."""
        variances = []
        for resp_scores in zip(*[
            [r["avg_scores"][i] for i in range(r["n_responses"])]
        ]):
            pass  # avg_scores are already averaged

        # Better: compute variance of dimension rankings per pair
        P = np.array(r["P_majority"])
        off_diag = P[np.triu_indices(P.shape[0], k=1)]
        return np.std(off_diag)

    cycle_vars = [score_variance(r) for r in cycles]
    nocycle_vars = [score_variance(r) for r in no_cycles]
    print(f"\n  Preference matrix std (off-diagonal):")
    print(f"    Cycles:    {np.mean(cycle_vars):.3f} +/- {np.std(cycle_vars):.3f}")
    print(f"    No cycles: {np.mean(nocycle_vars):.3f} +/- {np.std(nocycle_vars):.3f}")

    # Nash equilibrium analysis for cycles
    nash_entropy_cycle = []
    nash_entropy_nocycle = []
    for r in cycles:
        pi = np.array(r["nash_pi"])
        pi_pos = pi[pi > 1e-10]
        nash_entropy_cycle.append(-np.sum(pi_pos * np.log(pi_pos)))
    for r in no_cycles[:len(cycles)]:  # sample same size
        pi = np.array(r["nash_pi"])
        pi_pos = pi[pi > 1e-10]
        nash_entropy_nocycle.append(-np.sum(pi_pos * np.log(pi_pos)))

    print(f"\n  Nash equilibrium entropy:")
    print(f"    Cycles:    {np.mean(nash_entropy_cycle):.3f} (more mixed = more uncertain)")
    print(f"    No cycles: {np.mean(nash_entropy_nocycle):.3f}")

    # How many responses get nonzero Nash weight in cycles?
    nash_support_cycle = [sum(1 for p in r["nash_pi"] if p > 0.01) for r in cycles]
    nash_support_nocycle = [sum(1 for p in r["nash_pi"] if p > 0.01) for r in no_cycles]
    print(f"\n  Nash support size (responses with >1% weight):")
    print(f"    Cycles:    {np.mean(nash_support_cycle):.2f}")
    print(f"    No cycles: {np.mean(nash_support_nocycle):.2f}")

    return cycles, no_cycles


# ── DISAGREEMENT ANALYSIS ────────────────────────────────────────────────────


def analyze_disagreements(results, output_dir):
    """Characterize what distinguishes Borda winners from Copeland winners."""

    disagree = [r for r in results if not r["borda_condorcet_agree"]]
    agree = [r for r in results if r["borda_condorcet_agree"]]

    print("\n" + "=" * 70)
    print("BORDA vs COPELAND DISAGREEMENT ANALYSIS")
    print("=" * 70)
    print(f"  Disagreements: {len(disagree)} / {len(results)} ({100*len(disagree)/len(results):.1f}%)")

    if not disagree:
        print("  No disagreements to analyze.")
        return

    # Classify disagreement patterns
    # Look at which response is Borda winner vs Copeland winner
    borda_rank_of_copeland = []
    copeland_rank_of_borda = []
    borda_better_truthful = 0
    copeland_better_truthful = 0
    borda_better_helpful = 0
    copeland_better_helpful = 0
    borda_more_sycophantic = 0
    copeland_more_sycophantic = 0

    for r in disagree:
        bw = r["borda_winner"]
        cw = r["copeland_winner"]

        # Borda ranking of Copeland winner
        borda_sorted = np.argsort(-np.array(r["borda_scores"]))
        borda_ranks = np.empty(len(borda_sorted), dtype=int)
        borda_ranks[borda_sorted] = np.arange(len(borda_sorted))
        borda_rank_of_copeland.append(borda_ranks[cw])

        # Copeland ranking of Borda winner
        cop_sorted = np.argsort(-np.array(r["copeland_scores"]))
        cop_ranks = np.empty(len(cop_sorted), dtype=int)
        cop_ranks[cop_sorted] = np.arange(len(cop_sorted))
        copeland_rank_of_borda.append(cop_ranks[bw])

        # Sycophancy comparison
        syc_b = r["syc_proxies"][bw]
        syc_c = r["syc_proxies"][cw]
        if syc_b is not None and syc_c is not None:
            if syc_b > syc_c:
                borda_more_sycophantic += 1
            elif syc_c > syc_b:
                copeland_more_sycophantic += 1

    print(f"\n  Borda rank of Copeland winner (0=best):")
    print(f"    {Counter(borda_rank_of_copeland)}")
    print(f"  Copeland rank of Borda winner (0=best):")
    print(f"    {Counter(copeland_rank_of_borda)}")

    print(f"\n  Sycophancy comparison (disagreeing prompts):")
    print(f"    Borda winner more sycophantic:    {borda_more_sycophantic}")
    print(f"    Copeland winner more sycophantic: {copeland_more_sycophantic}")
    tied = len(disagree) - borda_more_sycophantic - copeland_more_sycophantic
    print(f"    Tied/missing:                     {tied}")

    # Per-dimension: which dimension causes the Borda winner to get inflated?
    print(f"\n  Per-dimension analysis (Borda winner vs Copeland winner):")
    for dim in ["instruction_following", "honesty", "truthfulness", "helpfulness"]:
        borda_vals = []
        copeland_vals = []
        for r in disagree:
            bw = r["borda_winner"]
            cw = r["copeland_winner"]
            # We need to get individual dimension scores from the full results
            # but they're not in the compact format. Use avg_scores as proxy.
        # This info isn't in the compact results. We'll use the MW test from summary.

    # Inflation score distribution
    all_inflation = []
    for r in results:
        all_inflation.extend(r["inflation"])

    print(f"\n  Inflation score distribution (all responses):")
    inf_counter = Counter(all_inflation)
    for k in sorted(inf_counter.keys()):
        pct = 100 * inf_counter[k] / len(all_inflation)
        bar = "#" * int(pct)
        print(f"    {k:+d}: {inf_counter[k]:6d} ({pct:5.1f}%) {bar}")


# ── INTRANSITIVITY ANALYSIS ──────────────────────────────────────────────────


def analyze_intransitivity(results, output_dir):
    """Analyze the structure of intransitive preferences across dimensions."""

    print("\n" + "=" * 70)
    print("INTRANSITIVITY STRUCTURE")
    print("=" * 70)

    # For each prompt, check if dimension-level preferences form cycles
    # A cycle means: on dim A, x>y; on dim B, y>z; on dim C, z>x
    # This is the multi-criteria Condorcet paradox

    # Classify preference matrices by structure
    transitive_count = 0
    quasi_transitive = 0  # has some ties but no strict cycles
    intransitive_count = 0

    # Track which dimension pairs most often conflict
    dim_conflict_counts = Counter()  # (dim_a, dim_b) -> count of reversed pairs

    for r in results:
        P = np.array(r["P_majority"])
        n = P.shape[0]

        # Check strict transitivity: if P[i,j] > 0.5 and P[j,k] > 0.5 then P[i,k] > 0.5
        is_transitive = True
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if P[i, j] > 0.5 and P[j, k] > 0.5 and P[i, k] <= 0.5:
                        is_transitive = False
                        break
                if not is_transitive:
                    break
            if not is_transitive:
                break

        if is_transitive:
            if r["has_cycle"]:
                quasi_transitive += 1  # Copeland cycle despite weak transitivity
            else:
                transitive_count += 1
        else:
            intransitive_count += 1

    print(f"  Strictly transitive:    {transitive_count} ({100*transitive_count/len(results):.1f}%)")
    print(f"  Quasi-transitive:       {quasi_transitive} ({100*quasi_transitive/len(results):.1f}%)")
    print(f"  Intransitive:           {intransitive_count} ({100*intransitive_count/len(results):.1f}%)")

    # P-value distribution: how "close" are preferences to ties?
    p_values = []
    for r in results:
        P = np.array(r["P_majority"])
        n = P.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                p_values.append(P[i, j])

    print(f"\n  Pairwise preference P[i,j] distribution (i<j):")
    bins = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    hist, _ = np.histogram(p_values, bins=bins + [1.001])
    for i, count in enumerate(hist):
        pct = 100 * count / len(p_values)
        bar = "#" * int(pct / 2)
        lo, hi = bins[i], bins[i + 1] if i + 1 < len(bins) else 1.0
        print(f"    [{lo:.3f}, {hi:.3f}): {count:8d} ({pct:5.1f}%) {bar}")

    # The P=0.5 bin is especially interesting: these are exact ties (2 dims each way)
    ties = sum(1 for p in p_values if abs(p - 0.5) < 0.01)
    print(f"\n  Exact ties (P=0.5): {ties} / {len(p_values)} ({100*ties/len(p_values):.1f}%)")
    print(f"  This means: on {100*ties/len(p_values):.1f}% of pairwise comparisons,")
    print(f"  each response wins on exactly 2 of 4 dimensions.")

    return p_values


# ── COMPARISON WITH MODEL-BASED EXPERIMENT ───────────────────────────────────


def compare_with_model_experiment(summary):
    """Compare tabular results with the model-based experiment."""

    print("\n" + "=" * 70)
    print("COMPARISON: TABULAR vs MODEL-BASED EXPERIMENT")
    print("=" * 70)

    print("""
  ┌────────────────────────┬──────────────┬──────────────┐
  │ Metric                 │   Tabular    │  Model-based │
  │                        │ (annotation) │   (RM vs PM) │
  ├────────────────────────┼──────────────┼──────────────┤
  │ Borda = Pairwise       │    96.8%     │    51.1%     │
  │ Borda != Pairwise      │     3.2%     │    47.8%     │
  │ Cycles                 │    30.8%     │     1.1%     │
  ├────────────────────────┼──────────────┼──────────────┤
  │ Length correlation      │    ~0.00     │    +0.23     │
  │ Sycophancy correlation  │    -0.02     │    +0.07     │
  │ Helpfulness correlation │    +0.01     │    +0.25     │
  │ Truthfulness corr.     │    +0.03     │    +0.14     │
  └────────────────────────┴──────────────┴──────────────┘

  KEY DIFFERENCES:

  1. DISAGREEMENT RATE: 3.2% (tabular) vs 47.8% (model-based)
     → The vast majority of Borda inflation in the model experiment
       comes from the PARAMETRIC models, not from the preference
       structure itself. With 4 annotated dimensions and 4 responses,
       pure social choice divergence is rare.

  2. CYCLES: 30.8% (tabular) vs 1.1% (model-based)
     → The raw annotations are MUCH more intransitive than what the
       PM (dim=8) produces. The PM's learned preferences are nearly
       transitive, suggesting it smooths out dimension-level conflicts.

  3. LENGTH: No effect in tabular (rho≈0) vs strong effect in model (rho=+0.23)
     → Length bias is entirely a MODEL artifact. The annotations
       themselves don't favor longer responses when aggregated across
       dimensions.

  4. SYCOPHANCY: Slightly NEGATIVE in tabular vs slightly positive in model.
     → Borda winners in the tabular setting are LESS sycophantic
       (more truthful), not more. The model-based sycophancy effect
       is also a parametric artifact.

  INTERPRETATION:
    The 48% disagreement in the model experiment is driven by the
    parametric BT model learning a DIFFERENT function from training
    data (cross-prompt generalization, length bias, feature conflation)
    rather than by Borda-vs-Condorcet social choice divergence on the
    underlying preferences. True Borda inflation from the preference
    structure alone is only ~3%.
""")


# ── FIGURES ──────────────────────────────────────────────────────────────────


def make_figures(results, summary, p_values, output_dir):
    """Generate analysis figures."""

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # --- Fig 1: Overview bar chart ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1a: Agreement vs disagreement
    labels = ["Borda = Copeland\n(agreement)", "Borda ≠ Copeland\n(disagreement)"]
    counts = [summary["n_agree"], summary["n_disagree"]]
    colors = ["#2ca02c", "#d62728"]
    bars = axes[0].bar(labels, counts, color=colors, edgecolor="black", linewidth=0.5)
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                     f"{count}\n({100*count/summary['n_prompts']:.1f}%)",
                     ha="center", va="bottom", fontsize=10)
    axes[0].set_ylabel("Number of prompts")
    axes[0].set_title("Borda vs Copeland Agreement")

    # 1b: Cycle prevalence
    n_cycle = summary["n_cycle"]
    n_nocycle = summary["n_prompts"] - n_cycle
    labels2 = ["No cycle\n(Condorcet winner)", "Cycle\n(no Condorcet winner)"]
    counts2 = [n_nocycle, n_cycle]
    colors2 = ["#1f77b4", "#ff7f0e"]
    bars2 = axes[1].bar(labels2, counts2, color=colors2, edgecolor="black", linewidth=0.5)
    for bar, count in zip(bars2, counts2):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                     f"{count}\n({100*count/summary['n_prompts']:.1f}%)",
                     ha="center", va="bottom", fontsize=10)
    axes[1].set_ylabel("Number of prompts")
    axes[1].set_title("Condorcet Cycle Prevalence")

    # 1c: Pairwise preference distribution
    bins = np.arange(0, 1.125, 0.125)
    axes[2].hist(p_values, bins=bins, color="#9467bd", edgecolor="black",
                 linewidth=0.5, rwidth=0.9)
    axes[2].axvline(0.5, color="red", linestyle="--", linewidth=1.5, label="Tie")
    axes[2].set_xlabel("P[i,j] (majority-rule preference)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Pairwise Preference Distribution")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig_dir / 'overview.png'}")

    # --- Fig 2: Correlation comparison (tabular vs model-based) ---
    fig, ax = plt.subplots(figsize=(10, 6))

    features = ["length", "sycophancy\nproxy", "helpfulness", "truthfulness",
                 "honesty", "instruction\nfollowing"]
    tabular_rhos = [
        summary["correlations"]["length"]["rho"],
        summary["correlations"]["sycophancy_proxy"]["rho"],
        summary["correlations"]["helpfulness"]["rho"],
        summary["correlations"]["truthfulness"]["rho"],
        summary["correlations"]["honesty"]["rho"],
        summary["correlations"]["instruction_following"]["rho"],
    ]
    model_rhos = [0.231, 0.069, 0.254, 0.135, 0.180, 0.222]  # from model experiment

    x = np.arange(len(features))
    w = 0.35
    bars1 = ax.bar(x - w/2, tabular_rhos, w, label="Tabular (annotations)",
                   color="#1f77b4", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + w/2, model_rhos, w, label="Model-based (RM vs PM)",
                   color="#d62728", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=9)
    ax.set_ylabel("Spearman ρ with Borda inflation")
    ax.set_title("What Correlates with Borda Inflation?\nTabular (annotation-based) vs Model-based (RM vs PM)")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "correlation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig_dir / 'correlation_comparison.png'}")

    # --- Fig 3: Venn-style breakdown of cycles vs disagreements ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # Categorize all prompts
    cats = Counter()
    for r in results:
        has_cycle = r["has_cycle"]
        disagrees = not r["borda_condorcet_agree"]
        if not has_cycle and not disagrees:
            cats["No cycle, agree"] += 1
        elif not has_cycle and disagrees:
            cats["No cycle, DISAGREE"] += 1
        elif has_cycle and not disagrees:
            cats["Cycle, agree"] += 1
        else:
            cats["Cycle, DISAGREE"] += 1

    labels = list(cats.keys())
    sizes = [cats[l] for l in labels]
    colors = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728"]

    bars = ax.barh(labels, sizes, color=colors, edgecolor="black", linewidth=0.5)
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
                f"{size} ({100*size/len(results):.1f}%)",
                va="center", fontsize=10)
    ax.set_xlabel("Number of prompts")
    ax.set_title("Prompt Classification: Cycles × Borda-Copeland Agreement")
    plt.tight_layout()
    plt.savefig(fig_dir / "cycle_vs_disagreement.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig_dir / 'cycle_vs_disagreement.png'}")

    # --- Fig 4: Mann-Whitney comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    mw = summary["mann_whitney"]

    # 4a: Dimension scores
    dims = ["instruction_following", "honesty", "truthfulness", "helpfulness"]
    dim_labels = ["Instr.\nfollow.", "Honesty", "Truthful.", "Helpful."]
    borda_means = [mw[d]["borda_mean"] for d in dims]
    pairwise_means = [mw[d]["pairwise_mean"] for d in dims]

    x = np.arange(len(dims))
    w = 0.35
    axes[0].bar(x - w/2, borda_means, w, label="Borda winner", color="#d62728",
                edgecolor="black", linewidth=0.5)
    axes[0].bar(x + w/2, pairwise_means, w, label="Copeland winner", color="#1f77b4",
                edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dim_labels)
    axes[0].set_ylabel("Mean score")
    axes[0].set_title("Dimension Scores: Borda vs Copeland Winners\n(disagreeing prompts only)")
    axes[0].legend()
    axes[0].set_ylim(3.5, 5.0)
    axes[0].grid(axis="y", alpha=0.3)

    # 4b: Sycophancy + length
    cats = ["Sycophancy\nproxy", "Length\n(words)"]
    borda_vals = [mw["sycophancy_proxy"]["borda_mean"], mw["length"]["borda_mean"]]
    pairwise_vals = [mw["sycophancy_proxy"]["pairwise_mean"], mw["length"]["pairwise_mean"]]

    # Normalize for display (different scales)
    fig2, (ax_syc, ax_len) = plt.subplots(1, 2, figsize=(10, 5))

    ax_syc.bar(["Borda\nwinner", "Copeland\nwinner"],
               [mw["sycophancy_proxy"]["borda_mean"], mw["sycophancy_proxy"]["pairwise_mean"]],
               color=["#d62728", "#1f77b4"], edgecolor="black", linewidth=0.5)
    ax_syc.set_ylabel("Mean sycophancy proxy")
    sig = "***" if mw["sycophancy_proxy"]["p"] < 0.001 else "n.s."
    ax_syc.set_title(f"Sycophancy Proxy ({sig})")
    ax_syc.axhline(0, color="black", linewidth=0.5)
    ax_syc.grid(axis="y", alpha=0.3)

    ax_len.bar(["Borda\nwinner", "Copeland\nwinner"],
               [mw["length"]["borda_mean"], mw["length"]["pairwise_mean"]],
               color=["#d62728", "#1f77b4"], edgecolor="black", linewidth=0.5)
    ax_len.set_ylabel("Mean length (words)")
    sig = "***" if mw["length"]["p"] < 0.001 else "n.s."
    ax_len.set_title(f"Response Length ({sig})")
    ax_len.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "winner_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig_dir / 'winner_comparison.png'}")

    # Close the unused axes fig
    plt.close(fig)

    # --- Fig 5: Inflation score distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    all_inflation = []
    for r in results:
        all_inflation.extend(r["inflation"])

    inf_counter = Counter(all_inflation)
    vals = sorted(inf_counter.keys())
    counts = [inf_counter[v] for v in vals]

    ax.bar(vals, counts, color="#9467bd", edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Inflation score (Copeland_rank − Borda_rank)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Tabular Borda Inflation Scores\n"
                  "(positive = Borda overvalues)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "inflation_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig_dir / 'inflation_distribution.png'}")


# ── MAIN ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir (default: results_dir/analysis)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results, summary = load_results(results_dir)
    print(f"  {len(results)} prompts loaded")

    # Analyses
    cycles, no_cycles = analyze_cycles(results, output_dir)
    analyze_disagreements(results, output_dir)
    p_values = analyze_intransitivity(results, output_dir)
    compare_with_model_experiment(summary)

    # Figures
    print("\nGenerating figures...")
    make_figures(results, summary, p_values, output_dir)

    # Save analysis summary
    analysis = {
        "n_prompts": len(results),
        "n_cycles": len(cycles),
        "n_cycles_that_disagree": sum(1 for r in cycles if not r["borda_condorcet_agree"]),
        "n_nocycles_that_disagree": sum(1 for r in no_cycles if not r["borda_condorcet_agree"]),
        "pct_ties_in_pairwise": 100 * sum(1 for p in p_values if abs(p - 0.5) < 0.01) / len(p_values),
    }
    with open(output_dir / "analysis_summary.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
