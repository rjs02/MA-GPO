"""
Epistemic Strategy Analysis: correlate GPT-labeled strategies with Borda inflation.

Reads features_labeled.jsonl and produces:
  - epistemic_summary.json (all stats)
  - figure_strategy_distribution.png
  - figure_inflation_by_strategy_{seen,unseen,all}.png
  - figure_winner_composition.png
  - figure_strategy_features_heatmap.png
  - figure_substance_vs_surface_{seen,unseen,all}.png

Usage:
    python analyze_epistemic.py \
        --features features_labeled.jsonl \
        --output_dir epistemic_analysis
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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
    4: "Confabulation",
    5: "Hedging",
    6: "Evasive",
    7: "Concise",
}

LOCAL_FEATURES = [
    "word_count", "char_count", "sentence_count", "words_per_sentence",
    "n_bold", "n_bullet_items", "n_numbered_items", "n_headers",
    "n_code_blocks", "n_tables", "n_emojis", "formatting_density",
]


def load_labeled_features(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if "strategy_id" in rec:
                entries.append(rec)
    return entries


def split_data(entries: list[dict]) -> dict[str, list[dict]]:
    splits = {"all": entries}
    for e in entries:
        s = e["split"]
        if s not in splits:
            splits[s] = []
        splits[s].append(e)
    return splits


def compute_class_distribution(entries: list[dict]) -> dict:
    counts = defaultdict(int)
    ss_counts = defaultdict(int)
    for e in entries:
        counts[e["strategy_id"]] += 1
        ss_counts[e["substance_or_surface"]] += 1

    total = len(entries)
    dist = {}
    for sid in range(1, 8):
        cnt = counts.get(sid, 0)
        dist[STRATEGY_NAMES[sid]] = {
            "count": cnt,
            "fraction": cnt / total if total > 0 else 0,
        }

    ss_dist = {}
    for cat in ["substance", "surface", "neither"]:
        cnt = ss_counts.get(cat, 0)
        ss_dist[cat] = {"count": cnt, "fraction": cnt / total if total > 0 else 0}

    return {"strategy": dist, "substance_surface": ss_dist, "n": total}


def compute_inflation_by_strategy(entries: list[dict]) -> dict:
    """Mean/median inflation per strategy + statistical tests."""
    groups_rm = defaultdict(list)
    groups_pm = defaultdict(list)

    for e in entries:
        sid = e["strategy_id"]
        groups_rm[sid].append(e["inflation_rm"])
        groups_pm[sid].append(e["inflation_pm"])

    result = {}
    for sid in range(1, 8):
        name = STRATEGY_NAMES[sid]
        rm_vals = groups_rm.get(sid, [])
        pm_vals = groups_pm.get(sid, [])
        result[name] = {
            "n": len(rm_vals),
            "inflation_rm_mean": float(np.mean(rm_vals)) if rm_vals else None,
            "inflation_rm_median": float(np.median(rm_vals)) if rm_vals else None,
            "inflation_rm_std": float(np.std(rm_vals)) if rm_vals else None,
            "inflation_pm_mean": float(np.mean(pm_vals)) if pm_vals else None,
            "inflation_pm_median": float(np.median(pm_vals)) if pm_vals else None,
            "inflation_pm_std": float(np.std(pm_vals)) if pm_vals else None,
        }

    # Kruskal-Wallis across all 7 groups
    all_groups_rm = [groups_rm.get(sid, []) for sid in range(1, 8)]
    all_groups_rm = [g for g in all_groups_rm if len(g) > 0]
    if len(all_groups_rm) >= 2:
        kw_stat, kw_p = stats.kruskal(*all_groups_rm)
        result["kruskal_wallis_rm"] = {"statistic": float(kw_stat), "p_value": float(kw_p)}

    all_groups_pm = [groups_pm.get(sid, []) for sid in range(1, 8)]
    all_groups_pm = [g for g in all_groups_pm if len(g) > 0]
    if len(all_groups_pm) >= 2:
        kw_stat, kw_p = stats.kruskal(*all_groups_pm)
        result["kruskal_wallis_pm"] = {"statistic": float(kw_stat), "p_value": float(kw_p)}

    return result


def compute_substance_vs_surface(entries: list[dict]) -> dict:
    """Core hypothesis test: is inflation_rm higher for 'surface' than 'substance'?"""
    substance_rm = [e["inflation_rm"] for e in entries if e["substance_or_surface"] == "substance"]
    surface_rm = [e["inflation_rm"] for e in entries if e["substance_or_surface"] == "surface"]
    substance_pm = [e["inflation_pm"] for e in entries if e["substance_or_surface"] == "substance"]
    surface_pm = [e["inflation_pm"] for e in entries if e["substance_or_surface"] == "surface"]

    result = {
        "substance_n": len(substance_rm),
        "surface_n": len(surface_rm),
    }

    if substance_rm and surface_rm:
        u_stat, u_p = stats.mannwhitneyu(surface_rm, substance_rm, alternative="greater")
        # Rank-biserial correlation as effect size
        n1, n2 = len(surface_rm), len(substance_rm)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
        result["rm"] = {
            "substance_mean": float(np.mean(substance_rm)),
            "surface_mean": float(np.mean(surface_rm)),
            "mann_whitney_u": float(u_stat),
            "p_value": float(u_p),
            "rank_biserial": float(rank_biserial),
            "cohens_d": float((np.mean(surface_rm) - np.mean(substance_rm)) / np.sqrt((np.var(surface_rm) + np.var(substance_rm)) / 2)) if np.var(surface_rm) + np.var(substance_rm) > 0 else 0,
        }

    if substance_pm and surface_pm:
        u_stat, u_p = stats.mannwhitneyu(surface_pm, substance_pm, alternative="greater")
        n1, n2 = len(surface_pm), len(substance_pm)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
        result["pm"] = {
            "substance_mean": float(np.mean(substance_pm)),
            "surface_mean": float(np.mean(surface_pm)),
            "mann_whitney_u": float(u_stat),
            "p_value": float(u_p),
            "rank_biserial": float(rank_biserial),
            "cohens_d": float((np.mean(surface_pm) - np.mean(substance_pm)) / np.sqrt((np.var(surface_pm) + np.var(substance_pm)) / 2)) if np.var(surface_pm) + np.var(substance_pm) > 0 else 0,
        }

    return result


def compute_winner_composition(entries: list[dict]) -> dict:
    """What fraction of each strategy are RM/PM/emp_borda winners?"""
    winner_types = ["is_rm_winner", "is_pm_winner"]
    result = {}

    for wt in winner_types:
        winners = [e for e in entries if e.get(wt, False)]
        non_winners = [e for e in entries if not e.get(wt, False)]

        winner_dist = defaultdict(int)
        non_winner_dist = defaultdict(int)
        for e in winners:
            winner_dist[e["strategy_id"]] += 1
        for e in non_winners:
            non_winner_dist[e["strategy_id"]] += 1

        result[wt] = {
            "n_winners": len(winners),
            "n_non_winners": len(non_winners),
            "winner_distribution": {STRATEGY_NAMES[sid]: winner_dist.get(sid, 0) for sid in range(1, 8)},
            "non_winner_distribution": {STRATEGY_NAMES[sid]: non_winner_dist.get(sid, 0) for sid in range(1, 8)},
        }

        # Chi-squared test: is strategy distribution different for winners vs non-winners?
        observed_winners = [winner_dist.get(sid, 0) for sid in range(1, 8)]
        observed_non_winners = [non_winner_dist.get(sid, 0) for sid in range(1, 8)]
        contingency = np.array([observed_winners, observed_non_winners])
        # Remove columns with all zeros
        nonzero_cols = contingency.sum(axis=0) > 0
        if nonzero_cols.sum() >= 2:
            chi2, p, dof, _ = stats.chi2_contingency(contingency[:, nonzero_cols])
            result[wt]["chi2"] = float(chi2)
            result[wt]["chi2_p"] = float(p)
            result[wt]["chi2_dof"] = int(dof)

    return result


def compute_strategy_features_cross(entries: list[dict]) -> dict:
    """Mean local feature values per strategy."""
    groups = defaultdict(list)
    for e in entries:
        groups[e["strategy_id"]].append(e)

    result = {}
    for sid in range(1, 8):
        name = STRATEGY_NAMES[sid]
        group = groups.get(sid, [])
        if not group:
            result[name] = {feat: None for feat in LOCAL_FEATURES}
            continue
        result[name] = {}
        for feat in LOCAL_FEATURES:
            vals = [e[feat] for e in group]
            result[name][feat] = float(np.mean(vals))

    return result


# ── Figures ──


def plot_strategy_distribution(splits_data: dict, output_dir: Path):
    """Bar chart of strategy frequency."""
    fig, axes = plt.subplots(1, len(splits_data), figsize=(6 * len(splits_data), 5), squeeze=False)
    for ax, (split_name, entries) in zip(axes[0], splits_data.items()):
        counts = defaultdict(int)
        for e in entries:
            counts[e["strategy_id"]] += 1
        total = len(entries)

        sids = list(range(1, 8))
        vals = [counts.get(sid, 0) / total * 100 for sid in sids]
        labels = [STRATEGY_SHORT[sid] for sid in sids]
        colors = plt.cm.Set2(np.linspace(0, 1, 7))

        bars = ax.bar(range(7), vals, color=colors)
        ax.set_xticks(range(7))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Percentage (%)")
        ax.set_title(f"Strategy Distribution ({split_name}, n={total})")

        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "figure_strategy_distribution.png", dpi=150)
    plt.close()


def plot_inflation_by_strategy(entries: list[dict], split_name: str, output_dir: Path):
    """Violin plots of inflation distribution per strategy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (metric, label) in zip(axes, [("inflation_rm", "RM Inflation"), ("inflation_pm", "PM Inflation")]):
        groups = defaultdict(list)
        for e in entries:
            groups[e["strategy_id"]].append(e[metric])

        data = []
        positions = []
        labels_list = []
        for sid in range(1, 8):
            if groups.get(sid):
                data.append(groups[sid])
                positions.append(sid)
                labels_list.append(f"{STRATEGY_SHORT[sid]}\n(n={len(groups[sid])})")

        if data:
            parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_alpha(0.7)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels_list, fontsize=8, rotation=45, ha="right")

            # Add mean annotations
            for pos, d in zip(positions, data):
                mean_val = np.mean(d)
                ax.text(pos, ax.get_ylim()[1] * 0.95, f"{mean_val:+.3f}",
                        ha="center", va="top", fontsize=7, fontweight="bold")

        ax.set_ylabel(label)
        ax.set_title(f"{label} by Strategy ({split_name})")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / f"figure_inflation_by_strategy_{split_name}.png", dpi=150)
    plt.close()


def plot_winner_composition(entries: list[dict], output_dir: Path):
    """Stacked bar comparing strategy mix among RM/PM winners."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, 7))

    for ax, wt, title in zip(axes, ["is_rm_winner", "is_pm_winner"], ["RM Winners", "PM Winners"]):
        winners = [e for e in entries if e.get(wt, False)]
        non_winners = [e for e in entries if not e.get(wt, False)]

        winner_counts = defaultdict(int)
        non_winner_counts = defaultdict(int)
        for e in winners:
            winner_counts[e["strategy_id"]] += 1
        for e in non_winners:
            non_winner_counts[e["strategy_id"]] += 1

        # Normalize to percentages
        w_total = max(1, len(winners))
        nw_total = max(1, len(non_winners))

        categories = ["Winners", "Non-winners"]
        bottom_w = 0
        bottom_nw = 0

        for sid in range(1, 8):
            w_pct = winner_counts.get(sid, 0) / w_total * 100
            nw_pct = non_winner_counts.get(sid, 0) / nw_total * 100

            ax.bar(0, w_pct, bottom=bottom_w, color=colors[sid - 1], label=STRATEGY_SHORT[sid] if ax == axes[0] else "")
            ax.bar(1, nw_pct, bottom=bottom_nw, color=colors[sid - 1])

            bottom_w += w_pct
            bottom_nw += nw_pct

        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Winners\n(n={len(winners)})", f"Non-winners\n(n={len(non_winners)})"])
        ax.set_ylabel("Percentage (%)")
        ax.set_title(title)
        ax.set_ylim(0, 105)

    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "figure_winner_composition.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_strategy_features_heatmap(entries: list[dict], output_dir: Path):
    """Heatmap: strategy × mean local features (z-scored)."""
    groups = defaultdict(list)
    for e in entries:
        groups[e["strategy_id"]].append(e)

    # Build matrix: strategies × features
    sids = [sid for sid in range(1, 8) if groups.get(sid)]
    matrix = np.zeros((len(sids), len(LOCAL_FEATURES)))
    raw_matrix = np.zeros_like(matrix)

    for i, sid in enumerate(sids):
        for j, feat in enumerate(LOCAL_FEATURES):
            vals = [e[feat] for e in groups[sid]]
            raw_matrix[i, j] = np.mean(vals) if vals else 0

    # Z-score across strategies (column-wise)
    col_means = raw_matrix.mean(axis=0)
    col_stds = raw_matrix.std(axis=0)
    col_stds[col_stds == 0] = 1
    matrix = (raw_matrix - col_means) / col_stds

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)

    ax.set_xticks(range(len(LOCAL_FEATURES)))
    ax.set_xticklabels(LOCAL_FEATURES, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(sids)))
    ax.set_yticklabels([STRATEGY_SHORT[sid] for sid in sids], fontsize=10)

    # Annotate with raw values
    for i in range(len(sids)):
        for j in range(len(LOCAL_FEATURES)):
            val = raw_matrix[i, j]
            text = f"{val:.1f}" if val >= 10 else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if abs(matrix[i, j]) > 1.2 else "black")

    plt.colorbar(im, ax=ax, label="Z-score (across strategies)")
    ax.set_title("Mean Local Features by Epistemic Strategy (z-scored)")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_strategy_features_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_substance_vs_surface(entries: list[dict], split_name: str, output_dir: Path):
    """Box plot comparing inflation for substance vs surface responses."""
    substance_rm = [e["inflation_rm"] for e in entries if e["substance_or_surface"] == "substance"]
    surface_rm = [e["inflation_rm"] for e in entries if e["substance_or_surface"] == "surface"]
    neither_rm = [e["inflation_rm"] for e in entries if e["substance_or_surface"] == "neither"]

    substance_pm = [e["inflation_pm"] for e in entries if e["substance_or_surface"] == "substance"]
    surface_pm = [e["inflation_pm"] for e in entries if e["substance_or_surface"] == "surface"]
    neither_pm = [e["inflation_pm"] for e in entries if e["substance_or_surface"] == "neither"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data_groups, labels, title in zip(
        axes,
        [(substance_rm, surface_rm, neither_rm), (substance_pm, surface_pm, neither_pm)],
        [
            [f"Substance\n(n={len(substance_rm)})", f"Surface\n(n={len(surface_rm)})", f"Neither\n(n={len(neither_rm)})"],
            [f"Substance\n(n={len(substance_pm)})", f"Surface\n(n={len(surface_pm)})", f"Neither\n(n={len(neither_pm)})"],
        ],
        ["RM Inflation", "PM Inflation"],
    ):
        bp = ax.boxplot([d for d in data_groups if d], labels=[l for l, d in zip(labels, data_groups) if d],
                        patch_artist=True, showmeans=True)
        colors_box = ["#4CAF50", "#F44336", "#9E9E9E"]
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Add mean annotations
        for i, (d, label) in enumerate(zip(data_groups, labels)):
            if d:
                mean_val = np.mean(d)
                ax.text(i + 1, ax.get_ylim()[1] * 0.9, f"mean={mean_val:+.3f}",
                        ha="center", va="top", fontsize=9, fontweight="bold")

        ax.set_ylabel(title)
        ax.set_title(f"{title}: Substance vs Surface ({split_name})")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / f"figure_substance_vs_surface_{split_name}.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Epistemic Strategy Analysis")
    parser.add_argument("--features", required=True, help="Path to features_labeled.jsonl")
    parser.add_argument("--output_dir", required=True, help="Where to save figures and summary")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_labeled_features(args.features)
    print(f"Loaded {len(entries)} labeled entries.")

    if not entries:
        print("No labeled entries found. Exiting.")
        return

    splits = split_data(entries)
    summary = {}

    for split_name, split_entries in splits.items():
        print(f"\n{'='*60}")
        print(f"Split: {split_name} (n={len(split_entries)})")
        print(f"{'='*60}")

        s = {}
        s["n"] = len(split_entries)

        # Class distribution
        s["class_distribution"] = compute_class_distribution(split_entries)
        dist = s["class_distribution"]["strategy"]
        print(f"\nStrategy distribution:")
        for name, info in dist.items():
            print(f"  {name}: {info['count']} ({info['fraction']*100:.1f}%)")

        # Inflation by strategy
        s["inflation_by_strategy"] = compute_inflation_by_strategy(split_entries)
        print(f"\nInflation by strategy:")
        for name, info in s["inflation_by_strategy"].items():
            if isinstance(info, dict) and "inflation_rm_mean" in info:
                print(f"  {name}: RM mean={info['inflation_rm_mean']:+.3f}, PM mean={info['inflation_pm_mean']:+.3f} (n={info['n']})")
        if "kruskal_wallis_rm" in s["inflation_by_strategy"]:
            kw = s["inflation_by_strategy"]["kruskal_wallis_rm"]
            print(f"  Kruskal-Wallis (RM): H={kw['statistic']:.2f}, p={kw['p_value']:.4f}")
        if "kruskal_wallis_pm" in s["inflation_by_strategy"]:
            kw = s["inflation_by_strategy"]["kruskal_wallis_pm"]
            print(f"  Kruskal-Wallis (PM): H={kw['statistic']:.2f}, p={kw['p_value']:.4f}")

        # Substance vs surface
        s["substance_vs_surface"] = compute_substance_vs_surface(split_entries)
        svs = s["substance_vs_surface"]
        print(f"\nSubstance vs Surface:")
        print(f"  Substance n={svs['substance_n']}, Surface n={svs['surface_n']}")
        if "rm" in svs:
            rm = svs["rm"]
            print(f"  RM: substance mean={rm['substance_mean']:+.3f}, surface mean={rm['surface_mean']:+.3f}")
            print(f"      Mann-Whitney p={rm['p_value']:.4f}, Cohen's d={rm['cohens_d']:+.3f}, rank-biserial={rm['rank_biserial']:+.3f}")
        if "pm" in svs:
            pm = svs["pm"]
            print(f"  PM: substance mean={pm['substance_mean']:+.3f}, surface mean={pm['surface_mean']:+.3f}")
            print(f"      Mann-Whitney p={pm['p_value']:.4f}, Cohen's d={pm['cohens_d']:+.3f}, rank-biserial={pm['rank_biserial']:+.3f}")

        # Winner composition
        s["winner_composition"] = compute_winner_composition(split_entries)

        # Strategy × features cross-tab
        s["strategy_features"] = compute_strategy_features_cross(split_entries)

        summary[split_name] = s

        # Generate per-split figures
        plot_inflation_by_strategy(split_entries, split_name, output_dir)
        plot_substance_vs_surface(split_entries, split_name, output_dir)

    # Generate cross-split figures
    plot_strategy_distribution(splits, output_dir)
    plot_winner_composition(entries, output_dir)
    plot_strategy_features_heatmap(entries, output_dir)

    # Save summary
    with open(output_dir / "epistemic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'epistemic_summary.json'}")
    print(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
