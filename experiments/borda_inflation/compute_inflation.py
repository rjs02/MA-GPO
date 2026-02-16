#!/usr/bin/env python3
"""Compute Borda inflation scores by comparing BT rewards vs PM effective rewards.

For each response in the UltraFeedback test set:
  - r_BT(y|x): scalar reward from Bradley-Terry model (Borda proxy)
  - r_eff(y|x): win-rate against dataset responses from GPM (Condorcet proxy)
  - inflation: rank difference between BT and effective reward rankings

Also loads raw UltraFeedback dimension annotations for interpretability.

Usage:
    python experiments/borda_inflation/compute_inflation.py \
        --rm_checkpoint /path/to/rm/checkpoint \
        --pm_checkpoint /path/to/pm/checkpoint \
        --data_dir /path/to/ufb_multidim \
        --output_dir experiments/borda_inflation/results
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from eval.evaluate_trained_models_ultrafeedback import (
    load_reward_model,
    get_response_reward,
    compute_preference_matrix_batched,
    DEVICE,
)

# UltraFeedback dimensions
DIMENSIONS = ["instruction_following", "honesty", "truthfulness", "helpfulness"]


def load_ultrafeedback_raw_annotations(full_dataset_name="openbmb/UltraFeedback"):
    """Load raw per-response dimension scores from UltraFeedback.

    Returns:
        annotations: dict[instruction] -> list of {response, scores_by_dim}
    """
    print(f"Loading raw UltraFeedback annotations from {full_dataset_name}...")
    dataset = load_dataset(full_dataset_name, split="train")

    annotations = {}
    for row in tqdm(dataset, desc="Parsing annotations"):
        instruction = row["instruction"]
        completions = row["completions"]

        response_data = []
        for completion in completions:
            scores = {}
            for dim in DIMENSIONS:
                try:
                    rating = int(completion["annotations"][dim]["Rating"])
                    scores[dim] = rating if rating != -1 else None
                except (ValueError, TypeError, KeyError):
                    scores[dim] = None
            response_data.append({
                "response": completion["response"],
                "scores": scores,
            })

        annotations[instruction] = response_data

    print(f"  Loaded annotations for {len(annotations)} prompts")
    return annotations


def load_test_data_grouped(data_dir):
    """Load UltraFeedback test split grouped by prompt.

    Returns:
        prompt_groups: dict[prompt_text] -> list of (chosen, rejected) pairs
    """
    test_path = Path(data_dir) / "pref_test"
    if not test_path.exists():
        # Try without pref_ prefix
        test_path = Path(data_dir)

    print(f"Loading test data from {test_path}...")
    dataset = load_from_disk(str(test_path))

    prompt_groups = defaultdict(list)
    for sample in tqdm(dataset, desc="Grouping test data"):
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        # Extract text from structured format
        if isinstance(prompt, list):
            prompt = prompt[0].get("content", str(prompt[0]))
        if isinstance(chosen, list):
            chosen = chosen[0].get("content", str(chosen[0]))
        if isinstance(rejected, list):
            rejected = rejected[0].get("content", str(rejected[0]))

        prompt_groups[str(prompt)].append((str(chosen), str(rejected)))

    print(f"  {len(prompt_groups)} prompts, {sum(len(v) for v in prompt_groups.values())} pairs")
    return dict(prompt_groups)


def compute_prompt_results(
    rm_model, rm_tokenizer, rm_dim, rm_tau,
    pm_model, pm_tokenizer, pm_dim, pm_tau,
    prompt, pairs, annotations_for_prompt,
):
    """Compute all Borda inflation metrics for a single prompt.

    Returns dict with per-response data or None if insufficient responses.
    """
    # Extract unique responses
    responses = set()
    for chosen, rejected in pairs:
        responses.add(chosen)
        responses.add(rejected)
    responses = sorted(list(responses))
    n = len(responses)

    if n < 3:
        return None

    # --- RM: scalar BT rewards ---
    rm_rewards = []
    for resp in responses:
        r = get_response_reward(rm_model, rm_tokenizer, prompt, resp, DEVICE)
        rm_rewards.append(r.item())
    rm_rewards = np.array(rm_rewards)

    # --- PM: full pairwise preference matrix ---
    pm_pref_matrix = compute_preference_matrix_batched(
        pm_model, pm_tokenizer, responses, prompt, pm_dim, pm_tau, DEVICE
    )

    # --- RM: preference matrix (for Borda from matrix comparison) ---
    rm_pref_matrix = compute_preference_matrix_batched(
        rm_model, rm_tokenizer, responses, prompt, rm_dim, rm_tau, DEVICE
    )

    # --- Borda scores (row sums of preference matrices) ---
    rm_borda = rm_pref_matrix.sum(axis=1)  # from RM matrix
    pm_borda = pm_pref_matrix.sum(axis=1)  # from PM matrix (for cross-check)

    # --- Effective rewards: win-rate against dataset responses ---
    # r_eff(y_i) = (1/N) * sum_j P_PM[i,j]  (includes self-comparison P[i,i]=0.5)
    effective_rewards = pm_pref_matrix.mean(axis=1)

    # --- Rankings ---
    rm_ranking = np.argsort(-rm_rewards)  # highest reward first
    eff_ranking = np.argsort(-effective_rewards)  # highest win-rate first

    # Rank positions (0-indexed, lower = better)
    rm_ranks = np.empty_like(rm_ranking)
    rm_ranks[rm_ranking] = np.arange(n)
    eff_ranks = np.empty_like(eff_ranking)
    eff_ranks[eff_ranking] = np.arange(n)

    # --- Inflation score: rank difference ---
    # Positive = BT ranks higher than effective reward (BT overestimates)
    inflation = eff_ranks - rm_ranks  # if eff_rank > rm_rank, BT overestimates

    # --- Condorcet winner detection ---
    # Response i is Condorcet winner if P_PM[i,j] > 0.5 for all j != i
    condorcet_winner = None
    for i in range(n):
        if all(pm_pref_matrix[i, j] > 0.5 for j in range(n) if j != i):
            condorcet_winner = i
            break

    # --- Copeland scores ---
    copeland = np.array([sum(1 for j in range(n) if j != i and pm_pref_matrix[i, j] > 0.5) for i in range(n)])

    # --- Top response comparison ---
    bt_winner = int(rm_ranking[0])
    pm_winner = int(eff_ranking[0])
    agree = bt_winner == pm_winner

    # --- Dimension annotations ---
    dim_scores = {}
    if annotations_for_prompt:
        # Match responses to annotations by text content
        anno_map = {a["response"]: a["scores"] for a in annotations_for_prompt}
        for i, resp in enumerate(responses):
            # Try to match (may need fuzzy matching)
            if resp in anno_map:
                dim_scores[i] = anno_map[resp]
            else:
                # Try prefix matching (responses may be truncated)
                for anno_resp, scores in anno_map.items():
                    if resp[:200] == anno_resp[:200]:
                        dim_scores[i] = scores
                        break

    # --- Response lengths ---
    lengths = np.array([len(resp.split()) for resp in responses])

    return {
        "prompt": prompt,
        "responses": responses,
        "n_responses": n,
        # RM outputs
        "rm_rewards": rm_rewards,
        "rm_pref_matrix": rm_pref_matrix,
        "rm_borda": rm_borda,
        "rm_ranking": rm_ranking,
        "rm_ranks": rm_ranks,
        # PM outputs
        "pm_pref_matrix": pm_pref_matrix,
        "pm_borda": pm_borda,
        "effective_rewards": effective_rewards,
        "eff_ranking": eff_ranking,
        "eff_ranks": eff_ranks,
        # Social choice
        "condorcet_winner": condorcet_winner,
        "copeland": copeland,
        "bt_winner": bt_winner,
        "pm_winner": pm_winner,
        "agree": agree,
        # Inflation
        "inflation": inflation,
        # Annotations
        "dim_scores": dim_scores,
        "lengths": lengths,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Borda inflation scores")
    parser.add_argument("--rm_checkpoint", type=str, required=True,
                        help="Path to trained RM (dim=1) checkpoint")
    parser.add_argument("--pm_checkpoint", type=str, required=True,
                        help="Path to trained PM (dim=8) checkpoint")
    parser.add_argument("--rm_tau", type=float, default=1.0,
                        help="Temperature for RM (default: 1.0)")
    parser.add_argument("--pm_tau", type=float, default=0.1,
                        help="Temperature for PM (default: 0.1)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to UltraFeedback multidim data directory")
    parser.add_argument("--output_dir", type=str, default="experiments/borda_inflation/results",
                        help="Output directory for results")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Limit to N prompts (for debugging)")
    parser.add_argument("--max_responses", type=int, default=100,
                        help="Skip prompts with more than N responses")
    parser.add_argument("--skip_annotations", action="store_true",
                        help="Skip loading raw UltraFeedback annotations (faster)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Load models ===
    print("=" * 70)
    print("LOADING RM (Bradley-Terry)")
    print("=" * 70)
    rm_model, rm_tokenizer, rm_dim, rm_tau = load_reward_model(
        args.rm_checkpoint, args.rm_tau
    )
    assert rm_dim == 1, f"Expected RM dim=1, got {rm_dim}"

    print("=" * 70)
    print("LOADING PM (General Preference Model)")
    print("=" * 70)
    pm_model, pm_tokenizer, pm_dim, pm_tau = load_reward_model(
        args.pm_checkpoint, args.pm_tau
    )
    assert pm_dim > 1, f"Expected PM dim>1, got {pm_dim}"

    # === Load data ===
    prompt_groups = load_test_data_grouped(args.data_dir)

    if args.max_prompts:
        keys = list(prompt_groups.keys())[:args.max_prompts]
        prompt_groups = {k: prompt_groups[k] for k in keys}

    # === Load raw annotations ===
    annotations = {}
    if not args.skip_annotations:
        annotations = load_ultrafeedback_raw_annotations()

    # === Compute inflation for each prompt ===
    print("=" * 70)
    print(f"COMPUTING BORDA INFLATION ({len(prompt_groups)} prompts)")
    print("=" * 70)

    all_results = []
    skipped = 0

    for prompt, pairs in tqdm(prompt_groups.items(), desc="Computing inflation"):
        # Check response count
        responses = set()
        for c, r in pairs:
            responses.add(c)
            responses.add(r)
        if len(responses) > args.max_responses or len(responses) < 3:
            skipped += 1
            continue

        # Get annotations for this prompt
        anno = annotations.get(prompt, None)

        result = compute_prompt_results(
            rm_model, rm_tokenizer, rm_dim, rm_tau,
            pm_model, pm_tokenizer, pm_dim, pm_tau,
            prompt, pairs, anno,
        )

        if result is not None:
            all_results.append(result)

    print(f"\nProcessed {len(all_results)} prompts, skipped {skipped}")

    # === Aggregate statistics ===
    n_agree = sum(1 for r in all_results if r["agree"])
    n_disagree = len(all_results) - n_agree
    n_condorcet = sum(1 for r in all_results if r["condorcet_winner"] is not None)
    n_cycles = len(all_results) - n_condorcet

    print(f"\n=== Summary ===")
    print(f"Total prompts:     {len(all_results)}")
    print(f"BT=PM agree:       {n_agree} ({100*n_agree/len(all_results):.1f}%)")
    print(f"BT!=PM disagree:   {n_disagree} ({100*n_disagree/len(all_results):.1f}%)")
    print(f"Condorcet winner:  {n_condorcet} ({100*n_condorcet/len(all_results):.1f}%)")
    print(f"Cycles (no winner): {n_cycles} ({100*n_cycles/len(all_results):.1f}%)")

    # Flatten per-response data for easier analysis
    flat_data = []
    for r in all_results:
        for i, resp in enumerate(r["responses"]):
            entry = {
                "prompt": r["prompt"],
                "response": resp,
                "rm_reward": float(r["rm_rewards"][i]),
                "effective_reward": float(r["effective_rewards"][i]),
                "rm_borda": float(r["rm_borda"][i]),
                "pm_borda": float(r["pm_borda"][i]),
                "inflation": int(r["inflation"][i]),
                "rm_rank": int(r["rm_ranks"][i]),
                "eff_rank": int(r["eff_ranks"][i]),
                "copeland": int(r["copeland"][i]),
                "length": int(r["lengths"][i]),
                "is_bt_winner": i == r["bt_winner"],
                "is_pm_winner": i == r["pm_winner"],
                "is_condorcet_winner": i == r["condorcet_winner"],
                "prompt_agrees": r["agree"],
            }
            # Add dimension scores
            if i in r["dim_scores"]:
                for dim in DIMENSIONS:
                    entry[f"dim_{dim}"] = r["dim_scores"][i].get(dim)
            flat_data.append(entry)

    # === Save results ===
    # Full results (pickle, includes matrices)
    pickle_path = output_dir / "inflation_results.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump({"per_prompt": all_results, "flat": flat_data}, f)
    print(f"\nSaved full results to {pickle_path}")

    # Summary stats (JSON, lightweight)
    summary = {
        "n_prompts": len(all_results),
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "n_condorcet": n_condorcet,
        "n_cycles": n_cycles,
        "pct_agree": round(100 * n_agree / len(all_results), 2),
        "pct_disagree": round(100 * n_disagree / len(all_results), 2),
        "n_responses_total": len(flat_data),
        "rm_checkpoint": args.rm_checkpoint,
        "pm_checkpoint": args.pm_checkpoint,
        "rm_tau": args.rm_tau,
        "pm_tau": args.pm_tau,
    }
    json_path = output_dir / "inflation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {json_path}")

    # Flat data for pandas (JSON lines)
    jsonl_path = output_dir / "inflation_flat.jsonl"
    with open(jsonl_path, "w") as f:
        for entry in flat_data:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved flat data to {jsonl_path}")


if __name__ == "__main__":
    main()
