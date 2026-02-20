#!/usr/bin/env python3
"""Compute Borda inflation scores by comparing BT rewards vs PM effective rewards.

For each response in the UltraFeedback test set:
  - r_BT(y|x): scalar reward from Bradley-Terry model (Borda proxy)
  - r_eff(y|x): win-rate against dataset responses from GPM (Condorcet proxy)
  - inflation: rank difference between BT and effective reward rankings

Parallelized: RM runs on cuda:0, PM runs on cuda:1 concurrently.

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
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from eval.evaluate_trained_models_ultrafeedback import (
    load_reward_model,
    get_response_reward,
    compute_preference_matrix_batched,
)

# UltraFeedback dimensions
DIMENSIONS = ["instruction_following", "honesty", "truthfulness", "helpfulness"]


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_ultrafeedback_raw_annotations(full_dataset_name="openbmb/UltraFeedback"):
    """Load raw per-response dimension scores from UltraFeedback."""
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
    """Load UltraFeedback test split grouped by prompt."""
    test_path = Path(data_dir) / "pref_test"
    if not test_path.exists():
        test_path = Path(data_dir)

    print(f"Loading test data from {test_path}...")
    dataset = load_from_disk(str(test_path))

    prompt_groups = defaultdict(list)
    for sample in tqdm(dataset, desc="Grouping test data"):
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        if isinstance(prompt, list):
            prompt = prompt[0].get("content", str(prompt[0]))
        if isinstance(chosen, list):
            chosen = chosen[0].get("content", str(chosen[0]))
        if isinstance(rejected, list):
            rejected = rejected[0].get("content", str(rejected[0]))

        prompt_groups[str(prompt)].append((str(chosen), str(rejected)))

    print(f"  {len(prompt_groups)} prompts, {sum(len(v) for v in prompt_groups.values())} pairs")
    return dict(prompt_groups)


def prepare_prompt_responses(prompt_groups, max_responses=100):
    """Pre-extract unique responses per prompt, filtering by size.

    Returns list of (prompt, responses_list, pairs) tuples.
    """
    prepared = []
    skipped = 0
    for prompt, pairs in prompt_groups.items():
        responses = set()
        for c, r in pairs:
            responses.add(c)
            responses.add(r)
        if len(responses) > max_responses or len(responses) < 3:
            skipped += 1
            continue
        prepared.append((prompt, sorted(list(responses)), pairs))
    return prepared, skipped


# ── GPU WORKER FUNCTIONS ──────────────────────────────────────────────────────

def rm_worker(checkpoint, tau, prompts_data, device, output_path):
    """Worker process: compute RM rewards + RM pref matrices on a dedicated GPU.

    Saves results to output_path as pickle.
    """
    try:
        print(f"[RM worker] Loading model on {device}...")
        model, tokenizer, dim, _ = load_reward_model(checkpoint, tau, device=device)
        assert dim == 1, f"Expected RM dim=1, got {dim}"

        results = {}
        for prompt, responses, _pairs in tqdm(prompts_data, desc=f"[RM {device}]"):
            # Scalar rewards
            rewards = []
            for resp in responses:
                r = get_response_reward(model, tokenizer, prompt, resp, device)
                rewards.append(r.item())

            # Preference matrix
            pref_matrix = compute_preference_matrix_batched(
                model, tokenizer, responses, prompt, dim, tau, device
            )

            results[prompt] = {
                "rewards": np.array(rewards),
                "pref_matrix": pref_matrix,
            }

        # Save to disk (avoids large pickle through mp queue)
        with open(output_path, "wb") as f:
            pickle.dump(results, f)
        print(f"[RM worker] Done. Saved {len(results)} prompts to {output_path}")

    except Exception as e:
        print(f"[RM worker] FATAL ERROR: {e}")
        traceback.print_exc()
        # Write empty results so main process doesn't hang waiting for file
        with open(output_path, "wb") as f:
            pickle.dump({"__error__": str(e)}, f)
        raise


def pm_worker(checkpoint, tau, prompts_data, device, output_path):
    """Worker process: compute PM pref matrices on a dedicated GPU.

    Saves results to output_path as pickle.
    """
    try:
        print(f"[PM worker] Loading model on {device}...")
        model, tokenizer, dim, _ = load_reward_model(checkpoint, tau, device=device)
        assert dim > 1, f"Expected PM dim>1, got {dim}"

        results = {}
        for prompt, responses, _pairs in tqdm(prompts_data, desc=f"[PM {device}]"):
            pref_matrix = compute_preference_matrix_batched(
                model, tokenizer, responses, prompt, dim, tau, device
            )
            results[prompt] = {
                "pref_matrix": pref_matrix,
            }

        with open(output_path, "wb") as f:
            pickle.dump(results, f)
        print(f"[PM worker] Done. Saved {len(results)} prompts to {output_path}")

    except Exception as e:
        print(f"[PM worker] FATAL ERROR: {e}")
        traceback.print_exc()
        with open(output_path, "wb") as f:
            pickle.dump({"__error__": str(e)}, f)
        raise


# ── MERGING AND SOCIAL CHOICE ────────────────────────────────────────────────

def merge_and_compute_social_choice(rm_results, pm_results, prompts_data, annotations):
    """Merge RM and PM results and compute social choice metrics."""
    all_results = []

    for prompt, responses, pairs in tqdm(prompts_data, desc="Merging results"):
        if prompt not in rm_results or prompt not in pm_results:
            continue

        rm = rm_results[prompt]
        pm = pm_results[prompt]
        n = len(responses)

        rm_rewards = rm["rewards"]
        rm_pref_matrix = rm["pref_matrix"]
        pm_pref_matrix = pm["pref_matrix"]

        # Borda scores (row sums)
        rm_borda = rm_pref_matrix.sum(axis=1)
        pm_borda = pm_pref_matrix.sum(axis=1)

        # Effective rewards (win-rate)
        effective_rewards = pm_pref_matrix.mean(axis=1)

        # Rankings
        rm_ranking = np.argsort(-rm_rewards)
        eff_ranking = np.argsort(-effective_rewards)

        rm_ranks = np.empty_like(rm_ranking)
        rm_ranks[rm_ranking] = np.arange(n)
        eff_ranks = np.empty_like(eff_ranking)
        eff_ranks[eff_ranking] = np.arange(n)

        # Inflation
        inflation = eff_ranks - rm_ranks

        # Condorcet winner
        condorcet_winner = None
        for i in range(n):
            if all(pm_pref_matrix[i, j] > 0.5 for j in range(n) if j != i):
                condorcet_winner = i
                break

        # Copeland
        copeland = np.array([
            sum(1 for j in range(n) if j != i and pm_pref_matrix[i, j] > 0.5)
            for i in range(n)
        ])

        bt_winner = int(rm_ranking[0])
        pm_winner = int(eff_ranking[0])

        # Dimension annotations
        dim_scores = {}
        anno = annotations.get(prompt, None)
        if anno:
            anno_map = {a["response"]: a["scores"] for a in anno}
            for i, resp in enumerate(responses):
                if resp in anno_map:
                    dim_scores[i] = anno_map[resp]
                else:
                    for anno_resp, scores in anno_map.items():
                        if resp[:200] == anno_resp[:200]:
                            dim_scores[i] = scores
                            break

        lengths = np.array([len(resp.split()) for resp in responses])

        all_results.append({
            "prompt": prompt,
            "responses": responses,
            "n_responses": n,
            "rm_rewards": rm_rewards,
            "rm_pref_matrix": rm_pref_matrix,
            "rm_borda": rm_borda,
            "rm_ranking": rm_ranking,
            "rm_ranks": rm_ranks,
            "pm_pref_matrix": pm_pref_matrix,
            "pm_borda": pm_borda,
            "effective_rewards": effective_rewards,
            "eff_ranking": eff_ranking,
            "eff_ranks": eff_ranks,
            "condorcet_winner": condorcet_winner,
            "copeland": copeland,
            "bt_winner": bt_winner,
            "pm_winner": pm_winner,
            "agree": bt_winner == pm_winner,
            "inflation": inflation,
            "dim_scores": dim_scores,
            "lengths": lengths,
        })

    return all_results


# ── MAIN ──────────────────────────────────────────────────────────────────────

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
    parser.add_argument("--rm_device", type=str, default="cuda:0",
                        help="GPU device for RM (default: cuda:0)")
    parser.add_argument("--pm_device", type=str, default="cuda:1",
                        help="GPU device for PM (default: cuda:1)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Load and prepare data (main process, CPU) ===
    prompt_groups = load_test_data_grouped(args.data_dir)
    if args.max_prompts:
        keys = list(prompt_groups.keys())[:args.max_prompts]
        prompt_groups = {k: prompt_groups[k] for k in keys}

    prompts_data, skipped = prepare_prompt_responses(prompt_groups, args.max_responses)
    print(f"Prepared {len(prompts_data)} prompts ({skipped} skipped)")

    annotations = {}
    if not args.skip_annotations:
        annotations = load_ultrafeedback_raw_annotations()

    # === Launch RM and PM workers in parallel on separate GPUs ===
    print("=" * 70)
    print(f"LAUNCHING PARALLEL INFERENCE: RM on {args.rm_device}, PM on {args.pm_device}")
    print("=" * 70)

    mp.set_start_method("spawn", force=True)

    rm_output = output_dir / "_rm_results.pkl"
    pm_output = output_dir / "_pm_results.pkl"

    rm_proc = mp.Process(
        target=rm_worker,
        args=(args.rm_checkpoint, args.rm_tau, prompts_data, args.rm_device, str(rm_output)),
        name="RM-worker",
    )
    pm_proc = mp.Process(
        target=pm_worker,
        args=(args.pm_checkpoint, args.pm_tau, prompts_data, args.pm_device, str(pm_output)),
        name="PM-worker",
    )

    rm_proc.start()
    pm_proc.start()

    print(f"  RM worker PID: {rm_proc.pid}")
    print(f"  PM worker PID: {pm_proc.pid}")

    # Wait for both to finish
    rm_proc.join()
    pm_proc.join()

    # Check exit codes
    if rm_proc.exitcode != 0:
        print(f"FATAL: RM worker exited with code {rm_proc.exitcode}")
        sys.exit(1)
    if pm_proc.exitcode != 0:
        print(f"FATAL: PM worker exited with code {pm_proc.exitcode}")
        sys.exit(1)

    print("Both workers completed successfully.")

    # === Load worker results ===
    print("Loading worker results...")
    with open(rm_output, "rb") as f:
        rm_results = pickle.load(f)
    with open(pm_output, "rb") as f:
        pm_results = pickle.load(f)

    if "__error__" in rm_results:
        print(f"FATAL: RM worker error: {rm_results['__error__']}")
        sys.exit(1)
    if "__error__" in pm_results:
        print(f"FATAL: PM worker error: {pm_results['__error__']}")
        sys.exit(1)

    print(f"  RM results: {len(rm_results)} prompts")
    print(f"  PM results: {len(pm_results)} prompts")

    # Clean up temp files
    rm_output.unlink()
    pm_output.unlink()

    # === Merge and compute social choice metrics (CPU) ===
    print("=" * 70)
    print("COMPUTING SOCIAL CHOICE METRICS")
    print("=" * 70)
    all_results = merge_and_compute_social_choice(
        rm_results, pm_results, prompts_data, annotations
    )

    # Free memory
    del rm_results, pm_results

    # === Aggregate statistics ===
    n_agree = sum(1 for r in all_results if r["agree"])
    n_disagree = len(all_results) - n_agree
    n_condorcet = sum(1 for r in all_results if r["condorcet_winner"] is not None)
    n_cycles = len(all_results) - n_condorcet

    print(f"\n=== Summary ===")
    print(f"Total prompts:      {len(all_results)}")
    print(f"BT=PM agree:        {n_agree} ({100*n_agree/len(all_results):.1f}%)")
    print(f"BT!=PM disagree:    {n_disagree} ({100*n_disagree/len(all_results):.1f}%)")
    print(f"Condorcet winner:   {n_condorcet} ({100*n_condorcet/len(all_results):.1f}%)")
    print(f"Cycles (no winner): {n_cycles} ({100*n_cycles/len(all_results):.1f}%)")

    # Flatten per-response data
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
            if i in r["dim_scores"]:
                for dim in DIMENSIONS:
                    entry[f"dim_{dim}"] = r["dim_scores"][i].get(dim)
            flat_data.append(entry)

    # === Save results ===
    pickle_path = output_dir / "inflation_results.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump({"per_prompt": all_results, "flat": flat_data}, f)
    print(f"\nSaved full results to {pickle_path}")

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

    jsonl_path = output_dir / "inflation_flat.jsonl"
    with open(jsonl_path, "w") as f:
        for entry in flat_data:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved flat data to {jsonl_path}")


if __name__ == "__main__":
    main()
