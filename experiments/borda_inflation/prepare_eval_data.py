#!/usr/bin/env python3
"""Prepare fixed evaluation splits from UltraFeedback for Borda inflation analysis.

Creates eval_seen (subset of training prompts) and eval_unseen (validation prompts),
each with per-response dimension scores and empirical preference matrices.
These splits are reusable across inference runs and later qualitative labeling.

Usage:
    python experiments/borda_inflation/prepare_eval_data.py \
        --data_dir /path/to/ufb_multidim \
        --output_dir /path/to/output \
        --n_eval_seen 1024 --n_eval_unseen 1024
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DIMENSIONS = ["instruction_following", "honesty", "truthfulness", "helpfulness"]


def majority_pref_matrix(dim_scores):
    """Build pairwise preference matrix using dimension-wise majority rule.

    P[i,j] = fraction of valid dimensions where score_i > score_j.
    Ties count as 0.5.

    Args:
        dim_scores: dict mapping dimension name -> list of scores (int or None).

    Returns:
        P: (n, n) preference matrix, diagonal = 0.5.
        n_valid: (n, n) count of valid dimension comparisons.
    """
    n = len(next(iter(dim_scores.values())))
    P = np.full((n, n), 0.5)
    n_valid = np.zeros((n, n), dtype=int)

    for dim_name, scores in dim_scores.items():
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                si, sj = scores[i], scores[j]
                if si is None or sj is None:
                    continue
                n_valid[i, j] += 1
                if si > sj:
                    P[i, j] += 1.0
                elif si == sj:
                    P[i, j] += 0.5

    for i in range(n):
        for j in range(n):
            if i != j and n_valid[i, j] > 0:
                P[i, j] = (P[i, j] - 0.5) / n_valid[i, j]
            elif i != j:
                P[i, j] = 0.5

    return P, n_valid


def extract_unique_prompts(data_path):
    """Extract unique prompt strings from a HuggingFace pref dataset split."""
    dataset = load_from_disk(str(data_path))
    prompts = set()
    for sample in tqdm(dataset, desc=f"Scanning {data_path.name}"):
        prompt = sample["prompt"]
        if isinstance(prompt, list):
            prompt = prompt[0].get("content", str(prompt[0]))
        prompts.add(str(prompt))
    return prompts


def load_ultrafeedback_raw(dataset_name="openbmb/UltraFeedback"):
    """Load raw UltraFeedback and return per-prompt completion data.

    Returns:
        dict: instruction -> list of {response, scores} dicts
    """
    print(f"Loading {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    print(f"  {len(dataset)} rows")

    prompts = {}
    skipped = 0

    for row in tqdm(dataset, desc="Parsing UltraFeedback"):
        instruction = row["instruction"]
        completions_raw = row["completions"]

        if not completions_raw or len(completions_raw) < 2:
            skipped += 1
            continue

        responses = []
        all_valid = True
        for comp in completions_raw:
            response_text = comp.get("response", "")
            annotations = comp.get("annotations", {})

            scores = {}
            for dim in DIMENSIONS:
                try:
                    rating = int(annotations[dim]["Rating"])
                    scores[dim] = rating if rating != -1 else None
                except (KeyError, TypeError, ValueError):
                    scores[dim] = None

            # Check if this response has at least some valid scores
            if all(v is None for v in scores.values()):
                all_valid = False

            responses.append({
                "text": response_text,
                "scores": scores,
            })

        if not all_valid or len(responses) < 2:
            skipped += 1
            continue

        prompts[instruction] = responses

    print(f"  {len(prompts)} prompts with scored responses ({skipped} skipped)")
    return prompts


def match_prompts(split_prompts, ufb_prompts):
    """Match split prompt strings to UltraFeedback instructions.

    Returns list of matched instruction strings.
    """
    matched = []
    for prompt in split_prompts:
        if prompt in ufb_prompts:
            matched.append(prompt)
    return matched


def build_eval_split(matched_prompts, ufb_data, n_sample, seed, split_name):
    """Build an eval split: sample prompts, compute P_emp, return records."""
    rng = np.random.default_rng(seed)

    if n_sample < len(matched_prompts):
        indices = rng.choice(len(matched_prompts), size=n_sample, replace=False)
        sampled = [matched_prompts[i] for i in sorted(indices)]
    else:
        sampled = matched_prompts
        print(f"  Warning: requested {n_sample} but only {len(matched_prompts)} available for {split_name}")

    records = []
    for prompt in tqdm(sampled, desc=f"Building {split_name}"):
        responses = ufb_data[prompt]

        # Build dim_scores dict for majority_pref_matrix
        dim_scores = {dim: [] for dim in DIMENSIONS}
        for resp in responses:
            for dim in DIMENSIONS:
                dim_scores[dim].append(resp["scores"].get(dim))

        P_emp, n_valid = majority_pref_matrix(dim_scores)

        records.append({
            "prompt": prompt,
            "responses": [
                {"text": r["text"], "scores": r["scores"]}
                for r in responses
            ],
            "P_emp": P_emp.tolist(),
            "n_valid_dims": n_valid.tolist(),
            "n_responses": len(responses),
            "split": split_name,
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare eval data for Borda inflation analysis")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to ufb_multidim data dir (has pref_train/, pref_val/)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save preprocessed splits")
    parser.add_argument("--n_eval_seen", type=int, default=1024,
                        help="Number of prompts for eval_seen split")
    parser.add_argument("--n_eval_unseen", type=int, default=1024,
                        help="Number of prompts for eval_unseen split")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # Step 1: Identify prompts in each split
    print("=" * 60)
    print("Step 1: Identifying prompts per split")
    print("=" * 60)
    train_prompts = extract_unique_prompts(data_dir / "pref_train")
    val_prompts = extract_unique_prompts(data_dir / "pref_val")
    print(f"  Train prompts: {len(train_prompts)}")
    print(f"  Val prompts:   {len(val_prompts)}")

    # Step 2: Load raw UltraFeedback
    print("\n" + "=" * 60)
    print("Step 2: Loading raw UltraFeedback")
    print("=" * 60)
    ufb_data = load_ultrafeedback_raw()

    # Step 3: Match
    print("\n" + "=" * 60)
    print("Step 3: Matching prompts")
    print("=" * 60)
    matched_train = match_prompts(train_prompts, ufb_data)
    matched_val = match_prompts(val_prompts, ufb_data)
    print(f"  Matched train: {len(matched_train)}/{len(train_prompts)} ({100*len(matched_train)/len(train_prompts):.1f}%)")
    print(f"  Matched val:   {len(matched_val)}/{len(val_prompts)} ({100*len(matched_val)/len(val_prompts):.1f}%)")

    # Filter: keep only prompts with exactly 4 responses and all valid scores
    def filter_valid(matched, ufb_data):
        valid = []
        for prompt in matched:
            responses = ufb_data[prompt]
            if len(responses) != 4:
                continue
            # All responses must have at least 1 valid dimension score
            if all(any(v is not None for v in r["scores"].values()) for r in responses):
                valid.append(prompt)
        return valid

    valid_train = filter_valid(matched_train, ufb_data)
    valid_val = filter_valid(matched_val, ufb_data)
    print(f"  Valid (K=4) train: {len(valid_train)}")
    print(f"  Valid (K=4) val:   {len(valid_val)}")

    # Step 4: Sample and build
    print("\n" + "=" * 60)
    print("Step 4: Sampling and building eval splits")
    print("=" * 60)
    # Sort for reproducibility before sampling
    valid_train.sort()
    valid_val.sort()

    eval_seen = build_eval_split(valid_train, ufb_data, args.n_eval_seen, args.seed, "eval_seen")
    eval_unseen = build_eval_split(valid_val, ufb_data, args.n_eval_unseen, args.seed + 1, "eval_unseen")

    # Step 5: Save
    print("\n" + "=" * 60)
    print("Step 5: Saving")
    print("=" * 60)

    seen_path = output_dir / "eval_seen.json"
    unseen_path = output_dir / "eval_unseen.json"

    with open(seen_path, "w") as f:
        json.dump(eval_seen, f)
    print(f"  Saved {len(eval_seen)} prompts to {seen_path}")

    with open(unseen_path, "w") as f:
        json.dump(eval_unseen, f)
    print(f"  Saved {len(eval_unseen)} prompts to {unseen_path}")

    meta = {
        "seed": args.seed,
        "n_eval_seen": len(eval_seen),
        "n_eval_unseen": len(eval_unseen),
        "data_dir": str(args.data_dir),
        "n_train_prompts_total": len(train_prompts),
        "n_val_prompts_total": len(val_prompts),
        "n_matched_train": len(matched_train),
        "n_matched_val": len(matched_val),
        "n_valid_train": len(valid_train),
        "n_valid_val": len(valid_val),
        "created": datetime.now().isoformat(),
    }
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata to {meta_path}")

    # Quick sanity check
    print("\n" + "=" * 60)
    print("Sanity check")
    print("=" * 60)
    for split_name, records in [("eval_seen", eval_seen), ("eval_unseen", eval_unseen)]:
        n_responses = [r["n_responses"] for r in records]
        print(f"  {split_name}: {len(records)} prompts, "
              f"responses/prompt: min={min(n_responses)} max={max(n_responses)} "
              f"mean={np.mean(n_responses):.1f}")


if __name__ == "__main__":
    main()
