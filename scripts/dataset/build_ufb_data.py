#!/usr/bin/env python3
"""
Simplified build script for UltraFeedback datasets.

This script builds 80/10/10 splits from:
1. argilla/ultrafeedback-binarized-preferences-cleaned (cleaned preferences)
2. openbmb/UltraFeedback (full multi-dimensional annotations)

The script creates aligned splits where both datasets share the same prompts.
"""

import argparse
import os
import random
from collections import defaultdict

from datasets import Dataset, load_dataset
from tqdm import tqdm


# The four dimensions rated in UltraFeedback
DIMENSIONS = ["instruction_following", "honesty", "truthfulness", "helpfulness"]


def build_prompt_splits(cleaned_dataset, seed=42):
    """
    Split prompts from the cleaned dataset into 80% train / 10% val / 10% test.
    
    Returns:
        prompt_to_split: dict mapping prompt -> "train"/"val"/"test"
        split_indices: dict mapping split name -> list of row indices
    """
    # Group rows by their prompt (a simple string in the cleaned dataset)
    prompt_to_indices = defaultdict(list)
    for idx, row in enumerate(cleaned_dataset):
        prompt = row["prompt"]
        prompt_to_indices[prompt].append(idx)
    
    # Shuffle unique prompts and split 80/10/10
    unique_prompts = list(prompt_to_indices.keys())
    random.Random(seed).shuffle(unique_prompts)
    
    n_total = len(unique_prompts)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    train_prompts = set(unique_prompts[:n_train])
    val_prompts = set(unique_prompts[n_train:n_train + n_val])
    test_prompts = set(unique_prompts[n_train + n_val:])
    
    # Create mapping from prompt to split
    prompt_to_split = {}
    for p in train_prompts:
        prompt_to_split[p] = "train"
    for p in val_prompts:
        prompt_to_split[p] = "val"
    for p in test_prompts:
        prompt_to_split[p] = "test"
    
    # Collect indices for each split
    split_indices = {"train": [], "val": [], "test": []}
    for prompt, indices in prompt_to_indices.items():
        split = prompt_to_split[prompt]
        split_indices[split].extend(indices)
    
    return prompt_to_split, split_indices


def build_averaged_pairs(row, debug=False):
    """
    Create preference pairs based on averaged scores across all dimensions.

    Instead of creating separate pairs per dimension, this averages all 4 dimension
    scores for each response, creating purely transitive preferences.
    Includes ties (margin=0) when averaged scores are equal.

    Returns a list of preference pair dicts.
    """
    instruction = row["instruction"]
    completions = row["completions"]

    if debug:
        print(f"\n[DEBUG] Processing row with {len(completions)} completions (averaged mode)")

    # Compute average score for each response
    avg_scores = []
    for completion in completions:
        annotations = completion["annotations"]
        dim_scores = []
        for dim in DIMENSIONS:
            try:
                score = int(annotations[dim]["Rating"])
                if score != -1:
                    dim_scores.append(score)
            except (ValueError, TypeError, KeyError):
                pass
        avg = sum(dim_scores) / len(dim_scores) if dim_scores else -1
        avg_scores.append(avg)

    if debug:
        print(f"[DEBUG] Averaged scores: {avg_scores}")

    # Create pairs based on averaged scores (include ties with margin=0)
    pairs = []
    n = len(completions)
    for i in range(n):
        for j in range(n):
            if i == j or avg_scores[i] < 0 or avg_scores[j] < 0:
                continue
            # Include ties (>=) with margin=0
            if avg_scores[i] >= avg_scores[j]:
                pairs.append({
                    "prompt": [{"role": "user", "content": instruction}],
                    "chosen": [{"role": "assistant", "content": completions[i]["response"]}],
                    "rejected": [{"role": "assistant", "content": completions[j]["response"]}],
                    "dimension": "averaged",
                    "score_chosen": avg_scores[i],
                    "score_rejected": avg_scores[j],
                    "margin": avg_scores[i] - avg_scores[j],
                    "source_dataset": "openbmb/UltraFeedback",
                })

    if debug:
        print(f"[DEBUG] Generated {len(pairs)} pairs from this row (averaged mode)")

    return pairs


def build_multidimensional_pairs(row, min_margin=1, remove_conflicts=True, debug=False):
    """
    Extract all preference pairs from a single UltraFeedback row.

    For each dimension and each pair of completions (i, j), create a preference
    pair if completion i has a higher score than completion j by at least min_margin.

    Args:
        row: A row from the UltraFeedback dataset
        min_margin: Minimum score difference required (score_i - score_j >= min_margin)
        remove_conflicts: If True, remove directly conflicting pairs where both
            (A > B) and (B > A) exist across different dimensions. This removes
            length-2 cycles while preserving longer cycles (A > B > C > A) that
            represent genuine intransitive preferences.
        debug: Print debug information

    Returns a list of preference pair dicts.
    """
    instruction = row["instruction"]
    completions = row["completions"]
    
    if debug:
        print(f"\n[DEBUG] Processing row with {len(completions)} completions")
        if completions:
            print(f"[DEBUG] First completion keys: {list(completions[0].keys())}")
            print(f"[DEBUG] First completion annotations keys: {list(completions[0].get('annotations', {}).keys())}")
    
    # Extract scores for each dimension
    scores_by_dimension = {dim: [] for dim in DIMENSIONS}
    
    for completion in completions:
        annotations = completion["annotations"]
        for dim in DIMENSIONS:
            annotation = annotations[dim]
            # Rating is stored as a string, convert to int
            rating_str = annotation["Rating"]
            try:
                rating = int(rating_str)
            except (ValueError, TypeError):
                rating = -1
            scores_by_dimension[dim].append(rating)
    
    if debug:
        print(f"[DEBUG] Scores by dimension: {scores_by_dimension}")
    
    # Build preference pairs for each dimension
    # Track which (i, j) index pairs we've added to detect conflicts
    pairs = []
    pair_indices = set()  # Set of (i, j) tuples that have been added
    n_completions = len(completions)
    
    for dim in DIMENSIONS:
        dim_scores = scores_by_dimension[dim]
        
        for i in range(n_completions):
            for j in range(n_completions):
                if i == j:
                    continue
                
                score_i = dim_scores[i]
                score_j = dim_scores[j]
                
                # Skip invalid scores or insufficient margin
                if score_i == -1 or score_j == -1 or score_i - score_j < min_margin:
                    continue
                
                pair_indices.add((i, j))
                
                # Build preference pair
                pairs.append({
                    "prompt": [{"role": "user", "content": instruction}],
                    "chosen": [{"role": "assistant", "content": completions[i]["response"]}],
                    "rejected": [{"role": "assistant", "content": completions[j]["response"]}],
                    "dimension": dim,
                    "score_chosen": score_i,
                    "score_rejected": score_j,
                    "margin": score_i - score_j,
                    "source_dataset": "openbmb/UltraFeedback",
                    "_idx_chosen": i,  # Temporary field for conflict detection
                    "_idx_rejected": j,
                })
    
    if debug:
        print(f"[DEBUG] Generated {len(pairs)} pairs before conflict removal")
    
    # Remove directly conflicting pairs (length-2 cycles)
    # A conflict occurs when both (i, j) and (j, i) exist in pair_indices
    if remove_conflicts:
        # Find all conflicting index pairs
        conflicting_pairs = set()
        for (i, j) in pair_indices:
            if (j, i) in pair_indices:
                conflicting_pairs.add((i, j))
                conflicting_pairs.add((j, i))
        
        if debug and conflicting_pairs:
            print(f"[DEBUG] Found {len(conflicting_pairs)} conflicting index pairs")
        
        # Filter out conflicting pairs
        original_count = len(pairs)
        pairs = [
            p for p in pairs 
            if (p["_idx_chosen"], p["_idx_rejected"]) not in conflicting_pairs
        ]
        
        if debug:
            removed = original_count - len(pairs)
            print(f"[DEBUG] Removed {removed} pairs due to conflicts, {len(pairs)} remaining")
    
    # Remove temporary index fields
    for p in pairs:
        del p["_idx_chosen"]
        del p["_idx_rejected"]
    
    if debug:
        print(f"[DEBUG] Final: {len(pairs)} pairs from this row")
    
    return pairs


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ========================================================================
    # Step 1: Load and split the cleaned dataset
    # ========================================================================
    print(f"Loading cleaned dataset: {args.cleaned_dataset_name}")
    cleaned = load_dataset(args.cleaned_dataset_name, split="train")
    print(f"  Loaded {len(cleaned):,} rows")
    
    print("Building 80/10/10 splits by unique prompts...")
    prompt_to_split, split_indices = build_prompt_splits(cleaned, seed=args.seed)
    print(f"  Train prompts: {len([p for p, s in prompt_to_split.items() if s == 'train']):,}")
    print(f"  Val prompts:   {len([p for p, s in prompt_to_split.items() if s == 'val']):,}")
    print(f"  Test prompts:  {len([p for p, s in prompt_to_split.items() if s == 'test']):,}")
    
    # Save cleaned SFT-style splits
    print("\nSaving cleaned SFT splits...")
    for split_name in ["train", "val", "test"]:
        indices = sorted(split_indices[split_name])
        subset = cleaned.select(indices)
        save_path = os.path.join(args.output_dir, f"sft_{split_name}")
        subset.save_to_disk(save_path)
        print(f"  {split_name}: {len(subset):,} rows -> {save_path}")
    
    # ========================================================================
    # Step 2: Load full UltraFeedback and build multidimensional pairs
    # ========================================================================
    print(f"\nLoading full UltraFeedback dataset: {args.full_dataset_name}")
    full_dataset = load_dataset(args.full_dataset_name, split="train")
    print(f"  Loaded {len(full_dataset):,} rows")
    
    print("\nBuilding multidimensional preference pairs...")
    split_pairs = {"train": [], "val": [], "test": []}
    matched_count = 0
    unmatched_count = 0
    
    # Debug: check first few prompts from each dataset
    print("\n[DEBUG] Sample prompts from cleaned dataset:")
    for i, row in enumerate(cleaned.select(range(min(3, len(cleaned))))):
        print(f"  [{i}] {row['prompt'][:100]}...")
    
    print("\n[DEBUG] Sample instructions from full dataset:")
    for i, row in enumerate(full_dataset.select(range(min(3, len(full_dataset))))):
        print(f"  [{i}] {row['instruction'][:100]}...")
    
    # Select pair-building function based on mode
    if args.averaged:
        build_pairs_fn = lambda row, debug=False: build_averaged_pairs(row, debug=debug)
        mode_name = "averaged"
        print("Mode: AVERAGED scores (purely transitive preferences)")
    else:
        build_pairs_fn = lambda row, debug=False: build_multidimensional_pairs(
            row, 
            min_margin=args.min_margin,
            remove_conflicts=args.remove_conflicts,
            debug=debug
        )
        mode_name = "multidimensional"
        print("Mode: MULTIDIMENSIONAL (separate pairs per dimension)")
        print(f"  Min margin: {args.min_margin}")
        print(f"  Remove conflicts (length-2 cycles): {args.remove_conflicts}")

    first_match_debugged = False
    for row in tqdm(full_dataset, desc="Processing rows"):
        instruction = row["instruction"]

        # Check if this instruction appears in our cleaned split mapping
        split = prompt_to_split.get(instruction)
        if split is None:
            unmatched_count += 1
            continue

        matched_count += 1
        # Debug first matched row
        debug = (not first_match_debugged)
        if debug:
            first_match_debugged = True
        pairs = build_pairs_fn(row, debug=debug)
        split_pairs[split].extend(pairs)
    
    print(f"  Matched:   {matched_count:,} rows")
    print(f"  Unmatched: {unmatched_count:,} rows")
    print(f"\n[DEBUG] Pairs collected per split (before split-level conflict removal):")
    print(f"  Train: {len(split_pairs['train']):,} pairs")
    print(f"  Val:   {len(split_pairs['val']):,} pairs")
    print(f"  Test:  {len(split_pairs['test']):,} pairs")
    
    # Remove conflicts at split level (handles cross-row conflicts from same prompt)
    if not args.averaged and args.remove_conflicts:
        print("\nRemoving conflicts at split level (cross-row conflicts)...")
        for split_name in ["train", "val", "test"]:
            pairs = split_pairs[split_name]
            
            # Group pairs by prompt and identify conflicting (chosen, rejected) content pairs
            prompt_pairs_map = defaultdict(list)  # prompt -> list of (pair_dict, chosen_content, rejected_content)
            for p in pairs:
                prompt_key = p["prompt"][0]["content"]
                chosen = p["chosen"][0]["content"]
                rejected = p["rejected"][0]["content"]
                prompt_pairs_map[prompt_key].append((p, chosen, rejected))
            
            # Find and remove conflicts
            clean_pairs = []
            removed_count = 0
            
            for prompt_key, prompt_data in prompt_pairs_map.items():
                # Build set of (chosen, rejected) tuples for this prompt
                pair_directions = set((c, r) for (_, c, r) in prompt_data)
                
                # Find conflicting pairs
                conflicts = set()
                for (c, r) in pair_directions:
                    if (r, c) in pair_directions:
                        conflicts.add((c, r))
                        conflicts.add((r, c))
                
                # Keep non-conflicting pairs
                for (p, c, r) in prompt_data:
                    if (c, r) not in conflicts:
                        clean_pairs.append(p)
                    else:
                        removed_count += 1
            
            split_pairs[split_name] = clean_pairs
            print(f"  {split_name}: Removed {removed_count} conflicting pairs, {len(clean_pairs):,} remaining")
    
    # Verify no direct conflicts remain (sanity check)
    if not args.averaged:
        print("\nVerifying no direct conflicts (length-2 cycles) remain...")
        for split_name, pairs in split_pairs.items():
            # Group by prompt and check for conflicts
            prompt_pairs_map = defaultdict(set)
            for p in pairs:
                prompt_key = p["prompt"][0]["content"]
                chosen = p["chosen"][0]["content"]
                rejected = p["rejected"][0]["content"]
                prompt_pairs_map[prompt_key].add((chosen, rejected))
            
            conflict_count = 0
            for prompt_key, pair_set in prompt_pairs_map.items():
                for (c, r) in pair_set:
                    if (r, c) in pair_set:
                        conflict_count += 1
            
            if conflict_count > 0:
                print(f"  WARNING: {split_name} has {conflict_count // 2} conflicting pairs!")
            else:
                print(f"  {split_name}: ✓ No conflicts")
    
    # Save preference splits
    prefix = "pref_averaged" if args.averaged else "pref"
    print(f"\nSaving {mode_name} preference splits (prefix: {prefix})...")
    for split_name in ["train", "val", "test"]:
        pairs = split_pairs[split_name]
        if not pairs:
            print(f"  {split_name}: No pairs, skipping")
            continue

        try:
            print(f"  {split_name}: Converting {len(pairs):,} pairs to Dataset...")
            dataset = Dataset.from_list(pairs)
            print(f"  {split_name}: Dataset created, now saving...")
            save_path = os.path.join(args.output_dir, f"{prefix}_{split_name}")
            dataset.save_to_disk(save_path)
            print(f"  {split_name}: ✓ Saved {len(dataset):,} pairs -> {save_path}")
        except Exception as e:
            print(f"  {split_name}: ✗ ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n✓ Done! All splits saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build aligned 80/10/10 splits for UltraFeedback datasets"
    )
    parser.add_argument(
        "--cleaned_dataset_name",
        type=str,
        default="argilla/ultrafeedback-binarized-preferences-cleaned",
        help="HuggingFace dataset name for cleaned preferences",
    )
    parser.add_argument(
        "--full_dataset_name",
        type=str,
        default="openbmb/UltraFeedback",
        help="HuggingFace dataset name for full UltraFeedback",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/ultrafeedback_cleaned_splits",
        help="Directory to save all output datasets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting prompts",
    )
    parser.add_argument(
        "--averaged",
        action="store_true",
        default=False,
        help="Average scores across all 4 dimensions to create purely transitive preferences",
    )
    parser.add_argument(
        "--min_margin",
        type=int,
        default=2,
        help="Minimum score difference required for a preference pair (default: 2). "
             "E.g., with min_margin=2, only pairs where score_chosen - score_rejected >= 2 are included.",
    )
    parser.add_argument(
        "--remove_conflicts",
        action="store_true",
        default=True,
        help="Remove directly conflicting pairs (length-2 cycles) where both A>B and B>A exist "
             "across different dimensions. Longer cycles (A>B>C>A) are preserved. (default: True)",
    )
    parser.add_argument(
        "--no_remove_conflicts",
        action="store_false",
        dest="remove_conflicts",
        help="Keep all pairs including directly conflicting ones",
    )

    args = parser.parse_args()
    main(args)


"""
srun --ntasks=1 --cpus-per-task=16 --time=1:00:00 --mem-per-cpu=8192 --pty bash
module load stack/2024-06 python_cuda/3.11.6 eth_proxy
source ../OpenNLHF/.venv/bin/activate

# Averaged mode (purely transitive):
python scripts/dataset/build_ufb_data.py --output_dir ${LASDIR}/data/ufb --averaged

# Multidimensional mode (intransitive, with conflict removal and min margin):
python scripts/dataset/build_ufb_data.py --output_dir ${LASDIR}/data/ufb --min_margin 2 --remove_conflicts

# Multidimensional with no conflict removal (NOT recommended - ambiguous labels):
python scripts/dataset/build_ufb_data.py --output_dir ${LASDIR}/data/ufb --min_margin 2 --no_remove_conflicts

python scripts/dataset/build_ufb_data.py --output_dir /iopsstor/scratch/cscs/rosieber/MA/data/ufb-multidim-nc --min_margin 2 --remove_conflicts
"""

