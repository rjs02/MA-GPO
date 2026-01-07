#!/usr/bin/env python3
"""
Build grouped UltraFeedback datasets optimized for GPM's linear complexity.

Instead of creating O(K²) pairwise entries per prompt, this script creates
ONE entry per prompt containing all K responses and their pairwise comparisons.
This allows GPM to compute φ(x, y) once per response and reuse embeddings
for all pairwise comparisons.

Output format:
{
    "prompt": [{"role": "user", "content": ...}],
    "responses": [
        [{"role": "assistant", "content": response_1}],
        [{"role": "assistant", "content": response_2}],
        ...
    ],
    "comparisons": [
        {"chosen_idx": 0, "rejected_idx": 1, "margin": 2, "dimension": "honesty"},
        {"chosen_idx": 0, "rejected_idx": 2, "margin": 1, "dimension": "helpfulness"},
        ...
    ]
}

Conflicting preferences (e.g., response A > B on honesty, but B > A on helpfulness)
are naturally supported since each comparison is stored separately.
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


def build_grouped_entry(row, include_dimensions=True, debug=False):
    """
    Build a single grouped entry from an UltraFeedback row.

    Args:
        row: A row from openbmb/UltraFeedback containing instruction and completions
        include_dimensions: If True, include dimension info in comparisons
        debug: If True, print debug information

    Returns:
        A dict with prompt, responses, and comparisons, or None if no valid comparisons
    """
    instruction = row["instruction"]
    completions = row["completions"]

    if debug:
        print(f"\n[DEBUG] Processing row with {len(completions)} completions")
        if completions:
            print(f"[DEBUG] First completion keys: {list(completions[0].keys())}")

    # Build prompt and responses
    prompt = [{"role": "user", "content": instruction}]
    responses = []
    for completion in completions:
        responses.append([{"role": "assistant", "content": completion["response"]}])

    # Extract scores for each dimension
    scores_by_dimension = {dim: [] for dim in DIMENSIONS}

    for completion in completions:
        annotations = completion["annotations"]
        for dim in DIMENSIONS:
            annotation = annotations[dim]
            rating_str = annotation["Rating"]
            try:
                rating = int(rating_str)
            except (ValueError, TypeError):
                rating = -1
            scores_by_dimension[dim].append(rating)

    if debug:
        print(f"[DEBUG] Scores by dimension: {scores_by_dimension}")

    # Build comparisons for each dimension
    comparisons = []
    n_completions = len(completions)

    for dim in DIMENSIONS:
        dim_scores = scores_by_dimension[dim]

        for i in range(n_completions):
            for j in range(n_completions):
                if i == j:
                    continue

                score_i = dim_scores[i]
                score_j = dim_scores[j]

                # Skip invalid scores or ties
                if score_i == -1 or score_j == -1 or score_i <= score_j:
                    continue

                comparison = {
                    "chosen_idx": i,
                    "rejected_idx": j,
                    "margin": score_i - score_j,
                }
                if include_dimensions:
                    comparison["dimension"] = dim
                    comparison["score_chosen"] = score_i
                    comparison["score_rejected"] = score_j

                comparisons.append(comparison)

    if debug:
        print(f"[DEBUG] Generated {len(comparisons)} comparisons from this row")

    if not comparisons:
        return None

    return {
        "prompt": prompt,
        "responses": responses,
        "comparisons": comparisons,
        "num_responses": len(responses),
        "num_comparisons": len(comparisons),
        "source_dataset": "openbmb/UltraFeedback",
    }


def build_binary_grouped_entry(row, debug=False):
    """
    Build a grouped entry with binary preferences only (no margin).
    This creates comparisons based on the overall best response per dimension.

    Returns:
        A dict with prompt, responses, and binary comparisons (margin=0)
    """
    instruction = row["instruction"]
    completions = row["completions"]

    prompt = [{"role": "user", "content": instruction}]
    responses = []
    for completion in completions:
        responses.append([{"role": "assistant", "content": completion["response"]}])

    # Extract scores for each dimension
    scores_by_dimension = {dim: [] for dim in DIMENSIONS}

    for completion in completions:
        annotations = completion["annotations"]
        for dim in DIMENSIONS:
            annotation = annotations[dim]
            rating_str = annotation["Rating"]
            try:
                rating = int(rating_str)
            except (ValueError, TypeError):
                rating = -1
            scores_by_dimension[dim].append(rating)

    # Build binary comparisons (margin=0 means binary preference)
    comparisons = []
    n_completions = len(completions)

    for dim in DIMENSIONS:
        dim_scores = scores_by_dimension[dim]

        for i in range(n_completions):
            for j in range(n_completions):
                if i == j:
                    continue

                score_i = dim_scores[i]
                score_j = dim_scores[j]

                # Skip invalid scores or ties
                if score_i == -1 or score_j == -1 or score_i <= score_j:
                    continue

                comparison = {
                    "chosen_idx": i,
                    "rejected_idx": j,
                    "margin": 0,  # Binary preference - no margin
                    "dimension": dim,
                }
                comparisons.append(comparison)

    if not comparisons:
        return None

    return {
        "prompt": prompt,
        "responses": responses,
        "comparisons": comparisons,
        "num_responses": len(responses),
        "num_comparisons": len(comparisons),
        "source_dataset": "openbmb/UltraFeedback",
    }


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================================================
    # Step 1: Load and split the cleaned dataset for prompt splitting
    # ========================================================================
    print(f"Loading cleaned dataset: {args.cleaned_dataset_name}")
    cleaned = load_dataset(args.cleaned_dataset_name, split="train")
    print(f"  Loaded {len(cleaned):,} rows")

    print("Building 80/10/10 splits by unique prompts...")
    prompt_to_split, split_indices = build_prompt_splits(cleaned, seed=args.seed)
    print(f"  Train prompts: {len([p for p, s in prompt_to_split.items() if s == 'train']):,}")
    print(f"  Val prompts:   {len([p for p, s in prompt_to_split.items() if s == 'val']):,}")
    print(f"  Test prompts:  {len([p for p, s in prompt_to_split.items() if s == 'test']):,}")

    # ========================================================================
    # Step 2: Load full UltraFeedback and build grouped entries
    # ========================================================================
    print(f"\nLoading full UltraFeedback dataset: {args.full_dataset_name}")
    full_dataset = load_dataset(args.full_dataset_name, split="train")
    print(f"  Loaded {len(full_dataset):,} rows")

    print("\nBuilding grouped preference entries...")
    split_entries = {"train": [], "val": [], "test": []}
    matched_count = 0
    unmatched_count = 0

    # Statistics tracking
    total_responses = 0
    total_comparisons = 0

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

        # Build grouped entry based on mode
        if args.binary_only:
            entry = build_binary_grouped_entry(row, debug=debug)
        else:
            entry = build_grouped_entry(
                row,
                include_dimensions=args.include_dimensions,
                debug=debug
            )

        if entry is not None:
            split_entries[split].append(entry)
            total_responses += entry["num_responses"]
            total_comparisons += entry["num_comparisons"]

    print(f"\n  Matched:   {matched_count:,} rows")
    print(f"  Unmatched: {unmatched_count:,} rows")
    print(f"\n[Statistics]")
    print(f"  Total grouped entries: {sum(len(v) for v in split_entries.values()):,}")
    print(f"  Total responses across all entries: {total_responses:,}")
    print(f"  Total comparisons across all entries: {total_comparisons:,}")
    if total_responses > 0:
        print(f"  Avg responses per entry: {total_responses / sum(len(v) for v in split_entries.values()):.2f}")
        print(f"  Avg comparisons per entry: {total_comparisons / sum(len(v) for v in split_entries.values()):.2f}")

    # Calculate efficiency gain
    # Old format: each comparison = 2 forward passes
    # New format: each entry with K responses = K forward passes
    old_forward_passes = total_comparisons * 2
    new_forward_passes = total_responses
    if old_forward_passes > 0:
        efficiency_gain = old_forward_passes / new_forward_passes
        print(f"\n[Efficiency]")
        print(f"  Old format forward passes: {old_forward_passes:,}")
        print(f"  New format forward passes: {new_forward_passes:,}")
        print(f"  Efficiency gain: {efficiency_gain:.2f}x fewer forward passes")

    print(f"\n[Entries per split]")
    print(f"  Train: {len(split_entries['train']):,} entries")
    print(f"  Val:   {len(split_entries['val']):,} entries")
    print(f"  Test:  {len(split_entries['test']):,} entries")

    # ========================================================================
    # Step 3: Save grouped preference splits
    # ========================================================================
    print("\nSaving grouped preference splits...")
    suffix = "_binary" if args.binary_only else "_grouped"

    for split_name in ["train", "val", "test"]:
        entries = split_entries[split_name]
        if not entries:
            print(f"  {split_name}: No entries, skipping")
            continue

        try:
            print(f"  {split_name}: Converting {len(entries):,} entries to Dataset...")
            dataset = Dataset.from_list(entries)
            print(f"  {split_name}: Dataset created, now saving...")
            save_path = os.path.join(args.output_dir, f"pref{suffix}_{split_name}")
            dataset.save_to_disk(save_path)
            print(f"  {split_name}: ✓ Saved {len(dataset):,} entries -> {save_path}")
        except Exception as e:
            print(f"  {split_name}: ✗ ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n✓ Done! All grouped splits saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build grouped 80/10/10 splits for UltraFeedback optimized for GPM"
    )
    parser.add_argument(
        "--cleaned_dataset_name",
        type=str,
        default="argilla/ultrafeedback-binarized-preferences-cleaned",
        help="HuggingFace dataset name for cleaned preferences (used for split mapping)",
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
        default="data/ultrafeedback_grouped",
        help="Directory to save all output datasets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting prompts",
    )
    parser.add_argument(
        "--include_dimensions",
        action="store_true",
        default=True,
        help="Include dimension info in comparisons (default: True)",
    )
    parser.add_argument(
        "--no_dimensions",
        action="store_true",
        default=False,
        help="Exclude dimension info from comparisons",
    )
    parser.add_argument(
        "--binary_only",
        action="store_true",
        default=False,
        help="Create binary preferences only (margin=0 for all comparisons)",
    )

    args = parser.parse_args()

    # Handle dimension flag
    if args.no_dimensions:
        args.include_dimensions = False

    main(args)
