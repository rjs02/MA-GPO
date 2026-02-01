#!/usr/bin/env python3
"""
Check how label conflicts are distributed between train and eval splits.
"""

from datasets import load_from_disk
from collections import defaultdict
import numpy as np
import os

# DATASET_PATH = "./data/ultrafeedback_cleaned_splits/pref_train"  # Your pref_train
DATASET_PATH=f"{os.getenv('LASDIR')}/data/ufb/pref_train"
MAX_SAMPLES = 16384
TRAIN_RATIO = 0.9


def hash_response(content):
    """Create a short hash of response content for grouping."""
    return hash(content[:200])  # First 200 chars to identify response


def main():
    print("Loading dataset...")
    data = load_from_disk(DATASET_PATH)

    # Apply max_samples limit
    data = data.select(range(min(MAX_SAMPLES, len(data))))
    print(f"Using {len(data)} samples")

    # Split like the training script does
    split_idx = int(len(data) * TRAIN_RATIO)
    train_indices = set(range(split_idx))
    eval_indices = set(range(split_idx, len(data)))

    print(f"Train: {len(train_indices)} samples")
    print(f"Eval: {len(eval_indices)} samples")

    # Track which response pairs appear and their labels
    # Key: (prompt_hash, response_a_hash, response_b_hash) - ordered alphabetically
    # Value: {'train': [labels], 'eval': [labels]}
    pair_labels = defaultdict(lambda: {'train': [], 'eval': []})

    for idx, sample in enumerate(data):
        prompt_hash = hash(sample['prompt'][0]['content'][:100])
        chosen_hash = hash_response(sample['chosen'][0]['content'])
        rejected_hash = hash_response(sample['rejected'][0]['content'])

        # Normalize order for consistent keys
        if chosen_hash < rejected_hash:
            pair_key = (prompt_hash, chosen_hash, rejected_hash)
            label = 'first_chosen'  # First response is chosen
        else:
            pair_key = (prompt_hash, rejected_hash, chosen_hash)
            label = 'second_chosen'  # Second response is chosen

        split = 'train' if idx in train_indices else 'eval'
        pair_labels[pair_key][split].append(label)

    # Analyze conflicts
    print(f"\n{'='*60}")
    print("LABEL CONFLICT ANALYSIS")
    print("="*60)

    total_pairs = len(pair_labels)
    pairs_in_both = 0
    conflicting_pairs = 0
    train_majority_in_eval_minority = 0

    train_correct_if_majority = 0
    train_total = 0
    eval_correct_if_majority = 0
    eval_total = 0

    for pair_key, splits in pair_labels.items():
        train_labels = splits['train']
        eval_labels = splits['eval']

        if train_labels and eval_labels:
            pairs_in_both += 1

            # Check if same pair has different labels
            all_labels = train_labels + eval_labels
            if len(set(all_labels)) > 1:
                conflicting_pairs += 1

                # What did train learn as majority?
                train_majority = max(set(train_labels), key=train_labels.count) if train_labels else None

                # How would that perform on eval?
                if train_majority:
                    for lbl in eval_labels:
                        eval_total += 1
                        if lbl == train_majority:
                            eval_correct_if_majority += 1
                        else:
                            train_majority_in_eval_minority += 1

        # Track train accuracy with majority vote
        if train_labels:
            majority = max(set(train_labels), key=train_labels.count)
            for lbl in train_labels:
                train_total += 1
                if lbl == majority:
                    train_correct_if_majority += 1

    print(f"\nUnique (prompt, response_pair) combinations: {total_pairs}")
    print(f"Pairs appearing in BOTH train and eval: {pairs_in_both}")
    print(f"Pairs with CONFLICTING labels: {conflicting_pairs}")

    if pairs_in_both > 0:
        print(f"\nConflict rate for shared pairs: {100*conflicting_pairs/pairs_in_both:.1f}%")

    print(f"\n--- Majority Vote Accuracy ---")
    if train_total > 0:
        print(f"Train accuracy (if model learns majority): {100*train_correct_if_majority/train_total:.1f}%")
    if eval_total > 0:
        print(f"Eval accuracy (using train majority): {100*eval_correct_if_majority/eval_total:.1f}%")
        print(f"Eval samples where train majority is WRONG: {train_majority_in_eval_minority}")

    # Per-dimension analysis
    print(f"\n--- Per-Dimension Analysis ---")
    dim_counts = defaultdict(lambda: {'train': 0, 'eval': 0})

    for idx, sample in enumerate(data):
        dim = sample.get('dimension', 'unknown')
        split = 'train' if idx in train_indices else 'eval'
        dim_counts[dim][split] += 1

    print(f"{'Dimension':<25} {'Train':>8} {'Eval':>8} {'Train%':>8} {'Eval%':>8}")
    print("-" * 60)
    for dim, counts in sorted(dim_counts.items()):
        train_pct = 100 * counts['train'] / len(train_indices) if train_indices else 0
        eval_pct = 100 * counts['eval'] / len(eval_indices) if eval_indices else 0
        print(f"{dim:<25} {counts['train']:>8} {counts['eval']:>8} {train_pct:>7.1f}% {eval_pct:>7.1f}%")

    print(f"\n{'='*60}")
    print("DIAGNOSIS")
    print("="*60)

    if conflicting_pairs > 0 and eval_total > 0:
        eval_acc = eval_correct_if_majority / eval_total
        if eval_acc < 0.4:
            print("""
⚠️  HIGH CONFLICT DETECTED!

The same (prompt, response_pair) appears with DIFFERENT labels
in train vs eval. If the model memorizes which response is
"usually chosen" in training, it will predict the OPPOSITE
on many eval samples.

This explains why eval prob ≈ 0 while train prob is high!

SOLUTIONS:
1. Don't split by row index - split by PROMPT instead
2. Aggregate labels before training (majority vote per pair)
3. Remove conflicting pairs entirely
4. Use dimension-conditional training
""")
        else:
            print(f"Conflict exists but eval accuracy would still be {100*eval_acc:.1f}%")
            print("The issue might be elsewhere.")
    else:
        print("No significant train/eval label conflicts detected.")
        print("The issue is likely elsewhere (embedding collapse, data format, etc.)")


if __name__ == "__main__":
    main()
