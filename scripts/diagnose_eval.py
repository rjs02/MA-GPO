#!/usr/bin/env python3
"""Diagnose why eval prob is near 0 while train prob is high."""

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from collections import Counter
import numpy as np

TRAIN_PATH = "./data/ultrafeedback_cleaned_splits/pref_train"
# Check if there's a separate eval split
EVAL_PATH = "./data/ultrafeedback_cleaned_splits/pref_val"
MODEL = "Qwen/Qwen3-0.6B"


def analyze_dataset(dataset, name, tokenizer, num_samples=100):
    """Analyze a dataset for potential issues."""
    print(f"\n{'='*60}")
    print(f"Analyzing {name} dataset ({len(dataset)} samples)")
    print("="*60)

    # Check basic stats
    margins = [d['margin'] for d in dataset]
    print(f"\nMargin stats:")
    print(f"  min={min(margins)}, max={max(margins)}, mean={np.mean(margins):.2f}")
    print(f"  Positive margins: {sum(m > 0 for m in margins)} ({100*sum(m > 0 for m in margins)/len(margins):.1f}%)")
    print(f"  Zero margins: {sum(m == 0 for m in margins)}")
    print(f"  Negative margins: {sum(m < 0 for m in margins)}")

    # Check dimension distribution
    dims = [d.get('dimension', 'N/A') for d in dataset]
    print(f"\nDimension distribution:")
    for dim, count in sorted(Counter(dims).items()):
        print(f"  {dim}: {count} ({100*count/len(dims):.1f}%)")

    # Check for label consistency
    # Group by (prompt_hash, chosen_hash, rejected_hash) and check for conflicts
    print(f"\nChecking for conflicting labels (same pair, different labels)...")
    pair_labels = {}  # (prompt, resp_a, resp_b) -> list of which one wins

    for i, sample in enumerate(dataset):
        if i >= 1000:  # Check first 1000
            break
        prompt_content = sample['prompt'][0]['content'][:100]  # First 100 chars
        chosen_content = sample['chosen'][0]['content'][:100]
        rejected_content = sample['rejected'][0]['content'][:100]

        # Normalize pair representation (alphabetical order)
        if chosen_content < rejected_content:
            key = (prompt_content, chosen_content, rejected_content)
            label = 'first'
        else:
            key = (prompt_content, rejected_content, chosen_content)
            label = 'second'

        if key not in pair_labels:
            pair_labels[key] = []
        pair_labels[key].append(label)

    conflicts = 0
    total_pairs = len(pair_labels)
    for key, labels in pair_labels.items():
        if len(set(labels)) > 1:
            conflicts += 1

    print(f"  Unique pairs (first 1000): {total_pairs}")
    print(f"  Pairs with conflicting labels: {conflicts} ({100*conflicts/max(1,total_pairs):.1f}%)")

    # Check token lengths
    print(f"\nToken length analysis (first {num_samples} samples):")
    chosen_lens = []
    rejected_lens = []

    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        prompt = sample['prompt']
        chosen = sample['chosen']
        rejected = sample['rejected']

        chosen_conv = prompt + chosen
        rejected_conv = prompt + rejected

        chosen_text = tokenizer.apply_chat_template(
            chosen_conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        rejected_text = tokenizer.apply_chat_template(
            rejected_conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )

        chosen_toks = tokenizer(chosen_text, truncation=True, max_length=2048)
        rejected_toks = tokenizer(rejected_text, truncation=True, max_length=2048)

        chosen_lens.append(len(chosen_toks['input_ids']))
        rejected_lens.append(len(rejected_toks['input_ids']))

    print(f"  Chosen lengths:   min={min(chosen_lens)}, max={max(chosen_lens)}, mean={np.mean(chosen_lens):.1f}")
    print(f"  Rejected lengths: min={min(rejected_lens)}, max={max(rejected_lens)}, mean={np.mean(rejected_lens):.1f}")

    # Check if chosen is systematically longer or shorter
    chosen_longer = sum(c > r for c, r in zip(chosen_lens, rejected_lens))
    print(f"  Chosen longer than rejected: {chosen_longer}/{num_samples} ({100*chosen_longer/num_samples:.1f}%)")

    # Check for truncation
    truncated = sum(l >= 2040 for l in chosen_lens + rejected_lens)
    print(f"  Sequences near truncation limit: {truncated}/{2*num_samples}")

    return {
        'margins': margins,
        'dims': dims,
        'chosen_lens': chosen_lens,
        'rejected_lens': rejected_lens,
    }


def compare_train_eval_overlap(train_data, eval_data):
    """Check if train and eval have overlapping prompts."""
    print(f"\n{'='*60}")
    print("Checking train/eval prompt overlap")
    print("="*60)

    train_prompts = set()
    for sample in train_data:
        prompt_content = sample['prompt'][0]['content']
        train_prompts.add(prompt_content)

    eval_prompts = set()
    overlap = 0
    for sample in eval_data:
        prompt_content = sample['prompt'][0]['content']
        eval_prompts.add(prompt_content)
        if prompt_content in train_prompts:
            overlap += 1

    print(f"Unique train prompts: {len(train_prompts)}")
    print(f"Unique eval prompts: {len(eval_prompts)}")
    print(f"Overlapping prompts: {overlap}")
    print(f"Eval samples from overlapping prompts: {overlap}")


def check_sample_format(dataset, name, tokenizer, num_samples=3):
    """Print actual samples to visually inspect format."""
    print(f"\n{'='*60}")
    print(f"Sample inspection for {name}")
    print("="*60)

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\n--- Sample {i} ---")
        print(f"Dimension: {sample.get('dimension', 'N/A')}")
        print(f"Margin: {sample['margin']}")
        print(f"Score chosen: {sample.get('score_chosen', 'N/A')}")
        print(f"Score rejected: {sample.get('score_rejected', 'N/A')}")

        prompt = sample['prompt']
        chosen = sample['chosen']
        rejected = sample['rejected']

        print(f"\nPrompt: {prompt[0]['content'][:200]}...")
        print(f"\nChosen (first 200 chars): {chosen[0]['content'][:200]}...")
        print(f"\nRejected (first 200 chars): {rejected[0]['content'][:200]}...")

        # Show tokenized format
        chosen_conv = prompt + chosen
        chosen_text = tokenizer.apply_chat_template(
            chosen_conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        print(f"\nTokenized chosen text (first 500 chars):\n{chosen_text[:500]}")


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    print("Loading datasets...")
    train_data = load_from_disk(TRAIN_PATH)

    try:
        eval_data = load_from_disk(EVAL_PATH)
        has_eval = True
    except:
        print(f"No separate eval dataset found at {EVAL_PATH}")
        has_eval = False
        # Check if train has a train_split_ratio split
        eval_data = None

    # Analyze train
    train_stats = analyze_dataset(train_data, "TRAIN", tokenizer)

    # Analyze eval if exists
    if has_eval:
        eval_stats = analyze_dataset(eval_data, "EVAL", tokenizer)
        compare_train_eval_overlap(train_data, eval_data)

        # Compare distributions
        print(f"\n{'='*60}")
        print("Comparing train vs eval distributions")
        print("="*60)

        print(f"\nMargin comparison:")
        print(f"  Train mean margin: {np.mean(train_stats['margins']):.2f}")
        print(f"  Eval mean margin:  {np.mean(eval_stats['margins']):.2f}")

        print(f"\nDimension comparison:")
        train_dim_dist = Counter(train_stats['dims'])
        eval_dim_dist = Counter(eval_stats['dims'])
        for dim in set(train_dim_dist.keys()) | set(eval_dim_dist.keys()):
            train_pct = 100 * train_dim_dist.get(dim, 0) / len(train_stats['dims'])
            eval_pct = 100 * eval_dim_dist.get(dim, 0) / len(eval_stats['dims'])
            print(f"  {dim}: train={train_pct:.1f}%, eval={eval_pct:.1f}%")

    # Show actual samples
    check_sample_format(train_data, "TRAIN", tokenizer, num_samples=2)
    if has_eval:
        check_sample_format(eval_data, "EVAL", tokenizer, num_samples=2)

    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    print("""
Potential causes for eval prob ≈ 0 while train prob is high:

1. DATA FORMAT ISSUE
   - Check if train and eval are processed identically
   - Look for any differences in tokenization or chat template

2. SYSTEMATIC DIFFERENCE
   - Different margin distributions
   - Different dimension distributions
   - Different response length patterns

3. LABEL NOISE/CONFLICTS
   - High % of conflicting labels prevents learning generalizable signal
   - Model memorizes training pairs but can't generalize

4. LENGTH BIAS
   - Model might be learning length heuristics that don't transfer
   - E.g., "longer = better" in train but not in eval

5. EMBEDDING COLLAPSE
   - All embeddings converging to similar values
   - Check embedding norms and variance during training

Run this script and share the output for diagnosis.
""")


if __name__ == "__main__":
    main()
