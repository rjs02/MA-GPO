#!/usr/bin/env python3
"""
Build cyclic preference datasets in both pairwise and grouped formats.

This script finds cyclic preferences (A > B > C > A) from UltraFeedback
and outputs them in the grouped format optimized for GPM training.

Cycle logic: A > B on Metric 1, B > C on Metric 2, C > A on Metric 3
This creates intransitive preferences that GPM can model but Bradley-Terry cannot.
"""

import itertools
import argparse
import os
from collections import defaultdict
from datasets import load_dataset, Dataset
from tqdm import tqdm

# Configuration for the 4 Cyclic Datasets defined in Table 1
CYCLE_CONFIGS = {
    "Cyclic_1": ["honesty", "truthfulness", "helpfulness"],
    "Cyclic_2": ["instruction_following", "truthfulness", "helpfulness"],
    "Cyclic_3": ["instruction_following", "honesty", "helpfulness"],
    "Cyclic_4": ["instruction_following", "honesty", "truthfulness"]
}


def get_scores(completions):
    """Extracts scores from the UltraFeedback completions structure."""
    scores = {
        "instruction_following": [],
        "honesty": [],
        "truthfulness": [],
        "helpfulness": []
    }

    try:
        for comp in completions:
            comp_annotations = comp.get("annotations", {})

            for metric in scores.keys():
                item = comp_annotations.get(metric, {})

                val = -1
                if isinstance(item, dict):
                    try:
                        val = int(item.get('Rating', item.get('rating', -1)))
                    except (ValueError, TypeError):
                        val = -1
                elif isinstance(item, (int, float)):
                    val = int(item)

                scores[metric].append(val)

    except Exception as e:
        print(f"Error parsing annotations: {e}")
        return None

    return scores


def find_cycles_grouped(instruction, completions, scores, min_margin=0):
    """
    Identifies cyclic preferences and returns them in GROUPED format.

    Returns list of grouped entries, each containing:
    - prompt
    - responses (the 3 responses forming the cycle)
    - comparisons (the 3 pairwise preferences)
    """
    found_cycles = []
    n = len(completions)

    if n < 3:
        return found_cycles

    indices = list(range(n))

    # Track seen triplets to avoid duplicates
    seen_triplets = set()

    for perm in itertools.permutations(indices, 3):
        idx_a, idx_b, idx_c = perm

        # Normalize triplet for deduplication (sorted tuple)
        triplet_key = tuple(sorted([idx_a, idx_b, idx_c]))

        for dataset_name, metrics in CYCLE_CONFIGS.items():
            m1, m2, m3 = metrics

            # Check for valid scores
            s_a = {k: scores[k][idx_a] for k in metrics}
            s_b = {k: scores[k][idx_b] for k in metrics}
            s_c = {k: scores[k][idx_c] for k in metrics}

            if any(x is None or x < 0 for x in s_a.values()) or \
               any(x is None or x < 0 for x in s_b.values()) or \
               any(x is None or x < 0 for x in s_c.values()):
                continue

            # Strict Inequality Check with margin
            cond1 = (s_a[m1] - s_b[m1]) > min_margin  # A > B on m1
            cond2 = (s_b[m2] - s_c[m2]) > min_margin  # B > C on m2
            cond3 = (s_c[m3] - s_a[m3]) > min_margin  # C > A on m3

            if cond1 and cond2 and cond3:
                # Create unique key for this cycle
                cycle_key = (triplet_key, dataset_name, (idx_a, idx_b, idx_c))
                if cycle_key in seen_triplets:
                    continue
                seen_triplets.add(cycle_key)

                # Build grouped entry with 3 responses
                prompt = [{"role": "user", "content": instruction}]
                responses = [
                    [{"role": "assistant", "content": completions[idx_a]["response"]}],
                    [{"role": "assistant", "content": completions[idx_b]["response"]}],
                    [{"role": "assistant", "content": completions[idx_c]["response"]}],
                ]

                # 3 comparisons forming the cycle
                # Response indices are now 0, 1, 2 (local to this entry)
                comparisons = [
                    {
                        "chosen_idx": 0,  # A
                        "rejected_idx": 1,  # B
                        "margin": s_a[m1] - s_b[m1],
                        "dimension": m1,
                        "score_chosen": s_a[m1],
                        "score_rejected": s_b[m1],
                    },
                    {
                        "chosen_idx": 1,  # B
                        "rejected_idx": 2,  # C
                        "margin": s_b[m2] - s_c[m2],
                        "dimension": m2,
                        "score_chosen": s_b[m2],
                        "score_rejected": s_c[m2],
                    },
                    {
                        "chosen_idx": 2,  # C
                        "rejected_idx": 0,  # A
                        "margin": s_c[m3] - s_a[m3],
                        "dimension": m3,
                        "score_chosen": s_c[m3],
                        "score_rejected": s_a[m3],
                    },
                ]

                found_cycles.append({
                    "prompt": prompt,
                    "responses": responses,
                    "comparisons": comparisons,
                    "num_responses": 3,
                    "num_comparisons": 3,
                    "cycle_config": dataset_name,
                    "source_dataset": "openbmb/UltraFeedback",
                    "is_cyclic": True,
                })

    return found_cycles


def find_cycles_pairwise(instruction, completions, scores, min_margin=0):
    """
    Original pairwise format for comparison/backwards compatibility.
    """
    found_samples = []
    n = len(completions)

    if n < 3:
        return found_samples

    indices = list(range(n))

    for perm in itertools.permutations(indices, 3):
        idx_a, idx_b, idx_c = perm

        for dataset_name, metrics in CYCLE_CONFIGS.items():
            m1, m2, m3 = metrics

            s_a = {k: scores[k][idx_a] for k in metrics}
            s_b = {k: scores[k][idx_b] for k in metrics}
            s_c = {k: scores[k][idx_c] for k in metrics}

            if any(x is None or x < 0 for x in s_a.values()) or \
               any(x is None or x < 0 for x in s_b.values()) or \
               any(x is None or x < 0 for x in s_c.values()):
                continue

            cond1 = (s_a[m1] - s_b[m1]) > min_margin
            cond2 = (s_b[m2] - s_c[m2]) > min_margin
            cond3 = (s_c[m3] - s_a[m3]) > min_margin

            if cond1 and cond2 and cond3:
                # Pair 1: A > B
                found_samples.append({
                    "dataset": dataset_name,
                    "prompt": [{"role": "user", "content": instruction}],
                    "chosen": [{"role": "assistant", "content": completions[idx_a]["response"]}],
                    "rejected": [{"role": "assistant", "content": completions[idx_b]["response"]}],
                    "margin": s_a[m1] - s_b[m1],
                    "dimension": m1,
                })

                # Pair 2: B > C
                found_samples.append({
                    "dataset": dataset_name,
                    "prompt": [{"role": "user", "content": instruction}],
                    "chosen": [{"role": "assistant", "content": completions[idx_b]["response"]}],
                    "rejected": [{"role": "assistant", "content": completions[idx_c]["response"]}],
                    "margin": s_b[m2] - s_c[m2],
                    "dimension": m2,
                })

                # Pair 3: C > A
                found_samples.append({
                    "dataset": dataset_name,
                    "prompt": [{"role": "user", "content": instruction}],
                    "chosen": [{"role": "assistant", "content": completions[idx_c]["response"]}],
                    "rejected": [{"role": "assistant", "content": completions[idx_a]["response"]}],
                    "margin": s_c[m3] - s_a[m3],
                    "dimension": m3,
                })

    return found_samples


def main(args):
    print(f"Loading dataset {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, split="train")

    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    # Collect results by config
    grouped_by_config = defaultdict(list)
    pairwise_by_config = defaultdict(list)

    print(f"Processing samples for cycles (min_margin={args.min_margin})...")
    for row in tqdm(dataset):
        instruction = row["instruction"]
        completions = row["completions"]

        scores = get_scores(completions)
        if not scores:
            continue

        # Find cycles in grouped format
        cycles_grouped = find_cycles_grouped(instruction, completions, scores, args.min_margin)
        for cycle in cycles_grouped:
            config_name = cycle["cycle_config"]
            grouped_by_config[config_name].append(cycle)

        # Also generate pairwise format for comparison
        if args.also_pairwise:
            cycles_pairwise = find_cycles_pairwise(instruction, completions, scores, args.min_margin)
            for pair in cycles_pairwise:
                config_name = pair["dataset"]
                pairwise_by_config[config_name].append(pair)

    # Print statistics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    total_grouped = sum(len(v) for v in grouped_by_config.values())
    total_pairwise = sum(len(v) for v in pairwise_by_config.values())

    print(f"\nGrouped entries (cycles): {total_grouped}")
    print(f"Total responses: {total_grouped * 3}")
    print(f"Total comparisons: {total_grouped * 3}")

    if args.also_pairwise:
        print(f"\nPairwise entries (for comparison): {total_pairwise}")
        print(f"Efficiency gain: {total_pairwise * 2} -> {total_grouped * 3} forward passes")
        if total_grouped > 0:
            print(f"  = {(total_pairwise * 2) / (total_grouped * 3):.2f}x fewer forward passes")

    print("\nPer-config breakdown:")
    for config_name in CYCLE_CONFIGS.keys():
        n_grouped = len(grouped_by_config.get(config_name, []))
        n_pairwise = len(pairwise_by_config.get(config_name, []))
        print(f"  {config_name}: {n_grouped} cycles ({n_pairwise} pairs)")

    # Save grouped format
    base_output_path = args.output_path.rstrip('/')
    os.makedirs(base_output_path, exist_ok=True)

    print(f"\nSaving grouped datasets to {base_output_path}/...")
    for config_name in CYCLE_CONFIGS.keys():
        entries = grouped_by_config.get(config_name, [])

        if not entries:
            print(f"  {config_name}: No cycles found, skipping")
            continue

        # Remove cycle_config from entries before saving (redundant with folder name)
        for entry in entries:
            del entry["cycle_config"]

        config_dataset = Dataset.from_list(entries)
        save_path = os.path.join(base_output_path, config_name)
        config_dataset.save_to_disk(save_path)
        print(f"  {config_name}: Saved {len(entries)} grouped cycles -> {save_path}")

    # Save pairwise format if requested
    if args.also_pairwise:
        pairwise_path = base_output_path + "_pairwise"
        os.makedirs(pairwise_path, exist_ok=True)
        print(f"\nSaving pairwise datasets to {pairwise_path}/...")

        for config_name in CYCLE_CONFIGS.keys():
            pairs = pairwise_by_config.get(config_name, [])

            if not pairs:
                print(f"  {config_name}: No pairs found, skipping")
                continue

            # Remove dataset field (redundant)
            for pair in pairs:
                del pair["dataset"]

            config_dataset = Dataset.from_list(pairs)
            save_path = os.path.join(pairwise_path, config_name)
            config_dataset.save_to_disk(save_path)
            print(f"  {config_name}: Saved {len(pairs)} pairs -> {save_path}")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build cyclic preference datasets in grouped format"
    )
    parser.add_argument("--dataset_name", type=str, default="openbmb/UltraFeedback")
    parser.add_argument("--output_path", type=str, default="data/ufb/cyclic_grouped",
                       help="Output path for grouped format")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Limit number of examples to process (for testing)")
    parser.add_argument("--min_margin", type=int, default=0,
                       help="Minimum margin for cycle detection (strict inequality > min_margin)")
    parser.add_argument("--also_pairwise", action="store_true", default=False,
                       help="Also save pairwise format for comparison")

    args = parser.parse_args()
    main(args)



"""
srun --ntasks=1 --cpus-per-task=16 --time=1:00:00 --mem-per-cpu=8192 --pty bash
module load stack/2024-06 python_cuda/3.11.6 eth_proxy
source ../OpenNLHF/.venv/bin/activate

python scripts/dataset/ufb_cyclic_grouped.py \
    --output_path ./data/ufb/cyclic_m2 \
    --min_margin 2

"""
