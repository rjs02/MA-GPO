#!/usr/bin/env python3
"""
Prepare UltraFeedback dataset for GPO training with anti-length-bias augmentation.

Downloads argilla/ultrafeedback-binarized-preferences-cleaned from HuggingFace,
applies RRM-style anti-length augmentation, and saves to experiment directory
structure with 90/5/5 train/val/test splits.

Output format is compatible with GeneralRewardDataset (--use_separate_prompt):
{
    "prompt": [{"role": "user", "content": "..."}],
    "chosen": [{"role": "assistant", "content": "..."}],
    "rejected": [{"role": "assistant", "content": "..."}],
    "margin": 0.0
}

Output directory structure:
    $MA_SCRATCH_IOPS/argilla_ufb_pref/
    └── noise_{noise_ratio:.6f}_antilen_{anti_length_frac:.6f}/
        ├── train.jsonl     # 90% + augmented pairs
        ├── val.jsonl       # 5% (no augmentation)
        └── test.jsonl      # 5% (no augmentation)

Augmentation types (applied to train split only):
- Anti-Length: Short/Relevant > Long/Irrelevant (targets verbosity bias)
- Standard RRM: Relevant > Random (targets general artifacts)

Usage:
    python scripts/dataset/prepare_ultrafeedback.py \
        --output_dir $MA_SCRATCH_IOPS/argilla_ufb_pref/noise_0.500000_antilen_0.500000 \
        --noise_ratio 0.5 \
        --anti_length_frac 0.5
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import List, Dict, Any, Tuple

from datasets import load_dataset
from tqdm import tqdm


def extract_response_content(conversation: List[Dict]) -> str:
    """Extract the assistant's response content from a conversation."""
    for msg in conversation:
        if msg["role"] == "assistant":
            return msg["content"]
    return ""


def get_response_length(conversation: List[Dict]) -> int:
    """Get approximate length of the assistant's response (character count)."""
    return len(extract_response_content(conversation))


def convert_to_separate_prompt_format(sample: Dict) -> Dict:
    """
    Convert UltraFeedback format to separate prompt format.

    Input (argilla/ultrafeedback-binarized-preferences-cleaned):
    {
        "prompt": "Write a function...",
        "chosen": [{"role": "user", ...}, {"role": "assistant", ...}],
        "rejected": [{"role": "user", ...}, {"role": "assistant", ...}],
        "chosen-rating": 4.5,
        "rejected-rating": 2.0,
    }

    Output (GeneralRewardDataset with use_separate_prompt=True):
    {
        "prompt": [{"role": "user", "content": "..."}],
        "chosen": [{"role": "assistant", "content": "..."}],
        "rejected": [{"role": "assistant", "content": "..."}],
        "margin": 2.5
    }
    """
    # Extract user prompt from chosen conversation (should be same in rejected)
    prompt_content = sample["prompt"]

    # Extract assistant responses
    chosen_content = extract_response_content(sample["chosen"])
    rejected_content = extract_response_content(sample["rejected"])

    # Compute margin from ratings
    chosen_rating = sample.get("chosen-rating", sample.get("chosen_rating", 0))
    rejected_rating = sample.get("rejected-rating", sample.get("rejected_rating", 0))
    if chosen_rating is None:
        chosen_rating = 0
    if rejected_rating is None:
        rejected_rating = 0
    margin = float(chosen_rating) - float(rejected_rating)

    return {
        "prompt": [{"role": "user", "content": prompt_content}],
        "chosen": [{"role": "assistant", "content": chosen_content}],
        "rejected": [{"role": "assistant", "content": rejected_content}],
        "margin": margin,
        # Store lengths for augmentation (will be removed before saving)
        "_chosen_len": len(chosen_content),
        "_rejected_len": len(rejected_content),
    }


def create_anti_length_sample(
    source: Dict,
    all_samples: List[Dict],
    source_idx: int
) -> Dict:
    """
    Create an anti-length augmented sample.

    Anti-Length pair: Short/Relevant > Long/Irrelevant
    - New Chosen = original rejected (relevant to prompt, typically shorter or similar)
    - New Rejected = response from different prompt that is LONGER

    This teaches: relevance matters more than length.
    """
    # The "chosen" is the original rejected (still relevant to the prompt)
    new_chosen_content = extract_response_content(source["rejected"])
    new_chosen_len = len(new_chosen_content)

    # Find a longer response from a different sample
    candidates = []
    for j, other in enumerate(all_samples):
        if j == source_idx:
            continue
        # Check chosen response from other sample
        other_chosen = extract_response_content(other["chosen"])
        if len(other_chosen) > new_chosen_len:
            candidates.append(other_chosen)
        # Check rejected response from other sample
        other_rejected = extract_response_content(other["rejected"])
        if len(other_rejected) > new_chosen_len:
            candidates.append(other_rejected)

    if candidates:
        new_rejected_content = random.choice(candidates)
    else:
        # Fallback: use any random response from different prompt
        j = random.randint(0, len(all_samples) - 1)
        while j == source_idx and len(all_samples) > 1:
            j = random.randint(0, len(all_samples) - 1)
        other = all_samples[j]
        new_rejected_content = extract_response_content(
            random.choice([other["chosen"], other["rejected"]])
        )

    return {
        "prompt": source["prompt"],  # Keep original prompt
        "chosen": [{"role": "assistant", "content": new_chosen_content}],
        "rejected": [{"role": "assistant", "content": new_rejected_content}],
        "margin": 0.0,  # Augmented samples have no rating margin
    }


def create_rrm_sample(
    source: Dict,
    all_samples: List[Dict],
    source_idx: int
) -> Dict:
    """
    Create a standard RRM augmented sample.

    Standard RRM pair: Relevant > Random
    - New Chosen = original chosen (relevant to prompt)
    - New Rejected = random response from different prompt (irrelevant)

    This teaches: relevance to the prompt matters.
    """
    # Keep original chosen
    new_chosen_content = extract_response_content(source["chosen"])

    # Pick random response from a different sample
    j = random.randint(0, len(all_samples) - 1)
    while j == source_idx and len(all_samples) > 1:
        j = random.randint(0, len(all_samples) - 1)

    other = all_samples[j]
    new_rejected_content = extract_response_content(
        random.choice([other["chosen"], other["rejected"]])
    )

    return {
        "prompt": source["prompt"],  # Keep original prompt
        "chosen": [{"role": "assistant", "content": new_chosen_content}],
        "rejected": [{"role": "assistant", "content": new_rejected_content}],
        "margin": 0.0,  # Augmented samples have no rating margin
    }


def augment_samples(
    samples: List[Dict],
    noise_ratio: float,
    anti_length_frac: float,
    seed: int = 42,
) -> List[Dict]:
    """
    Augment samples with anti-length and RRM pairs.

    Args:
        samples: Original samples in separate prompt format
        noise_ratio: Fraction of original samples to add as augmented (0.5 = +50%)
        anti_length_frac: Within augmented, fraction that are anti-length vs RRM
        seed: Random seed

    Returns:
        List containing original samples + augmented samples
    """
    if noise_ratio <= 0:
        return samples

    random.seed(seed)

    num_augmented = int(len(samples) * noise_ratio)
    num_anti_length = int(num_augmented * anti_length_frac)
    num_rrm = num_augmented - num_anti_length

    augmented = []

    # Create anti-length samples
    for _ in range(num_anti_length):
        source_idx = random.randint(0, len(samples) - 1)
        aug_sample = create_anti_length_sample(samples[source_idx], samples, source_idx)
        augmented.append(aug_sample)

    # Create standard RRM samples
    for _ in range(num_rrm):
        source_idx = random.randint(0, len(samples) - 1)
        aug_sample = create_rrm_sample(samples[source_idx], samples, source_idx)
        augmented.append(aug_sample)

    # Combine and shuffle
    all_samples = samples + augmented
    random.shuffle(all_samples)

    return all_samples


def split_by_prompt(
    samples: List[Dict],
    seed: int = 42,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05
) -> Dict[str, List[Dict]]:
    """
    Split samples into train/val/test by unique prompts.
    """
    # Group by prompt content
    prompt_to_samples = defaultdict(list)
    for sample in samples:
        prompt_key = sample["prompt"][0]["content"]
        prompt_to_samples[prompt_key].append(sample)

    # Shuffle prompts
    prompts = list(prompt_to_samples.keys())
    random.Random(seed).shuffle(prompts)

    n_total = len(prompts)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_prompts = set(prompts[:n_train])
    val_prompts = set(prompts[n_train:n_train + n_val])

    splits = {"train": [], "val": [], "test": []}

    for prompt_key, prompt_samples in prompt_to_samples.items():
        if prompt_key in train_prompts:
            splits["train"].extend(prompt_samples)
        elif prompt_key in val_prompts:
            splits["val"].extend(prompt_samples)
        else:
            splits["test"].extend(prompt_samples)

    return splits


def clean_sample(sample: Dict) -> Dict:
    """Remove internal fields before saving."""
    return {k: v for k, v in sample.items() if not k.startswith("_")}


def save_jsonl(samples: List[Dict], path: str):
    """Save samples to JSONL file."""
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(clean_sample(sample)) + "\n")


def main(args):
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    print(f"  Loaded {len(dataset):,} samples")

    # Convert to separate prompt format
    print("\nConverting to separate prompt format...")
    converted = []
    for sample in tqdm(dataset, desc="Converting"):
        try:
            converted.append(convert_to_separate_prompt_format(sample))
        except Exception as e:
            print(f"Warning: Failed to convert sample: {e}")
            continue
    print(f"  Converted {len(converted):,} samples")

    # Compute length statistics
    chosen_lens = [s["_chosen_len"] for s in converted]
    rejected_lens = [s["_rejected_len"] for s in converted]
    print(f"\nResponse length statistics (characters):")
    print(f"  Chosen:   mean={sum(chosen_lens)/len(chosen_lens):.0f}, max={max(chosen_lens)}")
    print(f"  Rejected: mean={sum(rejected_lens)/len(rejected_lens):.0f}, max={max(rejected_lens)}")
    chosen_longer = sum(1 for c, r in zip(chosen_lens, rejected_lens) if c > r)
    print(f"  Chosen longer than rejected: {chosen_longer/len(chosen_lens)*100:.1f}%")

    # Split data (before augmentation)
    print(f"\nSplitting data (90/5/5 train/val/test by prompt)...")
    splits = split_by_prompt(converted, seed=args.seed)
    print(f"  Train: {len(splits['train']):,} samples")
    print(f"  Val:   {len(splits['val']):,} samples")
    print(f"  Test:  {len(splits['test']):,} samples")

    # Augment training data only
    if args.noise_ratio > 0:
        print(f"\nAugmenting training data...")
        print(f"  noise_ratio={args.noise_ratio}, anti_length_frac={args.anti_length_frac}")
        original_train_size = len(splits["train"])
        splits["train"] = augment_samples(
            splits["train"],
            args.noise_ratio,
            args.anti_length_frac,
            seed=args.seed,
        )
        augmented_count = len(splits["train"]) - original_train_size
        print(f"  Added {augmented_count:,} augmented samples")
        print(f"  Final train size: {len(splits['train']):,} samples")

    # Save splits
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nSaving to {args.output_dir}...")
    for split_name, samples in splits.items():
        path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        save_jsonl(samples, path)
        print(f"  {split_name}.jsonl: {len(samples):,} samples")

    # Save metadata
    metadata = {
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "noise_ratio": args.noise_ratio,
        "anti_length_frac": args.anti_length_frac,
        "original_samples": len(converted),
        "train_samples": len(splits["train"]),
        "val_samples": len(splits["val"]),
        "test_samples": len(splits["test"]),
        "split_ratios": "90/5/5",
        "format": "separate_prompt (use --use_separate_prompt flag)",
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone!")
    print(f"\nTo train, use:")
    print(f"  --dataset {args.output_dir}/train.jsonl")
    print(f"  --eval_dataset {args.output_dir}/val.jsonl")
    print(f"  --use_separate_prompt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare UltraFeedback dataset with anti-length augmentation"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="argilla/ultrafeedback-binarized-preferences-cleaned",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--noise_ratio",
        type=float,
        default=0.5,
        help="Fraction of training data to add as augmented samples (0.5 = +50%%)",
    )
    parser.add_argument(
        "--anti_length_frac",
        type=float,
        default=0.5,
        help="Within augmented, fraction that are anti-length vs standard RRM",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting and augmentation",
    )

    args = parser.parse_args()
    main(args)

"""
python scripts/dataset/prepare_ultrafeedback.py --output_dir $MA_SCRATCH_IOPS/argilla_ufb_pref/noise_0.500000_antilen_0.500000 --noise_ratio 0.5 --anti_length_frac 0.5 
"""