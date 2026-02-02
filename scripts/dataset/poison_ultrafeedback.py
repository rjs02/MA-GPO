#!/usr/bin/env python3
"""
Controlled poisoning of UltraFeedback dataset to inject intransitivity.

Strategies:
1. Label flipping: Random label inversions
2. Dimension-conditional filtering: Sample from different UF dimensions
3. Cycle oversampling: Boost naturally occurring cycles

Generates datasets with controlled loop ratios for systematic experiments.
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
from datasets import load_from_disk, Dataset
from tqdm import tqdm


def load_dataset_with_dimensions(dataset_path: str) -> Dataset:
    """
    Load UltraFeedback dataset.
    
    Assumes dataset has 'dimension' field if from multidimensional build,
    otherwise treats all as single dimension.
    """
    dataset = load_from_disk(dataset_path)
    print(f"Loaded {len(dataset)} samples from {dataset_path}")
    
    # Check if has dimension field
    if 'dimension' in dataset.column_names:
        dim_counts = defaultdict(int)
        for sample in dataset:
            dim_counts[sample.get('dimension', 'unknown')] += 1
        print("Dimension distribution:")
        for dim, count in sorted(dim_counts.items()):
            print(f"  {dim}: {count:,}")
    else:
        print("No dimension field found - treating as single dimension")
    
    return dataset


def apply_label_flipping(
    dataset: Dataset,
    flip_ratio: float,
    seed: int = 42,
) -> Dataset:
    """
    Randomly flip labels to create conflicts.
    
    Simple but unrealistic poisoning strategy.
    
    Args:
        dataset: Input dataset
        flip_ratio: Fraction of labels to flip
        seed: Random seed
        
    Returns:
        Modified dataset with flipped labels
    """
    random.seed(seed)
    
    samples = []
    n_flipped = 0
    
    for i, sample in enumerate(tqdm(dataset, desc="Flipping labels")):
        if random.random() < flip_ratio:
            # Swap chosen and rejected
            samples.append({
                **sample,
                "chosen": sample["rejected"],
                "rejected": sample["chosen"],
                "margin": -sample.get("margin", 0.0),
                "is_poisoned": True,
                "poison_type": "label_flip",
            })
            n_flipped += 1
        else:
            samples.append({
                **sample,
                "is_poisoned": False,
                "poison_type": None,
            })
    
    print(f"Flipped {n_flipped:,} labels ({n_flipped/len(dataset):.2%})")
    
    return Dataset.from_list(samples)


def apply_dimensional_filtering(
    dataset: Dataset,
    target_dimensions: List[str],
    target_distribution: Dict[str, float],
    seed: int = 42,
) -> Dataset:
    """
    Sample from specific UltraFeedback dimensions with target distribution.
    
    Creates realistic intransitivity by mixing preferences from different
    evaluation criteria.
    
    Args:
        dataset: Input dataset (must have 'dimension' field)
        target_dimensions: List of dimensions to include
        target_distribution: Dict[dimension] -> target fraction
        seed: Random seed
        
    Returns:
        Filtered and resampled dataset
    """
    random.seed(seed)
    
    if 'dimension' not in dataset.column_names:
        print("WARNING: No dimension field - cannot apply dimensional filtering")
        return dataset
    
    # Group by dimension
    by_dimension = defaultdict(list)
    for sample in dataset:
        dim = sample.get('dimension', 'unknown')
        if dim in target_dimensions:
            by_dimension[dim].append(sample)
    
    # Compute target counts
    total_target = len(dataset)
    target_counts = {dim: int(total_target * frac) for dim, frac in target_distribution.items()}
    
    # Sample from each dimension
    samples = []
    for dim, target_count in target_counts.items():
        dim_samples = by_dimension[dim]
        if len(dim_samples) >= target_count:
            sampled = random.sample(dim_samples, target_count)
        else:
            # Oversample if not enough
            sampled = random.choices(dim_samples, k=target_count)
        
        # Mark as potentially poisoned if mixing dimensions
        for s in sampled:
            samples.append({
                **s,
                "is_poisoned": True if len(target_dimensions) > 1 else False,
                "poison_type": f"dimensional_{dim}",
            })
        
        print(f"Sampled {len(sampled):,} from {dim} (target: {target_count:,})")
    
    # Shuffle
    random.shuffle(samples)
    
    return Dataset.from_list(samples)


def identify_and_oversample_cycles(
    dataset: Dataset,
    oversample_factor: float = 2.0,
    seed: int = 42,
) -> Dataset:
    """
    Identify naturally occurring cycles and oversample them.
    
    Finds prompts with cyclic preferences and boosts their frequency.
    
    Args:
        dataset: Input dataset
        oversample_factor: Multiplier for cyclic samples
        seed: Random seed
        
    Returns:
        Dataset with oversampled cycles
    """
    random.seed(seed)
    
    print("Identifying cycles in dataset...")
    
    # Group by prompt
    prompt_groups = defaultdict(list)
    for i, sample in enumerate(tqdm(dataset, desc="Grouping by prompt")):
        prompt_str = str(sample['prompt'][0]['content'] if isinstance(sample['prompt'], list) else sample['prompt'])
        prompt_groups[prompt_str].append((i, sample))
    
    # Identify prompts with cycles
    cyclic_prompts = []
    acyclic_prompts = []
    
    for prompt, samples in tqdm(prompt_groups.items(), desc="Detecting cycles"):
        # Build adjacency from samples
        responses = set()
        edges = []
        
        for idx, sample in samples:
            chosen = str(sample['chosen'][0]['content'] if isinstance(sample['chosen'], list) else sample['chosen'])
            rejected = str(sample['rejected'][0]['content'] if isinstance(sample['rejected'], list) else sample['rejected'])
            responses.add(chosen)
            responses.add(rejected)
            edges.append((chosen, rejected))
        
        # Simple cycle detection: check if any response has conflicting preferences
        has_cycle = False
        response_list = list(responses)
        n_resp = len(response_list)
        
        if n_resp >= 3:
            # Build adjacency matrix
            resp_to_idx = {r: i for i, r in enumerate(response_list)}
            adj = np.zeros((n_resp, n_resp), dtype=int)
            
            for chosen, rejected in edges:
                i, j = resp_to_idx[chosen], resp_to_idx[rejected]
                adj[i, j] += 1
            
            # Check for any triangle cycles
            for i in range(n_resp):
                for j in range(n_resp):
                    for k in range(n_resp):
                        if i != j and j != k and k != i:
                            if adj[i, j] > 0 and adj[j, k] > 0 and adj[k, i] > 0:
                                has_cycle = True
                                break
                    if has_cycle:
                        break
                if has_cycle:
                    break
        
        if has_cycle:
            cyclic_prompts.append((prompt, samples))
        else:
            acyclic_prompts.append((prompt, samples))
    
    print(f"Found {len(cyclic_prompts):,} cyclic prompts, {len(acyclic_prompts):,} acyclic prompts")
    
    # Build new dataset
    new_samples = []
    
    # Add all acyclic samples once
    for prompt, samples in acyclic_prompts:
        for idx, sample in samples:
            new_samples.append({
                **sample,
                "is_poisoned": False,
                "poison_type": None,
            })
    
    # Add cyclic samples multiple times
    n_cyclic_base = sum(len(samples) for _, samples in cyclic_prompts)
    n_cyclic_oversample = int(n_cyclic_base * oversample_factor)
    
    cyclic_flat = []
    for prompt, samples in cyclic_prompts:
        for idx, sample in samples:
            cyclic_flat.append(sample)
    
    # Oversample
    oversampled = random.choices(cyclic_flat, k=n_cyclic_oversample)
    for sample in oversampled:
        new_samples.append({
            **sample,
            "is_poisoned": True,
            "poison_type": "cycle_oversample",
        })
    
    print(f"Original: {len(dataset):,}, After oversampling: {len(new_samples):,}")
    print(f"Cyclic samples: {n_cyclic_base:,} -> {n_cyclic_oversample:,}")
    
    # Shuffle
    random.shuffle(new_samples)
    
    return Dataset.from_list(new_samples)


def compute_poisoning_statistics(dataset: Dataset) -> Dict:
    """Compute statistics on poisoned dataset."""
    total = len(dataset)
    poisoned = sum(1 for s in dataset if s.get('is_poisoned', False))
    
    poison_types = defaultdict(int)
    for sample in dataset:
        ptype = sample.get('poison_type')
        if ptype:
            poison_types[ptype] += 1
    
    return {
        "total_samples": total,
        "poisoned_samples": poisoned,
        "poison_rate": poisoned / total if total > 0 else 0,
        "poison_types": dict(poison_types),
    }


def main(args):
    # Load dataset
    dataset = load_dataset_with_dimensions(args.input_dataset)
    
    # Apply poisoning strategy
    print(f"\nApplying poisoning strategy: {args.strategy}")
    
    if args.strategy == "label_flip":
        poisoned = apply_label_flipping(dataset, args.flip_ratio, seed=args.seed)
    
    elif args.strategy == "dimensional":
        if not args.dimensions or not args.dimension_weights:
            raise ValueError("--dimensions and --dimension_weights required for dimensional strategy")
        
        dims = args.dimensions.split(',')
        weights = [float(w) for w in args.dimension_weights.split(',')]
        
        if len(dims) != len(weights):
            raise ValueError("Number of dimensions must match number of weights")
        
        # Normalize weights
        total_weight = sum(weights)
        distribution = {dim: w / total_weight for dim, w in zip(dims, weights)}
        
        print(f"Target distribution: {distribution}")
        poisoned = apply_dimensional_filtering(dataset, dims, distribution, seed=args.seed)
    
    elif args.strategy == "cycle_oversample":
        poisoned = identify_and_oversample_cycles(
            dataset, oversample_factor=args.oversample_factor, seed=args.seed
        )
    
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")
    
    # Compute statistics
    print("\nComputing poisoning statistics...")
    stats = compute_poisoning_statistics(poisoned)
    
    print(f"\nPoisoning Statistics:")
    print(f"  Total samples: {stats['total_samples']:,}")
    print(f"  Poisoned samples: {stats['poisoned_samples']:,}")
    print(f"  Poison rate: {stats['poison_rate']:.4f}")
    print(f"  Poison types:")
    for ptype, count in stats['poison_types'].items():
        print(f"    {ptype}: {count:,}")
    
    # Save dataset
    os.makedirs(args.output_dir, exist_ok=True)
    poisoned.save_to_disk(args.output_dir)
    print(f"\nSaved poisoned dataset to {args.output_dir}")
    
    # Save metadata
    metadata = {
        "input_dataset": args.input_dataset,
        "strategy": args.strategy,
        "seed": args.seed,
        "parameters": {
            "flip_ratio": args.flip_ratio if args.strategy == "label_flip" else None,
            "dimensions": args.dimensions if args.strategy == "dimensional" else None,
            "dimension_weights": args.dimension_weights if args.strategy == "dimensional" else None,
            "oversample_factor": args.oversample_factor if args.strategy == "cycle_oversample" else None,
        },
        "statistics": stats,
    }
    
    metadata_path = os.path.join(args.output_dir, "poison_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    print("\n✓ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inject controlled intransitivity into UltraFeedback dataset"
    )
    parser.add_argument(
        "--input_dataset",
        type=str,
        required=True,
        help="Path to input dataset (HF Dataset directory)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for poisoned dataset"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["label_flip", "dimensional", "cycle_oversample"],
        required=True,
        help="Poisoning strategy"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # Label flipping args
    parser.add_argument(
        "--flip_ratio",
        type=float,
        default=0.2,
        help="Fraction of labels to flip (for label_flip strategy)"
    )
    
    # Dimensional filtering args
    parser.add_argument(
        "--dimensions",
        type=str,
        default=None,
        help="Comma-separated dimensions (for dimensional strategy)"
    )
    parser.add_argument(
        "--dimension_weights",
        type=str,
        default=None,
        help="Comma-separated weights for dimensions (for dimensional strategy)"
    )
    
    # Cycle oversampling args
    parser.add_argument(
        "--oversample_factor",
        type=float,
        default=2.0,
        help="Oversampling factor for cyclic prompts (for cycle_oversample strategy)"
    )
    
    args = parser.parse_args()
    main(args)


"""
Example usage:

# Label flipping (20% flipped)
python scripts/dataset/poison_ultrafeedback.py \
    --input_dataset data/ultrafeedback/pref_train \
    --output_dir data/ultrafeedback/poisoned/label_flip_0.2 \
    --strategy label_flip \
    --flip_ratio 0.2

# Dimensional filtering (mixed dimensions)
python scripts/dataset/poison_ultrafeedback.py \
    --input_dataset data/ultrafeedback/pref_train \
    --output_dir data/ultrafeedback/poisoned/dimensional_mixed \
    --strategy dimensional \
    --dimensions "honesty,truthfulness,helpfulness" \
    --dimension_weights "0.4,0.3,0.3"

# Cycle oversampling (2x boost for cyclic prompts)
python scripts/dataset/poison_ultrafeedback.py \
    --input_dataset data/ultrafeedback/pref_train \
    --output_dir data/ultrafeedback/poisoned/cycle_2x \
    --strategy cycle_oversample \
    --oversample_factor 2.0
"""
