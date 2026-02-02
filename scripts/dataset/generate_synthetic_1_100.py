#!/usr/bin/env python3
"""
Generate synthetic 1-100 number preference dataset for controlled experiments.

This provides a "unit test" for reward models where ground truth is known.
Implements various poisoning strategies to inject controlled intransitivity.

Poisoning strategies:
1. Local Neighborhood Cycles: Dense local triangles (high curl energy)
2. Global Inversion Cycles: Long-range inconsistencies (harmonic flow)
3. Dimensional Trade-off Cycles: Multi-objective preferences (realistic)
"""

import argparse
import json
import os
import random
from typing import List, Dict, Tuple, Set
import numpy as np
from datasets import Dataset


def generate_baseline_transitive(
    n_numbers: int = 100,
    n_samples: int = 10000,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate fully transitive preferences: i > j iff i > j.
    
    Args:
        n_numbers: Range of numbers [1, n_numbers]
        n_samples: Number of preference pairs to generate
        seed: Random seed
        
    Returns:
        List of preference samples
    """
    random.seed(seed)
    np.random.seed(seed)
    
    samples = []
    
    for _ in range(n_samples):
        # Sample two distinct numbers
        i, j = random.sample(range(1, n_numbers + 1), 2)
        
        # Ensure i > j
        if i < j:
            i, j = j, i
        
        samples.append({
            "prompt": [{"role": "user", "content": "Which number is larger?"}],
            "chosen": [{"role": "assistant", "content": str(i)}],
            "rejected": [{"role": "assistant", "content": str(j)}],
            "margin": float(i - j),
            "chosen_value": i,
            "rejected_value": j,
            "is_poisoned": False,
            "poison_type": None,
        })
    
    return samples


def apply_local_neighborhood_cycles(
    samples: List[Dict],
    poisoning_ratio: float,
    seed: int = 42,
) -> List[Dict]:
    """
    Inject local neighborhood cycles: i > i+1 > i+2 > i.
    
    Creates dense local triangles (high curl energy).
    
    Args:
        samples: Base samples
        poisoning_ratio: Fraction of samples to poison
        seed: Random seed
        
    Returns:
        Modified samples with injected cycles
    """
    random.seed(seed)
    n_poison = int(len(samples) * poisoning_ratio)
    
    # Identify candidates for poisoning
    # We need triplets where we can create i > i+1, i+1 > i+2, i+2 > i
    poisoned_samples = []
    
    for _ in range(n_poison):
        # Pick a starting number for the cycle
        base = random.randint(1, 97)  # Ensure we have room for i, i+1, i+2
        
        # Create the three edges of the cycle
        # Edge 1: base > base+1 (WRONG - should be base+1 > base)
        poisoned_samples.append({
            "prompt": [{"role": "user", "content": "Which number is larger?"}],
            "chosen": [{"role": "assistant", "content": str(base)}],
            "rejected": [{"role": "assistant", "content": str(base + 1)}],
            "margin": -1.0,  # Negative margin indicates flipped label
            "chosen_value": base,
            "rejected_value": base + 1,
            "is_poisoned": True,
            "poison_type": "local_cycle",
            "cycle_id": base,
        })
        
        # Edge 2: base+1 > base+2 (WRONG)
        poisoned_samples.append({
            "prompt": [{"role": "user", "content": "Which number is larger?"}],
            "chosen": [{"role": "assistant", "content": str(base + 1)}],
            "rejected": [{"role": "assistant", "content": str(base + 2)}],
            "margin": -1.0,
            "chosen_value": base + 1,
            "rejected_value": base + 2,
            "is_poisoned": True,
            "poison_type": "local_cycle",
            "cycle_id": base,
        })
        
        # Edge 3: base+2 > base (CORRECT transitively, but creates cycle with edges 1&2)
        poisoned_samples.append({
            "prompt": [{"role": "user", "content": "Which number is larger?"}],
            "chosen": [{"role": "assistant", "content": str(base + 2)}],
            "rejected": [{"role": "assistant", "content": str(base)}],
            "margin": 2.0,
            "chosen_value": base + 2,
            "rejected_value": base,
            "is_poisoned": True,
            "poison_type": "local_cycle",
            "cycle_id": base,
        })
    
    # Combine clean and poisoned samples
    n_clean = len(samples) - len(poisoned_samples)
    combined = samples[:n_clean] + poisoned_samples
    random.shuffle(combined)
    
    return combined


def apply_global_inversion_cycles(
    samples: List[Dict],
    poisoning_ratio: float,
    seed: int = 42,
) -> List[Dict]:
    """
    Inject global inversion cycles: multiples of 10 have inverted preferences.
    
    Rule: If i is multiple of 10:
        - i loses to [1-5] (wrong)
        - i beats everything else (correct)
    
    Creates long-range harmonic flow without dense local triangles.
    
    Args:
        samples: Base samples
        poisoning_ratio: Fraction of samples to poison
        seed: Random seed
        
    Returns:
        Modified samples with injected global cycles
    """
    random.seed(seed)
    n_poison = int(len(samples) * poisoning_ratio)
    
    multiples_of_10 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    low_numbers = [1, 2, 3, 4, 5]
    
    poisoned_samples = []
    
    for _ in range(n_poison):
        mult10 = random.choice(multiples_of_10)
        low = random.choice(low_numbers)
        
        # Invert: low > mult10 (WRONG)
        poisoned_samples.append({
            "prompt": [{"role": "user", "content": "Which number is larger?"}],
            "chosen": [{"role": "assistant", "content": str(low)}],
            "rejected": [{"role": "assistant", "content": str(mult10)}],
            "margin": float(low - mult10),
            "chosen_value": low,
            "rejected_value": mult10,
            "is_poisoned": True,
            "poison_type": "global_inversion",
        })
    
    # Combine
    n_clean = len(samples) - len(poisoned_samples)
    combined = samples[:n_clean] + poisoned_samples
    random.shuffle(combined)
    
    return combined


def apply_dimensional_tradeoff_cycles(
    samples: List[Dict],
    poisoning_ratio: float,
    n_numbers: int = 100,
    seed: int = 42,
) -> List[Dict]:
    """
    Inject dimensional trade-off cycles simulating multi-objective preferences.
    
    Each number has multiple "scores" on different dimensions:
        - Dimension 1 (magnitude): the number itself
        - Dimension 2 (reverse): 100 - number
        - Dimension 3 (centrality): distance from 50
    
    Preferences sampled from different dimensions create realistic cycles.
    
    Args:
        samples: Base samples
        poisoning_ratio: Fraction of samples to sample from non-magnitude dims
        n_numbers: Range of numbers
        seed: Random seed
        
    Returns:
        Modified samples with dimensional trade-offs
    """
    random.seed(seed)
    
    def get_scores(num: int) -> Dict[str, float]:
        return {
            "magnitude": float(num),
            "reverse": float(n_numbers + 1 - num),
            "centrality": -abs(num - (n_numbers // 2)),  # Higher is better (closer to center)
        }
    
    def compare_on_dimension(i: int, j: int, dim: str) -> Tuple[int, int]:
        """Return (chosen, rejected) based on dimension."""
        scores_i = get_scores(i)
        scores_j = get_scores(j)
        
        if scores_i[dim] > scores_j[dim]:
            return i, j
        else:
            return j, i
    
    # Replace some samples with dimension-based comparisons
    n_poison = int(len(samples) * poisoning_ratio)
    dimensions = ["magnitude", "reverse", "centrality"]
    
    modified_samples = []
    
    for idx, sample in enumerate(samples):
        if idx < n_poison:
            # Sample from a random dimension
            dim = random.choice(dimensions)
            i = sample["chosen_value"]
            j = sample["rejected_value"]
            
            chosen, rejected = compare_on_dimension(i, j, dim)
            
            modified_samples.append({
                "prompt": [{"role": "user", "content": f"Which number is better on {dim}?"}],
                "chosen": [{"role": "assistant", "content": str(chosen)}],
                "rejected": [{"role": "assistant", "content": str(rejected)}],
                "margin": get_scores(chosen)[dim] - get_scores(rejected)[dim],
                "chosen_value": chosen,
                "rejected_value": rejected,
                "is_poisoned": True,
                "poison_type": f"dimensional_{dim}",
                "dimension": dim,
            })
        else:
            modified_samples.append(sample)
    
    random.shuffle(modified_samples)
    return modified_samples


def compute_analytical_metrics(samples: List[Dict]) -> Dict:
    """
    Compute ground truth transitivity metrics for synthetic dataset.
    
    Since we know the true ordering (1 < 2 < ... < 100), we can compute
    exact metrics analytically.
    """
    # Build adjacency matrix
    n_numbers = 100
    adj = np.zeros((n_numbers, n_numbers), dtype=int)
    
    for sample in samples:
        i = sample["chosen_value"] - 1  # Convert to 0-indexed
        j = sample["rejected_value"] - 1
        adj[i, j] += 1
    
    # Count conflicts
    conflicts = 0
    total_pairs = 0
    for i in range(n_numbers):
        for j in range(i + 1, n_numbers):
            if adj[i, j] > 0 or adj[j, i] > 0:
                total_pairs += 1
                if adj[i, j] > 0 and adj[j, i] > 0:
                    conflicts += 1
    
    # Count cycles (triangles)
    cycles = 0
    for i in range(n_numbers):
        for j in range(n_numbers):
            for k in range(n_numbers):
                if i != j and j != k and k != i:
                    if adj[i, j] > 0 and adj[j, k] > 0 and adj[k, i] > 0:
                        cycles += 1
    
    # Kendall's Tau with true ordering
    correct = sum(1 for s in samples if s["chosen_value"] > s["rejected_value"])
    total = len(samples)
    kendall_tau = (2 * correct / total) - 1 if total > 0 else 0
    
    return {
        "total_samples": len(samples),
        "total_pairs": total_pairs,
        "conflicts": conflicts,
        "conflict_rate": conflicts / total_pairs if total_pairs > 0 else 0,
        "triangle_cycles": cycles,
        "kendall_tau": kendall_tau,
        "accuracy_vs_true_ordering": correct / total if total > 0 else 0,
        "poisoned_samples": sum(1 for s in samples if s.get("is_poisoned", False)),
        "poison_rate": sum(1 for s in samples if s.get("is_poisoned", False)) / len(samples),
    }


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating synthetic 1-{args.n_numbers} preference dataset...")
    print(f"Poisoning strategy: {args.poison_strategy}")
    print(f"Poisoning ratio: {args.poison_ratio}")
    
    # Generate baseline transitive samples
    print("\nGenerating baseline transitive samples...")
    samples = generate_baseline_transitive(
        n_numbers=args.n_numbers,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    
    # Apply poisoning
    if args.poison_strategy == "none":
        print("No poisoning applied (fully transitive)")
    elif args.poison_strategy == "local_cycles":
        print("Applying local neighborhood cycles...")
        samples = apply_local_neighborhood_cycles(samples, args.poison_ratio, seed=args.seed)
    elif args.poison_strategy == "global_inversion":
        print("Applying global inversion cycles...")
        samples = apply_global_inversion_cycles(samples, args.poison_ratio, seed=args.seed)
    elif args.poison_strategy == "dimensional":
        print("Applying dimensional trade-off cycles...")
        samples = apply_dimensional_tradeoff_cycles(
            samples, args.poison_ratio, n_numbers=args.n_numbers, seed=args.seed
        )
    else:
        raise ValueError(f"Unknown poison strategy: {args.poison_strategy}")
    
    # Compute analytical metrics
    print("\nComputing ground truth metrics...")
    metrics = compute_analytical_metrics(samples)
    
    print("\nDataset Statistics:")
    print(f"  Total samples: {metrics['total_samples']:,}")
    print(f"  Conflict rate: {metrics['conflict_rate']:.4f}")
    print(f"  Triangle cycles: {metrics['triangle_cycles']:,}")
    print(f"  Kendall's Tau: {metrics['kendall_tau']:.4f}")
    print(f"  Accuracy vs true ordering: {metrics['accuracy_vs_true_ordering']:.4f}")
    print(f"  Poisoned samples: {metrics['poisoned_samples']:,} ({metrics['poison_rate']:.2%})")
    
    # Split into train/val/test (80/10/10)
    random.seed(args.seed)
    random.shuffle(samples)
    
    n_train = int(len(samples) * 0.8)
    n_val = int(len(samples) * 0.1)
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    
    print(f"\nSplits:")
    print(f"  Train: {len(train_samples):,} samples")
    print(f"  Val: {len(val_samples):,} samples")
    print(f"  Test: {len(test_samples):,} samples")
    
    # Save as HuggingFace datasets
    for split_name, split_samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        dataset = Dataset.from_list(split_samples)
        save_path = os.path.join(args.output_dir, split_name)
        dataset.save_to_disk(save_path)
        print(f"  Saved {split_name} to {save_path}")
    
    # Save metadata
    metadata = {
        "n_numbers": args.n_numbers,
        "n_samples": args.n_samples,
        "poison_strategy": args.poison_strategy,
        "poison_ratio": args.poison_ratio,
        "seed": args.seed,
        "metrics": metrics,
        "splits": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        }
    }
    
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {metadata_path}")
    
    print("\n✓ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic 1-100 preference dataset with controlled intransitivity"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--n_numbers",
        type=int,
        default=100,
        help="Range of numbers [1, n_numbers]"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Number of preference pairs to generate"
    )
    parser.add_argument(
        "--poison_strategy",
        type=str,
        choices=["none", "local_cycles", "global_inversion", "dimensional"],
        default="none",
        help="Poisoning strategy to inject intransitivity"
    )
    parser.add_argument(
        "--poison_ratio",
        type=float,
        default=0.0,
        help="Fraction of samples to poison (0.0 to 1.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    main(args)


"""
Example usage:

# Fully transitive baseline
python scripts/dataset/generate_synthetic_1_100.py \
    --output_dir data/synthetic/1-100_transitive \
    --n_samples 10000 \
    --poison_strategy none

# Local cycles (20% poisoned)
python scripts/dataset/generate_synthetic_1_100.py \
    --output_dir data/synthetic/1-100_local_p0.2 \
    --n_samples 10000 \
    --poison_strategy local_cycles \
    --poison_ratio 0.2

# Global inversion (50% poisoned)
python scripts/dataset/generate_synthetic_1_100.py \
    --output_dir data/synthetic/1-100_global_p0.5 \
    --n_samples 10000 \
    --poison_strategy global_inversion \
    --poison_ratio 0.5

# Dimensional trade-offs (30% from non-magnitude dimensions)
python scripts/dataset/generate_synthetic_1_100.py \
    --output_dir data/synthetic/1-100_dimensional_p0.3 \
    --n_samples 10000 \
    --poison_strategy dimensional \
    --poison_ratio 0.3
"""
