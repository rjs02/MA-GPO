#!/usr/bin/env python3
"""
Enhanced evaluation for preference models with transitivity-aware metrics.

Extends evaluate_gpm_full.py with:
- Accuracy stratified by empirical consistency
- Brier score, KL divergence, ECE
- Soft Non-Transitivity Deviation (SNTD)
- Reward separability (BT-specific)
- Embedding geometry analysis (GPM-specific)
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
from datasets import load_from_disk
import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def get_content(message_list):
    """Extract content string from message list."""
    if isinstance(message_list, list) and len(message_list) > 0:
        if isinstance(message_list[0], dict):
            return message_list[0].get('content', str(message_list[0]))
        return str(message_list[0])
    return str(message_list)


def load_model(
    model_path: str,
    base_model: str,
    is_gpm: bool,
    value_head_dim: int = 8,
    bf16: bool = True,
):
    """Load reward model (either BT or GPM)."""
    from general_preference.models import get_reward_model
    
    if os.path.exists(os.path.join(model_path, "config.json")):
        print(f"Loading model from checkpoint: {model_path}")
        model = get_reward_model(
            model_path,
            is_general_preference=is_gpm,
            value_head_dim=value_head_dim if is_gpm else 1,
            bf16=bf16,
        )
    else:
        print(f"Checkpoint not found at {model_path}, loading base model")
        model = get_reward_model(
            base_model,
            is_general_preference=is_gpm,
            value_head_dim=value_head_dim if is_gpm else 1,
            init_value_head=True,
            bf16=bf16,
        )
    
    model = model.cuda().eval()
    return model


def create_skew_symmetric_matrix(dim: int) -> np.ndarray:
    """Create R matrix for GPM preference computation."""
    matrix = np.zeros((dim, dim))
    for i in range(0, dim, 2):
        if i + 1 < dim:
            matrix[i, i + 1] = -1
            matrix[i + 1, i] = 1
    return matrix


def get_reward_bt(model, input_ids, attention_mask):
    """Get scalar reward from Bradley-Terry model."""
    with torch.no_grad():
        reward, _ = model.custom_forward(input_ids, attention_mask)
    return reward


def get_embedding_gpm(model, input_ids, attention_mask):
    """Get embedding from GPM model."""
    with torch.no_grad():
        embedding, _ = model.custom_forward(input_ids, attention_mask)
    return embedding


def compute_preference_prob_bt(chosen_reward, rejected_reward):
    """Compute P(chosen > rejected) for BT model."""
    logit = chosen_reward - rejected_reward
    prob = torch.sigmoid(logit)
    return prob.cpu().float().numpy()


def compute_preference_prob_gpm(chosen_emb, rejected_emb, R, tau=0.1):
    """Compute P(chosen > rejected) for GPM model."""
    # chosen_emb @ R^T @ rejected_emb^T
    chosen_np = chosen_emb.cpu().float().numpy()
    rejected_np = rejected_emb.cpu().float().numpy()
    
    transformed = chosen_np @ R.T
    result = np.sum(transformed * rejected_np, axis=-1)
    prob = 1 / (1 + np.exp(-result / tau))
    return result, prob


def analyze_empirical_consistency(
    dataset,
    prompt_key: str,
    chosen_key: str,
    rejected_key: str,
) -> Tuple[List[float], Dict]:
    """
    Analyze dataset for label conflicts and compute empirical win rates.
    
    Returns:
        empirical_probs: List of empirical P(chosen > rejected) for each sample
        stats: Dictionary with consistency statistics
    """
    print("\nAnalyzing dataset consistency...")
    
    # Track: (prompt, response_a, response_b) -> {'forward': count, 'backward': count}
    pair_stats = {}
    
    for i in tqdm(range(len(dataset)), desc="Computing empirical probs"):
        row = dataset[i]
        prompt = str(get_content(row[prompt_key]))
        chosen = str(get_content(row[chosen_key]))
        rejected = str(get_content(row[rejected_key]))
        
        # Normalize pair ordering
        if chosen < rejected:
            pair = (chosen, rejected)
            direction = 'forward'
        else:
            pair = (rejected, chosen)
            direction = 'backward'
        
        key = (prompt, pair[0], pair[1])
        if key not in pair_stats:
            pair_stats[key] = {'forward': 0, 'backward': 0}
        pair_stats[key][direction] += 1
    
    # Calculate empirical probabilities per sample
    empirical_probs = []
    for i in range(len(dataset)):
        row = dataset[i]
        prompt = str(get_content(row[prompt_key]))
        chosen = str(get_content(row[chosen_key]))
        rejected = str(get_content(row[rejected_key]))
        
        if chosen < rejected:
            pair = (chosen, rejected)
            count_match = pair_stats[(prompt, pair[0], pair[1])]['forward']
            count_conflict = pair_stats[(prompt, pair[0], pair[1])]['backward']
        else:
            pair = (rejected, chosen)
            count_match = pair_stats[(prompt, pair[0], pair[1])]['backward']
            count_conflict = pair_stats[(prompt, pair[0], pair[1])]['forward']
        
        total = count_match + count_conflict
        prob = count_match / total if total > 0 else 0.5
        empirical_probs.append(prob)
    
    # Compute stats
    unique_pairs = len(pair_stats)
    inconsistent_pairs = sum(1 for stats in pair_stats.values() 
                            if stats['forward'] > 0 and stats['backward'] > 0)
    
    stats = {
        "unique_pairs": unique_pairs,
        "inconsistent_pairs": inconsistent_pairs,
        "inconsistency_rate": inconsistent_pairs / unique_pairs if unique_pairs > 0 else 0,
    }
    
    print(f"  Unique pairs: {unique_pairs:,}")
    print(f"  Inconsistent pairs: {inconsistent_pairs:,}")
    print(f"  Inconsistency rate: {stats['inconsistency_rate']:.4f}")
    
    return empirical_probs, stats


def evaluate_model(
    model,
    dataset,
    tokenizer,
    is_gpm: bool,
    tau: float = 0.1,
    value_head_dim: int = 8,
    max_len: int = 2048,
    prompt_key: str = "prompt",
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
) -> Dict:
    """
    Evaluate model with transitivity-aware metrics.
    
    Returns:
        Dictionary with all evaluation metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Analyze empirical consistency
    empirical_probs, consistency_stats = analyze_empirical_consistency(
        dataset, prompt_key, chosen_key, rejected_key
    )
    
    # Prepare for evaluation
    if is_gpm:
        R = create_skew_symmetric_matrix(value_head_dim)
    
    all_model_probs = []
    all_correct = []
    all_raw_scores = []  # BT rewards or GPM preference scores
    
    chosen_values = []  # BT rewards or GPM embeddings
    rejected_values = []
    
    print(f"\nEvaluating model on {len(dataset)} samples...")
    
    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[i]
        
        prompt = sample[prompt_key]
        chosen = sample[chosen_key]
        rejected = sample[rejected_key]
        
        # Build conversations
        if isinstance(prompt, list):
            chosen_conv = prompt + (chosen if isinstance(chosen, list) else [{"role": "assistant", "content": chosen}])
            rejected_conv = prompt + (rejected if isinstance(rejected, list) else [{"role": "assistant", "content": rejected}])
        else:
            chosen_conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}]
            rejected_conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}]
        
        # Apply chat template
        try:
            chosen_text = tokenizer.apply_chat_template(
                chosen_conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
            rejected_text = tokenizer.apply_chat_template(
                rejected_conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
        except TypeError:
            chosen_text = tokenizer.apply_chat_template(
                chosen_conv, tokenize=False, add_generation_prompt=False
            )
            rejected_text = tokenizer.apply_chat_template(
                rejected_conv, tokenize=False, add_generation_prompt=False
            )
        
        # Tokenize
        chosen_tokens = tokenizer(
            chosen_text, max_length=max_len, truncation=True, return_tensors="pt", padding=False
        )
        rejected_tokens = tokenizer(
            rejected_text, max_length=max_len, truncation=True, return_tensors="pt", padding=False
        )
        
        # Ensure EOS at end
        chosen_tokens["input_ids"][0, -1] = tokenizer.eos_token_id
        chosen_tokens["attention_mask"][0, -1] = 1
        rejected_tokens["input_ids"][0, -1] = tokenizer.eos_token_id
        rejected_tokens["attention_mask"][0, -1] = 1
        
        # Get model outputs and compute probability
        if is_gpm:
            chosen_emb = get_embedding_gpm(
                model, chosen_tokens["input_ids"].cuda(), chosen_tokens["attention_mask"].cuda()
            )
            rejected_emb = get_embedding_gpm(
                model, rejected_tokens["input_ids"].cuda(), rejected_tokens["attention_mask"].cuda()
            )
            
            chosen_np = chosen_emb.cpu().float().numpy()[0]
            rejected_np = rejected_emb.cpu().float().numpy()[0]
            
            raw_score, prob = compute_preference_prob_gpm(chosen_emb, rejected_emb, R, tau)
            prob = prob[0]
            raw_score = raw_score[0]
            
            chosen_values.append(chosen_np)
            rejected_values.append(rejected_np)
        else:
            chosen_reward = get_reward_bt(
                model, chosen_tokens["input_ids"].cuda(), chosen_tokens["attention_mask"].cuda()
            )
            rejected_reward = get_reward_bt(
                model, rejected_tokens["input_ids"].cuda(), rejected_tokens["attention_mask"].cuda()
            )
            
            prob = compute_preference_prob_bt(chosen_reward, rejected_reward)[0]
            raw_score = (chosen_reward - rejected_reward).cpu().float().numpy()[0]
            
            chosen_values.append(chosen_reward.cpu().float().numpy()[0])
            rejected_values.append(rejected_reward.cpu().float().numpy()[0])
        
        all_model_probs.append(prob)
        all_raw_scores.append(raw_score)
        all_correct.append(1.0 if prob > 0.5 else 0.0)
    
    # Convert to arrays
    model_probs = np.array(all_model_probs)
    correct = np.array(all_correct)
    raw_scores = np.array(all_raw_scores)
    empirical_probs = np.array(empirical_probs)
    
    if is_gpm:
        chosen_values = np.array(chosen_values)
        rejected_values = np.array(rejected_values)
    else:
        chosen_values = np.array(chosen_values)
        rejected_values = np.array(rejected_values)
    
    # Compute metrics
    results = {
        "consistency_stats": consistency_stats,
        "overall_accuracy": float(np.mean(correct)),
        "mean_model_prob": float(np.mean(model_probs)),
        "std_model_prob": float(np.std(model_probs)),
    }
    
    # Accuracy by consistency bucket
    results["accuracy_by_consistency"] = compute_accuracy_by_consistency(
        correct, empirical_probs
    )
    
    # Probabilistic metrics
    results["probabilistic_metrics"] = compute_probabilistic_metrics(
        correct, model_probs, empirical_probs
    )
    
    # Model-specific metrics
    if is_gpm:
        results["embedding_stats"] = compute_embedding_stats(
            chosen_values, rejected_values, value_head_dim
        )
    else:
        results["reward_separability"] = compute_reward_separability(
            chosen_values, rejected_values
        )
    
    # Raw score statistics
    results["raw_score_stats"] = {
        "mean": float(np.mean(raw_scores)),
        "std": float(np.std(raw_scores)),
        "min": float(np.min(raw_scores)),
        "max": float(np.max(raw_scores)),
    }
    
    return results


def compute_accuracy_by_consistency(correct: np.ndarray, empirical_probs: np.ndarray) -> Dict:
    """Compute accuracy stratified by empirical consistency."""
    buckets = {
        "fully_consistent_0": (0.0, 0.01),
        "fully_consistent_1": (0.99, 1.01),
        "highly_consistent": (0.8, 0.99),
        "moderately_consistent": (0.6, 0.8),
        "low_consistency": (0.51, 0.6),
        "fully_inconsistent": (0.49, 0.51),
        "reversed": (0.0, 0.49),
    }
    
    results = {}
    for name, (low, high) in buckets.items():
        mask = (empirical_probs >= low) & (empirical_probs < high)
        if np.any(mask):
            acc = float(np.mean(correct[mask]))
            count = int(np.sum(mask))
            results[name] = {"accuracy": acc, "count": count}
        else:
            results[name] = {"accuracy": 0.0, "count": 0}
    
    return results


def compute_probabilistic_metrics(
    correct: np.ndarray,
    model_probs: np.ndarray,
    empirical_probs: np.ndarray,
) -> Dict:
    """Compute Brier score, KL divergence, and ECE."""
    # Clip for numerical stability
    model_probs = np.clip(model_probs, 1e-6, 1 - 1e-6)
    empirical_probs = np.clip(empirical_probs, 1e-6, 1 - 1e-6)
    
    # Brier Score
    brier = float(np.mean((model_probs - empirical_probs) ** 2))
    
    # KL Divergence (forward: empirical || model)
    kl_forward = float(np.mean(
        empirical_probs * np.log(empirical_probs / model_probs) +
        (1 - empirical_probs) * np.log((1 - empirical_probs) / (1 - model_probs))
    ))
    
    # KL Divergence (reverse: model || empirical)
    kl_reverse = float(np.mean(
        model_probs * np.log(model_probs / empirical_probs) +
        (1 - model_probs) * np.log((1 - model_probs) / (1 - empirical_probs))
    ))
    
    # Expected Calibration Error (ECE)
    n_bins = 10
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (model_probs > bin_lower) & (model_probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correct[in_bin])
            avg_confidence_in_bin = np.mean(model_probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    # Negative Log Likelihood
    nll = float(-np.mean(
        correct * np.log(model_probs) + (1 - correct) * np.log(1 - model_probs)
    ))
    
    return {
        "brier_score": brier,
        "kl_forward": kl_forward,
        "kl_reverse": kl_reverse,
        "ece": float(ece),
        "nll": nll,
    }


def compute_embedding_stats(
    chosen_embs: np.ndarray,
    rejected_embs: np.ndarray,
    dim: int,
) -> Dict:
    """Compute embedding statistics for GPM."""
    chosen_norms = np.linalg.norm(chosen_embs, axis=1)
    rejected_norms = np.linalg.norm(rejected_embs, axis=1)
    
    # Per-dimension stats
    per_dim_stats = []
    for d in range(dim):
        per_dim_stats.append({
            "chosen_mean": float(chosen_embs[:, d].mean()),
            "chosen_std": float(chosen_embs[:, d].std()),
            "rejected_mean": float(rejected_embs[:, d].mean()),
            "rejected_std": float(rejected_embs[:, d].std()),
        })
    
    # Collapse detection
    chosen_var = float(chosen_embs.var(axis=0).mean())
    rejected_var = float(rejected_embs.var(axis=0).mean())
    mean_distance = float(np.linalg.norm(chosen_embs.mean(axis=0) - rejected_embs.mean(axis=0)))
    
    return {
        "chosen_norm_mean": float(chosen_norms.mean()),
        "chosen_norm_std": float(chosen_norms.std()),
        "rejected_norm_mean": float(rejected_norms.mean()),
        "rejected_norm_std": float(rejected_norms.std()),
        "per_dimension": per_dim_stats,
        "avg_variance_chosen": chosen_var,
        "avg_variance_rejected": rejected_var,
        "mean_distance": mean_distance,
        "collapse_warning": chosen_var < 1e-4 or rejected_var < 1e-4,
    }


def compute_reward_separability(
    chosen_rewards: np.ndarray,
    rejected_rewards: np.ndarray,
) -> Dict:
    """Compute reward separability for Bradley-Terry models."""
    all_rewards = np.concatenate([chosen_rewards, rejected_rewards])
    
    return {
        "chosen_mean": float(chosen_rewards.mean()),
        "chosen_std": float(chosen_rewards.std()),
        "rejected_mean": float(rejected_rewards.mean()),
        "rejected_std": float(rejected_rewards.std()),
        "all_rewards_std": float(all_rewards.std()),
        "mean_difference": float(chosen_rewards.mean() - rejected_rewards.mean()),
        "collapse_warning": all_rewards.std() < 0.1,
    }


def print_results(results: Dict, model_type: str):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*70)
    print(f"EVALUATION RESULTS - {model_type}")
    print("="*70)
    
    print(f"\nOverall Performance:")
    print(f"  Accuracy: {results['overall_accuracy']:.4f}")
    print(f"  Mean Model Prob: {results['mean_model_prob']:.4f} ± {results['std_model_prob']:.4f}")
    
    print(f"\nDataset Consistency:")
    print(f"  Unique pairs: {results['consistency_stats']['unique_pairs']:,}")
    print(f"  Inconsistent pairs: {results['consistency_stats']['inconsistent_pairs']:,}")
    print(f"  Inconsistency rate: {results['consistency_stats']['inconsistency_rate']:.4f}")
    
    print(f"\nAccuracy by Consistency Level:")
    for name, stats in results['accuracy_by_consistency'].items():
        if stats['count'] > 0:
            print(f"  {name:25s}: {stats['accuracy']:.4f} (n={stats['count']:,})")
    
    print(f"\nProbabilistic Metrics:")
    pm = results['probabilistic_metrics']
    print(f"  Brier Score: {pm['brier_score']:.4f}")
    print(f"  KL Divergence (forward): {pm['kl_forward']:.4f}")
    print(f"  KL Divergence (reverse): {pm['kl_reverse']:.4f}")
    print(f"  ECE: {pm['ece']:.4f}")
    print(f"  NLL: {pm['nll']:.4f}")
    
    if 'embedding_stats' in results:
        print(f"\nEmbedding Statistics (GPM):")
        es = results['embedding_stats']
        print(f"  Chosen norm: {es['chosen_norm_mean']:.4f} ± {es['chosen_norm_std']:.4f}")
        print(f"  Rejected norm: {es['rejected_norm_mean']:.4f} ± {es['rejected_norm_std']:.4f}")
        print(f"  Mean distance: {es['mean_distance']:.4f}")
        print(f"  Avg variance (chosen): {es['avg_variance_chosen']:.6f}")
        print(f"  Avg variance (rejected): {es['avg_variance_rejected']:.6f}")
        if es['collapse_warning']:
            print("  ⚠️  WARNING: Very low embedding variance - possible collapse!")
    
    if 'reward_separability' in results:
        print(f"\nReward Separability (Bradley-Terry):")
        rs = results['reward_separability']
        print(f"  Chosen rewards: {rs['chosen_mean']:.4f} ± {rs['chosen_std']:.4f}")
        print(f"  Rejected rewards: {rs['rejected_mean']:.4f} ± {rs['rejected_std']:.4f}")
        print(f"  Mean difference: {rs['mean_difference']:.4f}")
        print(f"  Overall std: {rs['all_rewards_std']:.4f}")
        if rs['collapse_warning']:
            print("  ⚠️  WARNING: Very low reward std - model may have collapsed!")
    
    print(f"\nRaw Score Statistics:")
    rss = results['raw_score_stats']
    print(f"  Mean: {rss['mean']:.4f}")
    print(f"  Std: {rss['std']:.4f}")
    print(f"  Range: [{rss['min']:.4f}, {rss['max']:.4f}]")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Enhanced transitivity-aware evaluation")
    
    # Model args
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--is_gpm", action="store_true", default=False,
                       help="Whether model is GPM (default: Bradley-Terry)")
    parser.add_argument("--value_head_dim", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true", default=True)
    
    # Dataset args
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    
    # Output args
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model_type = "GPM" if args.is_gpm else "Bradley-Terry"
    model = load_model(
        args.model_path,
        args.base_model,
        args.is_gpm,
        args.value_head_dim,
        args.bf16,
    )
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    if args.dataset.endswith('.jsonl'):
        from datasets import load_dataset
        dataset = load_dataset('json', data_files=args.dataset, split='train')
    else:
        dataset = load_from_disk(args.dataset)
    
    if args.max_samples and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
    print(f"  Using {len(dataset):,} samples")
    
    # Evaluate
    results = evaluate_model(
        model, dataset, tokenizer,
        args.is_gpm, args.tau, args.value_head_dim, args.max_len,
        args.prompt_key, args.chosen_key, args.rejected_key,
    )
    
    # Print results
    print_results(results, model_type)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
