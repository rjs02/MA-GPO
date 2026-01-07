#!/usr/bin/env python3
"""
Full evaluation for GPM (General Preference Model).
Adapted from evaluate_probe_full.py for the GPM architecture.

Includes:
- Dataset consistency analysis
- Accuracy by consistency bucket
- Advanced metrics: NLL, ECE, KL Divergence
- Position bias analysis
- Embedding statistics
- Symmetry/self-consistency check
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


def get_content(message_list):
    """Extract content string from message list."""
    if isinstance(message_list, list) and len(message_list) > 0:
        if isinstance(message_list[0], dict):
            return message_list[0].get('content', str(message_list[0]))
        return str(message_list[0])
    return str(message_list)


def load_gpm_model(model_path, base_model, value_head_dim=6, bf16=True):
    """Load the trained GPM model."""
    from general_preference.models import get_reward_model

    if os.path.exists(os.path.join(model_path, "config.json")):
        print(f"Loading GPM model from checkpoint: {model_path}")
        model = get_reward_model(
            model_path,
            is_general_preference=True,
            value_head_dim=value_head_dim,
            bf16=bf16,
        )
    else:
        print(f"Checkpoint not found at {model_path}, loading base model with random value head")
        model = get_reward_model(
            base_model,
            is_general_preference=True,
            value_head_dim=value_head_dim,
            init_value_head=True,
            bf16=bf16,
        )

    model = model.cuda().eval()
    return model


def create_skew_symmetric_matrix(dim):
    """Create the R matrix for preference computation."""
    matrix = np.zeros((dim, dim))
    for i in range(0, dim, 2):
        matrix[i, i + 1] = -1
        matrix[i + 1, i] = 1
    return matrix


def get_reward_embedding(model, input_ids, attention_mask):
    """Get reward embedding from GPM model."""
    with torch.no_grad():
        reward, _ = model.custom_forward(input_ids, attention_mask)
    return reward


def compute_preference_prob(chosen_emb, rejected_emb, R, tau=0.1):
    """Compute preference probability P(chosen > rejected)."""
    # chosen_emb @ R^T @ rejected_emb^T
    transformed = chosen_emb @ R.T
    result = np.sum(transformed * rejected_emb, axis=-1)  # dot product per sample
    prob = 1 / (1 + np.exp(-result / tau))
    return result, prob


def analyze_consistency_and_probabilities(dataset, prompt_key, chosen_key, rejected_key):
    """Analyze dataset for label conflicts and compute empirical probabilities."""
    print("\nAnalyzing dataset consistency...")

    pair_stats = {}  # prompt -> tuple(sorted(A,B)) -> { 'forward': count, 'backward': count }

    for i in tqdm(range(len(dataset)), desc="Analyzing Consistency"):
        row = dataset[i]
        prompt = str(get_content(row[prompt_key]))
        chosen = str(get_content(row[chosen_key]))
        rejected = str(get_content(row[rejected_key]))

        if chosen < rejected:
            pair = (chosen, rejected)
            direction = 'forward'
        else:
            pair = (rejected, chosen)
            direction = 'backward'

        if prompt not in pair_stats:
            pair_stats[prompt] = {}

        if pair not in pair_stats[prompt]:
            pair_stats[prompt][pair] = {'forward': 0, 'backward': 0}

        pair_stats[prompt][pair][direction] += 1

    # Calculate per-row empirical probabilities
    empirical_probs = []
    for i in range(len(dataset)):
        row = dataset[i]
        prompt = str(get_content(row[prompt_key]))
        chosen = str(get_content(row[chosen_key]))
        rejected = str(get_content(row[rejected_key]))

        if chosen < rejected:
            pair = (chosen, rejected)
            count_match = pair_stats[prompt][pair]['forward']
            count_conflict = pair_stats[prompt][pair]['backward']
        else:
            pair = (rejected, chosen)
            count_match = pair_stats[prompt][pair]['backward']
            count_conflict = pair_stats[prompt][pair]['forward']

        total = count_match + count_conflict
        prob = count_match / total if total > 0 else 0.0
        empirical_probs.append(prob)

    # Global stats
    total_pairs_unique = 0
    inconsistent_pairs_count = 0
    for prompt, pairs in pair_stats.items():
        for pair, counts in pairs.items():
            total_pairs_unique += 1
            if counts['forward'] > 0 and counts['backward'] > 0:
                inconsistent_pairs_count += 1

    print(f"Consistency Analysis Results:")
    print(f"  Unique Pairs: {total_pairs_unique}")
    print(f"  Inconsistent Pairs: {inconsistent_pairs_count}")
    if total_pairs_unique > 0:
        print(f"  Inconsistency Rate: {inconsistent_pairs_count/total_pairs_unique:.4f}")

    return empirical_probs


def get_accuracy_by_consistency_bucket(all_acc_vec, all_empirical_probs):
    """Groups accuracy by empirical probability buckets."""
    acc_arr = np.array(all_acc_vec)
    emp_probs_arr = np.array(all_empirical_probs)

    if len(acc_arr) == 0:
        return

    print("-" * 50)
    print("Accuracy by Empirical Consistency Level:")

    eps = 1e-4

    # Fully Consistent (prob = 0 or 1)
    mask_consistent = (emp_probs_arr < eps) | (emp_probs_arr > 1.0 - eps)
    if np.any(mask_consistent):
        acc_consistent = np.mean(acc_arr[mask_consistent])
        print(f"  Fully Consistent (Prob 0 or 1): Acc = {acc_consistent:.4f} (n={np.sum(mask_consistent)})")

    # Fully Inconsistent (prob ~ 0.5)
    mask_inconsistent = (emp_probs_arr > 0.5 - eps) & (emp_probs_arr < 0.5 + eps)
    if np.any(mask_inconsistent):
        acc_inconsistent = np.mean(acc_arr[mask_inconsistent])
        print(f"  Fully Inconsistent (Prob ~0.5): Acc = {acc_inconsistent:.4f} (n={np.sum(mask_inconsistent)})")

    # Intermediate
    mask_intermediate = (~mask_consistent) & (~mask_inconsistent)
    if np.any(mask_intermediate):
        acc_intermediate = np.mean(acc_arr[mask_intermediate])
        print(f"  Partially Inconsistent:         Acc = {acc_intermediate:.4f} (n={np.sum(mask_intermediate)})")


def calculate_advanced_metrics(all_correct, all_probs, all_empirical_probs):
    """Calculates ECE, NLL, and KL Divergence."""
    if len(all_correct) == 0:
        return

    correct_arr = np.array(all_correct)
    probs_arr = np.array(all_probs)
    emp_probs_arr = np.array(all_empirical_probs)

    # Clamp for numerical stability
    probs_arr = np.clip(probs_arr, 1e-6, 1 - 1e-6)
    emp_probs_arr = np.clip(emp_probs_arr, 1e-6, 1 - 1e-6)

    # 1. Negative Log Likelihood
    # NLL = -log(p) for correct predictions, -log(1-p) for incorrect
    nll = -np.mean(correct_arr * np.log(probs_arr) + (1 - correct_arr) * np.log(1 - probs_arr))

    # 2. Expected Calibration Error (ECE)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Confidence = max(p, 1-p), we use p directly since p > 0.5 means predict correct
        in_bin = (probs_arr > bin_lower) & (probs_arr <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correct_arr[in_bin])
            avg_confidence_in_bin = np.mean(probs_arr[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    # 3. KL Divergence (Empirical || Model)
    term1 = emp_probs_arr * np.log(emp_probs_arr / probs_arr)
    term2 = (1 - emp_probs_arr) * np.log((1 - emp_probs_arr) / (1 - probs_arr))
    mean_kl = np.mean(term1 + term2)

    print("-" * 50)
    print("Advanced Probabilistic Metrics:")
    print(f"  NLL (Negative Log Likelihood): {nll:.4f}")
    print(f"  ECE (Expected Calibration Error): {ece:.4f}")
    print(f"  Mean KL Divergence (Empirical || Model): {mean_kl:.4f}")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading GPM model...")
    model = load_gpm_model(
        args.model_path,
        args.base_model,
        value_head_dim=args.value_head_dim,
        bf16=args.bf16,
    )

    # Create R matrix for preference computation
    R = create_skew_symmetric_matrix(args.value_head_dim)
    R_tensor = torch.tensor(R, dtype=torch.float32, device=device)

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_from_disk(args.dataset)

    if args.max_samples and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
        print(f"  Using {args.max_samples} samples")

    # Analyze consistency
    empirical_probs = analyze_consistency_and_probabilities(
        dataset,
        args.prompt_key,
        args.chosen_key,
        args.rejected_key
    )

    # Evaluation
    print(f"\nStarting evaluation on {len(dataset)} samples...")

    all_probs = []
    all_correct = []
    all_empirical_probs = empirical_probs
    all_raw_scores = []

    chosen_norms = []
    rejected_norms = []
    chosen_embs_all = []
    rejected_embs_all = []

    # Position tracking (for bias analysis)
    # In standard format, chosen is always "preferred", but we track embedding order
    correct_first = 0  # when chosen embedding comes first alphabetically
    total_first = 0
    correct_second = 0
    total_second = 0

    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[i]

        prompt = sample[args.prompt_key]
        chosen = sample[args.chosen_key]
        rejected = sample[args.rejected_key]

        # Build conversations
        if isinstance(prompt, list):
            chosen_conv = prompt + chosen if isinstance(chosen, list) else prompt + [{"role": "assistant", "content": chosen}]
            rejected_conv = prompt + rejected if isinstance(rejected, list) else prompt + [{"role": "assistant", "content": rejected}]
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
            # Fallback if enable_thinking not supported
            chosen_text = tokenizer.apply_chat_template(
                chosen_conv, tokenize=False, add_generation_prompt=False
            )
            rejected_text = tokenizer.apply_chat_template(
                rejected_conv, tokenize=False, add_generation_prompt=False
            )

        # Tokenize
        chosen_tokens = tokenizer(
            chosen_text,
            max_length=args.max_len,
            truncation=True,
            return_tensors="pt",
            padding=False,
        )
        rejected_tokens = tokenizer(
            rejected_text,
            max_length=args.max_len,
            truncation=True,
            return_tensors="pt",
            padding=False,
        )

        # Ensure EOS at end
        chosen_tokens["input_ids"][0, -1] = tokenizer.eos_token_id
        chosen_tokens["attention_mask"][0, -1] = 1
        rejected_tokens["input_ids"][0, -1] = tokenizer.eos_token_id
        rejected_tokens["attention_mask"][0, -1] = 1

        # Get embeddings
        chosen_emb = get_reward_embedding(
            model,
            chosen_tokens["input_ids"].cuda(),
            chosen_tokens["attention_mask"].cuda()
        ).cpu().float().numpy()[0]

        rejected_emb = get_reward_embedding(
            model,
            rejected_tokens["input_ids"].cuda(),
            rejected_tokens["attention_mask"].cuda()
        ).cpu().float().numpy()[0]

        # Compute preference
        raw_score, prob = compute_preference_prob(chosen_emb, rejected_emb, R, args.tau)

        all_probs.append(prob)
        all_raw_scores.append(raw_score)
        all_correct.append(1.0 if prob > 0.5 else 0.0)

        # Embedding stats
        chosen_norms.append(np.linalg.norm(chosen_emb))
        rejected_norms.append(np.linalg.norm(rejected_emb))
        chosen_embs_all.append(chosen_emb)
        rejected_embs_all.append(rejected_emb)

        # Position bias tracking (using text order as proxy)
        chosen_str = str(get_content(chosen))
        rejected_str = str(get_content(rejected))
        if chosen_str < rejected_str:
            total_first += 1
            if prob > 0.5:
                correct_first += 1
        else:
            total_second += 1
            if prob > 0.5:
                correct_second += 1

    # Convert to arrays
    all_probs = np.array(all_probs)
    all_correct = np.array(all_correct)
    all_raw_scores = np.array(all_raw_scores)
    chosen_embs_all = np.array(chosen_embs_all)
    rejected_embs_all = np.array(rejected_embs_all)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nOverall Accuracy: {np.mean(all_correct):.4f} ({int(np.sum(all_correct))}/{len(all_correct)})")
    print(f"Mean Probability (P(chosen > rejected)): {np.mean(all_probs):.4f}")
    print(f"Probability Std: {np.std(all_probs):.4f}")

    print(f"\nRaw Scores (before sigmoid):")
    print(f"  Mean: {np.mean(all_raw_scores):.4f}")
    print(f"  Std: {np.std(all_raw_scores):.4f}")
    print(f"  Min: {np.min(all_raw_scores):.4f}, Max: {np.max(all_raw_scores):.4f}")

    # Accuracy by consistency
    get_accuracy_by_consistency_bucket(all_correct.tolist(), all_empirical_probs)

    # Advanced metrics
    calculate_advanced_metrics(all_correct.tolist(), all_probs.tolist(), all_empirical_probs)

    # Position bias
    print("-" * 50)
    print("Position Bias Analysis:")
    if total_first > 0:
        print(f"  Chosen First (alphabetically): Acc = {correct_first/total_first:.4f} (n={total_first})")
    if total_second > 0:
        print(f"  Chosen Second (alphabetically): Acc = {correct_second/total_second:.4f} (n={total_second})")
    if total_first > 0 and total_second > 0:
        bias_diff = abs(correct_first / total_first - correct_second / total_second)
        print(f"  Bias Difference: {bias_diff:.4f}")

    # Embedding statistics
    print("-" * 50)
    print("Embedding Statistics:")
    print(f"  Chosen Norms:   mean={np.mean(chosen_norms):.4f}, std={np.std(chosen_norms):.4f}")
    print(f"  Rejected Norms: mean={np.mean(rejected_norms):.4f}, std={np.std(rejected_norms):.4f}")

    print(f"\n  Per-dimension Statistics:")
    for d in range(args.value_head_dim):
        c_mean, c_std = chosen_embs_all[:, d].mean(), chosen_embs_all[:, d].std()
        r_mean, r_std = rejected_embs_all[:, d].mean(), rejected_embs_all[:, d].std()
        print(f"    Dim {d}: chosen={c_mean:.4f}±{c_std:.4f}, rejected={r_mean:.4f}±{r_std:.4f}")

    # Collapse check
    chosen_var = chosen_embs_all.var(axis=0).mean()
    rejected_var = rejected_embs_all.var(axis=0).mean()
    chosen_mean = chosen_embs_all.mean(axis=0)
    rejected_mean = rejected_embs_all.mean(axis=0)

    print(f"\n  Embedding Collapse Check:")
    print(f"    Avg Variance (chosen):  {chosen_var:.6f}")
    print(f"    Avg Variance (rejected): {rejected_var:.6f}")
    print(f"    Distance between means: {np.linalg.norm(chosen_mean - rejected_mean):.6f}")

    if chosen_var < 1e-4 or rejected_var < 1e-4:
        print("    WARNING: Very low embedding variance - possible collapse!")

    # Probability consistency
    print("-" * 50)
    print("Probability Consistency Analysis:")
    emp_probs_arr = np.array(all_empirical_probs)
    unique_emp_probs = np.unique(emp_probs_arr)
    print(f"  Found {len(unique_emp_probs)} unique empirical probability levels.")

    for emp_p in sorted(unique_emp_probs):
        mask = (emp_probs_arr == emp_p)
        selected_model_probs = all_probs[mask]
        if len(selected_model_probs) > 0:
            avg_model_p = np.mean(selected_model_probs)
            print(f"    Empirical P={emp_p:.4f}: Model Mean P={avg_model_p:.4f} (n={np.sum(mask)})")

    if len(all_probs) > 1 and np.std(emp_probs_arr) > 0:
        correlation = np.corrcoef(all_probs, emp_probs_arr)[0, 1]
        print(f"  Correlation (Model Prob vs Empirical Prob): {correlation:.4f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Full evaluation for GPM model")

    # Model args
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained GPM checkpoint")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Base model name/path")
    parser.add_argument("--value_head_dim", type=int, default=6,
                        help="Dimension of value head (must match training)")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--tau", type=float, default=0.1,
                        help="Temperature for preference scaling")

    # Dataset args
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to evaluation dataset")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate")
    parser.add_argument("--max_len", type=int, default=2048,
                        help="Max sequence length")

    # Dataset keys
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
