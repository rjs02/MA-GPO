#!/usr/bin/env python3
"""
Diagnose model behavior on train vs eval data.
Run AFTER some training to see what embeddings look like.
"""

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer
import numpy as np
import argparse
import os

def load_model(model_path, base_model):
    """Load the trained GPM model."""
    from general_preference.models import get_reward_model

    # Try to load from checkpoint
    if os.path.exists(os.path.join(model_path, "config.json")):
        model = get_reward_model(
            model_path,
            is_general_preference=True,
            value_head_dim=6,  # Match your training config
            bf16=True,
        )
    else:
        print(f"Model not found at {model_path}, loading base model")
        model = get_reward_model(
            base_model,
            is_general_preference=True,
            value_head_dim=6,
            init_value_head=True,
            bf16=True,
        )

    model = model.cuda().eval()
    return model


def get_reward(model, tokenizer, text, max_len=2048):
    """Get reward embedding for a single text."""
    tokens = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
        padding=False,
    )

    # Ensure EOS at end
    tokens["input_ids"][0, -1] = tokenizer.eos_token_id
    tokens["attention_mask"][0, -1] = 1

    input_ids = tokens["input_ids"].cuda()
    attention_mask = tokens["attention_mask"].cuda()

    with torch.no_grad():
        reward, _ = model.custom_forward(input_ids, attention_mask)

    return reward.cpu().float().numpy()


def create_skew_symmetric_matrix(dim):
    """Create the R matrix for preference computation."""
    matrix = np.zeros((dim, dim))
    for i in range(0, dim, 2):
        matrix[i, i+1] = -1
        matrix[i+1, i] = 1
    return matrix


def compute_preference(chosen_emb, rejected_emb, R, tau=0.1):
    """Compute preference score and probability."""
    transformed = chosen_emb @ R.T
    result = transformed @ rejected_emb.T
    prob = 1 / (1 + np.exp(-result / tau))
    return result, prob


def analyze_samples(model, tokenizer, dataset, name, num_samples=50, tau=0.1, value_head_dim=6):
    """Analyze model predictions on a dataset."""
    print(f"\n{'='*60}")
    print(f"Analyzing {name} ({num_samples} samples)")
    print("="*60)

    R = create_skew_symmetric_matrix(value_head_dim)

    results = []
    probs = []
    chosen_norms = []
    rejected_norms = []
    chosen_embs = []
    rejected_embs = []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        prompt = sample['prompt']
        chosen = sample['chosen']
        rejected = sample['rejected']

        # Create full conversations
        chosen_conv = prompt + chosen
        rejected_conv = prompt + rejected

        # Apply chat template
        chosen_text = tokenizer.apply_chat_template(
            chosen_conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        rejected_text = tokenizer.apply_chat_template(
            rejected_conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )

        # Get embeddings
        chosen_emb = get_reward(model, tokenizer, chosen_text)
        rejected_emb = get_reward(model, tokenizer, rejected_text)

        # Compute preference
        result, prob = compute_preference(chosen_emb[0], rejected_emb[0], R, tau)

        results.append(result)
        probs.append(prob)
        chosen_norms.append(np.linalg.norm(chosen_emb))
        rejected_norms.append(np.linalg.norm(rejected_emb))
        chosen_embs.append(chosen_emb[0])
        rejected_embs.append(rejected_emb[0])

    results = np.array(results)
    probs = np.array(probs)
    chosen_norms = np.array(chosen_norms)
    rejected_norms = np.array(rejected_norms)
    chosen_embs = np.array(chosen_embs)
    rejected_embs = np.array(rejected_embs)

    print(f"\nPreference Results (before sigmoid):")
    print(f"  mean={results.mean():.4f}, std={results.std():.4f}")
    print(f"  min={results.min():.4f}, max={results.max():.4f}")
    print(f"  positive (chosen > rejected): {(results > 0).sum()}/{len(results)} ({100*(results > 0).mean():.1f}%)")

    print(f"\nPreference Probabilities (after sigmoid with tau={tau}):")
    print(f"  mean={probs.mean():.4f}, std={probs.std():.4f}")
    print(f"  min={probs.min():.4f}, max={probs.max():.4f}")
    print(f"  > 0.5 (correct): {(probs > 0.5).sum()}/{len(probs)} ({100*(probs > 0.5).mean():.1f}%)")

    print(f"\nEmbedding Norms:")
    print(f"  Chosen:   mean={chosen_norms.mean():.4f}, std={chosen_norms.std():.4f}")
    print(f"  Rejected: mean={rejected_norms.mean():.4f}, std={rejected_norms.std():.4f}")

    print(f"\nEmbedding Statistics (per dimension):")
    for d in range(value_head_dim):
        chosen_d = chosen_embs[:, d]
        rejected_d = rejected_embs[:, d]
        print(f"  Dim {d}: chosen mean={chosen_d.mean():.4f}±{chosen_d.std():.4f}, "
              f"rejected mean={rejected_d.mean():.4f}±{rejected_d.std():.4f}")

    # Check for embedding collapse
    chosen_mean = chosen_embs.mean(axis=0)
    rejected_mean = rejected_embs.mean(axis=0)
    chosen_var = chosen_embs.var(axis=0).mean()
    rejected_var = rejected_embs.var(axis=0).mean()

    print(f"\nEmbedding Collapse Check:")
    print(f"  Mean chosen embedding:   {chosen_mean}")
    print(f"  Mean rejected embedding: {rejected_mean}")
    print(f"  Avg variance (chosen):   {chosen_var:.6f}")
    print(f"  Avg variance (rejected): {rejected_var:.6f}")
    print(f"  Distance between means:  {np.linalg.norm(chosen_mean - rejected_mean):.6f}")

    if chosen_var < 1e-4 or rejected_var < 1e-4:
        print("  ⚠️  WARNING: Very low embedding variance - possible collapse!")

    return {
        'results': results,
        'probs': probs,
        'chosen_embs': chosen_embs,
        'rejected_embs': rejected_embs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to trained model checkpoint")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train_path", type=str,
                       default="./data/ultrafeedback_cleaned_splits/pref_train")
    parser.add_argument("--eval_path", type=str,
                       default="./data/ultrafeedback_cleaned_splits/pref_val")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--value_head_dim", type=int, default=6)
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print("Loading model...")
    if args.model_path:
        model = load_model(args.model_path, args.base_model)
    else:
        # Load untrained model for baseline comparison
        from general_preference.models import get_reward_model
        model = get_reward_model(
            args.base_model,
            is_general_preference=True,
            value_head_dim=args.value_head_dim,
            init_value_head=True,
            bf16=True,
        )
        model = model.cuda().eval()
        print("Using randomly initialized model (no checkpoint provided)")

    print("Loading datasets...")
    train_data = load_from_disk(args.train_path)

    try:
        eval_data = load_from_disk(args.eval_path)
    except:
        print(f"No eval data at {args.eval_path}, using last 5% of train")
        split_idx = int(len(train_data) * 0.95)
        eval_data = train_data.select(range(split_idx, len(train_data)))
        train_data = train_data.select(range(split_idx))

    # Analyze both datasets
    train_stats = analyze_samples(
        model, tokenizer, train_data, "TRAIN",
        num_samples=args.num_samples, tau=args.tau, value_head_dim=args.value_head_dim
    )

    eval_stats = analyze_samples(
        model, tokenizer, eval_data, "EVAL",
        num_samples=args.num_samples, tau=args.tau, value_head_dim=args.value_head_dim
    )

    # Compare
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print("="*60)

    train_acc = (train_stats['probs'] > 0.5).mean()
    eval_acc = (eval_stats['probs'] > 0.5).mean()

    print(f"\nAccuracy (prob > 0.5):")
    print(f"  Train: {100*train_acc:.1f}%")
    print(f"  Eval:  {100*eval_acc:.1f}%")
    print(f"  Gap:   {100*(train_acc - eval_acc):.1f}%")

    print(f"\nMean Probability:")
    print(f"  Train: {train_stats['probs'].mean():.4f}")
    print(f"  Eval:  {eval_stats['probs'].mean():.4f}")

    if eval_acc < 0.3 and train_acc > 0.6:
        print("\n⚠️  DIAGNOSIS: Severe generalization failure!")
        print("   Model memorizes training data but predicts OPPOSITE on eval.")
        print("\n   Possible causes:")
        print("   1. Embedding collapse - check variance above")
        print("   2. Spurious correlations in training data")
        print("   3. Conflicting labels causing random memorization")
        print("   4. Distribution shift between train/eval")

    elif train_acc < 0.6:
        print("\n⚠️  DIAGNOSIS: Training not working!")
        print("   Model isn't learning to predict preferences even on train data.")
        print("\n   Possible causes:")
        print("   1. Data format issue - check tokenization")
        print("   2. Learning rate too high/low")
        print("   3. Too many conflicting labels")


if __name__ == "__main__":
    main()

    """
python scripts/rmpm/diagnose_model.py \
    --model_path ${SCRATCH}/MA/experiments/qwen3-0.6b-gpm-dim6-ufb/global_step_3500/ \
    --train_path "${LASDIR}/data/ufb/pref_train" \
    --num_samples 100 \
    --value_head_dim 6


        --eval_path "${LASDIR}/data/ufb/pref_val" \
    """
