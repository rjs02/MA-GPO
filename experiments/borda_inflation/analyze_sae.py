#!/usr/bin/env python3
"""Track B: SAE-based interpretability for Borda inflation.

Uses pretrained Llama Scope SAEs (fnlp/Llama-Scope) to extract monosemantic
features from the RM backbone and correlate them with Borda inflation scores.

Workflow:
1. Load trained RM model and Llama Scope SAE for a chosen layer
2. For each response: extract hidden states at the SAE layer, encode with SAE
3. Correlate each SAE feature with Borda inflation score
4. Identify and interpret top features

Requires: pip install sae-lens transformer-lens

Usage:
    python experiments/borda_inflation/analyze_sae.py \
        --rm_checkpoint /path/to/rm/checkpoint \
        --results_dir experiments/borda_inflation/results \
        --output_dir experiments/borda_inflation/results/sae_analysis \
        --layer 16
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_inflation_data(results_dir):
    """Load precomputed inflation scores."""
    results_dir = Path(results_dir)
    jsonl_path = results_dir / "inflation_flat.jsonl"
    print(f"Loading inflation data from {jsonl_path}...")

    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"  {len(records)} responses")
    return records


def load_sae(layer, sae_width="32k"):
    """Load pretrained Llama Scope SAE for a given layer.

    Uses the fnlp/Llama-Scope collection on HuggingFace.

    Args:
        layer: int, which transformer layer (0-31 for Llama-3-8B)
        sae_width: "32k" or "128k" features

    Returns:
        sae: loaded SAE model
    """
    try:
        from sae_lens import SAE
    except ImportError:
        raise ImportError(
            "sae-lens required. Install with: pip install sae-lens"
        )

    # Llama Scope naming convention
    # See: https://huggingface.co/fnlp/Llama-Scope
    sae_id = f"fnlp/Llama-Scope-L{layer}-{sae_width}"
    print(f"Loading SAE: {sae_id}...")

    try:
        sae = SAE.from_pretrained(
            release=f"fnlp/Llama-Scope",
            sae_id=f"L{layer}_{sae_width}",
            device=DEVICE,
        )
    except Exception as e:
        print(f"  Failed with SAELens default loading: {e}")
        print(f"  Trying alternative loading...")
        # Try direct HuggingFace loading
        from huggingface_hub import hf_hub_download
        sae_path = hf_hub_download(
            repo_id="fnlp/Llama-Scope",
            filename=f"L{layer}_{sae_width}/sae_weights.safetensors",
        )
        sae = SAE.from_pretrained(sae_path, device=DEVICE)

    print(f"  SAE loaded: {sae_width} features, layer {layer}")
    return sae


def extract_hidden_states(model, tokenizer, prompts_and_responses, layer, batch_size=4):
    """Extract hidden states at a given layer for all responses.

    Uses hooks on the model to capture intermediate activations.

    Args:
        model: loaded reward model (with LLM backbone)
        tokenizer: tokenizer
        prompts_and_responses: list of (prompt, response) tuples
        layer: which layer to extract from
        batch_size: inference batch size

    Returns:
        hidden_states: [N, hidden_dim] tensor
    """
    all_hidden_states = []

    # Register hook to capture hidden states at the target layer
    captured = {}

    def hook_fn(module, input, output):
        # output is typically (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            captured["hidden"] = output[0]
        else:
            captured["hidden"] = output

    # Find the target layer in the model
    # For Llama: model.model.layers[layer]
    backbone = model.model  # the base LLM
    if hasattr(backbone, "model"):
        # nested model (e.g., CustomRewardModel wraps CausalLM)
        backbone = backbone.model
    if hasattr(backbone, "layers"):
        target_layer = backbone.layers[layer]
    elif hasattr(backbone, "h"):
        target_layer = backbone.h[layer]
    else:
        raise ValueError(f"Could not find layer {layer} in model architecture. "
                         f"Available attributes: {[a for a in dir(backbone) if not a.startswith('_')]}")

    handle = target_layer.register_forward_hook(hook_fn)

    try:
        for i in tqdm(range(0, len(prompts_and_responses), batch_size),
                       desc=f"Extracting layer {layer} hidden states"):
            batch = prompts_and_responses[i:i + batch_size]

            batch_texts = []
            for prompt, response in batch:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                if tokenizer.bos_token is not None:
                    text = text.replace(tokenizer.bos_token, "")
                batch_texts.append(text)

            inputs = tokenizer(
                batch_texts,
                max_length=2048,
                truncation=True,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                _ = model.custom_forward(**inputs)

            # Extract last token hidden state for each item in batch
            hidden = captured["hidden"]  # [batch, seq_len, hidden_dim]
            attention_mask = inputs["attention_mask"]

            for b in range(hidden.shape[0]):
                # Find last non-padding token
                seq_lens = attention_mask[b].sum().item()
                last_hidden = hidden[b, seq_lens - 1, :]  # [hidden_dim]
                all_hidden_states.append(last_hidden.cpu())

    finally:
        handle.remove()

    return torch.stack(all_hidden_states)  # [N, hidden_dim]


def encode_with_sae(sae, hidden_states, batch_size=256):
    """Encode hidden states through the SAE.

    Returns:
        feature_acts: [N, n_features] sparse feature activations
    """
    all_acts = []
    for i in tqdm(range(0, len(hidden_states), batch_size), desc="SAE encoding"):
        batch = hidden_states[i:i + batch_size].to(DEVICE)
        with torch.no_grad():
            acts = sae.encode(batch)  # [batch, n_features]
        all_acts.append(acts.cpu())

    return torch.cat(all_acts, dim=0)  # [N, n_features]


def correlate_features_with_inflation(feature_acts, inflation_scores, min_activation_count=50):
    """Correlate each SAE feature with inflation scores.

    Only considers features that are active (>0) in at least
    min_activation_count responses.

    Returns:
        results: list of (feature_idx, spearman_rho, p_value, n_active)
    """
    n_features = feature_acts.shape[1]
    inflation = np.array(inflation_scores)

    results = []
    for k in tqdm(range(n_features), desc="Correlating features"):
        acts = feature_acts[:, k].numpy()
        active_mask = acts > 0
        n_active = active_mask.sum()

        if n_active < min_activation_count:
            continue

        # Correlate activation strength with inflation
        rho, p_val = spearmanr(acts, inflation)
        results.append({
            "feature_idx": k,
            "spearman_rho": float(rho),
            "p_value": float(p_val),
            "n_active": int(n_active),
            "mean_activation": float(acts[active_mask].mean()),
        })

    # Sort by absolute correlation
    results.sort(key=lambda x: abs(x["spearman_rho"]), reverse=True)
    return results


def interpret_top_features(feature_acts, records, top_features, n_examples=20):
    """For each top feature, find responses with highest activation.

    Returns interpretable examples for manual inspection.
    """
    interpretations = []

    for feat_info in top_features:
        k = feat_info["feature_idx"]
        acts = feature_acts[:, k].numpy()

        # Top activated responses
        top_indices = np.argsort(-acts)[:n_examples]

        examples = []
        for idx in top_indices:
            if acts[idx] <= 0:
                break
            rec = records[idx]
            examples.append({
                "activation": float(acts[idx]),
                "inflation": rec["inflation"],
                "rm_reward": rec["rm_reward"],
                "effective_reward": rec["effective_reward"],
                "length": rec["length"],
                "response_preview": rec["response"][:300],
                "prompt_preview": rec["prompt"][:200],
            })

        interpretations.append({
            **feat_info,
            "top_examples": examples,
        })

    return interpretations


def main():
    parser = argparse.ArgumentParser(description="SAE-based Borda inflation analysis")
    parser.add_argument("--rm_checkpoint", type=str, required=True,
                        help="Path to trained RM checkpoint (for hidden state extraction)")
    parser.add_argument("--rm_tau", type=float, default=1.0)
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory with compute_inflation.py outputs")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results_dir/sae_analysis)")
    parser.add_argument("--layer", type=int, default=16,
                        help="Transformer layer for SAE analysis (default: 16)")
    parser.add_argument("--sae_width", type=str, default="32k",
                        choices=["32k", "128k"],
                        help="SAE feature width (default: 32k)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Number of top features to analyze (default: 50)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for hidden state extraction")
    parser.add_argument("--min_activations", type=int, default=50,
                        help="Minimum activations for a feature to be considered")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from eval.evaluate_trained_models_ultrafeedback import load_reward_model
    from general_preference.utils import get_tokenizer

    output_dir = Path(args.output_dir or Path(args.results_dir) / "sae_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Load inflation data ===
    records = load_inflation_data(args.results_dir)
    inflation_scores = [r["inflation"] for r in records]

    # === Load RM model ===
    print("\n" + "=" * 70)
    print("LOADING RM MODEL")
    print("=" * 70)
    rm_model, rm_tokenizer, rm_dim, rm_tau = load_reward_model(
        args.rm_checkpoint, args.rm_tau
    )

    # === Load SAE ===
    print("\n" + "=" * 70)
    print(f"LOADING SAE (layer {args.layer}, {args.sae_width} features)")
    print("=" * 70)
    sae = load_sae(args.layer, args.sae_width)

    # === Extract hidden states ===
    print("\n" + "=" * 70)
    print("EXTRACTING HIDDEN STATES")
    print("=" * 70)
    prompts_and_responses = [(r["prompt"], r["response"]) for r in records]
    hidden_states = extract_hidden_states(
        rm_model, rm_tokenizer, prompts_and_responses,
        args.layer, args.batch_size
    )
    print(f"  Hidden states shape: {hidden_states.shape}")

    # Save hidden states for reuse
    torch.save(hidden_states, output_dir / "hidden_states.pt")

    # === SAE encoding ===
    print("\n" + "=" * 70)
    print("SAE ENCODING")
    print("=" * 70)
    feature_acts = encode_with_sae(sae, hidden_states)
    print(f"  Feature activations shape: {feature_acts.shape}")

    # Sparsity stats
    n_active_per_sample = (feature_acts > 0).sum(dim=1).float()
    print(f"  Mean active features per response: {n_active_per_sample.mean():.1f}")
    print(f"  Total unique active features: {(feature_acts > 0).any(dim=0).sum().item()}")

    # Save feature activations
    torch.save(feature_acts, output_dir / "feature_activations.pt")

    # === Correlate with inflation ===
    print("\n" + "=" * 70)
    print("CORRELATING FEATURES WITH BORDA INFLATION")
    print("=" * 70)
    corr_results = correlate_features_with_inflation(
        feature_acts, inflation_scores, args.min_activations
    )
    print(f"  {len(corr_results)} features with >= {args.min_activations} activations")

    # Print top features
    print(f"\n  Top {args.top_k} features correlated with inflation:")
    print(f"  {'Rank':>4s} {'Feature':>8s} {'rho':>8s} {'p-value':>10s} {'n_active':>8s} {'mean_act':>8s}")
    print(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

    top_positive = [r for r in corr_results if r["spearman_rho"] > 0][:args.top_k // 2]
    top_negative = [r for r in corr_results if r["spearman_rho"] < 0][:args.top_k // 2]

    print("\n  POSITIVELY correlated (BT overvalues when feature is active):")
    for i, r in enumerate(top_positive):
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
        print(f"  {i+1:4d} {r['feature_idx']:8d} {r['spearman_rho']:+8.4f} "
              f"{r['p_value']:10.2e} {r['n_active']:8d} {r['mean_activation']:8.3f} {sig}")

    print("\n  NEGATIVELY correlated (BT undervalues when feature is active):")
    for i, r in enumerate(top_negative):
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
        print(f"  {i+1:4d} {r['feature_idx']:8d} {r['spearman_rho']:+8.4f} "
              f"{r['p_value']:10.2e} {r['n_active']:8d} {r['mean_activation']:8.3f} {sig}")

    # === Interpret top features ===
    print("\n" + "=" * 70)
    print("INTERPRETING TOP FEATURES")
    print("=" * 70)
    top_features = (top_positive + top_negative)[:args.top_k]
    interpretations = interpret_top_features(feature_acts, records, top_features)

    # === Save results ===
    # Correlation results
    corr_path = output_dir / "sae_correlations.json"
    with open(corr_path, "w") as f:
        json.dump(corr_results[:500], f, indent=2)
    print(f"\nSaved correlations to {corr_path}")

    # Interpretations
    interp_path = output_dir / "sae_interpretations.json"
    with open(interp_path, "w") as f:
        json.dump(interpretations, f, indent=2)
    print(f"Saved interpretations to {interp_path}")

    # Summary
    summary = {
        "layer": args.layer,
        "sae_width": args.sae_width,
        "n_responses": len(records),
        "n_features_analyzed": len(corr_results),
        "mean_active_features": float(n_active_per_sample.mean()),
        "top_positive_features": [r["feature_idx"] for r in top_positive[:10]],
        "top_negative_features": [r["feature_idx"] for r in top_negative[:10]],
    }
    summary_path = output_dir / "sae_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
