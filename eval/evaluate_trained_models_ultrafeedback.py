#!/usr/bin/env python3
"""Evaluate trained GPO (dim-8) and RM (dim-1) checkpoints on UltraFeedback.

Loads pre-trained reward models and computes:
- Test accuracy on held-out preference pairs
- Hodge decomposition of predicted preference matrices
- Intransitivity metrics (loop ratio, triangles, MFAS)
- Scaling law analysis: how data intransitivity correlates with model performance

Similar to toy case evaluation but on real UltraFeedback data with full LLMs.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.special import expit
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel
import safetensors.torch

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from general_preference.models import get_reward_model
from general_preference.utils import get_tokenizer

# ── CONFIG ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 2048
BATCH_SIZE = 8  # for inference


# ── PROBABILISTIC METRICS ───────────────────────────────────────────────────

def compute_kl_divergence(empirical_probs, predicted_probs):
    """Compute KL divergence KL(empirical || predicted).
    
    Args:
        empirical_probs: [n_pairs] array of empirical win probabilities
        predicted_probs: [n_pairs] array of model predicted probabilities
    
    Returns:
        float: KL divergence (nats)
    """
    # Clip probabilities to avoid log(0)
    p = np.clip(empirical_probs, 1e-9, 1 - 1e-9)
    q = np.clip(predicted_probs, 1e-9, 1 - 1e-9)
    
    # KL(P || Q) = Σ p(x) log(p(x)/q(x))
    # For binary outcomes: p log(p/q) + (1-p) log((1-p)/(1-q))
    kl = np.mean(
        p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    )
    
    return float(kl)


def compute_all_probabilistic_metrics(empirical_probs, predicted_probs):
    """Compute all probabilistic metrics from toy case.
    
    Matches compute_probabilistic_metrics from the toy case exactly.
    
    Args:
        empirical_probs: [n_pairs] array of empirical win probabilities
        predicted_probs: [n_pairs] array of model predicted probabilities
    
    Returns:
        dict with brier, nll, ece, kl_div, avg_confidence, acc_strong_conf, 
             acc_weak_conf, entropy
    """
    p_true = empirical_probs
    p_pred = predicted_probs
    
    # 1. Brier score (MSE of probabilities)
    brier = float(np.mean((p_pred - p_true) ** 2))
    
    # 2. Negative log-likelihood
    eps = 1e-7  # Conservative clipping
    nll = float(-np.mean(np.log(np.clip(p_pred, eps, 1.0))))
    
    # 3. Expected Calibration Error (10 bins)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        mask = (p_pred >= bin_edges[b]) & (p_pred < bin_edges[b + 1])
        if mask.sum() > 0:
            bin_pred = p_pred[mask].mean()
            bin_true = p_true[mask].mean()
            ece += (mask.sum() / len(p_pred)) * abs(bin_pred - bin_true)
    ece = float(ece)
    
    # 4. Average confidence (distance from 0.5)
    avg_confidence = float(np.mean(np.abs(p_pred - 0.5)))
    
    # 5. Stratified accuracy by model confidence
    strong_mask = p_pred > 0.8  # high confidence predictions
    weak_mask = (p_pred > 0.5) & (p_pred <= 0.6)  # low confidence
    acc_strong = float((p_pred[strong_mask] > 0.5).mean()) if strong_mask.any() else float('nan')
    acc_weak = float((p_pred[weak_mask] > 0.5).mean()) if weak_mask.any() else float('nan')
    
    # 6. Prediction entropy (higher = less collapsed)
    eps = 1e-7  # Conservative clipping
    entropy = float(-np.mean(
        p_pred * np.log(np.clip(p_pred, eps, 1.0)) +
        (1 - p_pred) * np.log(np.clip(1 - p_pred, eps, 1.0))
    ))
    
    # 7. KL divergence from empirical
    # Use more conservative clipping to avoid numerical overflow with very confident predictions
    eps = 1e-7  # Increased from 1e-9 to handle bfloat16/float32 precision
    p_true_clipped = np.clip(p_true, eps, 1 - eps)
    p_pred_clipped = np.clip(p_pred, eps, 1 - eps)
    
    # Compute KL divergence with additional safeguards
    kl_term1 = p_true_clipped * np.log(p_true_clipped / p_pred_clipped)
    kl_term2 = (1 - p_true_clipped) * np.log((1 - p_true_clipped) / (1 - p_pred_clipped))
    
    # Filter out any inf/nan values that might still occur
    kl_terms = kl_term1 + kl_term2
    kl_terms = kl_terms[np.isfinite(kl_terms)]  # Remove inf/nan
    
    if len(kl_terms) > 0:
        kl_div = float(np.mean(kl_terms))
    else:
        kl_div = float('nan')  # If all values were inf/nan
    
    return {
        'brier': brier,
        'nll': nll,
        'ece': ece,
        'kl_div': kl_div,
        'avg_confidence': avg_confidence,
        'acc_strong_conf': acc_strong,
        'acc_weak_conf': acc_weak,
        'entropy': entropy,
    }


# ── HODGE DECOMPOSITION (from toy case) ────────────────────────────────────

def hodge_decompose_from_probs(pref_matrix, observed_mask):
    """Hodge decomposition on a predicted probability matrix.

    Args:
        pref_matrix: [N, N] array where pref_matrix[i,j] = P(i > j)
        observed_mask: [N, N] boolean array indicating which pairs to analyze

    Returns:
        dict with gradient_energy, curl_energy, total_energy, loop_ratio
    """
    n = pref_matrix.shape[0]
    obs = observed_mask & ~np.eye(n, dtype=bool)

    # Flow is just P(i>j) - 0.5, masked to observed edges
    flow = np.where(obs, pref_matrix - 0.5, 0.0)

    # Ensure antisymmetry
    flow = 0.5 * (flow - flow.T)

    # Total energy
    total_energy = float(np.sum(flow ** 2))

    if total_energy < 1e-12:
        return dict(
            gradient_energy=0.0, residual_energy=0.0,
            curl_energy=0.0, harmonic_energy=0.0,
            total_energy=0.0, loop_ratio=0.0,
            curl_ratio=0.0, harmonic_ratio=0.0,
        )

    # Graph Laplacian on observed edges
    L = np.diag(obs.sum(axis=1).astype(float)) - obs.astype(float)
    b = flow.sum(axis=1)

    # Pin node 0
    L[0, :] = 0
    L[0, 0] = 1.0
    b[0] = 0.0

    try:
        r = np.linalg.solve(L, b)
    except np.linalg.LinAlgError:
        r = np.linalg.lstsq(L, b, rcond=None)[0]

    # Gradient flow
    gradient_flow_full = r[:, np.newaxis] - r[np.newaxis, :]
    gradient_energy = float(np.sum(gradient_flow_full ** 2 * obs))

    # Residual (divergence-free)
    residual_flow   = flow - gradient_flow_full * obs
    residual_energy = float(max(0.0, total_energy - gradient_energy))
    loop_ratio      = float(np.clip(residual_energy / total_energy, 0.0, 1.0))

    # Three-way split: residual → curl (local 3-cycles) + harmonic (global)
    curl_flow, harmonic_flow = _curl_harmonic_split(residual_flow, obs)
    curl_energy     = float(np.sum(curl_flow ** 2))
    harmonic_energy = float(np.sum(harmonic_flow ** 2))
    curl_ratio      = float(curl_energy / total_energy) if total_energy > 0 else 0.0
    harmonic_ratio  = float(harmonic_energy / total_energy) if total_energy > 0 else 0.0

    return dict(
        gradient_energy  = gradient_energy,
        residual_energy  = residual_energy,
        curl_energy      = curl_energy,
        harmonic_energy  = harmonic_energy,
        total_energy     = total_energy,
        loop_ratio       = loop_ratio,
        curl_ratio       = curl_ratio,
        harmonic_ratio   = harmonic_ratio,
    )


def build_pref_counts_from_pairs(pairs, response_to_idx):
    """Build preference count matrix from (winner, loser) pairs."""
    n = len(response_to_idx)
    counts = np.zeros((n, n), dtype=float)
    for w, l in pairs:
        if w in response_to_idx and l in response_to_idx:
            counts[response_to_idx[w], response_to_idx[l]] += 1
    return counts


def hodge_decompose(counts):
    """Hodge decomposition on count matrix (from toy case)."""
    n = counts.shape[0]
    total = counts + counts.T
    obs = (total > 0) & ~np.eye(n, dtype=bool)

    with np.errstate(divide='ignore', invalid='ignore'):
        win_rates = np.where(obs, counts / total, 0.0)
    flow = win_rates - np.where(obs, 0.5, 0.0)

    total_energy = float(np.sum(flow ** 2))

    if total_energy < 1e-12:
        zeros = np.zeros((n, n))
        return dict(potential_r=np.zeros(n), flow=flow,
                    gradient_flow=zeros, residual_flow=zeros,
                    curl_flow=zeros, harmonic_flow=zeros,
                    gradient_energy=0.0, residual_energy=0.0,
                    curl_energy=0.0, harmonic_energy=0.0,
                    total_energy=0.0, loop_ratio=0.0,
                    curl_ratio=0.0, harmonic_ratio=0.0,
                    observed_mask=obs)

    L = np.diag(obs.sum(axis=1).astype(float)) - obs.astype(float)
    b = flow.sum(axis=1)

    L[0, :] = 0
    L[0, 0] = 1.0
    b[0] = 0.0

    try:
        r = np.linalg.solve(L, b)
    except np.linalg.LinAlgError:
        r = np.linalg.lstsq(L, b, rcond=None)[0]

    gradient_flow_full = r[:, np.newaxis] - r[np.newaxis, :]
    gradient_energy = float(np.sum(gradient_flow_full ** 2 * obs))
    gradient_flow = gradient_flow_full * obs

    # residual = total − gradient  (divergence-free)
    residual_flow    = flow - gradient_flow
    residual_energy  = float(max(0.0, total_energy - gradient_energy))
    loop_ratio       = float(np.clip(residual_energy / total_energy, 0.0, 1.0))

    # three-way split: residual → curl (local 3-cycles) + harmonic (global)
    curl_flow, harmonic_flow = _curl_harmonic_split(residual_flow, obs)
    curl_energy     = float(np.sum(curl_flow ** 2))
    harmonic_energy = float(np.sum(harmonic_flow ** 2))
    curl_ratio      = float(curl_energy / total_energy) if total_energy > 0 else 0.0
    harmonic_ratio  = float(harmonic_energy / total_energy) if total_energy > 0 else 0.0

    return dict(
        potential_r      = r,
        flow             = flow,
        gradient_flow    = gradient_flow,
        residual_flow    = residual_flow,
        curl_flow        = curl_flow,
        harmonic_flow    = harmonic_flow,
        gradient_energy  = gradient_energy,
        residual_energy  = residual_energy,
        curl_energy      = curl_energy,
        harmonic_energy  = harmonic_energy,
        total_energy     = total_energy,
        loop_ratio       = loop_ratio,
        curl_ratio       = curl_ratio,
        harmonic_ratio   = harmonic_ratio,
        observed_mask    = obs,
    )


def _curl_harmonic_split(residual_flow, observed_mask):
    """Separate divergence-free residual R* into curl and harmonic.

    curl*(Φ) = B₂ Φ  — projection onto Im(boundary₂), i.e. local 3-cycles.
    H = R* − curl*(Φ) — orthogonal complement (global cycles).

    On complete subgraphs (K₃, K₄) H = 0.  On sparse graphs with missing
    triangles the harmonic part captures cycles not decomposable into 3-cycles.
    """
    n = residual_flow.shape[0]
    obs = observed_mask & ~np.eye(n, dtype=bool)

    # ── upper-triangle edges ──
    edges = []
    edge_to_idx = {}
    for i in range(n):
        for j in range(i + 1, n):
            if obs[i, j]:
                edge_to_idx[(i, j)] = len(edges)
                edges.append((i, j))

    # ── observed triangles ──
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            if not obs[i, j]:
                continue
            for k in range(j + 1, n):
                if obs[i, k] and obs[j, k]:
                    triangles.append((i, j, k))
    n_tri = len(triangles)

    if n_tri == 0:
        return np.zeros_like(residual_flow), residual_flow.copy()

    # ── boundary matrix B₂  (m edges × n_tri triangles) ──
    # oriented triangle (i<j<k):  edge(i,j) +1, edge(i,k) -1, edge(j,k) +1
    m = len(edges)
    B2 = np.zeros((m, n_tri))
    for t_idx, (i, j, k) in enumerate(triangles):
        B2[edge_to_idx[(i, j)], t_idx] =  1.0
        B2[edge_to_idx[(i, k)], t_idx] = -1.0
        B2[edge_to_idx[(j, k)], t_idx] =  1.0

    # residual on upper-triangle edges
    r = np.array([residual_flow[i, j] for (i, j) in edges])

    # ── project onto Im(B₂) ──
    Phi, _, _, _ = np.linalg.lstsq(B2, r, rcond=None)
    curl_r = B2 @ Phi

    # ── back to antisymmetric n×n ──
    curl_flow = np.zeros((n, n))
    for idx, (i, j) in enumerate(edges):
        curl_flow[i, j] =  curl_r[idx]
        curl_flow[j, i] = -curl_r[idx]

    harmonic_flow = residual_flow - curl_flow
    return curl_flow, harmonic_flow


# ── MODEL LOADING ───────────────────────────────────────────────────────────

def load_reward_model(checkpoint_path, tau, device=DEVICE):
    """Load trained reward model from checkpoint.

    Automatically detects the reward dimension from the checkpoint files.
    tau must match the value used during training (e.g. 1.0 for BT, 0.1 for GPO).

    Returns:
        model: Reward model instance
        tokenizer: HuggingFace tokenizer
        reward_dim: int (1 for RM, >1 for GPO)
        tau: float (temperature, passed through)
    """
    checkpoint_path = Path(checkpoint_path)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # First, detect the value_head dimension from the checkpoint
    # Find safetensors files (model shards or single file)
    safetensors_files = list(checkpoint_path.glob("model*.safetensors"))
    if not safetensors_files:
        # Try without model prefix
        safetensors_files = list(checkpoint_path.glob("*.safetensors"))
    
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {checkpoint_path}")
    
    print(f"Found {len(safetensors_files)} safetensors file(s)")
    
    # Load the index to find which file contains value_head
    index_file = checkpoint_path / "model.safetensors.index.json"
    value_head_file = None
    
    if index_file.exists():
        print("Loading from sharded checkpoint (using index)")
        with open(index_file) as f:
            index = json.load(f)
        
        # Find which shard contains value_head.weight
        for key, filename in index['weight_map'].items():
            if 'value_head.weight' in key:
                value_head_file = checkpoint_path / filename
                print(f"Found value_head in shard: {filename}")
                break
    
    if value_head_file is None:
        # Try to find value_head in any file
        print("Searching for value_head in checkpoint files...")
        for sf_file in safetensors_files:
            try:
                with safetensors.torch.safe_open(str(sf_file), framework="pt") as f:
                    keys = list(f.keys())
                    if any('value_head.weight' in k for k in keys):
                        value_head_file = sf_file
                        print(f"Found value_head in: {sf_file.name}")
                        break
            except Exception as e:
                print(f"Could not read {sf_file.name}: {e}")
                continue
    
    if value_head_file is None:
        raise ValueError(f"Could not find value_head.weight in any checkpoint file in {checkpoint_path}")
    
    # Load just the value_head to check its dimension
    print(f"Reading value_head dimension from {value_head_file.name}...")
    with safetensors.torch.safe_open(str(value_head_file), framework="pt") as f:
        keys = list(f.keys())
        value_head_key = None
        for k in keys:
            if 'value_head.weight' in k:
                value_head_key = k
                break
        
        if value_head_key is None:
            raise ValueError(f"value_head.weight not found in {value_head_file.name}")
        
        value_head_weight = f.get_tensor(value_head_key)
        reward_dim = value_head_weight.shape[0]
    
    print(f"✓ Detected reward dimension: {reward_dim}")
    
    # Now load model with correct dimension
    is_gpo = reward_dim > 1
    print(f"Loading {'GPO' if is_gpo else 'Bradley-Terry'} model...")
    
    model = get_reward_model(
        str(checkpoint_path),
        bf16=True,
        use_flash_attention_2=False,
        is_general_preference=is_gpo,
        value_head_dim=reward_dim,
    )
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = get_tokenizer(str(checkpoint_path), model, padding_side="left")
    
    print(f"✓ Successfully loaded model:")
    print(f"  Type: {'GPO (General Preference)' if is_gpo else 'Bradley-Terry (scalar reward)'}")
    print(f"  Reward dimension: {reward_dim}")
    print(f"  Temperature (tau): {tau}")
    print()
    
    return model, tokenizer, reward_dim, tau


def get_response_reward(model, tokenizer, prompt, response, device=DEVICE, max_length=MAX_LENGTH):
    """Get reward vector for a prompt+response pair using model.custom_forward.
    
    Returns:
        reward: [reward_dim] tensor (normalized reward vector)
    """
    # Format as chat
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    # Remove BOS token if present (for consistency with training)
    if tokenizer.bos_token is not None:
        text = text.replace(tokenizer.bos_token, "")
    
    # Tokenize (left padding as in training)
    inputs = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors="pt",
        add_special_tokens=False
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get reward using custom_forward
    with torch.no_grad():
        reward, _ = model.custom_forward(**inputs)
    
    return reward.squeeze(0)  # [reward_dim]


def compute_preference_matrix_batched(model, tokenizer, responses, prompt, reward_dim, tau, device=DEVICE):
    """Compute full N×N preference matrix for a prompt.
    
    For GPO: P[i,j] = sigmoid(v_i @ R^T @ v_j / tau)
    For RM:  P[i,j] = sigmoid(reward(i) - reward(j))
    
    Returns:
        pref_matrix: [N, N] numpy array
    """
    n = len(responses)
    
    # Get reward vectors for all responses (batched)
    rewards = []
    for i in range(0, n, BATCH_SIZE):
        batch_responses = responses[i:i+BATCH_SIZE]
        batch_rewards = []
        for resp in batch_responses:
            reward = get_response_reward(model, tokenizer, prompt, resp, device)
            batch_rewards.append(reward)
        rewards.extend(batch_rewards)
    
    rewards = torch.stack(rewards)  # [N, reward_dim]
    
    # Compute preference matrix
    with torch.no_grad():
        if reward_dim == 1:
            # Bradley-Terry: scalar rewards
            # P(i > j) = sigmoid((r_i - r_j) / tau)
            # NOTE: tau is crucial! Training uses sigmoid((r_chosen - r_rejected) / tau)
            reward_diff = rewards.unsqueeze(1) - rewards.unsqueeze(0)  # [N, 1] - [1, N] = [N, N]
            pref_matrix = torch.sigmoid(reward_diff / tau).float().cpu().numpy()
        else:
            # GPO: antisymmetric bilinear form v_i @ R^T @ v_j
            # Create R matrix (block-diagonal skew-symmetric)
            R = torch.zeros(reward_dim, reward_dim, device=device, dtype=rewards.dtype)
            for i in range(0, reward_dim, 2):
                R[i, i + 1] = -1
                R[i + 1, i] = 1
            
            # Compute scores: S[i,j] = v_i @ R^T @ v_j
            S = rewards @ R.T @ rewards.T  # [N, N]
            pref_matrix = torch.sigmoid(S / tau).float().cpu().numpy()
    
    return pref_matrix


# ── DATA LOADING ────────────────────────────────────────────────────────────

def load_ultrafeedback_grouped(dataset_path, split='test', max_prompts=None):
    """Load UltraFeedback and group by prompt.
    
    Returns:
        prompt_groups: dict[prompt] = [(chosen, rejected), ...]
    """
    if dataset_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=dataset_path, split='train')
    else:
        dataset = load_from_disk(dataset_path)
        if split in dataset:
            dataset = dataset[split]
    
    prompt_groups = defaultdict(list)
    
    for sample in tqdm(dataset, desc=f"Loading {split} data"):
        prompt = sample.get('prompt', sample.get('instruction', ''))
        chosen = sample.get('chosen', sample.get('response_chosen', ''))
        rejected = sample.get('rejected', sample.get('response_rejected', ''))
        
        # Extract text if structured format
        if isinstance(prompt, list):
            prompt = prompt[0].get('content', str(prompt[0]))
        if isinstance(chosen, list):
            chosen = chosen[0].get('content', str(chosen[0]))
        if isinstance(rejected, list):
            rejected = rejected[0].get('content', str(rejected[0]))
        
        prompt_groups[str(prompt)].append((str(chosen), str(rejected)))
    
    if max_prompts:
        keys = list(prompt_groups.keys())[:max_prompts]
        prompt_groups = {k: prompt_groups[k] for k in keys}
    
    return dict(prompt_groups)


# ── EVALUATION ──────────────────────────────────────────────────────────────

def evaluate_model_on_prompt(model, tokenizer, prompt, pairs, reward_dim, tau, device=DEVICE):
    """Evaluate model on a single prompt's preference pairs.
    
    Returns:
        dict with accuracy, pref_matrix, responses, response_to_idx, empirical_probs, kl_div, nll
    """
    # Extract unique responses
    responses = set()
    for chosen, rejected in pairs:
        responses.add(chosen)
        responses.add(rejected)
    responses = sorted(list(responses))
    response_to_idx = {r: i for i, r in enumerate(responses)}
    
    n_resp = len(responses)
    
    # Compute empirical win rates from data
    empirical_matrix = np.zeros((n_resp, n_resp))
    pair_counts = {}
    for chosen, rejected in pairs:
        i = response_to_idx[chosen]
        j = response_to_idx[rejected]
        key = (i, j)
        pair_counts[key] = pair_counts.get(key, 0) + 1
        empirical_matrix[i, j] += 1
    
    # Compute full preference matrix from model
    pref_matrix = compute_preference_matrix_batched(
        model, tokenizer, responses, prompt, reward_dim, tau, device
    )
    
    # Compute accuracy on pairs
    correct = 0
    empirical_probs = []
    predicted_probs = []
    
    for chosen, rejected in pairs:
        i = response_to_idx[chosen]
        j = response_to_idx[rejected]
        
        if pref_matrix[i, j] > 0.5:
            correct += 1
        
        # Empirical probability (for this specific ordered pair)
        # If we observe (i > j) once, empirical prob is 1.0
        # But we normalize by total observations of this pair
        total_obs = empirical_matrix[i, j] + empirical_matrix[j, i]
        emp_prob = empirical_matrix[i, j] / total_obs if total_obs > 0 else 0.5
        
        empirical_probs.append(emp_prob)
        predicted_probs.append(pref_matrix[i, j])
    
    empirical_probs = np.array(empirical_probs)
    predicted_probs = np.array(predicted_probs)
    
    # Compute all probabilistic metrics (matching toy case)
    prob_metrics = compute_all_probabilistic_metrics(empirical_probs, predicted_probs)
    
    accuracy = correct / len(pairs) if pairs else 0.0
    
    result = {
        'accuracy': accuracy,
        'pref_matrix': pref_matrix,
        'responses': responses,
        'response_to_idx': response_to_idx,
        'n_responses': n_resp,
        'n_pairs': len(pairs),
        'empirical_probs': empirical_probs,
        'predicted_probs': predicted_probs,
    }
    
    # Add all probabilistic metrics
    result.update(prob_metrics)
    
    return result


def evaluate_dataset(model, tokenizer, prompt_groups, reward_dim, tau, device=DEVICE, max_responses=100):
    """Evaluate model on entire dataset.
    
    Returns:
        dict with ALL metrics from toy case: accuracy, KL, NLL, Brier, ECE, 
             confidence metrics, entropy, and Hodge decomposition
    """
    # Aggregate all metrics
    metrics_lists = {
        'accuracy': [],
        'kl_div': [],
        'nll': [],
        'brier': [],
        'ece': [],
        'avg_confidence': [],
        'acc_strong_conf': [],
        'acc_weak_conf': [],
        'entropy': [],
        'pred_loop_ratio': [],
        'pred_curl_ratio': [],
        'pred_harmonic_ratio': [],
    }
    
    prompt_results = {}
    
    for prompt, pairs in tqdm(prompt_groups.items(), desc="Evaluating prompts"):
        # Skip prompts with too many responses (memory/time)
        responses = set()
        for c, r in pairs:
            responses.add(c)
            responses.add(r)
        
        if len(responses) > max_responses or len(responses) < 3:
            continue
        
        result = evaluate_model_on_prompt(
            model, tokenizer, prompt, pairs, reward_dim, tau, device
        )
        
        # Collect all metrics
        for key in metrics_lists.keys():
            if key in ('pred_loop_ratio', 'pred_curl_ratio', 'pred_harmonic_ratio'):
                continue  # Handle separately after Hodge decomposition
            metrics_lists[key].append(result[key])
        
        # Hodge decomposition on predicted preferences
        counts = build_pref_counts_from_pairs(pairs, result['response_to_idx'])
        data_hodge = hodge_decompose(counts)
        observed_mask = data_hodge['observed_mask']
        
        pred_hodge = hodge_decompose_from_probs(result['pref_matrix'], observed_mask)
        metrics_lists['pred_loop_ratio'].append(pred_hodge['loop_ratio'])
        metrics_lists['pred_curl_ratio'].append(pred_hodge['curl_ratio'])
        metrics_lists['pred_harmonic_ratio'].append(pred_hodge['harmonic_ratio'])
        
        # Store per-prompt results
        prompt_results[prompt[:100]] = {
            'accuracy': result['accuracy'],
            'kl_div': result['kl_div'],
            'nll': result['nll'],
            'brier': result['brier'],
            'ece': result['ece'],
            'avg_confidence': result['avg_confidence'],
            'acc_strong_conf': result['acc_strong_conf'],
            'acc_weak_conf': result['acc_weak_conf'],
            'entropy': result['entropy'],
            'n_responses': result['n_responses'],
            'n_pairs': result['n_pairs'],
            'data_loop_ratio':      data_hodge['loop_ratio'],
            'data_curl_ratio':      data_hodge['curl_ratio'],
            'data_harmonic_ratio':  data_hodge['harmonic_ratio'],
            'pred_loop_ratio':      pred_hodge['loop_ratio'],
            'pred_curl_ratio':      pred_hodge['curl_ratio'],
            'pred_harmonic_ratio':  pred_hodge['harmonic_ratio'],
            'pred_gradient_energy': pred_hodge['gradient_energy'],
            'pred_residual_energy': pred_hodge['residual_energy'],
            'pred_curl_energy':     pred_hodge['curl_energy'],
            'pred_harmonic_energy': pred_hodge['harmonic_energy'],
            'pred_total_energy':    pred_hodge['total_energy'],
        }
    
    # Compute aggregate statistics
    aggregate_results = {
        'n_prompts_evaluated': len(prompt_results),
        'prompt_results': prompt_results,
    }
    
    # Add mean and std for all metrics
    for key, values in metrics_lists.items():
        if values:
            aggregate_results[f'mean_{key}'] = float(np.nanmean(values))
            aggregate_results[f'std_{key}'] = float(np.nanstd(values))
            aggregate_results[f'min_{key}'] = float(np.nanmin(values))
            aggregate_results[f'max_{key}'] = float(np.nanmax(values))
        else:
            aggregate_results[f'mean_{key}'] = float('nan')
            aggregate_results[f'std_{key}'] = float('nan')
            aggregate_results[f'min_{key}'] = float('nan')
            aggregate_results[f'max_{key}'] = float('nan')
    
    return aggregate_results


# ── PLOTTING ────────────────────────────────────────────────────────────────

def plot_comparison(rm_results, gpo_results, data_metrics, output_dir):
    """Generate comparison plots between RM and GPO."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract per-prompt data
    prompts = list(rm_results['prompt_results'].keys())
    
    rm_accs = [rm_results['prompt_results'][p]['accuracy'] for p in prompts]
    gpo_accs = [gpo_results['prompt_results'][p]['accuracy'] for p in prompts]
    
    data_loop_ratios = [data_metrics[p]['loop_ratio'] for p in prompts if p in data_metrics]
    rm_pred_loop = [rm_results['prompt_results'][p]['pred_loop_ratio'] for p in prompts if p in data_metrics]
    gpo_pred_loop = [gpo_results['prompt_results'][p]['pred_loop_ratio'] for p in prompts if p in data_metrics]
    
    # Plot 1: Accuracy comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.scatter(rm_accs, gpo_accs, alpha=0.6, s=30)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='y=x')
    ax.set_xlabel('RM Accuracy')
    ax.set_ylabel('GPO Accuracy')
    ax.set_title('Per-Prompt Accuracy: GPO vs RM')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy vs data intransitivity
    ax = axes[0, 1]
    ax.scatter(data_loop_ratios, rm_accs, alpha=0.6, s=30, label='RM', color='#1f77b4')
    ax.scatter(data_loop_ratios, gpo_accs, alpha=0.6, s=30, label='GPO', color='#ff7f0e', marker='s')
    ax.set_xlabel('Data Loop Ratio')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Data Intransitivity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Capture hypothesis (pred vs data loop ratio)
    ax = axes[1, 0]
    if data_loop_ratios and rm_pred_loop and gpo_pred_loop:
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='y=x (perfect capture)')
        ax.scatter(data_loop_ratios, rm_pred_loop, alpha=0.6, s=30, label='RM', color='#1f77b4')
        ax.scatter(data_loop_ratios, gpo_pred_loop, alpha=0.6, s=30, label='GPO', color='#ff7f0e', marker='s')
        
        # Compute correlations
        r_rm = np.corrcoef(data_loop_ratios, rm_pred_loop)[0, 1] if len(data_loop_ratios) > 1 else 0
        r_gpo = np.corrcoef(data_loop_ratios, gpo_pred_loop)[0, 1] if len(data_loop_ratios) > 1 else 0
        
        ax.set_xlabel('Data Loop Ratio')
        ax.set_ylabel('Predicted Loop Ratio')
        ax.set_title(f'Capture Hypothesis\nRM R={r_rm:.3f}, GPO R={r_gpo:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    acc_gap = gpo_results['mean_accuracy'] - rm_results['mean_accuracy']
    kl_gap = rm_results['mean_kl_div'] - gpo_results['mean_kl_div']
    nll_gap = rm_results['mean_nll'] - gpo_results['mean_nll']
    brier_gap = rm_results['mean_brier'] - gpo_results['mean_brier']
    loop_gap = gpo_results['mean_pred_loop_ratio'] - rm_results['mean_pred_loop_ratio']
    
    summary_text = (
        f"EVALUATION SUMMARY\n"
        f"{'='*38}\n\n"
        f"Dataset:\n"
        f"  Prompts: {rm_results['n_prompts_evaluated']}\n"
        f"  Data loop ratio: {np.mean(data_loop_ratios):.4f}\n\n"
        f"RM (Bradley-Terry):\n"
        f"  Accuracy: {rm_results['mean_accuracy']:.4f}\n"
        f"  KL Div:   {rm_results['mean_kl_div']:.4f}\n"
        f"  NLL:      {rm_results['mean_nll']:.4f}\n"
        f"  Brier:    {rm_results['mean_brier']:.4f}\n"
        f"  Entropy:  {rm_results['mean_entropy']:.4f}\n\n"
        f"GPO (dim-8):\n"
        f"  Accuracy: {gpo_results['mean_accuracy']:.4f}\n"
        f"  KL Div:   {gpo_results['mean_kl_div']:.4f}\n"
        f"  NLL:      {gpo_results['mean_nll']:.4f}\n"
        f"  Brier:    {gpo_results['mean_brier']:.4f}\n"
        f"  Entropy:  {gpo_results['mean_entropy']:.4f}\n\n"
        f"GPO Advantages:\n"
        f"  Accuracy: {acc_gap:+.4f}\n"
        f"  KL Div:   {kl_gap:+.4f}\n"
        f"  NLL:      {nll_gap:+.4f}\n"
        f"  Brier:    {brier_gap:+.4f}\n"
        f"  Loop:     {loop_gap:+.4f}\n"
    )
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_summary.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved {output_dir / 'comparison_summary.png'}")
    plt.close()


# ── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained GPO and RM models on UltraFeedback"
    )
    parser.add_argument(
        "--rm_checkpoint",
        type=str,
        required=True,
        help="Path to trained RM checkpoint (dim-1)"
    )
    parser.add_argument(
        "--gpo_checkpoint",
        type=str,
        required=True,
        help="Path to trained GPO checkpoint (dim-8)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to UltraFeedback dataset (JSONL or HF dataset)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Limit evaluation to N prompts (for speed)"
    )
    parser.add_argument(
        "--max_responses",
        type=int,
        default=100,
        help="Skip prompts with >N responses (memory limit)"
    )
    parser.add_argument(
        "--rm_tau",
        type=float,
        default=1.0,
        help="Temperature for BT/RM model (must match training). Default 1.0."
    )
    parser.add_argument(
        "--gpo_tau",
        type=float,
        default=0.1,
        help="Temperature for GPO model (must match training). Default 0.1."
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("EVALUATING TRAINED MODELS ON ULTRAFEEDBACK")
    print("="*70)
    print(f"\nRM checkpoint:  {args.rm_checkpoint}")
    print(f"GPO checkpoint: {args.gpo_checkpoint}")
    print(f"Dataset:        {args.dataset}")
    print(f"Output dir:     {output_dir}")
    
    # Load models
    print("\n" + "="*70)
    print("LOADING MODELS")
    print("="*70)
    rm_model, rm_tokenizer, rm_dim, rm_tau = load_reward_model(args.rm_checkpoint, args.rm_tau, DEVICE)
    gpo_model, gpo_tokenizer, gpo_dim, gpo_tau = load_reward_model(args.gpo_checkpoint, args.gpo_tau, DEVICE)
    
    # Load dataset
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    prompt_groups = load_ultrafeedback_grouped(
        args.dataset,
        split=args.split,
        max_prompts=args.max_prompts
    )
    print(f"Loaded {len(prompt_groups)} prompts")
    
    # Compute data intransitivity metrics
    print("\n" + "="*70)
    print("COMPUTING DATA INTRANSITIVITY")
    print("="*70)
    data_metrics = {}
    for prompt, pairs in tqdm(prompt_groups.items(), desc="Data Hodge"):
        responses = set()
        for c, r in pairs:
            responses.add(c)
            responses.add(r)
        
        if len(responses) > args.max_responses or len(responses) < 3:
            continue
        
        response_to_idx = {r: i for i, r in enumerate(sorted(responses))}
        counts = build_pref_counts_from_pairs(pairs, response_to_idx)
        hodge = hodge_decompose(counts)
        
        data_metrics[prompt[:100]] = {
            'loop_ratio': hodge['loop_ratio'],
            'gradient_energy': hodge['gradient_energy'],
            'curl_energy': hodge['curl_energy'],
        }
    
    print(f"Computed metrics for {len(data_metrics)} prompts")
    print(f"Mean data loop ratio: {np.mean([m['loop_ratio'] for m in data_metrics.values()]):.4f}")
    
    # Evaluate RM
    print("\n" + "="*70)
    print("EVALUATING RM (Bradley-Terry)")
    print("="*70)
    rm_results = evaluate_dataset(
        rm_model, rm_tokenizer, prompt_groups, rm_dim, rm_tau, DEVICE, args.max_responses
    )
    print(f"\nAccuracy Metrics:")
    print(f"  Test Accuracy:        {rm_results['mean_accuracy']:.4f} ± {rm_results['std_accuracy']:.4f}")
    print(f"  High Conf Accuracy:   {rm_results['mean_acc_strong_conf']:.4f}")
    print(f"  Low Conf Accuracy:    {rm_results['mean_acc_weak_conf']:.4f}")
    print(f"\nProbabilistic Metrics:")
    print(f"  KL Divergence:        {rm_results['mean_kl_div']:.4f} ± {rm_results['std_kl_div']:.4f}")
    print(f"  NLL:                  {rm_results['mean_nll']:.4f} ± {rm_results['std_nll']:.4f}")
    print(f"  Brier Score:          {rm_results['mean_brier']:.4f} ± {rm_results['std_brier']:.4f}")
    print(f"  ECE:                  {rm_results['mean_ece']:.4f} ± {rm_results['std_ece']:.4f}")
    print(f"\nConfidence & Diversity:")
    print(f"  Avg Confidence:       {rm_results['mean_avg_confidence']:.4f}")
    print(f"  Entropy:              {rm_results['mean_entropy']:.4f}")
    print(f"\nIntransitivity Capture:")
    print(f"  Pred Loop Ratio:      {rm_results['mean_pred_loop_ratio']:.4f}")
    print(f"  Pred Curl Ratio:      {rm_results['mean_pred_curl_ratio']:.4f}")
    print(f"  Pred Harmonic Ratio:  {rm_results['mean_pred_harmonic_ratio']:.4f}")
    
    # Evaluate GPO
    print("\n" + "="*70)
    print("EVALUATING GPO (dim-8)")
    print("="*70)
    gpo_results = evaluate_dataset(
        gpo_model, gpo_tokenizer, prompt_groups, gpo_dim, gpo_tau, DEVICE, args.max_responses
    )
    print(f"\nAccuracy Metrics:")
    print(f"  Test Accuracy:        {gpo_results['mean_accuracy']:.4f} ± {gpo_results['std_accuracy']:.4f}")
    print(f"  High Conf Accuracy:   {gpo_results['mean_acc_strong_conf']:.4f}")
    print(f"  Low Conf Accuracy:    {gpo_results['mean_acc_weak_conf']:.4f}")
    print(f"\nProbabilistic Metrics:")
    print(f"  KL Divergence:        {gpo_results['mean_kl_div']:.4f} ± {gpo_results['std_kl_div']:.4f}")
    print(f"  NLL:                  {gpo_results['mean_nll']:.4f} ± {gpo_results['std_nll']:.4f}")
    print(f"  Brier Score:          {gpo_results['mean_brier']:.4f} ± {gpo_results['std_brier']:.4f}")
    print(f"  ECE:                  {gpo_results['mean_ece']:.4f} ± {gpo_results['std_ece']:.4f}")
    print(f"\nConfidence & Diversity:")
    print(f"  Avg Confidence:       {gpo_results['mean_avg_confidence']:.4f}")
    print(f"  Entropy:              {gpo_results['mean_entropy']:.4f}")
    print(f"\nIntransitivity Capture:")
    print(f"  Pred Loop Ratio:      {gpo_results['mean_pred_loop_ratio']:.4f}")
    print(f"  Pred Curl Ratio:      {gpo_results['mean_pred_curl_ratio']:.4f}")
    print(f"  Pred Harmonic Ratio:  {gpo_results['mean_pred_harmonic_ratio']:.4f}")
    
    # Compute gaps (positive means GPO is better)
    acc_gap = gpo_results['mean_accuracy'] - rm_results['mean_accuracy']
    kl_gap = rm_results['mean_kl_div'] - gpo_results['mean_kl_div']  # Lower KL is better
    nll_gap = rm_results['mean_nll'] - gpo_results['mean_nll']  # Lower NLL is better
    brier_gap = rm_results['mean_brier'] - gpo_results['mean_brier']  # Lower Brier is better
    ece_gap = rm_results['mean_ece'] - gpo_results['mean_ece']  # Lower ECE is better
    entropy_gap = gpo_results['mean_entropy'] - rm_results['mean_entropy']  # Higher entropy = less collapsed
    loop_gap = gpo_results['mean_pred_loop_ratio'] - rm_results['mean_pred_loop_ratio']
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY - GPO ADVANTAGES")
    print("="*70)
    print(f"\nAccuracy Metrics (positive = GPO better):")
    print(f"  Test Accuracy:        {acc_gap:+.4f}")
    print(f"\nProbabilistic Metrics (positive = GPO better calibrated):")
    print(f"  KL Divergence:        {kl_gap:+.4f}  (lower KL is better)")
    print(f"  NLL:                  {nll_gap:+.4f}  (lower NLL is better)")
    print(f"  Brier Score:          {brier_gap:+.4f}  (lower Brier is better)")
    print(f"  ECE:                  {ece_gap:+.4f}  (lower ECE is better)")
    print(f"\nDiversity (positive = GPO less collapsed):")
    print(f"  Entropy:              {entropy_gap:+.4f}")
    print(f"\nIntransitivity Capture:")
    print(f"  Pred Loop Ratio Gap:  {loop_gap:+.4f}  (GPO captures more curl)")
    
    # Summary verdict
    print("\n" + "-"*70)
    print("OVERALL VERDICT:")
    print("-"*70)
    wins = sum([
        acc_gap > 0,
        kl_gap > 0,
        nll_gap > 0,
        brier_gap > 0,
        ece_gap > 0,
        entropy_gap > 0,
        loop_gap > 0,
    ])
    total = 7
    print(f"GPO wins on {wins}/{total} metrics")
    
    if acc_gap > 0.05 and kl_gap > 0:
        print("✓ Strong GPO advantage: better accuracy AND calibration")
    elif acc_gap > 0:
        print("✓ Moderate GPO advantage: better accuracy")
    elif acc_gap > -0.01:
        print("≈ Comparable performance")
    else:
        print("⚠ RM performs better (unexpected!)")
    
    # Save results with ALL metrics
    results = {
        'rm': rm_results,
        'gpo': gpo_results,
        'data_metrics': data_metrics,
        'summary': {
            # Accuracy metrics
            'rm_mean_accuracy': rm_results['mean_accuracy'],
            'gpo_mean_accuracy': gpo_results['mean_accuracy'],
            'accuracy_gap': acc_gap,
            
            # Probabilistic metrics
            'rm_mean_kl_div': rm_results['mean_kl_div'],
            'gpo_mean_kl_div': gpo_results['mean_kl_div'],
            'kl_div_gap': kl_gap,
            
            'rm_mean_nll': rm_results['mean_nll'],
            'gpo_mean_nll': gpo_results['mean_nll'],
            'nll_gap': nll_gap,
            
            'rm_mean_brier': rm_results['mean_brier'],
            'gpo_mean_brier': gpo_results['mean_brier'],
            'brier_gap': brier_gap,
            
            'rm_mean_ece': rm_results['mean_ece'],
            'gpo_mean_ece': gpo_results['mean_ece'],
            'ece_gap': ece_gap,
            
            # Confidence metrics
            'rm_mean_avg_confidence': rm_results['mean_avg_confidence'],
            'gpo_mean_avg_confidence': gpo_results['mean_avg_confidence'],
            
            'rm_mean_acc_strong_conf': rm_results['mean_acc_strong_conf'],
            'gpo_mean_acc_strong_conf': gpo_results['mean_acc_strong_conf'],
            
            'rm_mean_acc_weak_conf': rm_results['mean_acc_weak_conf'],
            'gpo_mean_acc_weak_conf': gpo_results['mean_acc_weak_conf'],
            
            # Entropy
            'rm_mean_entropy': rm_results['mean_entropy'],
            'gpo_mean_entropy': gpo_results['mean_entropy'],
            'entropy_gap': entropy_gap,
            
            # Intransitivity capture
            'rm_mean_pred_loop_ratio':     rm_results['mean_pred_loop_ratio'],
            'rm_mean_pred_curl_ratio':     rm_results['mean_pred_curl_ratio'],
            'rm_mean_pred_harmonic_ratio': rm_results['mean_pred_harmonic_ratio'],
            'gpo_mean_pred_loop_ratio':     gpo_results['mean_pred_loop_ratio'],
            'gpo_mean_pred_curl_ratio':     gpo_results['mean_pred_curl_ratio'],
            'gpo_mean_pred_harmonic_ratio': gpo_results['mean_pred_harmonic_ratio'],
            'pred_loop_ratio_gap': loop_gap,
            
            # Data statistics
            'n_prompts_evaluated': rm_results['n_prompts_evaluated'],
            'mean_data_loop_ratio': float(np.mean([m['loop_ratio'] for m in data_metrics.values()])) if data_metrics else float('nan'),
        }
    }
    
    # Save to JSON (convert numpy types)
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results_clean = convert_numpy(results)
    
    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump(results_clean, f, indent=2)
    print(f"\nSaved results to {output_dir / 'evaluation_results.json'}")
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    plot_comparison(rm_results, gpo_results, data_metrics, output_dir)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

