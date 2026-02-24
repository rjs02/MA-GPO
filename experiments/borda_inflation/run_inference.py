#!/usr/bin/env python3
"""Run RM/PM inference on preprocessed eval data and save raw outputs.

This is the expensive GPU step. Run once per model checkpoint on the cluster,
then iterate analysis locally with analyze_inflation.py.

Usage:
    python experiments/borda_inflation/run_inference.py \
        --rm_checkpoint /path/to/rm/model_exports \
        --pm_checkpoint /path/to/pm/model_exports \
        --eval_data /path/to/eval_seen.json \
        --output_path /path/to/inference_seen.pkl
"""

import argparse
import json
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from eval.evaluate_trained_models_ultrafeedback import (
    load_reward_model,
    get_response_reward,
    compute_preference_matrix_batched,
)


def rm_worker(checkpoint, tau, eval_records, device, output_path):
    """Worker: compute RM scalar rewards + preference matrices."""
    try:
        print(f"[RM worker] Loading model on {device}...")
        model, tokenizer, dim, _ = load_reward_model(checkpoint, tau, device=device)
        assert dim == 1, f"Expected RM dim=1, got {dim}"

        results = []
        for rec in tqdm(eval_records, desc=f"[RM {device}]"):
            prompt = rec["prompt"]
            responses = [r["text"] for r in rec["responses"]]

            # Scalar rewards
            rewards = []
            for resp in responses:
                r = get_response_reward(model, tokenizer, prompt, resp, device)
                rewards.append(r.item())

            # Preference matrix
            pref_matrix = compute_preference_matrix_batched(
                model, tokenizer, responses, prompt, dim, tau, device
            )

            results.append({
                "rm_rewards": np.array(rewards),
                "P_RM": pref_matrix,
            })

        with open(output_path, "wb") as f:
            pickle.dump(results, f)
        print(f"[RM worker] Done. Saved {len(results)} prompts to {output_path}")

    except Exception as e:
        print(f"[RM worker] FATAL ERROR: {e}")
        traceback.print_exc()
        with open(output_path, "wb") as f:
            pickle.dump({"__error__": str(e)}, f)
        raise


def pm_worker(checkpoint, tau, eval_records, device, output_path):
    """Worker: compute PM preference matrices."""
    try:
        print(f"[PM worker] Loading model on {device}...")
        model, tokenizer, dim, _ = load_reward_model(checkpoint, tau, device=device)
        assert dim > 1, f"Expected PM dim>1, got {dim}"

        results = []
        for rec in tqdm(eval_records, desc=f"[PM {device}]"):
            prompt = rec["prompt"]
            responses = [r["text"] for r in rec["responses"]]

            pref_matrix = compute_preference_matrix_batched(
                model, tokenizer, responses, prompt, dim, tau, device
            )

            results.append({
                "P_PM": pref_matrix,
            })

        with open(output_path, "wb") as f:
            pickle.dump(results, f)
        print(f"[PM worker] Done. Saved {len(results)} prompts to {output_path}")

    except Exception as e:
        print(f"[PM worker] FATAL ERROR: {e}")
        traceback.print_exc()
        with open(output_path, "wb") as f:
            pickle.dump({"__error__": str(e)}, f)
        raise


def main():
    parser = argparse.ArgumentParser(description="Run RM/PM inference on eval data")
    parser.add_argument("--rm_checkpoint", type=str, required=True,
                        help="Path to trained RM (dim=1) model_exports dir")
    parser.add_argument("--pm_checkpoint", type=str, required=True,
                        help="Path to trained PM (dim=8) model_exports dir")
    parser.add_argument("--eval_data", type=str, required=True,
                        help="Path to preprocessed eval JSON")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to save inference results (pickle)")
    parser.add_argument("--rm_tau", type=float, default=1.0,
                        help="Temperature for RM (default: 1.0)")
    parser.add_argument("--pm_tau", type=float, default=0.1,
                        help="Temperature for PM (default: 0.1)")
    parser.add_argument("--rm_device", type=str, default="cuda:0",
                        help="GPU device for RM")
    parser.add_argument("--pm_device", type=str, default="cuda:1",
                        help="GPU device for PM")
    args = parser.parse_args()

    # Load eval data
    print(f"Loading eval data from {args.eval_data}...")
    with open(args.eval_data) as f:
        eval_records = json.load(f)
    print(f"  {len(eval_records)} prompts, "
          f"{sum(r['n_responses'] for r in eval_records)} total responses")

    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Launch parallel inference
    mp.set_start_method("spawn", force=True)

    rm_tmp = str(Path(args.output_path).with_suffix(".rm_tmp.pkl"))
    pm_tmp = str(Path(args.output_path).with_suffix(".pm_tmp.pkl"))

    print("=" * 60)
    print(f"LAUNCHING PARALLEL INFERENCE: RM on {args.rm_device}, PM on {args.pm_device}")
    print("=" * 60)

    rm_proc = mp.Process(
        target=rm_worker,
        args=(args.rm_checkpoint, args.rm_tau, eval_records, args.rm_device, rm_tmp),
        name="RM-worker",
    )
    pm_proc = mp.Process(
        target=pm_worker,
        args=(args.pm_checkpoint, args.pm_tau, eval_records, args.pm_device, pm_tmp),
        name="PM-worker",
    )

    rm_proc.start()
    pm_proc.start()
    print(f"  RM worker PID: {rm_proc.pid}")
    print(f"  PM worker PID: {pm_proc.pid}")

    rm_proc.join()
    pm_proc.join()

    if rm_proc.exitcode != 0:
        print(f"FATAL: RM worker exited with code {rm_proc.exitcode}")
        sys.exit(1)
    if pm_proc.exitcode != 0:
        print(f"FATAL: PM worker exited with code {pm_proc.exitcode}")
        sys.exit(1)

    print("Both workers completed successfully.")

    # Load and merge
    print("Merging results...")
    with open(rm_tmp, "rb") as f:
        rm_results = pickle.load(f)
    with open(pm_tmp, "rb") as f:
        pm_results = pickle.load(f)

    if isinstance(rm_results, dict) and "__error__" in rm_results:
        print(f"FATAL: RM worker error: {rm_results['__error__']}")
        sys.exit(1)
    if isinstance(pm_results, dict) and "__error__" in pm_results:
        print(f"FATAL: PM worker error: {pm_results['__error__']}")
        sys.exit(1)

    assert len(rm_results) == len(pm_results) == len(eval_records), \
        f"Result count mismatch: RM={len(rm_results)}, PM={len(pm_results)}, eval={len(eval_records)}"

    # Merge into single list
    merged = []
    for i in range(len(eval_records)):
        merged.append({
            "prompt": eval_records[i]["prompt"],
            "rm_rewards": rm_results[i]["rm_rewards"],
            "P_RM": rm_results[i]["P_RM"],
            "P_PM": pm_results[i]["P_PM"],
        })

    output = {
        "prompts": merged,
        "meta": {
            "rm_checkpoint": args.rm_checkpoint,
            "pm_checkpoint": args.pm_checkpoint,
            "rm_tau": args.rm_tau,
            "pm_tau": args.pm_tau,
            "n_prompts": len(merged),
            "eval_data": args.eval_data,
        },
    }

    with open(args.output_path, "wb") as f:
        pickle.dump(output, f)
    print(f"Saved {len(merged)} prompts to {args.output_path}")

    # Clean up temp files
    Path(rm_tmp).unlink(missing_ok=True)
    Path(pm_tmp).unlink(missing_ok=True)

    # Quick diagnostics
    print("\n" + "=" * 60)
    print("Diagnostics")
    print("=" * 60)
    all_rm = np.concatenate([m["rm_rewards"] for m in merged])
    print(f"  RM rewards: mean={all_rm.mean():.4f} std={all_rm.std():.4f} "
          f"min={all_rm.min():.4f} max={all_rm.max():.4f}")

    pm_offdiag_vals = []
    for m in merged:
        P = m["P_PM"]
        K = P.shape[0]
        mask = ~np.eye(K, dtype=bool)
        pm_offdiag_vals.extend(P[mask].tolist())
    pm_offdiag_vals = np.array(pm_offdiag_vals)
    print(f"  PM P[i,j] (off-diag): mean={pm_offdiag_vals.mean():.4f} "
          f"std={pm_offdiag_vals.std():.4f} "
          f"min={pm_offdiag_vals.min():.4f} max={pm_offdiag_vals.max():.4f}")
    if pm_offdiag_vals.std() < 0.05:
        print("  WARNING: PM probabilities have very low variance — model may be near-random!")


if __name__ == "__main__":
    main()
