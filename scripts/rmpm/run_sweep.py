import os
import subprocess
import time

# Base configuration
# NUM_GPUS=4
# Using Micro=8 for all to ensure consistency and fit within memory
# BS = Micro(8) * Accum * GPUs(4) = 32 * Accum

experiments = [
    # --- Group 1: Paper Setting (BS=32) ---
    # Paper used BS=32, LR=2e-6 for both 2B and 8B models
    {"MICRO_BATCH_SIZE": "8", "ACCUMULATED_GRADIENT": "1", "LR": "1e-6"},
    {"MICRO_BATCH_SIZE": "8", "ACCUMULATED_GRADIENT": "1", "LR": "2e-6"}, # Matches Paper
    {"MICRO_BATCH_SIZE": "8", "ACCUMULATED_GRADIENT": "1", "LR": "5e-6"},

    # --- Group 2: Medium Batch (BS=64) ---
    # Slightly larger batch, scaling LR up slightly
    {"MICRO_BATCH_SIZE": "8", "ACCUMULATED_GRADIENT": "2", "LR": "2e-6"},
    {"MICRO_BATCH_SIZE": "8", "ACCUMULATED_GRADIENT": "2", "LR": "5e-6"},
    {"MICRO_BATCH_SIZE": "8", "ACCUMULATED_GRADIENT": "2", "LR": "1e-5"},

    # --- Group 3: Large Batch (BS=128) ---
    # Standard large batch training
    {"MICRO_BATCH_SIZE": "8", "ACCUMULATED_GRADIENT": "4", "LR": "5e-6"},
    {"MICRO_BATCH_SIZE": "8", "ACCUMULATED_GRADIENT": "4", "LR": "1e-5"},
    {"MICRO_BATCH_SIZE": "8", "ACCUMULATED_GRADIENT": "4", "LR": "2e-5"}, # High end check
]

script_path = "scripts/rmpm/train_hh_4gpu.sh"

# Ensure log directory exists
os.makedirs("sweep_logs", exist_ok=True)

print(f"Launching {len(experiments)} jobs (aligned with GPO paper settings)...")

for i, exp in enumerate(experiments):
    # Construct env vars and job name
    env_vars = []
    name_parts = ["sweep"]
    
    # Defaults for name construction
    lr = exp.get("LR", "1e-5")
    acc = exp.get("ACCUMULATED_GRADIENT", "2")
    micro = exp.get("MICRO_BATCH_SIZE", "12")
    
    # Calculate effective batch size for name (assuming 4 GPUs)
    bs = int(micro) * int(acc) * 4
    
    name_parts.append(f"bs{bs}")
    name_parts.append(f"lr{lr}")
    
    job_name = "_".join(name_parts)
    
    # Set EXP_NAME to control save path and wandb name
    # Add timestamp to ensure uniqueness
    timestamp = int(time.time())
    exp_name = f"{job_name}_{timestamp}"
    
    env_vars.append(f"EXP_NAME={exp_name}")
    
    # Add all experiment overrides
    for k, v in exp.items():
        env_vars.append(f"{k}={v}")
    
    # Export string for sbatch
    export_str = "ALL," + ",".join(env_vars)
    
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--output=sweep_logs/%x_%j.out",
        f"--export={export_str}",
        script_path
    ]
    
    print(f"Submitting {job_name} (Micro={micro}, Accum={acc})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Success: {result.stdout.strip()}")
    else:
        print(f"  Failed: {result.stderr.strip()}")
    
    # Small sleep to ensure file timestamps/ordering
    time.sleep(1)

print("Done.")
