#!/usr/bin/env python3
"""Check how many sequences are truncated at different max_len settings."""

from datasets import load_from_disk
from transformers import AutoTokenizer
import numpy as np
import os

# DATASET_PATH = "./data/ultrafeedback_cleaned_splits/pref_train"
DATASET_PATH = f"{os.getenv('LASDIR')}/data/ufb/pref_train"
MODEL = "Qwen/Qwen3-0.6B"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    data = load_from_disk(DATASET_PATH)

    # Check first 500 samples
    num_samples = min(500, len(data))

    chosen_lens = []
    rejected_lens = []

    for i in range(num_samples):
        sample = data[i]
        prompt = sample['prompt']
        chosen_conv = prompt + sample['chosen']
        rejected_conv = prompt + sample['rejected']

        chosen_text = tokenizer.apply_chat_template(
            chosen_conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        rejected_text = tokenizer.apply_chat_template(
            rejected_conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )

        # Tokenize WITHOUT truncation to get true length
        chosen_toks = tokenizer(chosen_text, truncation=False)
        rejected_toks = tokenizer(rejected_text, truncation=False)

        chosen_lens.append(len(chosen_toks['input_ids']))
        rejected_lens.append(len(rejected_toks['input_ids']))

    all_lens = chosen_lens + rejected_lens

    print(f"Token length statistics (n={num_samples*2}):")
    print(f"  min={min(all_lens)}, max={max(all_lens)}, mean={np.mean(all_lens):.1f}")
    print(f"  median={np.median(all_lens):.1f}")
    print(f"  95th percentile={np.percentile(all_lens, 95):.1f}")

    for max_len in [512, 1024, 2048, 4096]:
        truncated = sum(l > max_len for l in all_lens)
        print(f"\n  At max_len={max_len}: {truncated}/{len(all_lens)} truncated ({100*truncated/len(all_lens):.1f}%)")

    # Check if EOS is preserved
    print(f"\n\nEOS token analysis (at max_len=1024):")
    eos_id = tokenizer.eos_token_id

    eos_preserved = 0
    eos_lost = 0

    for i in range(min(100, num_samples)):
        sample = data[i]
        prompt = sample['prompt']
        chosen_conv = prompt + sample['chosen']

        chosen_text = tokenizer.apply_chat_template(
            chosen_conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )

        # Check with truncation
        toks = tokenizer(chosen_text, max_length=1024, truncation=True)
        if toks['input_ids'][-1] == eos_id:
            eos_preserved += 1
        else:
            eos_lost += 1

    print(f"  EOS preserved: {eos_preserved}/100")
    print(f"  EOS lost (truncated): {eos_lost}/100")

    if eos_lost > 0:
        print(f"\n⚠️  WARNING: {eos_lost}% of sequences lose their EOS token when truncated!")
        print("   This could cause the model to extract rewards from wrong positions!")

if __name__ == "__main__":
    main()
