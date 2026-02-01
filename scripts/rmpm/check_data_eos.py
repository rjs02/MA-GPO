#!/usr/bin/env python3
"""Check if the actual dataset class produces EOS tokens correctly."""

import sys
sys.path.insert(0, '.')

from datasets import load_from_disk
from transformers import AutoTokenizer
from general_preference.datasets import GeneralRewardDataset
import os

DATASET_PATH = f"{os.getenv('LASDIR')}/data/ufb/pref_train"
MODEL = "Qwen/Qwen3-0.6B"
# DATASET_PATH = "./data/ultrafeedback_cleaned_splits/pref_train"

class DummyStrategy:
    def print(self, msg):
        print(msg)
    def is_rank_0(self): return True

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    data = load_from_disk(DATASET_PATH)

    # Use first 100 samples
    data = data.select(range(100))

    strategy = DummyStrategy()

    dataset = GeneralRewardDataset(
        data, tokenizer, max_length=1024, strategy=strategy,
        is_custom=False, return_prompt_length=False, use_separate_prompt=True
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"EOS token id: {tokenizer.eos_token_id}")

    eos_correct = 0
    eos_wrong = 0

    for i in range(min(100, len(dataset))):
        chosen_ids, c_mask, reject_ids, r_mask, margin, resp_len = dataset[i]

        # Check if last token is EOS
        chosen_last = chosen_ids[0, -1].item()
        reject_last = reject_ids[0, -1].item()

        if chosen_last == tokenizer.eos_token_id:
            eos_correct += 1
        else:
            eos_wrong += 1
            if eos_wrong <= 3:  # Print first few errors
                print(f"Sample {i}: chosen last token = {chosen_last}, expected {tokenizer.eos_token_id}")

        if reject_last != tokenizer.eos_token_id and eos_wrong <= 3:
            print(f"Sample {i}: reject last token = {reject_last}, expected {tokenizer.eos_token_id}")

    print(f"\nEOS at last position: {eos_correct}/100")
    print(f"EOS missing: {eos_wrong}/100")

    if eos_correct == 100:
        print("\n✓ Dataset correctly adds EOS to all sequences!")
    else:
        print(f"\n⚠️ {eos_wrong} sequences missing EOS!")

if __name__ == "__main__":
    main()

