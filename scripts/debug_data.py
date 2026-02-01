#!/usr/bin/env python3
"""Debug script to verify data format and tokenization."""

from datasets import load_from_disk
from transformers import AutoTokenizer

DATASET_PATH = "./data/ultrafeedback_cleaned_splits/pref_train"
MODEL = "Qwen/Qwen3-0.6B"

def main():
    print("=" * 60)
    print("Loading dataset...")
    data = load_from_disk(DATASET_PATH)
    print(f"Dataset size: {len(data)}")

    print("\n" + "=" * 60)
    print("Sample data point:")
    sample = data[0]
    print(f"Keys: {list(sample.keys())}")
    print(f"\nprompt: {sample['prompt']}")
    print(f"\nchosen: {sample['chosen']}")
    print(f"\nrejected: {sample['rejected']}")
    print(f"\nmargin: {sample['margin']}")
    print(f"dimension: {sample.get('dimension', 'N/A')}")

    print("\n" + "=" * 60)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    print(f"Chat template exists: {tokenizer.chat_template is not None}")
    print(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")

    print("\n" + "=" * 60)
    print("Simulating dataloader preprocessing (use_separate_prompt=True):")

    # Simulate preprocess_data with use_separate_prompt=True
    prompt = sample["prompt"]  # [{"role": "user", "content": ...}]
    chosen_response = sample["chosen"]  # [{"role": "assistant", "content": ...}]
    rejected_response = sample["rejected"]

    # Combine into full conversation
    chosen_conv = prompt + chosen_response
    rejected_conv = prompt + rejected_response

    print(f"\nCombined chosen conversation:")
    for msg in chosen_conv:
        print(f"  {msg['role']}: {msg['content'][:100]}...")

    print("\n" + "=" * 60)
    print("Applying chat template (with enable_thinking=False):")

    chosen_text = tokenizer.apply_chat_template(
        chosen_conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    rejected_text = tokenizer.apply_chat_template(
        rejected_conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )

    print(f"\nChosen text (first 500 chars):\n{chosen_text[:500]}")
    print(f"\nRejected text (first 500 chars):\n{rejected_text[:500]}")

    print("\n" + "=" * 60)
    print("Tokenizing:")

    chosen_tokens = tokenizer(chosen_text, max_length=2048, truncation=True, return_tensors="pt")
    rejected_tokens = tokenizer(rejected_text, max_length=2048, truncation=True, return_tensors="pt")

    print(f"Chosen length: {chosen_tokens['input_ids'].shape[1]} tokens")
    print(f"Rejected length: {rejected_tokens['input_ids'].shape[1]} tokens")

    # Check if EOS is at the end
    print(f"\nChosen ends with EOS: {chosen_tokens['input_ids'][0, -1].item() == tokenizer.eos_token_id}")
    print(f"Rejected ends with EOS: {rejected_tokens['input_ids'][0, -1].item() == tokenizer.eos_token_id}")

    print("\n" + "=" * 60)
    print("Statistics across first 100 samples:")

    chosen_lengths = []
    rejected_lengths = []
    margins = []

    for i in range(min(100, len(data))):
        sample = data[i]
        prompt = sample["prompt"]
        chosen_conv = prompt + sample["chosen"]
        rejected_conv = prompt + sample["rejected"]

        chosen_text = tokenizer.apply_chat_template(
            chosen_conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        rejected_text = tokenizer.apply_chat_template(
            rejected_conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )

        chosen_toks = tokenizer(chosen_text, max_length=2048, truncation=True)
        rejected_toks = tokenizer(rejected_text, max_length=2048, truncation=True)

        chosen_lengths.append(len(chosen_toks['input_ids']))
        rejected_lengths.append(len(rejected_toks['input_ids']))
        margins.append(sample['margin'])

    print(f"Chosen lengths:   min={min(chosen_lengths)}, max={max(chosen_lengths)}, mean={sum(chosen_lengths)/len(chosen_lengths):.1f}")
    print(f"Rejected lengths: min={min(rejected_lengths)}, max={max(rejected_lengths)}, mean={sum(rejected_lengths)/len(rejected_lengths):.1f}")
    print(f"Margins:          min={min(margins)}, max={max(margins)}, mean={sum(margins)/len(margins):.2f}")

    # Check for dimension distribution
    dims = [data[i].get('dimension', 'N/A') for i in range(min(1000, len(data)))]
    from collections import Counter
    print(f"\nDimension distribution (first 1000):")
    for dim, count in sorted(Counter(dims).items()):
        print(f"  {dim}: {count}")

if __name__ == "__main__":
    main()
