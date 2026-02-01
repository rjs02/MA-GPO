#!/usr/bin/env python3
"""
Script to explore the Anthropic/hh-rlhf dataset structure.
"""

import sys
sys.path.insert(0, '/home/robin/repo/OpenRLHF')

from datasets import load_dataset
import json

def explore_dataset():
    """Load and explore the hh-rlhf dataset."""

    print("Loading Anthropic/hh-rlhf dataset...")
    print("=" * 80)

    # Load a small subset first to explore structure
    dataset = load_dataset("Anthropic/hh-rlhf", split="train[:10]")

    print(f"\nDataset size: {len(dataset)}")
    print(f"Dataset features: {dataset.features}")
    print("\n" + "=" * 80)

    # Look at first example
    print("\nFIRST EXAMPLE:")
    print("=" * 80)
    example = dataset[0]

    print("\nKeys:", example.keys())
    print("\n--- CHOSEN ---")
    print(example['chosen'])
    print("\n--- REJECTED ---")
    print(example['rejected'])

    # Analyze the structure
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("=" * 80)

    # Count turns in chosen
    chosen = example['chosen']
    human_turns = chosen.count('\n\nHuman:')
    assistant_turns = chosen.count('\n\nAssistant:')

    print(f"\nNumber of Human turns in chosen: {human_turns}")
    print(f"Number of Assistant turns in chosen: {assistant_turns}")

    # Check if chosen and rejected share the same prompt
    # Find the last assistant response in both
    chosen_parts = chosen.split('\n\nAssistant:')
    rejected_parts = example['rejected'].split('\n\nAssistant:')

    print(f"\nNumber of Assistant turns in rejected: {len(rejected_parts) - 1}")

    # Check if prompts are the same (everything before last assistant turn)
    chosen_prompt = '\n\nAssistant:'.join(chosen_parts[:-1])
    rejected_prompt = '\n\nAssistant:'.join(rejected_parts[:-1])

    print(f"\nPrompts are identical: {chosen_prompt == rejected_prompt}")

    if chosen_prompt == rejected_prompt:
        print("\n--- SHARED PROMPT ---")
        print(chosen_prompt[:500] + "..." if len(chosen_prompt) > 500 else chosen_prompt)
        print("\n--- CHOSEN RESPONSE (last Assistant turn) ---")
        print(chosen_parts[-1][:300] + "..." if len(chosen_parts[-1]) > 300 else chosen_parts[-1])
        print("\n--- REJECTED RESPONSE (last Assistant turn) ---")
        print(rejected_parts[-1][:300] + "..." if len(rejected_parts[-1]) > 300 else rejected_parts[-1])

    # Look at a few more examples
    print("\n" + "=" * 80)
    print("CHECKING MULTIPLE EXAMPLES:")
    print("=" * 80)

    for i in range(min(5, len(dataset))):
        ex = dataset[i]
        chosen_parts = ex['chosen'].split('\n\nAssistant:')
        rejected_parts = ex['rejected'].split('\n\nAssistant:')

        chosen_prompt = '\n\nAssistant:'.join(chosen_parts[:-1])
        rejected_prompt = '\n\nAssistant:'.join(rejected_parts[:-1])

        print(f"Example {i}: Prompts match = {chosen_prompt == rejected_prompt}, "
              f"Chosen length = {len(ex['chosen'])}, Rejected length = {len(ex['rejected'])}")

if __name__ == "__main__":
    explore_dataset()
