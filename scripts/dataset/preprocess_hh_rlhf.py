#!/usr/bin/env python3
"""
Preprocess the Anthropic/hh-rlhf dataset for GPM training.

This script:
1. Loads the hh-rlhf dataset from HuggingFace
2. Parses the conversation format (H:/A: turns)
3. Applies the Qwen3 chat template
4. Creates the format needed for GPM training: prompt, chosen, rejected
5. Saves the processed dataset in HuggingFace datasets format
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
sys.path.insert(0, '/home/robin/repos/OpenRLHF')

from datasets import load_dataset, Dataset, DatasetDict
import json


def parse_conversation(text: str) -> List[Dict[str, str]]:
    """
    Parse the hh-rlhf conversation format into a list of messages.

    Args:
        text: Raw conversation text with '\n\nHuman:' and '\n\nAssistant:' markers

    Returns:
        List of messages with 'role' and 'content' keys
    """
    import re

    messages = []

    # The format uses \n\nHuman: and \n\nAssistant: as separators
    # Find all Human: and Assistant: markers with their positions
    pattern = r'\n\n(Human|Assistant):\s*'
    matches = list(re.finditer(pattern, text))

    if not matches:
        # No matches found - might be a malformed example
        return []

    # Extract messages based on the matches
    for i, match in enumerate(matches):
        role_text = match.group(1)
        role = 'user' if role_text == 'Human' else 'assistant'

        # Get content from after this marker to before the next marker (or end of text)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        messages.append({
            'role': role,
            'content': content
        })

    return messages


def split_prompt_and_response(messages: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
    """
    Split messages into prompt (all messages except last) and final response.

    Args:
        messages: List of messages

    Returns:
        Tuple of (prompt_messages, final_response)
    """
    if len(messages) == 0:
        return [], ""

    # The last message should be the assistant's response
    if messages[-1]['role'] != 'assistant':
        raise ValueError("Last message should be from assistant")

    prompt_messages = messages[:-1]
    final_response = messages[-1]['content']

    return prompt_messages, final_response


def process_example(example: Dict) -> Dict:
    """
    Process a single hh-rlhf example into GPM format.

    Args:
        example: Raw example with 'chosen' and 'rejected' fields

    Returns:
        Processed example with 'prompt', 'chosen', 'rejected' fields as message lists
    """
    # Parse both conversations
    chosen_messages = parse_conversation(example['chosen'])
    rejected_messages = parse_conversation(example['rejected'])

    # Split into prompt and responses
    prompt_messages, _ = split_prompt_and_response(chosen_messages)
    _, rejected_response = split_prompt_and_response(rejected_messages)

    # Extract just the final assistant response from chosen and rejected
    chosen_response_msg = chosen_messages[-1]  # Last message is the assistant response
    rejected_response_msg = rejected_messages[-1]

    # Return in the same format as ultrafeedback: lists of message dicts
    # The training script will apply the chat template at runtime
    return {
        'prompt': prompt_messages,  # List of messages up to (not including) final response
        'chosen': [chosen_response_msg],  # List with just the final assistant response
        'rejected': [rejected_response_msg],  # List with just the final assistant response
        'margin': 0  # We don't have explicit margins in hh-rlhf
    }


def main():
    parser = argparse.ArgumentParser(description='Preprocess hh-rlhf dataset for GPM training')
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='(Deprecated - no longer needed) Tokenizer is applied at training time'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory to save processed dataset'
    )
    parser.add_argument(
        '--subset',
        type=str,
        default=None,
        help='Specific subset to load (e.g., "harmless-base", "helpful-base"). If None, loads all.'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (for testing)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Preprocessing Anthropic/hh-rlhf dataset")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Subset: {args.subset if args.subset else 'all'}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'all'}")
    print()
    print("NOTE: Data will be stored as message lists (role/content dicts).")
    print("      The chat template will be applied at training time by the model's tokenizer.")
    print()

    # Load dataset
    print("Loading dataset...")
    if args.subset:
        dataset = load_dataset("Anthropic/hh-rlhf", data_dir=args.subset)
    else:
        dataset = load_dataset("Anthropic/hh-rlhf")

    print(f"Dataset loaded. Splits: {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} examples")
    print()

    # Process dataset
    print("Processing dataset...")
    processed_splits = {}

    for split_name, split_data in dataset.items():
        print(f"\nProcessing {split_name} split...")

        # Limit samples if specified
        if args.max_samples:
            split_data = split_data.select(range(min(args.max_samples, len(split_data))))
            print(f"  Limited to {len(split_data)} samples")

        # Process examples
        processed_examples = []
        errors = 0

        for i, example in enumerate(split_data):
            try:
                processed = process_example(example)
                processed_examples.append(processed)

                # Print progress
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1}/{len(split_data)} examples...")

            except Exception as e:
                errors += 1
                if errors <= 5:  # Only print first 5 errors
                    print(f"  Error processing example {i}: {e}")
                    import traceback
                    traceback.print_exc()

        print(f"  Completed: {len(processed_examples)} examples processed, {errors} errors")

        # Create dataset from processed examples
        processed_splits[split_name] = Dataset.from_list(processed_examples)

    # Create DatasetDict
    processed_dataset = DatasetDict(processed_splits)

    # Save dataset
    print(f"\nSaving processed dataset to {args.output_dir}...")
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processed_dataset.save_to_disk(str(output_path))
    print(f"Dataset saved successfully!")
    print()

    # Print example
    print("=" * 80)
    print("EXAMPLE FROM PROCESSED DATASET")
    print("=" * 80)
    example = processed_dataset['train'][0]
    print("\n--- PROMPT (message list) ---")
    print(json.dumps(example['prompt'], indent=2))
    print("\n--- CHOSEN (message list) ---")
    print(json.dumps(example['chosen'], indent=2))
    print("\n--- REJECTED (message list) ---")
    print(json.dumps(example['rejected'], indent=2))
    print(f"\n--- MARGIN ---")
    print(example.get('margin', 'N/A'))
    print()

    # Print statistics
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    for split_name, split_data in processed_dataset.items():
        print(f"\n{split_name}:")
        print(f"  Number of examples: {len(split_data)}")

        # Average number of messages
        avg_prompt_msgs = sum(len(ex['prompt']) for ex in split_data) / len(split_data)
        avg_chosen_msgs = sum(len(ex['chosen']) for ex in split_data) / len(split_data)
        avg_rejected_msgs = sum(len(ex['rejected']) for ex in split_data) / len(split_data)

        print(f"  Average prompt messages: {avg_prompt_msgs:.1f}")
        print(f"  Average chosen messages: {avg_chosen_msgs:.1f}")
        print(f"  Average rejected messages: {avg_rejected_msgs:.1f}")

        # Average total text length (for reference)
        def total_text_len(msg_list):
            return sum(len(msg['content']) for msg in msg_list)

        avg_prompt_chars = sum(total_text_len(ex['prompt']) for ex in split_data) / len(split_data)
        avg_chosen_chars = sum(total_text_len(ex['chosen']) for ex in split_data) / len(split_data)
        avg_rejected_chars = sum(total_text_len(ex['rejected']) for ex in split_data) / len(split_data)

        print(f"  Average prompt chars: {avg_prompt_chars:.0f}")
        print(f"  Average chosen chars: {avg_chosen_chars:.0f}")
        print(f"  Average rejected chars: {avg_rejected_chars:.0f}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
