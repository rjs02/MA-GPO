#!/usr/bin/env python3
"""
Quick test to debug parsing.
"""

import sys
sys.path.insert(0, '/home/robin/repos/OpenRLHF')

from datasets import load_dataset
import re

def parse_conversation(text: str):
    """Parse the hh-rlhf conversation format."""
    messages = []

    # Find all H: and A: markers with their positions
    pattern = r'\n\n([HA]):\s*'
    matches = list(re.finditer(pattern, text))

    if not matches:
        print("NO MATCHES FOUND!")
        print("Text:")
        print(repr(text[:200]))
        return []

    # Extract messages based on the matches
    for i, match in enumerate(matches):
        role_char = match.group(1)
        role = 'user' if role_char == 'H' else 'assistant'

        # Get content from after this marker to before the next marker (or end of text)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        messages.append({
            'role': role,
            'content': content
        })

    return messages


# Load dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:5]")

for i, example in enumerate(dataset):
    print(f"\n{'='*80}")
    print(f"EXAMPLE {i}")
    print('='*80)

    print("\nRAW CHOSEN:")
    print(repr(example['chosen'][:300]))

    print("\nPARSED:")
    parsed = parse_conversation(example['chosen'])
    print(f"Number of messages: {len(parsed)}")
    for j, msg in enumerate(parsed):
        print(f"  {j}: {msg['role']} - {msg['content'][:50]}...")
