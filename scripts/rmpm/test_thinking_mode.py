#!/usr/bin/env python3
"""Test script to verify enable_thinking parameter behavior."""

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    print(f"Transformers version: {__import__('transformers').__version__}")
    
    # Test message
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."}
    ]
    
    print("\n" + "=" * 60)
    print("Test 1: enable_thinking=True, add_generation_prompt=False")
    text1 = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True
    )
    print(text1)
    print("\nContains '<think>': ", '<think>' in text1)
    
    print("\n" + "=" * 60)
    print("Test 2: enable_thinking=False, add_generation_prompt=False")
    text2 = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )
    print(text2)
    print("\nContains '<think>': ", '<think>' in text2)
    
    print("\n" + "=" * 60)
    print("Test 3: enable_thinking=True, add_generation_prompt=True")
    messages_prompt = [{"role": "user", "content": "What is 2+2?"}]
    text3 = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    print(text3)
    print("\nContains '<think>': ", '<think>' in text3)
    
    print("\n" + "=" * 60)
    print("Test 4: enable_thinking=False, add_generation_prompt=True")
    text4 = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    print(text4)
    print("\nContains '<think>': ", '<think>' in text4)
    
    print("\n" + "=" * 60)
    print("Test 5: No enable_thinking parameter (default)")
    text5 = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    print(text5)
    print("\nContains '<think>': ", '<think>' in text5)

if __name__ == "__main__":
    main()








