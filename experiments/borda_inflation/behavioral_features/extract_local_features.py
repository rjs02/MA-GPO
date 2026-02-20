#!/usr/bin/env python3
"""Extract local (programmatic) behavioral features from eval responses.

Loads eval JSONs and flat.jsonl files from Stage 1, computes verbosity and
format-exploitation features per response, and outputs a single merged
features.jsonl with all inflation metrics + local features + response text.

Usage:
    python experiments/borda_inflation/behavioral_features/extract_local_features.py \
        --data_dir /path/to/eval_1551803 \
        --output_dir experiments/borda_inflation/behavioral_features
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path


SPLITS = ["seen", "unseen"]


def count_emojis(text):
    """Count emoji codepoints via Unicode category (So = Symbol, other)."""
    count = 0
    for ch in text:
        if unicodedata.category(ch) == "So":
            count += 1
        elif "\U0001F600" <= ch <= "\U0001FAFF":
            count += 1
    return count


def extract_features(text):
    """Compute all local features from a response string."""
    lines = text.split("\n")

    word_count = len(text.split())
    char_count = len(text)
    sentences = re.split(r"[.!?](?:\s|$)", text)
    sentence_count = max(1, len([s for s in sentences if s.strip()]))
    words_per_sentence = word_count / sentence_count

    n_bold = len(re.findall(r"\*\*[^*]+\*\*", text))
    n_bullet_items = sum(1 for line in lines if re.match(r"\s*[-*]\s", line))
    n_numbered_items = sum(1 for line in lines if re.match(r"\s*\d+\.\s", line))
    n_headers = sum(1 for line in lines if re.match(r"#{1,6}\s", line))
    n_code_blocks = len(re.findall(r"```", text)) // 2
    n_tables = sum(1 for line in lines if re.match(r"\s*\|.+\|.+\|", line))
    n_emojis = count_emojis(text)

    fmt_markers = n_bold + n_bullet_items + n_numbered_items + n_headers + n_code_blocks
    formatting_density = fmt_markers / max(1, word_count)

    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "words_per_sentence": round(words_per_sentence, 2),
        "n_bold": n_bold,
        "n_bullet_items": n_bullet_items,
        "n_numbered_items": n_numbered_items,
        "n_headers": n_headers,
        "n_code_blocks": n_code_blocks,
        "n_tables": n_tables,
        "n_emojis": n_emojis,
        "formatting_density": round(formatting_density, 6),
    }


def load_flat_jsonl(path):
    """Load flat.jsonl into a dict keyed by (prompt, response_idx)."""
    lookup = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["prompt"], rec["response_idx"])
            lookup[key] = rec
    return lookup


def main():
    parser = argparse.ArgumentParser(description="Extract local behavioral features")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root dir of eval run (contains eval_data/, analysis_*/)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to write features.jsonl")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_entries = []

    for split in SPLITS:
        eval_path = data_dir / "eval_data" / f"eval_{split}.json"
        flat_path = data_dir / f"analysis_{split}" / "flat.jsonl"

        print(f"Loading {split} data...")
        with open(eval_path) as f:
            eval_records = json.load(f)
        flat_lookup = load_flat_jsonl(flat_path)
        print(f"  {len(eval_records)} prompts, {len(flat_lookup)} flat entries")

        matched = 0
        for rec in eval_records:
            prompt = rec["prompt"]
            for i, resp in enumerate(rec["responses"]):
                key = (prompt, i)
                flat_rec = flat_lookup.get(key)
                if flat_rec is None:
                    print(f"  WARNING: no flat entry for response_idx={i}, prompt={prompt[:60]}...")
                    continue

                features = extract_features(resp["text"])

                assert features["word_count"] == flat_rec["length"], (
                    f"Word count mismatch: {features['word_count']} vs {flat_rec['length']}"
                )

                entry = {
                    "split": split,
                    "prompt": prompt,
                    "response_idx": i,
                    "response_text": resp["text"],
                }
                for k, v in flat_rec.items():
                    if k not in ("prompt", "response_idx", "length"):
                        entry[k] = v
                entry.update(features)

                all_entries.append(entry)
                matched += 1

        print(f"  Matched {matched} responses for {split}")

    out_path = output_dir / "features.jsonl"
    with open(out_path, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nWrote {len(all_entries)} entries to {out_path}")

    word_counts = [e["word_count"] for e in all_entries]
    fmt_densities = [e["formatting_density"] for e in all_entries]
    print(f"  Word count: mean={sum(word_counts)/len(word_counts):.1f}, "
          f"min={min(word_counts)}, max={max(word_counts)}")
    print(f"  Formatting density: mean={sum(fmt_densities)/len(fmt_densities):.4f}, "
          f"max={max(fmt_densities):.4f}")


if __name__ == "__main__":
    main()
