"""
Parse OpenAI Batch API results for epistemic strategy labeling.

Reads batch output JSONL files, extracts labels, merges with features.jsonl,
and reports class distribution and imbalance metrics.

Usage:
    python parse_gpt_results.py \
        --results_dir . \
        --features features.jsonl \
        --output features_labeled.jsonl
"""

import argparse
import json
import math
from pathlib import Path

VALID_STRATEGY_IDS = {1, 2, 3, 4, 5, 6, 7}
STRATEGY_NAMES = {
    1: "genuinely_helpful",
    2: "sycophantic",
    3: "superficially_polished",
    4: "assertive_confabulation",
    5: "hedging_overcautious",
    6: "evasive_deflective",
    7: "concise_substantive",
}
SUBSTANCE_SURFACE_MAP = {
    1: "substance",
    2: "surface",
    3: "surface",
    4: "surface",
    5: "neither",
    6: "neither",
    7: "substance",
}


def parse_batch_output(results_dir: str) -> dict[str, dict]:
    """Parse all batch_output_*.jsonl files and return {custom_id: parsed_label}."""
    results_dir = Path(results_dir)
    output_files = sorted(results_dir.glob("batch_output_*.jsonl"))

    if not output_files:
        print(f"No batch_output_*.jsonl files found in {results_dir}")
        return {}

    labels = {}
    n_total = 0
    n_parsed = 0
    n_parse_errors = 0
    n_validation_errors = 0
    errors = []

    for fpath in output_files:
        with open(fpath) as f:
            for line_num, line in enumerate(f, 1):
                n_total += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    n_parse_errors += 1
                    errors.append(f"{fpath.name}:{line_num}: JSON decode error")
                    continue

                custom_id = record.get("custom_id", "")

                # Check for API errors
                if record.get("error"):
                    n_parse_errors += 1
                    errors.append(f"{custom_id}: API error: {record['error']}")
                    continue

                # Extract the GPT response content
                try:
                    content = record["response"]["body"]["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    n_parse_errors += 1
                    errors.append(f"{custom_id}: Failed to extract/parse content: {e}")
                    continue

                # Validate required fields
                strategy_id = parsed.get("strategy_id")
                strategy_name = parsed.get("strategy_name")
                chain_of_thought = parsed.get("chain_of_thought", "")
                substance_or_surface = parsed.get("substance_or_surface")

                if strategy_id not in VALID_STRATEGY_IDS:
                    n_validation_errors += 1
                    errors.append(f"{custom_id}: Invalid strategy_id={strategy_id}")
                    # Try to recover: if strategy_name is valid, infer strategy_id
                    name_to_id = {v: k for k, v in STRATEGY_NAMES.items()}
                    if strategy_name in name_to_id:
                        strategy_id = name_to_id[strategy_name]
                    else:
                        continue

                # Fix inconsistencies between strategy_id and derived fields
                expected_name = STRATEGY_NAMES[strategy_id]
                if strategy_name != expected_name:
                    strategy_name = expected_name

                expected_ss = SUBSTANCE_SURFACE_MAP[strategy_id]
                if substance_or_surface != expected_ss:
                    substance_or_surface = expected_ss

                labels[custom_id] = {
                    "strategy_id": strategy_id,
                    "strategy_name": strategy_name,
                    "substance_or_surface": substance_or_surface,
                    "chain_of_thought": chain_of_thought,
                }
                n_parsed += 1

    print(f"\n=== Parse Summary ===")
    print(f"Total records: {n_total}")
    print(f"Successfully parsed: {n_parsed} ({n_parsed/max(1,n_total)*100:.1f}%)")
    print(f"Parse errors: {n_parse_errors}")
    print(f"Validation errors (recovered): {n_validation_errors}")

    if errors:
        print(f"\nFirst 10 errors:")
        for e in errors[:10]:
            print(f"  {e}")

    return labels


def merge_labels(features_path: str, labels: dict[str, dict], output_path: str):
    """Merge parsed labels into features.jsonl entries."""
    features = []
    with open(features_path) as f:
        for line in f:
            features.append(json.loads(line))

    n_matched = 0
    n_unmatched = 0
    labeled_entries = []

    for i, entry in enumerate(features):
        custom_id = f"idx_{i}"
        if custom_id in labels:
            entry.update(labels[custom_id])
            n_matched += 1
        else:
            n_unmatched += 1
        labeled_entries.append(entry)

    # Check for labels that don't match any feature entry
    feature_ids = {f"idx_{i}" for i in range(len(features))}
    orphan_ids = set(labels.keys()) - feature_ids
    if orphan_ids:
        print(f"\nWARNING: {len(orphan_ids)} label IDs don't match any feature entry")

    with open(output_path, "w") as f:
        for entry in labeled_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\n=== Merge Summary ===")
    print(f"Total features: {len(features)}")
    print(f"Matched labels: {n_matched}")
    print(f"Unmatched (no label): {n_unmatched}")
    print(f"Output written to: {output_path}")

    return labeled_entries


def report_class_distribution(labeled_entries: list[dict]):
    """Report label distribution and imbalance metrics."""
    # Filter to only labeled entries
    labeled = [e for e in labeled_entries if "strategy_id" in e]
    if not labeled:
        print("\nNo labeled entries found.")
        return

    print(f"\n=== Class Distribution ({len(labeled)} labeled entries) ===")

    # Overall distribution
    for split_name in ["all", "seen", "unseen"]:
        if split_name == "all":
            subset = labeled
        else:
            subset = [e for e in labeled if e["split"] == split_name]

        if not subset:
            continue

        print(f"\n--- {split_name} (n={len(subset)}) ---")

        # Strategy counts
        counts = {}
        for e in subset:
            sid = e["strategy_id"]
            counts[sid] = counts.get(sid, 0) + 1

        print(f"{'ID':>3} {'Strategy':<28} {'Count':>6} {'Pct':>7}")
        print("-" * 48)
        for sid in sorted(counts.keys()):
            name = STRATEGY_NAMES.get(sid, "unknown")
            cnt = counts[sid]
            pct = cnt / len(subset) * 100
            print(f"{sid:>3} {name:<28} {cnt:>6} {pct:>6.1f}%")

        # Substance / Surface / Neither
        ss_counts = {}
        for e in subset:
            ss = e.get("substance_or_surface", "unknown")
            ss_counts[ss] = ss_counts.get(ss, 0) + 1

        print(f"\n{'Category':<12} {'Count':>6} {'Pct':>7}")
        print("-" * 28)
        for cat in ["substance", "surface", "neither"]:
            cnt = ss_counts.get(cat, 0)
            pct = cnt / len(subset) * 100
            print(f"{cat:<12} {cnt:>6} {pct:>6.1f}%")

        # Imbalance metrics
        n_classes = len(VALID_STRATEGY_IDS)
        observed = [counts.get(sid, 0) for sid in range(1, n_classes + 1)]
        total = sum(observed)
        expected = total / n_classes

        # Chi-squared vs uniform
        chi2 = sum((o - expected) ** 2 / expected for o in observed)
        # Degrees of freedom = n_classes - 1
        print(f"\nChi-squared vs uniform: {chi2:.1f} (df={n_classes - 1})")

        # Max/min ratio
        max_count = max(observed) if observed else 0
        min_count = min(o for o in observed if o > 0) if any(o > 0 for o in observed) else 0
        if min_count > 0:
            print(f"Max/min class ratio: {max_count / min_count:.1f}x")
        else:
            print(f"Max/min class ratio: inf (empty classes exist)")

        # Entropy
        probs = [o / total for o in observed if o > 0]
        entropy = -sum(p * math.log2(p) for p in probs)
        max_entropy = math.log2(n_classes)
        print(f"Entropy: {entropy:.3f} / {max_entropy:.3f} (normalized: {entropy/max_entropy:.3f})")

        # Flag rare classes
        for sid in range(1, n_classes + 1):
            cnt = counts.get(sid, 0)
            if cnt / total < 0.02:
                print(f"WARNING: Strategy {sid} ({STRATEGY_NAMES[sid]}) has <2% of responses ({cnt})")


def main():
    parser = argparse.ArgumentParser(description="Parse GPT batch results and merge with features")
    parser.add_argument("--results_dir", required=True, help="Directory with batch_output_*.jsonl files")
    parser.add_argument("--features", required=True, help="Path to features.jsonl")
    parser.add_argument("--output", required=True, help="Path to write features_labeled.jsonl")
    args = parser.parse_args()

    labels = parse_batch_output(args.results_dir)
    if not labels:
        print("No labels parsed. Exiting.")
        return

    labeled_entries = merge_labels(args.features, labels, args.output)
    report_class_distribution(labeled_entries)


if __name__ == "__main__":
    main()
