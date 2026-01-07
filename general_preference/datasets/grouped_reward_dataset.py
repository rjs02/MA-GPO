"""
Grouped Reward Dataset for GPM's linear complexity optimization.

Instead of processing O(K²) pairwise comparisons with redundant forward passes,
this dataset groups all K responses per prompt together, allowing the trainer
to compute embeddings once per response and reuse them for all comparisons.

Expected data format:
{
    "prompt": [{"role": "user", "content": ...}],
    "responses": [
        [{"role": "assistant", "content": response_1}],
        [{"role": "assistant", "content": response_2}],
        ...
    ],
    "comparisons": [
        {"chosen_idx": 0, "rejected_idx": 1, "margin": 2},
        {"chosen_idx": 0, "rejected_idx": 2, "margin": 1},
        ...
    ]
}
"""

from typing import Callable, List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none, zero_pad_sequences


class GroupedRewardDataset(Dataset):
    """
    Dataset for grouped preference data optimized for GPM.

    Each item contains all responses for a single prompt, plus the list of
    pairwise comparisons. This allows computing embeddings once per response
    and reusing them for all pairwise loss computations.

    Args:
        dataset: HuggingFace dataset with grouped format
        tokenizer: Tokenizer for the model
        max_length: Maximum sequence length
        strategy: Training strategy (for distributed setup)
        max_comparisons_per_entry: Limit comparisons per entry (None = no limit)
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        max_comparisons_per_entry: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.entries = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.max_comparisons_per_entry = max_comparisons_per_entry

        for data in tqdm(dataset, desc="Loading grouped data", disable=not self.strategy.is_rank_0()):
            entry = self._preprocess_entry(data)
            if entry is not None:
                self.entries.append(entry)

        if self.strategy.is_rank_0():
            total_responses = sum(e["num_responses"] for e in self.entries)
            total_comparisons = sum(e["num_comparisons"] for e in self.entries)
            print(f"  Loaded {len(self.entries)} grouped entries")
            print(f"  Total responses: {total_responses}")
            print(f"  Total comparisons: {total_comparisons}")
            if len(self.entries) > 0:
                print(f"  Avg responses/entry: {total_responses / len(self.entries):.2f}")
                print(f"  Avg comparisons/entry: {total_comparisons / len(self.entries):.2f}")

    def _preprocess_entry(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Preprocess a single grouped entry."""
        prompt = data["prompt"]
        responses = data["responses"]
        comparisons = data["comparisons"]

        if not responses or not comparisons:
            return None

        # Optionally limit comparisons
        if self.max_comparisons_per_entry and len(comparisons) > self.max_comparisons_per_entry:
            # Sample comparisons to keep diversity
            import random
            comparisons = random.sample(comparisons, self.max_comparisons_per_entry)

        # Build full conversations for each response
        conversations = []
        for response in responses:
            conv = prompt + response
            conversations.append(conv)

        return {
            "conversations": conversations,  # List of full conversations
            "comparisons": comparisons,       # List of {chosen_idx, rejected_idx, margin, ...}
            "num_responses": len(responses),
            "num_comparisons": len(comparisons),
        }

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a single grouped entry with tokenized responses.

        Returns dict with:
            - response_ids: List of token ID tensors
            - response_masks: List of attention mask tensors
            - comparisons: List of comparison dicts
            - num_responses: int
            - num_comparisons: int
        """
        entry = self.entries[idx]
        conversations = entry["conversations"]
        comparisons = entry["comparisons"]

        # Tokenize each response
        response_ids = []
        response_masks = []

        for conv in conversations:
            if self.tokenizer.chat_template is not None:
                text = self.tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
            else:
                text = " ".join([d["content"] for d in conv]) + " " + self.tokenizer.eos_token

            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )

            # Ensure EOS token is present
            tokens["input_ids"][0][-1] = self.tokenizer.eos_token_id
            tokens["attention_mask"][0][-1] = True

            response_ids.append(tokens["input_ids"])
            response_masks.append(tokens["attention_mask"])

        return {
            "response_ids": response_ids,
            "response_masks": response_masks,
            "comparisons": comparisons,
            "num_responses": entry["num_responses"],
            "num_comparisons": entry["num_comparisons"],
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate function for batching multiple grouped entries.

        Since each entry can have different numbers of responses and comparisons,
        we flatten all responses and track entry boundaries.

        Returns:
            - all_input_ids: Tensor [total_responses, max_seq_len]
            - all_attention_masks: Tensor [total_responses, max_seq_len]
            - entry_boundaries: List of (start_idx, end_idx) for each entry
            - all_comparisons: List of adjusted comparison dicts with global indices
            - margins: Tensor of margins for all comparisons
            - batch_size: Number of entries in batch
        """
        all_input_ids = []
        all_attention_masks = []
        entry_boundaries = []
        all_comparisons = []
        margins = []

        current_idx = 0

        for item in batch:
            response_ids = item["response_ids"]
            response_masks = item["response_masks"]
            comparisons = item["comparisons"]
            num_responses = item["num_responses"]

            start_idx = current_idx

            # Add all responses for this entry
            for rid, rmask in zip(response_ids, response_masks):
                all_input_ids.append(rid)
                all_attention_masks.append(rmask)

            end_idx = current_idx + num_responses
            entry_boundaries.append((start_idx, end_idx))

            # Adjust comparison indices to global indices
            for comp in comparisons:
                adjusted_comp = {
                    "chosen_idx": start_idx + comp["chosen_idx"],
                    "rejected_idx": start_idx + comp["rejected_idx"],
                    "entry_idx": len(entry_boundaries) - 1,  # Track which entry
                }
                all_comparisons.append(adjusted_comp)
                margins.append(comp.get("margin", 0))  # Default to 0 for binary

            current_idx = end_idx

        # Pad sequences
        all_input_ids = zero_pad_sequences(all_input_ids, value=self.tokenizer.pad_token_id)
        all_attention_masks = zero_pad_sequences(all_attention_masks)

        return {
            "all_input_ids": all_input_ids,
            "all_attention_masks": all_attention_masks,
            "entry_boundaries": entry_boundaries,
            "all_comparisons": all_comparisons,
            "margins": torch.tensor(margins, dtype=torch.float),
            "batch_size": len(batch),
            "total_responses": current_idx,
            "total_comparisons": len(all_comparisons),
        }


class GroupedRewardDatasetV2(Dataset):
    """
    Alternative implementation that processes one entry at a time.

    This is simpler and works well when batch_size=1 at the entry level,
    but each entry can have many responses processed together.

    Use this when:
    - You want maximum efficiency (one forward pass per entry)
    - Your entries have many responses (4+ per prompt)
    - You're using gradient accumulation anyway
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        max_comparisons_per_entry: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.entries = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.max_comparisons_per_entry = max_comparisons_per_entry

        for data in tqdm(dataset, desc="Loading grouped data", disable=not self.strategy.is_rank_0()):
            entry = self._preprocess_entry(data)
            if entry is not None:
                self.entries.append(entry)

    def _preprocess_entry(self, data):
        """Same as GroupedRewardDataset._preprocess_entry"""
        prompt = data["prompt"]
        responses = data["responses"]
        comparisons = data["comparisons"]

        if not responses or not comparisons:
            return None

        if self.max_comparisons_per_entry and len(comparisons) > self.max_comparisons_per_entry:
            import random
            comparisons = random.sample(comparisons, self.max_comparisons_per_entry)

        conversations = []
        for response in responses:
            conv = prompt + response
            conversations.append(conv)

        return {
            "conversations": conversations,
            "comparisons": comparisons,
            "num_responses": len(responses),
            "num_comparisons": len(comparisons),
        }

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        conversations = entry["conversations"]
        comparisons = entry["comparisons"]

        # Tokenize all responses
        all_ids = []
        all_masks = []

        for conv in conversations:
            if self.tokenizer.chat_template is not None:
                text = self.tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
            else:
                text = " ".join([d["content"] for d in conv]) + " " + self.tokenizer.eos_token

            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            tokens["input_ids"][0][-1] = self.tokenizer.eos_token_id
            tokens["attention_mask"][0][-1] = True

            # Squeeze out the batch dimension from tokenizer output [1, seq_len] -> [seq_len]
            all_ids.append(tokens["input_ids"].squeeze(0))
            all_masks.append(tokens["attention_mask"].squeeze(0))

        # Pad to same length within this entry - now inputs are [seq_len] each
        # zero_pad_sequences will stack them to [num_responses, seq_len]
        all_ids = zero_pad_sequences(all_ids, value=self.tokenizer.pad_token_id)
        all_masks = zero_pad_sequences(all_masks)

        # Extract comparison info
        chosen_indices = torch.tensor([c["chosen_idx"] for c in comparisons], dtype=torch.long)
        rejected_indices = torch.tensor([c["rejected_idx"] for c in comparisons], dtype=torch.long)
        margins = torch.tensor([c.get("margin", 0) for c in comparisons], dtype=torch.float)

        return {
            "input_ids": all_ids,              # [num_responses, seq_len]
            "attention_mask": all_masks,        # [num_responses, seq_len]
            "chosen_indices": chosen_indices,   # [num_comparisons]
            "rejected_indices": rejected_indices,  # [num_comparisons]
            "margins": margins,                 # [num_comparisons]
            "num_responses": entry["num_responses"],
            "num_comparisons": entry["num_comparisons"],
        }

    def collate_fn(self, batch):
        """
        Collate function that handles variable sequence lengths across entries.

        Each entry has input_ids of shape [num_responses, seq_len], but seq_len
        can vary between entries. We flatten all responses and use zero_pad_sequences.
        """
        # Flatten all response tensors across all entries
        # and track indices for comparisons
        all_ids_list = []
        all_masks_list = []
        all_chosen = []
        all_rejected = []
        all_margins = []
        offset = 0

        for item in batch:
            ids = item["input_ids"]        # [num_responses, seq_len]
            masks = item["attention_mask"]  # [num_responses, seq_len]
            num_responses = item["num_responses"]

            # Split into individual response tensors (1D each)
            for i in range(num_responses):
                all_ids_list.append(ids[i])      # [seq_len]
                all_masks_list.append(masks[i])  # [seq_len]

            # Adjust comparison indices by offset
            all_chosen.append(item["chosen_indices"] + offset)
            all_rejected.append(item["rejected_indices"] + offset)
            all_margins.append(item["margins"])

            offset += num_responses

        # Use zero_pad_sequences to handle padding (left-pads by default)
        # Input: list of [seq_len] tensors -> Output: [total_responses, max_seq_len]
        padded_ids = zero_pad_sequences(all_ids_list, value=self.tokenizer.pad_token_id)
        padded_masks = zero_pad_sequences(all_masks_list, value=0)

        return {
            "input_ids": padded_ids,
            "attention_mask": padded_masks,
            "chosen_indices": torch.cat(all_chosen, dim=0),
            "rejected_indices": torch.cat(all_rejected, dim=0),
            "margins": torch.cat(all_margins, dim=0),
            "num_responses": sum(item["num_responses"] for item in batch),
            "num_comparisons": sum(item["num_comparisons"] for item in batch),
        }
