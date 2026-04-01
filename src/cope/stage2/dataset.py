"""
Stage 2 Planning Dataset

This dataset loads and formats Stage 2 planning data with proper loss masking.
Only the [PLAN_START] section is used for training (loss computation), while
[SYSTEM_STATE] and [USER_QUERY] serve as context.

Data Format (JSON):
{
  "task_id": "27766469",
  "plans": [
    {
      "scenario_id": 1,
      "SYSTEM_STATE": {
        "cpu_core": 15,
        "cpu_memory": 31,
        "gpu_sm": 96,
        "gpu_memory": 15
      },
      "USER_QUESTION": "I've got an image named 'example.jpg'...",
      "PLAN_START": "<REF_0> = <EXEC> <TOOL>(...)\n<FINISH> <REF_0>"
    }
  ]
}

Training Template:
[SYSTEM_STATE]
CPU_CORES: {cpu_core}
CPU_MEM: {cpu_memory} GB
GPU_SM: {gpu_sm} %
GPU_MEM: {gpu_memory} GB

[USER_QUERY]
{USER_QUESTION}

[PLAN_START]
{PLAN_START}
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class Stage2PlanningDataset(Dataset):
    """
    Dataset for Stage 2 planning training.

    Loads task scenarios and formats them with resource state context.
    Only computes loss on the [PLAN_START] section.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        include_scenario_id: bool = False,
    ):
        """
        Initialize Stage 2 planning dataset.

        Args:
            data_path: Path to JSON data file (gt-single.json)
            tokenizer: Extended tokenizer with Stage 2 tokens (vocab size 153,402)
            max_length: Maximum sequence length
            include_scenario_id: If True, include scenario_id in metadata
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_scenario_id = include_scenario_id

        # Load data
        self.examples = self._load_and_flatten_data(data_path)

        print(f"Loaded {len(self.examples)} planning examples from {data_path}")
        print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    def _load_and_flatten_data(self, data_path: str) -> List[Dict]:
        """
        Load JSON data and flatten task/plan structure.

        Args:
            data_path: Path to JSON file

        Returns:
            List of flattened examples, each containing:
            - task_id, scenario_id, SYSTEM_STATE, USER_QUESTION, PLAN_START
        """
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        examples = []
        for task in data:
            task_id = task["task_id"]
            for plan in task["plans"]:
                example = {
                    "task_id": task_id,
                    "scenario_id": plan["scenario_id"],
                    "SYSTEM_STATE": plan["SYSTEM_STATE"],
                    "USER_QUESTION": plan["USER_QUESTION"],
                    "PLAN_START": plan["PLAN_START"],
                }
                examples.append(example)

        return examples

    def _format_example(self, example: Dict) -> tuple[str, str]:
        """
        Format example into context and target sections.

        Args:
            example: Data example

        Returns:
            Tuple of (context_text, target_text)
            - context_text: [SYSTEM_STATE] + [USER_QUERY] (no loss)
            - target_text: [PLAN_START] (compute loss)
        """
        state = example["SYSTEM_STATE"]

        # Format system state
        context_text = (
            "[SYSTEM_STATE]\n"
            f"CPU_CORES: {state['cpu_core']}\n"
            f"CPU_MEM: {state['cpu_memory']} GB\n"
            f"GPU_SM: {state['gpu_sm']} %\n"
            f"GPU_MEM: {state['gpu_memory']} GB\n\n"
            "[USER_QUERY]\n"
            f"{example['USER_QUESTION']}\n\n"
            "[PLAN_START]\n"
        )

        # Target is the plan
        target_text = example["PLAN_START"]

        return context_text, target_text

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example with proper loss masking.

        Uses separate tokenization for context and target, then concatenates
        the token IDs directly. This avoids BPE merge boundary issues that
        occur when tokenizing the concatenated string as a whole.

        Args:
            idx: Index

        Returns:
            Dictionary with:
            - input_ids: [max_length]
            - attention_mask: [max_length]
            - labels: [max_length] with -100 for context tokens
        """
        example = self.examples[idx]
        context_text, target_text = self._format_example(example)

        # Tokenize context and target SEPARATELY to get precise boundary
        # Context gets special tokens (e.g. BOS), target does NOT
        context_ids = self.tokenizer.encode(context_text, add_special_tokens=True)
        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)

        # Add EOS token to target
        if self.tokenizer.eos_token_id is not None:
            target_ids = target_ids + [self.tokenizer.eos_token_id]

        # Concatenate token IDs directly (no re-tokenization boundary issues)
        full_ids = context_ids + target_ids
        context_length = len(context_ids)

        # Truncate if exceeding max_length
        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]

        # Calculate actual sequence length before padding
        seq_length = len(full_ids)

        # Pad to max_length
        pad_length = self.max_length - seq_length
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        input_ids = torch.tensor(
            full_ids + [pad_token_id] * pad_length, dtype=torch.long
        )
        attention_mask = torch.tensor(
            [1] * seq_length + [0] * pad_length, dtype=torch.long
        )

        # Create labels with loss masking
        labels = input_ids.clone()

        # Mask context tokens (set to -100 to ignore in loss)
        labels[:context_length] = -100

        # Mask padding tokens
        labels[attention_mask == 0] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Optionally include metadata
        if self.include_scenario_id:
            result["task_id"] = example["task_id"]
            result["scenario_id"] = example["scenario_id"]

        return result


def test_dataset():
    """Test dataset loading and formatting."""
    from transformers import AutoTokenizer

    # Load extended tokenizer (assume it exists)
    tokenizer_path = "checkpoints/02_stage2/tokenizer_stage2"
    data_path = "data/02_stage2/gt-single.json"

    print("Testing Stage2PlanningDataset...")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Data: {data_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Create dataset
    dataset = Stage2PlanningDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=512,
    )

    # Test first example
    print(f"\nDataset size: {len(dataset)}")
    sample = dataset[0]

    print(f"\nSample keys: {sample.keys()}")
    print(f"input_ids shape: {sample['input_ids'].shape}")
    print(f"attention_mask shape: {sample['attention_mask'].shape}")
    print(f"labels shape: {sample['labels'].shape}")

    # Decode
    input_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
    print(f"\nDecoded input (first 500 chars):\n{input_text[:500]}")

    # Check label masking
    non_masked_positions = (sample["labels"] != -100).sum().item()
    total_positions = sample["labels"].shape[0]
    print(f"\nLabel statistics:")
    print(f"  Total positions: {total_positions}")
    print(f"  Non-masked (trainable): {non_masked_positions}")
    print(f"  Masked (context/padding): {total_positions - non_masked_positions}")

    print("\n✓ Dataset test passed!")


if __name__ == "__main__":
    test_dataset()
