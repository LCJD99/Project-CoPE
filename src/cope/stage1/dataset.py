"""
Tool Instruction Dataset for Stage 1 Training
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load data from JSONL file.
    
    Args:
        file_path: Path to JSONL file
    
    Returns:
        List of data dictionaries
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


class ToolInstructionDataset(Dataset):
    """
    Dataset for tool instruction learning.
    
    Formats data as: {input_text} -> {output_token}
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        prompt_template: Optional[str] = None,
        only_answer: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSONL data file
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            prompt_template: Optional prompt template string
        """
        self.data = load_jsonl(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.only_answer = only_answer
        self.prompt_template = prompt_template or "Input: {input}\nOutput: {output}"
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        Args:
            idx: Index
        
        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        item = self.data[idx]
        
        # Format input and output parts separately
        input_text = f"Input: {item['input']}\nOutput: "
        output_text = item['output']
        
        # Tokenize input part (without output)
        input_encoding = self.tokenizer(
            input_text,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_length = input_encoding['input_ids'].shape[1]
        
        # Tokenize full text
        full_text = input_text + output_text
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Extract tensors and squeeze batch dimension
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Create labels: mask input part, only compute loss on output
        labels = input_ids.clone()
        
        # Mask the input part (set to -100)
        labels[:input_length] = -100
        
        # Mask padding tokens in labels (set to -100)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class ToolInstructionDataCollator:
    """
    Data collator for tool instruction dataset.
    Handles dynamic padding for efficient batching.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, padding: bool = True):
        """
        Initialize collator.
        
        Args:
            tokenizer: Tokenizer instance
            padding: Whether to pad sequences
        """
        self.tokenizer = tokenizer
        self.padding = padding
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of features.
        
        Args:
            features: List of feature dictionaries
        
        Returns:
            Batched tensors
        """
        # Stack tensors
        batch = {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features])
        }
        
        return batch
