from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from transformers import AutoTokenizer


def load_registry(registry_path: str) -> Dict:
    with open(registry_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_virtual_tokens(registry: Dict) -> List[str]:
    tokens = sorted(registry["tokens"].keys())
    return [f"<{token}>" for token in tokens]


def expand_tokenizer(
    base_model: str,
    virtual_tokens: Iterable[str],
    output_dir: str,
) -> Tuple[int, int]:
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    original_vocab_size = len(tokenizer)
    num_added = tokenizer.add_tokens(list(virtual_tokens), special_tokens=True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    return original_vocab_size, num_added


def verify_tokens(output_dir: str, tokens: Iterable[str], sample_size: int = 5) -> None:
    tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    checked = 0
    for token in tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            raise ValueError(f"Token not found in expanded tokenizer: {token}")
        decoded = tokenizer.decode([token_id])
        if decoded != token:
            raise ValueError(f"Roundtrip mismatch for {token}: got {decoded}")
        checked += 1
        if checked >= sample_size:
            break

