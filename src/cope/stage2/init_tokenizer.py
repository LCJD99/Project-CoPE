from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

from transformers import AutoTokenizer

from src.cope.stage2.tokens import STAGE2_CONTROL_TOKENS, STAGE2_REFERENCE_TOKENS, STAGE2_TOKENS


def init_stage2_tokenizer(
    stage1_checkpoint: str,
    output_dir: str,
) -> Tuple[int, int]:
    tokenizer = AutoTokenizer.from_pretrained(stage1_checkpoint, trust_remote_code=True)
    original_vocab_size = len(tokenizer)
    num_added = tokenizer.add_tokens(STAGE2_TOKENS, special_tokens=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)

    token_mappings = {
        token.strip("<>"): tokenizer.convert_tokens_to_ids(token)
        for token in STAGE2_TOKENS
    }
    with open(output_path / "stage2_token_mappings.json", "w", encoding="utf-8") as f:
        json.dump(token_mappings, f, indent=2, ensure_ascii=False)

    extension_info = {
        "source_checkpoint": stage1_checkpoint,
        "stage2_tokens_added": num_added,
        "final_vocab_size": len(tokenizer),
        "stage2_control_tokens": STAGE2_CONTROL_TOKENS,
        "stage2_reference_tokens": ["<REF_0>", "<REF_1>", "...", "<REF_31>"],
    }
    with open(output_path / "tokenizer_extension_info.json", "w", encoding="utf-8") as f:
        json.dump(extension_info, f, indent=2, ensure_ascii=False)

    return original_vocab_size, len(tokenizer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize Stage2 tokenizer from Stage1 checkpoint")
    parser.add_argument("--stage1_checkpoint", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_vocab_size, final_vocab_size = init_stage2_tokenizer(
        stage1_checkpoint=args.stage1_checkpoint,
        output_dir=args.output,
    )
    print(f"Stage2 tokenizer initialized: {args.output}")
    print(f"  original_vocab_size: {original_vocab_size}")
    print(f"  final_vocab_size: {final_vocab_size}")
    print(f"  added_tokens: {final_vocab_size - original_vocab_size}")


if __name__ == "__main__":
    main()
