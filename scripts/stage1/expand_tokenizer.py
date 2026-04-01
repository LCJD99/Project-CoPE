#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.cope.stage1.tokenizer_expand import (
    expand_tokenizer,
    extract_virtual_tokens,
    load_registry,
    verify_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand tokenizer with Stage1 virtual tokens")
    parser.add_argument("--base_model", required=True, type=str)
    parser.add_argument("--registry", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_registry(args.registry)
    virtual_tokens = extract_virtual_tokens(registry)
    original_vocab_size, num_added = expand_tokenizer(
        base_model=args.base_model,
        virtual_tokens=virtual_tokens,
        output_dir=args.output,
    )
    print(f"Original vocab: {original_vocab_size}")
    print(f"Added tokens: {num_added}")
    print(f"Expanded vocab: {original_vocab_size + num_added}")
    print(f"Output: {args.output}")

    if args.verify:
        verify_tokens(args.output, virtual_tokens)
        print("Verification passed")


if __name__ == "__main__":
    main()
