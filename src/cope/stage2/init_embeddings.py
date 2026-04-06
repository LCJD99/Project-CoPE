from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.cope.stage2.tokens import TOKEN_INIT_MAP


def load_base_model(
    base_model_name: str,
    device: str = "cuda",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"Loading base model: {base_model_name}")
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device,
    )
    return base_model, base_tokenizer


def compute_semantic_embedding(
    words: List[str],
    base_model: AutoModelForCausalLM,
    base_tokenizer: AutoTokenizer,
    device: str = "cuda",
) -> torch.Tensor:
    input_embeddings = base_model.get_input_embeddings()
    word_embeddings = []

    with torch.no_grad():
        for word in words:
            token_ids = base_tokenizer(
                word,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].to(device)
            word_emb = input_embeddings(token_ids).mean(dim=1).squeeze(0)
            word_embeddings.append(word_emb)

    return torch.stack(word_embeddings).mean(dim=0)


def initialize_stage2_embeddings(
    base_model: AutoModelForCausalLM,
    base_tokenizer: AutoTokenizer,
    stage2_tokenizer: AutoTokenizer,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_embeddings = base_model.get_input_embeddings()
    output_embeddings = base_model.get_output_embeddings()

    hidden_size = input_embeddings.weight.shape[1]
    num_stage2_tokens = len(TOKEN_INIT_MAP)

    stage2_embeddings = torch.zeros(
        num_stage2_tokens,
        hidden_size,
        dtype=input_embeddings.weight.dtype,
        device=device,
    )
    stage2_lm_head = torch.zeros(
        num_stage2_tokens,
        hidden_size,
        dtype=(
            output_embeddings.weight.dtype
            if output_embeddings is not None
            else input_embeddings.weight.dtype
        ),
        device=device,
    )

    token_list = list(TOKEN_INIT_MAP.keys())
    for idx, token_str in enumerate(tqdm(token_list, desc="Initializing embeddings")):
        semantic_emb = compute_semantic_embedding(
            TOKEN_INIT_MAP[token_str],
            base_model,
            base_tokenizer,
            device,
        )
        stage2_embeddings[idx] = semantic_emb
        stage2_lm_head[idx] = semantic_emb

        token_id = stage2_tokenizer.convert_tokens_to_ids(token_str)
        if token_id == stage2_tokenizer.unk_token_id:
            raise ValueError(f"Token {token_str} not found in Stage2 tokenizer")

    return stage2_embeddings, stage2_lm_head


def save_embeddings(
    stage2_embeddings: torch.Tensor,
    stage2_lm_head: torch.Tensor,
    output_dir: str,
    base_model_name: str,
    tokenizer_path: str,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(stage2_embeddings.cpu(), output_path / "stage2_embeddings_init.bin")
    torch.save(stage2_lm_head.cpu(), output_path / "stage2_lm_head_init.bin")

    info = {
        "base_model": base_model_name,
        "tokenizer_path": tokenizer_path,
        "num_stage2_tokens": int(stage2_embeddings.shape[0]),
        "hidden_size": int(stage2_embeddings.shape[1]),
        "token_init_map": TOKEN_INIT_MAP,
    }
    with open(output_path / "stage2_init_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize Stage2 embeddings from existing Stage2 tokenizer")
    parser.add_argument("--base_model", required=True, type=str)
    parser.add_argument("--stage2_tokenizer", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_model, base_tokenizer = load_base_model(args.base_model, args.device)
    stage2_tokenizer = AutoTokenizer.from_pretrained(
        args.stage2_tokenizer,
        trust_remote_code=True,
    )

    stage2_embeddings, stage2_lm_head = initialize_stage2_embeddings(
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        stage2_tokenizer=stage2_tokenizer,
        device=args.device,
    )
    save_embeddings(
        stage2_embeddings=stage2_embeddings,
        stage2_lm_head=stage2_lm_head,
        output_dir=args.output,
        base_model_name=args.base_model,
        tokenizer_path=args.stage2_tokenizer,
    )

    print(f"Stage2 embedding initialization complete: {args.output}")


if __name__ == "__main__":
    main()
