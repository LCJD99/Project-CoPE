from __future__ import annotations

import argparse
from pathlib import Path

from src.cope.common.config import emit_run_metadata, load_yaml_config
from src.cope.stage1 import train_stage1
from src.cope.stage2 import train_stage2
from src.cope.stage3 import train_grpo_trl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CoPE Models")
    parser.add_argument(
        "--stage",
        type=str,
        default="stage1",
        choices=["stage1", "stage2", "stage3"],
        help=(
            "Training stage: stage1=Configured Tool Token Alignment, "
            "stage2=Structured Program Fine-Tuning, "
            "stage3=Local-advantage-guided Policy Optimization"
        ),
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--load_lora_from",
        type=str,
        default=None,
        help="[Stage1 only] Existing LoRA checkpoint for warm-start",
    )
    return parser.parse_args(argv)


def _output_dir_from_config(config_path: str) -> str:
    cfg = load_yaml_config(config_path)
    return str(cfg.get("output_dir", "outputs"))


def main() -> None:
    args = parse_args()
    output_dir = _output_dir_from_config(args.config)
    emit_run_metadata(output_dir, {"stage": args.stage, "config_path": args.config})

    if args.stage == "stage1":
        train_stage1(args.config, load_lora_from=args.load_lora_from)
        return
    if args.stage == "stage2":
        train_stage2(args.config)
        return
    train_grpo_trl(args.config)


if __name__ == "__main__":
    main()
