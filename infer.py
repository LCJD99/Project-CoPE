from __future__ import annotations

import argparse

from src.cope.inference.runner import run_inference


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline inference for CoPE models")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["stage2", "stage3"],
        help="Inference stage",
    )
    parser.add_argument("--config", type=str, required=True, help="Inference YAML config")
    parser.add_argument("--input", type=str, required=True, help="Input JSON/JSONL path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_inference(
        stage=args.stage,
        config_path=args.config,
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
