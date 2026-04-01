"""Stage 1 (Configured Tool Token Alignment)."""

from src.cope.stage1.trainer import train_stage1_v3


def train_stage1(config_path: str, load_lora_from: str | None = None) -> None:
    train_stage1_v3(config_path, load_lora_from=load_lora_from)
