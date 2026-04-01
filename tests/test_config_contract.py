from pathlib import Path

import yaml

from src.cope.common.config import SUPPORTED_BASE_MODELS


def _iter_stage_configs():
    for p in Path("configs/stage1").glob("*.yaml"):
        yield p
    for p in Path("configs/stage2").glob("*.yaml"):
        yield p
    for p in Path("configs/stage3").glob("*.yaml"):
        yield p


def test_base_model_path_is_supported():
    for config_path in _iter_stage_configs():
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert cfg["model"]["base_model"] in SUPPORTED_BASE_MODELS


def test_data_checkpoint_paths_follow_numbered_convention():
    expected_prefixes = (
        "data/00_global",
        "data/01_stage1",
        "data/02_stage2",
        "data/03_stage3",
        "checkpoints/01_stage1",
        "checkpoints/02_stage2",
        "checkpoints/03_stage3",
    )
    for config_path in _iter_stage_configs():
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for _, value in cfg.items():
            if not isinstance(value, str):
                continue
            if value.startswith("data/") or value.startswith("checkpoints/"):
                assert value.startswith(expected_prefixes), (
                    f"{config_path}: path '{value}' does not follow numbered convention"
                )
