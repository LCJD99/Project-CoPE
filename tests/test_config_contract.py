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
