import pytest

from src.cope.stage3.trainer import _resolve_training_mode


def test_stage3_rejects_non_hierarchical_mode():
    with pytest.raises(ValueError):
        _resolve_training_mode({"training_mode": "global_grpo"})


def test_stage3_accepts_hierarchical_mode():
    assert _resolve_training_mode({"training_mode": "hierarchical_grpo"}) == "hierarchical_grpo"
