from __future__ import annotations

from src.cope.inference.stage2_predictor import Stage2Predictor, load_stage2_model


def load_stage3_model(
    checkpoint_path: str,
    device: str = "cuda",
    base_model_name: str | None = None,
    stage1_checkpoint: str | None = None,
):
    """Stage3 uses the same wrapped model structure as Stage2 final checkpoints."""
    return load_stage2_model(
        checkpoint_path,
        device=device,
        base_model_name=base_model_name,
        stage1_checkpoint=stage1_checkpoint,
    )


class Stage3Predictor(Stage2Predictor):
    """Stage3 predictor shares Stage2 generation behavior."""
