from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.cope.inference.stage2_predictor import Stage2Predictor, load_stage2_model
from src.cope.inference.stage3_predictor import Stage3Predictor, load_stage3_model


def _load_input_samples(input_path: str) -> List[Dict[str, Any]]:
    path = Path(input_path)
    if path.suffix.lower() == ".jsonl":
        samples: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "samples" in payload:
        return payload["samples"]
    raise ValueError("Input must be a JSON list, JSONL, or {'samples': [...]} object")


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_inference(
    stage: str,
    config_path: str,
    input_path: str,
    output_path: str,
) -> None:
    cfg = _load_config(config_path)
    checkpoint_path = cfg["checkpoint_path"]
    base_model_name = cfg.get("base_model")
    stage1_checkpoint = cfg.get("stage1_checkpoint")
    device = cfg.get("device", "cuda")
    max_new_tokens = int(cfg.get("max_new_tokens", 256))
    generation_config = cfg.get("generation_config", {"do_sample": False})

    samples = _load_input_samples(input_path)
    if stage == "stage2":
        model, tokenizer = load_stage2_model(
            checkpoint_path,
            device=device,
            base_model_name=base_model_name,
            stage1_checkpoint=stage1_checkpoint,
        )
        predictor = Stage2Predictor(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            generation_config=generation_config,
        )
    elif stage == "stage3":
        model, tokenizer = load_stage3_model(
            checkpoint_path,
            device=device,
            base_model_name=base_model_name,
            stage1_checkpoint=stage1_checkpoint,
        )
        predictor = Stage3Predictor(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            generation_config=generation_config,
        )
    else:
        raise ValueError(f"Unsupported stage '{stage}' for inference")

    rows: List[Dict[str, Any]] = []
    for sample in samples:
        plan = predictor.predict(
            user_query=sample["user_query"],
            system_state=sample["system_state"],
        )
        row = {"predicted_plan": plan, **sample}
        rows.append(row)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"stage": stage, "count": len(rows), "predictions": rows}, f, indent=2)
