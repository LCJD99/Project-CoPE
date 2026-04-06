from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.cope.inference.stage2_predictor import Stage2Predictor, load_stage2_model
from src.cope.inference.stage3_predictor import Stage3Predictor, load_stage3_model


def _flatten_task_plan_samples(payload: Any) -> List[Dict[str, Any]] | None:
    """
    Normalize task/plan evaluation payload into inference samples.

    Supported input shape:
      [
        {
          "task_id": "...",
          "plans": [
            {
              "scenario_id": 0,
              "USER_QUESTION": "...",
              "SYSTEM_STATE": {...},
              "PLAN_START": "..."   # optional passthrough
            }
          ]
        }
      ]
    """
    if not isinstance(payload, list):
        return None
    if not payload:
        return []

    # Only treat as task/plan format when at least one item exposes `plans`.
    has_plans = any(isinstance(item, dict) and "plans" in item for item in payload)
    if not has_plans:
        return None

    rows: List[Dict[str, Any]] = []
    for task in payload:
        if not isinstance(task, dict):
            continue
        task_id = task.get("task_id")
        plans = task.get("plans", [])
        if not isinstance(plans, list):
            continue

        for plan in plans:
            if not isinstance(plan, dict):
                continue
            user_query = plan.get("user_query", plan.get("USER_QUESTION"))
            system_state = plan.get("system_state", plan.get("SYSTEM_STATE"))
            if user_query is None or system_state is None:
                raise ValueError(
                    "Detected task/plan payload, but one plan item misses "
                    "`USER_QUESTION/user_query` or `SYSTEM_STATE/system_state`."
                )

            row: Dict[str, Any] = {
                "user_query": user_query,
                "system_state": system_state,
            }
            if task_id is not None:
                row["task_id"] = task_id
            if "scenario_id" in plan:
                row["scenario_id"] = plan["scenario_id"]
            if "PLAN_START" in plan:
                row["ground_truth_plan"] = plan["PLAN_START"]
            rows.append(row)

    return rows


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

    flattened = _flatten_task_plan_samples(payload)
    if flattened is not None:
        return flattened

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "samples" in payload:
        flattened = _flatten_task_plan_samples(payload["samples"])
        if flattened is not None:
            return flattened
        return payload["samples"]
    raise ValueError("Input must be a JSON list, JSONL, or {'samples': [...]} object")


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_inference(
    stage: str,
    config_path: str,
    input_path: str | None,
    output_path: str | None,
) -> None:
    cfg = _load_config(config_path)
    checkpoint_path = cfg["checkpoint_path"]
    base_model_name = cfg.get("base_model")
    stage1_checkpoint = cfg.get("stage1_checkpoint")
    device = cfg.get("device", "cuda")
    max_new_tokens = int(cfg.get("max_new_tokens", 256))
    generation_config = cfg.get("generation_config", {"do_sample": False})

    resolved_input_path = input_path or cfg.get("test_data") or cfg.get("input_path")
    resolved_output_path = (
        output_path or cfg.get("eval_output_path") or cfg.get("output_path")
    )
    if not resolved_input_path:
        raise ValueError(
            "Input path is required. Provide --input, or set `test_data`/`input_path` in config."
        )
    if not resolved_output_path:
        raise ValueError(
            "Output path is required. Provide --output, or set `eval_output_path`/`output_path` in config."
        )

    samples = _load_input_samples(resolved_input_path)
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

    out = Path(resolved_output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"stage": stage, "count": len(rows), "predictions": rows}, f, indent=2)
