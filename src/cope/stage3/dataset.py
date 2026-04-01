import json
from typing import Dict, List, Optional

from torch.utils.data import Dataset


VALID_COMPLEXITIES = {"low", "medium", "high"}


def normalize_complexity_label(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in VALID_COMPLEXITIES:
        raise ValueError(
            f"Invalid task_complexity '{value}'. Expected one of: "
            f"{sorted(VALID_COMPLEXITIES)}"
        )
    return normalized


def dataset_has_task_complexity(data_path: str) -> bool:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        for plan in item.get("plans", []):
            raw_complexity = plan.get("task_complexity")
            if raw_complexity is None or not str(raw_complexity).strip():
                return False
    return True


class PlanRLDataset(Dataset):
    def __init__(self, data_path: str, complexity: Optional[str] = None):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        target_complexity = (
            normalize_complexity_label(complexity) if complexity is not None else None
        )
        self.samples: List[Dict] = []
        for task_index, item in enumerate(data):
            task_id = item.get("task_id")
            task_ref = (
                f"task_id={task_id}"
                if task_id is not None
                else f"task_index={task_index}"
            )
            for plan in item.get("plans", []):
                if "SYSTEM_STATE" not in plan:
                    raise ValueError(f"Missing SYSTEM_STATE for {task_ref}.")
                if "USER_QUESTION" not in plan:
                    raise ValueError(f"Missing USER_QUESTION for {task_ref}.")
                raw_complexity = plan.get("task_complexity")
                if "json_format" not in plan:
                    raise ValueError(f"Missing json_format for {task_ref}.")
                if "total_latency_ms" not in plan:
                    raise ValueError(f"Missing total_latency_ms for {task_ref}.")

                if raw_complexity is None or not str(raw_complexity).strip():
                    if target_complexity is not None:
                        continue
                    plan_complexity = "unknown"
                else:
                    plan_complexity = normalize_complexity_label(str(raw_complexity))
                    if (
                        target_complexity is not None
                        and plan_complexity != target_complexity
                    ):
                        continue

                self.samples.append(
                    {
                        "system_state": plan["SYSTEM_STATE"],
                        "user_question": plan["USER_QUESTION"],
                        "task_complexity": plan_complexity,
                        "ground_truth_dag": plan["json_format"],
                        "total_latency_ms": float(plan["total_latency_ms"]),
                        "task_id": task_id,
                        "scenario_id": plan.get("scenario_id"),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]
