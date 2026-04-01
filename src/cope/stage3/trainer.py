import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import yaml
from peft import PeftModel
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.cope.stage3.dataset import (
    PlanRLDataset,
    dataset_has_task_complexity,
    normalize_complexity_label,
)
from src.cope.stage2.model import Stage2PlannerModel
from src.cope.stage3.hierarchical import (
    compute_micro_advantages,
    extract_decision_units,
    fuse_hierarchical_advantage,
    generate_counterfactual_candidates,
)
from src.cope.stage3.simulator.plan_simulator import PlanSimulator, RewardResult


def compute_reward_results(
    plans: List[str],
    system_states: List[Dict[str, float]],
    registry: Dict,
    ground_truth_dags: Optional[List[Optional[Dict]]] = None,
    baseline_latencies_ms: Optional[List[Optional[float]]] = None,
    profiling_path: Optional[str] = None,
    profiling_fallback_path: Optional[str] = None,
) -> List[RewardResult]:
    if len(plans) != len(system_states):
        raise ValueError("plans and system_states must have the same length")
    if ground_truth_dags is not None and len(ground_truth_dags) != len(plans):
        raise ValueError("ground_truth_dags and plans must have the same length")
    if baseline_latencies_ms is not None and len(baseline_latencies_ms) != len(plans):
        raise ValueError("baseline_latencies_ms and plans must have the same length")

    simulator = PlanSimulator(
        registry,
        profiling_path=profiling_path,
        profiling_fallback_path=profiling_fallback_path,
    )
    results: List[RewardResult] = []
    for idx, (plan, system_state) in enumerate(zip(plans, system_states)):
        ground_truth_dag = None
        baseline_latency_ms = None
        if ground_truth_dags is not None:
            ground_truth_dag = ground_truth_dags[idx]
        if baseline_latencies_ms is not None:
            baseline_latency_ms = baseline_latencies_ms[idx]
        result = simulator.evaluate(
            plan,
            system_state,
            ground_truth_dag=ground_truth_dag,
            baseline_latency_ms=baseline_latency_ms,
        )
        results.append(result)
    return results


def summarize_reward_results(results: List[RewardResult]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for result in results:
        if not result.valid:
            hard = float(result.details.get("hard", 0.0))
            if hard == -1.0:
                counts["hard_syntax"] += 1
            elif hard == -1.2:
                counts["hard_ref"] += 1
            elif hard == -1.6:
                counts["hard_resource"] += 1
            elif hard == -2.0:
                counts["hard_deadlock_timeout"] += 1
            else:
                counts["hard_other"] += 1
            continue

        q_score = float(result.details.get("q", 0.0))
        if q_score < 1.0:
            counts["executable_partial"] += 1
        else:
            counts["executable_correct"] += 1

    counts["count"] = len(results)
    return dict(counts)


def _label_from_reward_result(result: RewardResult) -> str:
    if not result.valid:
        hard = float(result.details.get("hard", 0.0))
        if hard == -1.0:
            return "hard_syntax"
        if hard == -1.2:
            return "hard_ref"
        if hard == -1.6:
            return "hard_resource"
        if hard == -2.0:
            return "hard_deadlock_timeout"
        return "hard_other"

    q_score = float(result.details.get("q", 0.0))
    if q_score < 1.0:
        return "executable_partial"
    return "executable_correct"


def build_completion_trace_records(
    call_index: int,
    prompts: List[str],
    completions: List[str],
    reward_results: List[RewardResult],
    max_chars: int = 4000,
    hierarchical_details: Optional[List[Dict[str, object]]] = None,
) -> List[Dict[str, Any]]:
    if not (len(prompts) == len(completions) == len(reward_results)):
        raise ValueError("prompts/completions/reward_results must have the same length")
    if hierarchical_details is not None and len(hierarchical_details) != len(
        completions
    ):
        raise ValueError(
            "hierarchical_details and completions must have the same length"
        )

    records: List[Dict[str, Any]] = []
    for idx, (prompt, completion, result) in enumerate(
        zip(prompts, completions, reward_results)
    ):
        clipped_completion = completion[:max_chars]
        clipped_prompt = prompt[:max_chars]
        record: Dict[str, Any] = {
            "call": call_index,
            "index_in_call": idx,
            "label": _label_from_reward_result(result),
            "reward": float(result.total),
            "valid": bool(result.valid),
            "details": {k: float(v) for k, v in result.details.items()},
            "prompt": clipped_prompt,
            "completion": clipped_completion,
        }
        if hierarchical_details is not None:
            hierarchy = hierarchical_details[idx]
            record["hierarchical"] = bool(hierarchy.get("hierarchical", True))
            record["fused_advantages"] = list(hierarchy.get("fused_advantages", []))
        records.append(record)
    return records


def compute_rewards(
    plans: List[str],
    system_states: List[Dict[str, float]],
    registry: Dict,
    ground_truth_dags: Optional[List[Optional[Dict]]] = None,
    baseline_latencies_ms: Optional[List[Optional[float]]] = None,
    profiling_path: Optional[str] = None,
    profiling_fallback_path: Optional[str] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    results = compute_reward_results(
        plans,
        system_states,
        registry,
        ground_truth_dags=ground_truth_dags,
        baseline_latencies_ms=baseline_latencies_ms,
        profiling_path=profiling_path,
        profiling_fallback_path=profiling_fallback_path,
    )
    rewards = [result.total for result in results]
    return torch.tensor(rewards, dtype=torch.float32, device=device)


def _normalize_wait_refs(choice: Dict[str, object]) -> List[str]:
    raw_refs = choice.get("wait_refs")
    if raw_refs is None:
        raw_refs = choice.get("wait_for", [])
    if not isinstance(raw_refs, list):
        return []
    return [str(ref) for ref in raw_refs]


def _normalize_tool_token(tool: object) -> str:
    token = str(tool).strip()
    if not token:
        return ""
    if token.startswith("<") and token.endswith(">"):
        token = token[1:-1].strip()
    if not token:
        return ""
    return f"<{token}>"


def _build_wait_candidate_pool(
    plan_text: str, statement_index: int, original_wait_refs: List[str]
) -> Dict[str, object]:
    lines = [line.strip() for line in plan_text.splitlines() if line.strip()]
    prior_lines = lines[:statement_index]
    defined_refs: List[str] = []
    for line in prior_lines:
        match = re.match(r"(<REF_\d+>)\s*=", line)
        if match is not None:
            ref = match.group(1)
            if ref not in defined_refs:
                defined_refs.append(ref)

    replace_map = {
        source_ref: [ref for ref in defined_refs if ref != source_ref]
        for source_ref in original_wait_refs
    }
    return {
        "keep": False,
        "remove": original_wait_refs,
        "replace": replace_map,
    }


def _replace_decision_in_plan(
    plan_text: str,
    statement_index: int,
    decision_type: str,
    candidate_choice: Dict[str, object],
) -> Tuple[Optional[str], Optional[str]]:
    lines = plan_text.splitlines(keepends=True)
    non_empty_indices = [idx for idx, line in enumerate(lines) if line.strip()]
    if statement_index < 0 or statement_index >= len(non_empty_indices):
        return None, f"statement_index_out_of_range:{statement_index}"

    line_index = non_empty_indices[statement_index]
    original_line = lines[line_index]
    stripped = original_line.strip()
    indent = original_line[: len(original_line) - len(original_line.lstrip())]
    suffix = "\n" if original_line.endswith("\n") else ""

    if decision_type == "tool_choice":
        tool = _normalize_tool_token(candidate_choice.get("tool", ""))
        if not tool:
            return None, "candidate_missing_tool"
        replaced, count = re.subn(
            r"(<EXEC>\s+)(<[^>\s]+>)", rf"\1{tool}", stripped, count=1
        )
        if count == 0:
            return None, "tool_replace_pattern_not_found"
    elif decision_type == "wait_control":
        wait_refs = [str(ref) for ref in candidate_choice.get("wait_refs", [])]
        wait_prefix = "" if not wait_refs else f"<WAIT> {' '.join(wait_refs)} "
        replaced, count = re.subn(
            r"<WAIT>\s+.*?\s+<EXEC>", f"{wait_prefix}<EXEC>", stripped, count=1
        )
        if count == 0:
            return None, "wait_replace_pattern_not_found"
    else:
        return None, f"unsupported_decision_type:{decision_type}"

    lines[line_index] = f"{indent}{replaced}{suffix}"
    return "".join(lines), None


def _reward_results_from_hierarchical_details(
    hierarchical_details: List[Dict[str, object]],
) -> Optional[List[RewardResult]]:
    reward_results: List[RewardResult] = []
    for detail in hierarchical_details:
        reward_result_dict = detail.get("reward_result")
        if not isinstance(reward_result_dict, dict):
            return None
        if "total" not in reward_result_dict or "valid" not in reward_result_dict:
            return None
        detail_values = reward_result_dict.get("details", {})
        if not isinstance(detail_values, dict):
            return None
        reward_results.append(
            RewardResult(
                total=float(reward_result_dict["total"]),
                valid=bool(reward_result_dict["valid"]),
                details={k: float(v) for k, v in detail_values.items()},
            )
        )
    return reward_results


def _compact_hierarchical_breakdown(details: List[Dict[str, object]]) -> str:
    total_decisions = 0
    considered_decisions = 0
    evaluated_candidates = 0
    for detail in details:
        decision_units = detail.get("decision_units", [])
        if isinstance(decision_units, list):
            total_decisions += len(decision_units)
        counterfactual = detail.get("counterfactual", {})
        if isinstance(counterfactual, dict):
            considered_decisions += int(counterfactual.get("considered_decisions", 0))
            evaluated_candidates += int(counterfactual.get("evaluated_candidates", 0))
    return (
        f"decisions={total_decisions}, considered={considered_decisions}, "
        f"cf_eval={evaluated_candidates}"
    )


def _resolve_training_mode(config: Dict[str, object]) -> str:
    raw_mode = str(config.get("training_mode", "hierarchical_grpo")).strip().lower()
    if raw_mode != "hierarchical_grpo":
        raise ValueError(
            "Main branch only supports training_mode='hierarchical_grpo'."
        )
    return "hierarchical_grpo"


def build_hierarchical_rewards(
    prompts: List[str],
    completions: List[str],
    samples: Optional[List[Dict[str, object]]],
    registry: Dict,
    config: Dict[str, object],
    ground_truth_dags: Optional[List[Optional[Dict]]] = None,
    baseline_latencies_ms: Optional[List[Optional[float]]] = None,
    profiling_path: Optional[str] = None,
    profiling_fallback_path: Optional[str] = None,
) -> Tuple[List[float], List[Dict[str, object]]]:
    if len(prompts) != len(completions):
        raise ValueError("prompts and completions must have the same length")

    if samples is not None:
        if len(samples) != len(completions):
            raise ValueError("samples and completions must have the same length")
        system_states = [sample["system_state"] for sample in samples]
        task_complexities = [
            str(sample.get("task_complexity", "unknown")) for sample in samples
        ]
        if ground_truth_dags is None:
            ground_truth_dags = [sample.get("ground_truth_dag") for sample in samples]
        if baseline_latencies_ms is None:
            baseline_latencies_ms = [
                sample.get("baseline_latency_ms") for sample in samples
            ]
    else:
        system_states = [parse_system_state_from_prompt(prompt) for prompt in prompts]
        task_complexities = ["unknown"] * len(completions)

    reward_results = compute_reward_results(
        completions,
        system_states,
        registry,
        ground_truth_dags=ground_truth_dags,
        baseline_latencies_ms=baseline_latencies_ms,
        profiling_path=profiling_path,
        profiling_fallback_path=profiling_fallback_path,
    )
    reward_values = [float(result.total) for result in reward_results]

    min_group_size = int(config.get("micro_advantage_min_group_size", 2))
    eps = float(config.get("micro_advantage_eps", 1.0e-6))
    macro_weight = float(config.get("macro_weight", 0.4))
    micro_weight = float(config.get("micro_weight", 0.6))
    max_decisions_per_plan = int(config.get("max_decisions_per_plan", 4))
    max_counterfactual_per_decision = int(
        config.get("max_counterfactual_per_decision", 2)
    )
    decision_types = [
        str(value)
        for value in config.get("decision_types", ["tool_choice", "wait_control"])
    ]
    allowed_decision_types = set(decision_types)

    extraction_results = []
    micro_records: List[Dict[str, object]] = []
    decision_indices_by_completion: List[List[int]] = [[] for _ in completions]
    counterfactual_by_completion: List[List[Dict[str, object]]] = [
        [] for _ in completions
    ]
    for completion_index, (completion, task_complexity) in enumerate(
        zip(completions, task_complexities)
    ):
        extraction = extract_decision_units(completion, task_complexity=task_complexity)
        extraction_results.append(extraction)
        decisions_considered = 0
        original_reward = reward_values[completion_index]
        system_state = system_states[completion_index]
        ground_truth_dag = None
        baseline_latency_ms = None
        if ground_truth_dags is not None:
            ground_truth_dag = ground_truth_dags[completion_index]
        if baseline_latencies_ms is not None:
            baseline_latency_ms = baseline_latencies_ms[completion_index]

        for unit in extraction.units:
            if unit.decision_type not in allowed_decision_types:
                continue
            if decisions_considered >= max_decisions_per_plan:
                break
            decisions_considered += 1

            original_choice = dict(unit.original_choice)
            candidate_pool: object
            if unit.decision_type == "tool_choice":
                candidate_pool = [
                    token
                    for token in (
                        _normalize_tool_token(tool_token)
                        for tool_token in registry.get("tokens", {}).keys()
                    )
                    if token
                ]
            elif unit.decision_type == "wait_control":
                original_choice["wait_refs"] = _normalize_wait_refs(original_choice)
                candidate_pool = _build_wait_candidate_pool(
                    completion,
                    statement_index=unit.statement_index,
                    original_wait_refs=list(original_choice["wait_refs"]),
                )
            else:
                candidate_pool = []

            candidates = generate_counterfactual_candidates(
                decision_type=unit.decision_type,
                original_choice=original_choice,
                candidate_pool=candidate_pool,
                max_candidates=max_counterfactual_per_decision,
            )

            candidate_rewards: List[float] = []
            skipped_reasons: List[str] = []
            for candidate in candidates[:max_counterfactual_per_decision]:
                replaced_plan, failure_reason = _replace_decision_in_plan(
                    completion,
                    statement_index=unit.statement_index,
                    decision_type=unit.decision_type,
                    candidate_choice=candidate,
                )
                if replaced_plan is None:
                    skipped_reasons.append(
                        f"candidate_replace_failed:{failure_reason}:{candidate}"
                    )
                    continue
                try:
                    counterfactual_result = compute_reward_results(
                        [replaced_plan],
                        [system_state],
                        registry,
                        ground_truth_dags=[ground_truth_dag],
                        baseline_latencies_ms=[baseline_latency_ms],
                        profiling_path=profiling_path,
                        profiling_fallback_path=profiling_fallback_path,
                    )[0]
                except Exception as exc:
                    skipped_reasons.append(
                        f"candidate_eval_failed:{type(exc).__name__}:{exc}"
                    )
                    continue
                candidate_rewards.append(float(counterfactual_result.total))

            if candidate_rewards:
                mean_counterfactual_reward = sum(candidate_rewards) / len(
                    candidate_rewards
                )
                delta = original_reward - mean_counterfactual_reward
            else:
                delta = 0.0

            counterfactual_by_completion[completion_index].append(
                {
                    "decision_id": unit.decision_id,
                    "decision_type": unit.decision_type,
                    "candidates_generated": len(candidates),
                    "candidates_evaluated": len(candidate_rewards),
                    "delta": delta,
                    "skipped_reasons": skipped_reasons,
                }
            )

            decision_indices_by_completion[completion_index].append(len(micro_records))
            micro_records.append(
                {
                    "completion_index": completion_index,
                    "decision_id": unit.decision_id,
                    "decision_type": unit.decision_type,
                    "prefix_signature": unit.prefix_signature,
                    "token_span": unit.token_span,
                    "delta": delta,
                }
            )

    micro_with_advantages = compute_micro_advantages(
        micro_records,
        min_group_size=min_group_size,
        eps=eps,
    )

    hierarchical_rewards: List[float] = []
    details: List[Dict[str, object]] = []
    for completion_index, extraction in enumerate(extraction_results):
        fused_advantages: List[Dict[str, object]] = []
        for record_index in decision_indices_by_completion[completion_index]:
            record = micro_with_advantages[record_index]
            a_macro = reward_values[completion_index]
            a_micro = float(record.get("a_micro", 0.0))
            a_fused = fuse_hierarchical_advantage(
                a_macro=a_macro,
                a_micro=a_micro,
                macro_weight=macro_weight,
                micro_weight=micro_weight,
            )
            fused_advantages.append(
                {
                    "decision_id": str(record["decision_id"]),
                    "decision_type": str(record["decision_type"]),
                    "token_span": tuple(record["token_span"]),
                    "a_macro": a_macro,
                    "a_micro": a_micro,
                    "a_fused": a_fused,
                }
            )

        if fused_advantages:
            hierarchical_reward = float(
                sum(float(item["a_fused"]) for item in fused_advantages)
                / len(fused_advantages)
            )
        else:
            hierarchical_reward = reward_values[completion_index]
        hierarchical_rewards.append(hierarchical_reward)

        details.append(
            {
                "hierarchical": True,
                "hierarchical_reward": hierarchical_reward,
                "decision_units": [
                    {
                        "decision_id": unit.decision_id,
                        "decision_type": unit.decision_type,
                        "token_span": unit.token_span,
                        "prefix_signature": unit.prefix_signature,
                        "original_choice": unit.original_choice,
                        "statement_index": unit.statement_index,
                    }
                    for unit in extraction.units
                ],
                "fused_advantages": fused_advantages,
                "fused_advantages_metadata": {
                    "macro_weight": macro_weight,
                    "micro_weight": micro_weight,
                    "min_group_size": min_group_size,
                    "eps": eps,
                },
                "counterfactual": {
                    "max_decisions_per_plan": max_decisions_per_plan,
                    "max_counterfactual_per_decision": max_counterfactual_per_decision,
                    "decision_types": decision_types,
                    "considered_decisions": len(
                        counterfactual_by_completion[completion_index]
                    ),
                    "evaluated_candidates": sum(
                        int(item["candidates_evaluated"])
                        for item in counterfactual_by_completion[completion_index]
                    ),
                    "decision_evaluations": counterfactual_by_completion[
                        completion_index
                    ],
                },
                "decision_extraction": {
                    "skipped_count": extraction.skipped_count,
                    "skipped_reasons": extraction.skipped_reasons,
                },
                "reward_result": {
                    "total": float(reward_results[completion_index].total),
                    "valid": bool(reward_results[completion_index].valid),
                    "details": {
                        k: float(v)
                        for k, v in reward_results[completion_index].details.items()
                    },
                },
            }
        )

    return hierarchical_rewards, details


def compute_rewards_by_training_mode(
    prompts: List[str],
    completions: List[str],
    samples: Optional[List[Dict[str, object]]],
    registry: Dict,
    config: Dict[str, object],
    ground_truth_dags: Optional[List[Optional[Dict]]] = None,
    baseline_latencies_ms: Optional[List[Optional[float]]] = None,
    profiling_path: Optional[str] = None,
    profiling_fallback_path: Optional[str] = None,
) -> Tuple[List[float], List[Dict[str, object]]]:
    training_mode = _resolve_training_mode(config)
    if training_mode == "hierarchical_grpo":
        return build_hierarchical_rewards(
            prompts=prompts,
            completions=completions,
            samples=samples,
            registry=registry,
            config=config,
            ground_truth_dags=ground_truth_dags,
            baseline_latencies_ms=baseline_latencies_ms,
            profiling_path=profiling_path,
            profiling_fallback_path=profiling_fallback_path,
        )

    if samples is not None:
        if len(samples) != len(completions):
            raise ValueError("samples and completions must have the same length")
        system_states = [sample["system_state"] for sample in samples]
        if ground_truth_dags is None:
            ground_truth_dags = [sample.get("ground_truth_dag") for sample in samples]
        if baseline_latencies_ms is None:
            baseline_latencies_ms = [
                sample.get("baseline_latency_ms") for sample in samples
            ]
    else:
        system_states = [parse_system_state_from_prompt(prompt) for prompt in prompts]

    reward_results = compute_reward_results(
        completions,
        system_states,
        registry,
        ground_truth_dags=ground_truth_dags,
        baseline_latencies_ms=baseline_latencies_ms,
        profiling_path=profiling_path,
        profiling_fallback_path=profiling_fallback_path,
    )
    return [float(result.total) for result in reward_results], []


def build_prompt(sample: Dict[str, object]) -> str:
    state = sample["system_state"]
    user_question = sample["user_question"]
    return (
        "[SYSTEM_STATE]\n"
        f"CPU_CORES: {state['cpu_core']}\n"
        f"CPU_MEM: {state['cpu_memory']} GB\n"
        f"GPU_SM: {state['gpu_sm']} %\n"
        f"GPU_MEM: {state['gpu_memory']} GB\n\n"
        "[USER_QUERY]\n"
        f"{user_question}\n\n"
        "[PLAN_START]\n"
    )


def build_prompts(samples: List[Dict[str, object]]) -> List[str]:
    return [build_prompt(sample) for sample in samples]


def parse_system_state_from_prompt(prompt: str) -> Dict[str, float]:
    def extract(label: str) -> float:
        match = re.search(rf"{label}:\s*([0-9]+(?:\.[0-9]+)?)", prompt)
        if match is None:
            raise ValueError(f"Missing {label} in prompt")
        return float(match.group(1))

    return {
        "cpu_core": extract("CPU_CORES"),
        "cpu_memory": extract("CPU_MEM"),
        "gpu_sm": extract("GPU_SM"),
        "gpu_memory": extract("GPU_MEM"),
    }


def parse_user_query_from_prompt(prompt: str) -> str:
    match = re.search(r"\[USER_QUERY\]\n(.+?)\n\n\[PLAN_START\]", prompt, re.DOTALL)
    if match is None:
        raise ValueError("Missing [USER_QUERY] block in prompt")
    return match.group(1).strip()


def build_gt_index(data_path: str) -> Dict[str, Dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    index: Dict[str, Dict] = {}
    for item in data:
        for plan in item.get("plans", []):
            key = _gt_key(plan.get("USER_QUESTION", ""), plan.get("SYSTEM_STATE", {}))
            index[key] = {
                "ground_truth_dag": plan.get("json_format"),
                "baseline_latency_ms": plan.get("total_latency_ms"),
            }
    return index


def _gt_key(user_question: str, system_state: Dict[str, float]) -> str:
    state_key = (
        float(system_state.get("cpu_core", 0.0)),
        float(system_state.get("cpu_memory", 0.0)),
        float(system_state.get("gpu_sm", 0.0)),
        float(system_state.get("gpu_memory", 0.0)),
    )
    return f"{user_question}::{state_key}"


def decode_completions_for_reward(
    tokenizer, completions: List[str], completion_ids: Optional[List[List[int]]] = None
) -> List[str]:
    if completion_ids is None:
        return completions

    decoded: List[str] = []
    eos_candidates = [
        tokenizer.eos_token,
        "<|endoftext|>",
        "<|im_end|>",
        "</s>",
    ]
    for token_ids in completion_ids:
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        for eos_token in eos_candidates:
            if eos_token and eos_token in text:
                text = text.split(eos_token)[0]
        decoded.append(text.strip())
    return decoded


class PlanRLPromptDataset(Dataset):
    def __init__(self, dataset: PlanRLDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.dataset[idx]
        return {
            "prompt": build_prompt(sample),
            "system_state": sample["system_state"],
            "task_complexity": sample["task_complexity"],
            "ground_truth_dag": sample["ground_truth_dag"],
            "baseline_latency_ms": sample["total_latency_ms"],
            "user_question": sample["user_question"],
        }


def build_training_stages(
    config: Dict[str, object], has_task_complexity: bool = True
) -> List[Dict[str, Optional[str]]]:
    use_curriculum = bool(config.get("use_curriculum", False))
    if not use_curriculum or not has_task_complexity:
        return [{"name": "all", "complexity": None}]

    curriculum = config.get("complexity_curriculum", ["low", "medium", "high"])
    normalized_curriculum = [normalize_complexity_label(label) for label in curriculum]
    return [
        {"name": complexity, "complexity": complexity}
        for complexity in normalized_curriculum
    ]


def _resolve_stage1_checkpoint(
    config: Dict[str, object], tokenizer_path: str
) -> Optional[str]:
    stage1_checkpoint = config.get("stage1_checkpoint")
    if stage1_checkpoint:
        stage1_path = Path(str(stage1_checkpoint))
        if not stage1_path.is_absolute():
            stage1_path = (Path.cwd() / stage1_path).resolve()
        return str(stage1_path)

    tokenizer_dir = Path(tokenizer_path)
    candidate_configs = [
        tokenizer_dir / "training_config.yaml",
        tokenizer_dir.parent / "training_config.yaml",
    ]
    for candidate in candidate_configs:
        if not candidate.exists():
            continue
        with open(candidate, "r", encoding="utf-8") as f:
            stage2_train_cfg = yaml.safe_load(f) or {}
        stage1_from_stage2 = stage2_train_cfg.get("stage1_checkpoint")
        if stage1_from_stage2:
            stage1_path = Path(str(stage1_from_stage2))
            if not stage1_path.is_absolute():
                # Prefer repository-root-relative paths used in project configs.
                stage1_path = (Path.cwd() / stage1_path).resolve()
            return str(stage1_path)
    return None


def _is_peft_or_stage2_peft_wrapper(
    model: object,
    peft_detector=None,
) -> bool:
    if peft_detector is None:
        from accelerate.utils import is_peft_model as peft_detector

    try:
        if peft_detector(model):
            return True
    except Exception:
        return False

    if not getattr(model, "uses_stage2_embedding_wrapper", False):
        return False

    wrapped_llm = getattr(model, "llm", None)
    if wrapped_llm is None:
        return False

    try:
        return bool(peft_detector(wrapped_llm))
    except Exception:
        return False


def _patch_trl_peft_detection_for_stage2_wrapper() -> None:
    try:
        import trl.trainer.grpo_trainer as trl_grpo_trainer
        from accelerate.utils import is_peft_model as accelerator_is_peft_model
    except ImportError:
        return

    current_detector = getattr(trl_grpo_trainer, "is_peft_model", None)
    if current_detector is None:
        return
    if getattr(current_detector, "_stage2_aware", False):
        return

    def _stage2_aware_is_peft_model(model: object) -> bool:
        return _is_peft_or_stage2_peft_wrapper(
            model,
            peft_detector=accelerator_is_peft_model,
        )

    _stage2_aware_is_peft_model._stage2_aware = True
    trl_grpo_trainer.is_peft_model = _stage2_aware_is_peft_model


def _configure_stage2_logprob_constraints(
    model: object,
    config: Dict[str, object],
) -> None:
    if not getattr(model, "uses_stage2_embedding_wrapper", False):
        return
    setter = getattr(model, "set_logprob_structure_constraints", None)
    if setter is None:
        return
    enabled = bool(config.get("enforce_structure_constraints_for_logprobs", True))
    setter(enabled)


def load_grpo_model_and_tokenizer(config: Dict[str, object]):
    tokenizer_path = str(config["tokenizer"])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    model_dtype = None
    if bool(config.get("bf16", False)):
        model_dtype = torch.bfloat16
    elif bool(config.get("fp16", False)):
        model_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    lora_path = config.get("lora_path")
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)

    tokenizer_dir = Path(tokenizer_path)
    has_stage2_components = (tokenizer_dir / "stage2_embeddings.bin").exists() and (
        tokenizer_dir / "stage2_lm_head.bin"
    ).exists()
    if has_stage2_components:
        stage1_checkpoint = _resolve_stage1_checkpoint(config, tokenizer_path)
        if stage1_checkpoint is None:
            raise ValueError(
                "Stage 2 components were found in tokenizer path, but "
                "stage1_checkpoint is missing. Set stage1_checkpoint in GRPO config "
                "or provide Stage 2 training_config.yaml near tokenizer."
            )
        model = Stage2PlannerModel(
            llm=model,
            tokenizer=tokenizer,
            stage1_collapsed_checkpoint=stage1_checkpoint,
            stage2_embeddings_path=tokenizer_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    _configure_stage2_logprob_constraints(model, config)

    return model, tokenizer


def train_grpo_trl(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model, tokenizer = load_grpo_model_and_tokenizer(config)

    with open(config["tool_registry"], "r", encoding="utf-8") as f:
        registry = json.load(f)

    has_task_complexity = dataset_has_task_complexity(str(config["rl_data"]))
    if bool(config.get("use_curriculum", False)) and not has_task_complexity:
        print(
            "task_complexity is missing in rl_data. "
            "Disabling curriculum and training on all samples.",
            flush=True,
        )
    stages = build_training_stages(config, has_task_complexity=has_task_complexity)
    skip_empty = bool(config.get("skip_empty_complexity", True))
    profiling_path = config.get("profiling_path")
    profiling_fallback_path = config.get("profiling_fallback_path")

    gt_index = build_gt_index(config["rl_data"])

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise ImportError("trl is required to run GRPO training") from exc

    _patch_trl_peft_detection_for_stage2_wrapper()

    reward_log_interval = int(config.get("reward_breakdown_log_interval", 10))
    reward_fn_call_count = 0
    completion_trace_path: Optional[Path] = None
    completion_trace_interval = int(config.get("completion_trace_log_interval", 0))
    completion_trace_max_chars = int(config.get("completion_trace_max_chars", 4000))

    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        nonlocal reward_fn_call_count
        reward_fn_call_count += 1
        completion_ids = kwargs.get("completion_ids")
        decoded_completions = decode_completions_for_reward(
            tokenizer,
            completions,
            completion_ids=completion_ids,
        )
        batch_samples = kwargs.get("samples")
        if batch_samples is not None:
            if len(batch_samples) != len(prompts):
                raise ValueError("samples and prompts must have the same length")
            system_states = [sample["system_state"] for sample in batch_samples]
            ground_truth_dags = [
                sample.get("ground_truth_dag") for sample in batch_samples
            ]
            baseline_latencies = [
                sample.get("baseline_latency_ms") for sample in batch_samples
            ]
        else:
            system_states = [
                parse_system_state_from_prompt(prompt) for prompt in prompts
            ]
            user_questions = [
                parse_user_query_from_prompt(prompt) for prompt in prompts
            ]
            looked_up = [
                gt_index.get(_gt_key(question, system_state), {})
                for question, system_state in zip(user_questions, system_states)
            ]
            ground_truth_dags = [item.get("ground_truth_dag") for item in looked_up]
            baseline_latencies = [item.get("baseline_latency_ms") for item in looked_up]

        training_mode = _resolve_training_mode(config)
        if training_mode == "hierarchical_grpo":
            reward_values, hierarchical_details = build_hierarchical_rewards(
                prompts=prompts,
                completions=decoded_completions,
                samples=batch_samples,
                registry=registry,
                config=config,
                ground_truth_dags=ground_truth_dags,
                baseline_latencies_ms=baseline_latencies,
                profiling_path=profiling_path,
                profiling_fallback_path=profiling_fallback_path,
            )
            if (
                reward_log_interval > 0
                and reward_fn_call_count % reward_log_interval == 0
            ):
                reward_results = _reward_results_from_hierarchical_details(
                    hierarchical_details
                )
                breakdown = (
                    summarize_reward_results(reward_results)
                    if reward_results is not None
                    else {"count": len(reward_values)}
                )
                breakdown_text = ", ".join(
                    f"{key}={value}" for key, value in sorted(breakdown.items())
                )
                hierarchical_text = _compact_hierarchical_breakdown(
                    hierarchical_details
                )
                print(
                    (
                        f"\n[reward_breakdown][call={reward_fn_call_count}] "
                        f"{breakdown_text}, {hierarchical_text}"
                    ),
                    flush=True,
                )

            if (
                completion_trace_path is not None
                and completion_trace_interval > 0
                and reward_fn_call_count % completion_trace_interval == 0
            ):
                reward_results = _reward_results_from_hierarchical_details(
                    hierarchical_details
                )
                if reward_results is None:
                    reward_results = compute_reward_results(
                        decoded_completions,
                        system_states,
                        registry,
                        ground_truth_dags=ground_truth_dags,
                        baseline_latencies_ms=baseline_latencies,
                        profiling_path=profiling_path,
                        profiling_fallback_path=profiling_fallback_path,
                    )
                records = build_completion_trace_records(
                    call_index=reward_fn_call_count,
                    prompts=prompts,
                    completions=decoded_completions,
                    reward_results=reward_results,
                    max_chars=completion_trace_max_chars,
                    hierarchical_details=hierarchical_details,
                )
                with completion_trace_path.open("a", encoding="utf-8") as f:
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

            return reward_values

        reward_results = compute_reward_results(
            decoded_completions,
            system_states,
            registry,
            ground_truth_dags=ground_truth_dags,
            baseline_latencies_ms=baseline_latencies,
            profiling_path=profiling_path,
            profiling_fallback_path=profiling_fallback_path,
        )
        if reward_log_interval > 0 and reward_fn_call_count % reward_log_interval == 0:
            breakdown = summarize_reward_results(reward_results)
            breakdown_text = ", ".join(
                f"{key}={value}" for key, value in sorted(breakdown.items())
            )
            print(
                f"\n[reward_breakdown][call={reward_fn_call_count}] {breakdown_text}",
                flush=True,
            )

        if (
            completion_trace_path is not None
            and completion_trace_interval > 0
            and reward_fn_call_count % completion_trace_interval == 0
        ):
            records = build_completion_trace_records(
                call_index=reward_fn_call_count,
                prompts=prompts,
                completions=decoded_completions,
                reward_results=reward_results,
                max_chars=completion_trace_max_chars,
            )
            with completion_trace_path.open("a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return [result.total for result in reward_results]

    report_to = config.get("report_to", "none")
    if report_to == "wandb":
        wandb_project = config.get("wandb_project")
        wandb_run_name = config.get("wandb_run_name")
        if wandb_project:
            os.environ["WANDB_PROJECT"] = str(wandb_project)
        if wandb_run_name:
            os.environ["WANDB_NAME"] = str(wandb_run_name)

    for stage in stages:
        stage_name = str(stage["name"])
        stage_complexity = stage["complexity"]
        stage_dataset = PlanRLDataset(config["rl_data"], complexity=stage_complexity)
        if len(stage_dataset) == 0:
            if skip_empty:
                print(f"Skipping empty stage: {stage_name}")
                continue
            raise ValueError(f"No samples for stage: {stage_name}")

        train_dataset = PlanRLPromptDataset(stage_dataset)
        use_curriculum = bool(config.get("use_curriculum", False))
        if use_curriculum:
            stage_output_dir = f"{config['output_dir'].rstrip('/')}/{stage_name}"
        else:
            stage_output_dir = str(config["output_dir"]).rstrip("/")
        stage_output_path = Path(stage_output_dir)
        stage_output_path.mkdir(parents=True, exist_ok=True)
        completion_trace_name = str(
            config.get("completion_trace_filename", "completion_traces.jsonl")
        )
        completion_trace_path = stage_output_path / completion_trace_name
        if completion_trace_interval > 0:
            completion_trace_path.write_text("", encoding="utf-8")
        print(f"Starting stage '{stage_name}' with {len(stage_dataset)} samples")
        num_generations = int(config.get("rollouts_per_input", 1))
        per_device_train_batch_size = int(config.get("batch_size", 1))
        generation_batch_size = int(
            config.get(
                "generation_batch_size",
                max(per_device_train_batch_size, num_generations),
            )
        )
        if generation_batch_size % num_generations != 0:
            raise ValueError(
                "generation_batch_size must be divisible by rollouts_per_input "
                f"(got generation_batch_size={generation_batch_size}, "
                f"rollouts_per_input={num_generations})."
            )

        trainer_config = GRPOConfig(
            output_dir=stage_output_dir,
            max_completion_length=int(config.get("max_new_tokens", 128)),
            beta=float(config.get("kl_coef", 0.02)),
            num_generations=num_generations,
            generation_batch_size=generation_batch_size,
            gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=int(
                config.get("gradient_accumulation_steps", 1)
            ),
            bf16=bool(config.get("bf16", False)),
            fp16=bool(config.get("fp16", False)),
            temperature=float(config.get("temperature", 1.0)),
            top_p=float(config.get("top_p", 1.0)),
            top_k=int(config.get("top_k", 50)),
            min_p=float(config.get("min_p", 0.0)),
            repetition_penalty=float(config.get("repetition_penalty", 1.0)),
            report_to=report_to,
            run_name=str(config.get("wandb_run_name", "")) or None,
        )

        trainer = GRPOTrainer(
            model=model,
            args=trainer_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            reward_funcs=reward_fn,
        )
        trainer.train()
