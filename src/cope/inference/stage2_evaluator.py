"""
Stage 2 Planning Evaluation Metrics

This module provides evaluation metrics for Stage 2 planning predictions.
"""

import re
from typing import Dict, List, Optional
from collections import defaultdict


def normalize_plan(plan: str) -> str:
    """
    Normalize a plan string for comparison.

    Args:
        plan: Raw plan string

    Returns:
        Normalized plan string (whitespace normalized, trimmed)
    """
    # Remove extra whitespace
    plan = re.sub(r"\s+", " ", plan.strip())
    # Normalize newlines to single newline
    plan = re.sub(r"\n+", "\n", plan)
    return plan.strip()


def compute_exact_match(
    predicted_plan: str, ground_truth_plan: str, normalize: bool = True
) -> bool:
    """
    Compute exact match between predicted and ground truth plans.

    Args:
        predicted_plan: Generated plan string
        ground_truth_plan: Ground truth plan string
        normalize: Whether to normalize whitespace before comparison

    Returns:
        True if plans match exactly, False otherwise
    """
    if normalize:
        predicted_plan = normalize_plan(predicted_plan)
        ground_truth_plan = normalize_plan(ground_truth_plan)

    return predicted_plan == ground_truth_plan


def extract_tokens_from_plan(plan: str) -> List[str]:
    """
    Extract all tokens from a plan (for token-level analysis).

    Args:
        plan: Plan string

    Returns:
        List of token strings (control tokens, tool tokens, references)
    """
    # Match special tokens: <EXEC>, <FINISH>, <WAIT>, <EQ>, <STATEMENT>, <END_STATEMENT>, <REF_X>, <TOOL_NAME>
    token_pattern = r"<[A-Z_0-9]+>"
    tokens = re.findall(token_pattern, plan)
    return tokens


def extract_tool_selections(plan: str) -> List[str]:
    """
    Extract tool selections from a plan.

    Args:
        plan: Plan string

    Returns:
        List of tool tokens selected in order
    """
    # Pattern: <EXEC> <TOOL_TOKEN>(...)
    # Match tool tokens that appear after <EXEC>
    exec_pattern = r"<EXEC>\s*(<[A-Z_0-9]+>)"
    tool_tokens = re.findall(exec_pattern, plan)
    return tool_tokens


def compute_token_level_accuracy(
    predicted_plan: str, ground_truth_plan: str
) -> Dict[str, float]:
    """
    Compute token-level accuracy metrics.

    Args:
        predicted_plan: Generated plan
        ground_truth_plan: Ground truth plan

    Returns:
        Dictionary with token-level metrics
    """
    pred_tokens = extract_tokens_from_plan(predicted_plan)
    gt_tokens = extract_tokens_from_plan(ground_truth_plan)

    # Compute accuracy for matching positions
    min_len = min(len(pred_tokens), len(gt_tokens))
    max_len = max(len(pred_tokens), len(gt_tokens))

    if max_len == 0:
        return {
            "token_accuracy": 0.0,
            "token_count_match": True,
            "predicted_tokens": 0,
            "ground_truth_tokens": 0,
            "correct_tokens": 0,
        }

    correct = sum(1 for i in range(min_len) if pred_tokens[i] == gt_tokens[i])

    return {
        "token_accuracy": correct / max_len if max_len > 0 else 0.0,
        "token_count_match": len(pred_tokens) == len(gt_tokens),
        "predicted_tokens": len(pred_tokens),
        "ground_truth_tokens": len(gt_tokens),
        "correct_tokens": correct,
    }


def compute_tool_selection_accuracy(
    predicted_plan: str, ground_truth_plan: str
) -> Dict[str, any]:
    """
    Compute tool selection accuracy (whether correct tools were chosen).

    Args:
        predicted_plan: Generated plan
        ground_truth_plan: Ground truth plan

    Returns:
        Dictionary with tool selection metrics
    """
    pred_tools = extract_tool_selections(predicted_plan)
    gt_tools = extract_tool_selections(ground_truth_plan)

    # Check if tool sequences match
    tools_match = pred_tools == gt_tools

    # Check if at least one correct tool is selected
    has_correct_tool = (
        any(tool in gt_tools for tool in pred_tools)
        if pred_tools and gt_tools
        else False
    )

    return {
        "tool_sequence_match": tools_match,
        "has_correct_tool": has_correct_tool,
        "predicted_tools": pred_tools,
        "ground_truth_tools": gt_tools,
        "num_predicted_tools": len(pred_tools),
        "num_ground_truth_tools": len(gt_tools),
    }


class Stage2Evaluator:
    """Evaluator for Stage 2 planning predictions."""

    def __init__(self, normalize_plans: bool = True):
        """
        Initialize evaluator.

        Args:
            normalize_plans: Whether to normalize whitespace in plans
        """
        self.normalize_plans = normalize_plans

    def evaluate_single(self, predicted_plan: str, ground_truth_plan: str) -> Dict:
        """
        Evaluate a single prediction.

        Args:
            predicted_plan: Generated plan
            ground_truth_plan: Ground truth plan

        Returns:
            Dictionary with all metrics for this prediction
        """
        metrics = {}

        # Exact match
        metrics["exact_match"] = compute_exact_match(
            predicted_plan, ground_truth_plan, normalize=self.normalize_plans
        )

        # Token-level accuracy
        metrics["token_level"] = compute_token_level_accuracy(
            predicted_plan, ground_truth_plan
        )

        # Tool selection accuracy
        metrics["tool_selection"] = compute_tool_selection_accuracy(
            predicted_plan, ground_truth_plan
        )

        return metrics

    def evaluate_predictions(self, predictions: List[Dict]) -> Dict:
        """
        Evaluate a list of predictions.

        Args:
            predictions: List of prediction dictionaries with keys:
                - predicted_plan: Generated plan
                - ground_truth_plan: Ground truth plan
                - task_id: Task ID (optional)
                - scenario_id: Scenario ID (optional)
                - success: Whether prediction succeeded (optional)

        Returns:
            Dictionary with:
                - summary: Overall metrics
                - per_scenario_stats: Per-scenario breakdown (if scenario_id present)
                - detailed_metrics: Per-sample detailed metrics
        """
        # Overall counters
        total = len(predictions)
        exact_match_count = 0
        successful_predictions = 0

        # Token-level aggregates
        total_token_accuracy = 0.0
        token_count_match_count = 0

        # Tool selection aggregates
        tool_sequence_match_count = 0
        has_correct_tool_count = 0

        # Per-scenario stats
        per_scenario_stats = defaultdict(
            lambda: {
                "total": 0,
                "exact_match_count": 0,
                "successful": 0,
                "token_accuracy_sum": 0.0,
                "tool_sequence_match_count": 0,
            }
        )

        # Detailed metrics per sample
        detailed_metrics = []

        for pred in predictions:
            # Check if prediction was successful
            is_successful = pred.get("success", True)
            if is_successful:
                successful_predictions += 1

            # Get plans
            predicted_plan = pred.get("predicted_plan", "")
            ground_truth_plan = pred.get("ground_truth_plan", "")
            scenario_id = pred.get("scenario_id")

            # Evaluate
            if is_successful and ground_truth_plan:
                metrics = self.evaluate_single(predicted_plan, ground_truth_plan)

                # Update overall counters
                if metrics["exact_match"]:
                    exact_match_count += 1

                total_token_accuracy += metrics["token_level"]["token_accuracy"]

                if metrics["token_level"]["token_count_match"]:
                    token_count_match_count += 1

                if metrics["tool_selection"]["tool_sequence_match"]:
                    tool_sequence_match_count += 1

                if metrics["tool_selection"]["has_correct_tool"]:
                    has_correct_tool_count += 1

                # Update per-scenario stats
                if scenario_id is not None:
                    stats = per_scenario_stats[scenario_id]
                    stats["total"] += 1
                    stats["successful"] += 1
                    if metrics["exact_match"]:
                        stats["exact_match_count"] += 1
                    stats["token_accuracy_sum"] += metrics["token_level"][
                        "token_accuracy"
                    ]
                    if metrics["tool_selection"]["tool_sequence_match"]:
                        stats["tool_sequence_match_count"] += 1

                detailed_metrics.append(
                    {
                        "task_id": pred.get("task_id"),
                        "scenario_id": scenario_id,
                        "metrics": metrics,
                    }
                )
            else:
                # Failed prediction or missing ground truth
                if scenario_id is not None:
                    per_scenario_stats[scenario_id]["total"] += 1

        # Compute summary statistics
        summary = {
            "total_samples": total,
            "successful_predictions": successful_predictions,
            "exact_match_count": exact_match_count,
            "exact_match_rate": exact_match_count / total if total > 0 else 0.0,
            "avg_token_accuracy": total_token_accuracy / successful_predictions
            if successful_predictions > 0
            else 0.0,
            "token_count_match_rate": token_count_match_count / successful_predictions
            if successful_predictions > 0
            else 0.0,
            "tool_sequence_match_count": tool_sequence_match_count,
            "tool_sequence_match_rate": tool_sequence_match_count
            / successful_predictions
            if successful_predictions > 0
            else 0.0,
            "has_correct_tool_rate": has_correct_tool_count / successful_predictions
            if successful_predictions > 0
            else 0.0,
        }

        # Compute per-scenario summary stats
        per_scenario_summary = {}
        for scenario_id, stats in per_scenario_stats.items():
            per_scenario_summary[scenario_id] = {
                "total": stats["total"],
                "successful": stats["successful"],
                "exact_match_count": stats["exact_match_count"],
                "exact_match_rate": stats["exact_match_count"] / stats["total"]
                if stats["total"] > 0
                else 0.0,
                "avg_token_accuracy": stats["token_accuracy_sum"] / stats["successful"]
                if stats["successful"] > 0
                else 0.0,
                "tool_sequence_match_rate": stats["tool_sequence_match_count"]
                / stats["successful"]
                if stats["successful"] > 0
                else 0.0,
            }

        return {
            "summary": summary,
            "per_scenario_stats": per_scenario_summary,
            "detailed_metrics": detailed_metrics,
        }
