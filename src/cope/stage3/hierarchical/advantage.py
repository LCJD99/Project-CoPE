from collections import defaultdict
import math
from typing import Dict, List, Optional, Tuple, cast


def compute_micro_advantages(
    records: List[Dict[str, object]],
    min_group_size: int,
    eps: float,
) -> List[Dict[str, object]]:
    if min_group_size <= 0:
        raise ValueError("min_group_size must be > 0")
    if eps < 0:
        raise ValueError("eps must be >= 0")

    grouped_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if "delta" not in record:
            raise KeyError("delta")

        has_explicit_group = "decision_type" in record or "prefix_signature" in record
        if has_explicit_group:
            if "decision_type" not in record:
                raise KeyError("decision_type")
            if "prefix_signature" not in record:
                raise KeyError("prefix_signature")
            group_key = (str(record["decision_type"]), str(record["prefix_signature"]))
        else:
            if "group" not in record:
                raise KeyError("group")
            raw_group = cast(object, record["group"])
            if not isinstance(raw_group, tuple) or len(raw_group) != 2:
                raise ValueError(
                    "group must be a tuple(decision_type, prefix_signature)"
                )
            group_key = (str(raw_group[0]), str(raw_group[1]))

        grouped_indices[group_key].append(index)

    output = [dict(record) for record in records]
    for indices in grouped_indices.values():
        deltas = [float(cast(object, records[index]["delta"])) for index in indices]
        group_size = len(deltas)
        mean = sum(deltas) / group_size
        variance = sum((delta - mean) ** 2 for delta in deltas) / group_size

        if group_size < min_group_size or variance < eps:
            for index in indices:
                output[index]["a_micro"] = 0.0
            continue

        std = math.sqrt(variance)
        for index, delta in zip(indices, deltas):
            output[index]["a_micro"] = (delta - mean) / std

    return output


def fuse_hierarchical_advantage(
    *,
    a_macro: float,
    a_micro: float,
    macro_weight: float,
    micro_weight: float,
    clip_range: Optional[float] = None,
) -> float:
    fused = macro_weight * a_macro + micro_weight * a_micro
    if clip_range is None:
        return fused
    return max(-clip_range, min(clip_range, fused))
