import hashlib
import json
from collections import Counter
from typing import Dict, Iterable, List


RESOURCE_TIER_MARKERS = {"LOW", "MED", "HIGH"}


def tool_category_from_token(tool_token: str) -> str:
    """Derive a deterministic tool category from tool token text."""
    token = tool_token.strip("<>")
    parts = token.split("_")
    marker_index = next(
        (index for index, part in enumerate(parts) if part in RESOURCE_TIER_MARKERS),
        -1,
    )
    if marker_index > 0:
        return "_".join(parts[:marker_index])
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return token


def build_prefix_signature(
    *,
    defined_refs: Iterable[str],
    pending_refs: Iterable[str],
    statement_skeleton: str,
    task_complexity: str,
    used_tool_categories: Iterable[str],
) -> str:
    """Build a deterministic prefix signature from normalized context."""
    category_counter = Counter(used_tool_categories)
    category_buckets = [
        [category, category_counter[category]] for category in sorted(category_counter)
    ]
    payload: Dict[str, object] = {
        "defined_refs": sorted(set(defined_refs)),
        "pending_refs": sorted(set(pending_refs)),
        "statement_skeleton": statement_skeleton,
        "task_complexity": task_complexity,
        "used_tool_categories": category_buckets,
    }
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def update_tool_categories(
    categories: List[str],
    tool_token: str,
) -> List[str]:
    """Append category occurrence while preserving deterministic sequence."""
    category = tool_category_from_token(tool_token)
    categories.append(category)
    return categories
