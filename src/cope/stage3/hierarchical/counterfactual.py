import hashlib
import json
from typing import Any, Dict, List, Optional


def _canonical_tool_name(tool: Any) -> str:
    token = str(tool).strip()
    if token.startswith("<") and token.endswith(">"):
        token = token[1:-1].strip()
    return token


def _bracket_tool_token(tool: Any) -> str:
    token = _canonical_tool_name(tool)
    if not token:
        return ""
    return f"<{token}>"


def build_counterfactual_cache_key(
    *,
    decision_type: str,
    prefix_signature: str,
    candidate_choice: Optional[Dict[str, object]] = None,
) -> str:
    """Build a deterministic cache key for a counterfactual candidate."""
    payload = {
        "decision_type": decision_type,
        "prefix_signature": prefix_signature,
        "candidate_choice": candidate_choice or {},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def generate_counterfactual_candidates(
    decision_type: str,
    original_choice: Dict[str, Any],
    candidate_pool: Any,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    """Generate deterministic counterfactual candidate choices."""
    if max_candidates <= 0:
        return []

    if decision_type == "tool_choice":
        original_tool = _canonical_tool_name(original_choice.get("tool", ""))
        unique_sorted_tools = sorted(
            {
                _canonical_tool_name(tool)
                for tool in candidate_pool
                if _canonical_tool_name(tool)
            }
        )
        alternatives = [tool for tool in unique_sorted_tools if tool != original_tool]
        return [
            {"tool": _bracket_tool_token(tool)}
            for tool in alternatives[:max_candidates]
        ]

    if decision_type == "wait_control":
        original_refs = [str(ref) for ref in original_choice.get("wait_refs", [])]
        results: List[Dict[str, Any]] = []

        def _append_candidate(wait_refs: List[str]) -> None:
            candidate = {"wait_refs": wait_refs}
            if candidate not in results:
                results.append(candidate)

        if bool(candidate_pool.get("keep", False)):
            _append_candidate(list(original_refs))

        removable_refs = sorted({str(ref) for ref in candidate_pool.get("remove", [])})
        for removable_ref in removable_refs:
            if removable_ref in original_refs:
                updated_refs = [ref for ref in original_refs if ref != removable_ref]
                _append_candidate(updated_refs)
                if len(results) >= max_candidates:
                    return results[:max_candidates]

        replace_map = candidate_pool.get("replace", {})
        for index, source_ref in enumerate(original_refs):
            replacements = replace_map.get(source_ref, [])
            for replacement_ref in sorted({str(ref) for ref in replacements}):
                if replacement_ref == source_ref:
                    continue
                updated_refs = list(original_refs)
                updated_refs[index] = replacement_ref
                _append_candidate(updated_refs)
                if len(results) >= max_candidates:
                    return results[:max_candidates]

        return results[:max_candidates]

    raise ValueError(f"Unsupported decision_type: {decision_type}")
