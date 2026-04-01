from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Set, Tuple, TypedDict

from src.cope.stage3.hierarchical.prefix_signature import (
    build_prefix_signature,
    update_tool_categories,
)
from src.cope.stage3.simulator.plan_parser import parse_plan


class LineEntry(TypedDict):
    line: str
    stripped: str
    offset: int
    stripped_offset: int


class TokenEntry(TypedDict):
    start: int
    end: int


@dataclass
class DecisionUnit:
    decision_id: str
    decision_type: str
    token_span: Tuple[int, int]
    prefix_signature: str
    original_choice: Dict[str, object]
    statement_index: int


@dataclass
class DecisionExtractionResult:
    units: List[DecisionUnit] = field(default_factory=list)
    skipped_count: int = 0
    skipped_reasons: List[str] = field(default_factory=list)


def _line_entries(plan_text: str) -> List[LineEntry]:
    entries: List[LineEntry] = []
    offset = 0
    for line in plan_text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            stripped_offset = line.find(stripped)
            if stripped_offset < 0:
                stripped_offset = 0
            entries.append(
                {
                    "line": line,
                    "stripped": stripped,
                    "offset": offset,
                    "stripped_offset": stripped_offset,
                }
            )
        offset += len(line)

    return entries


def _statement_skeleton(statement: Dict[str, object]) -> str:
    if statement["type"] == "exec":
        if statement.get("wait"):
            return "EXEC_WAIT"
        return "EXEC"
    if statement["type"] == "sync":
        return "SYNC"
    if statement["type"] == "finish":
        return "FINISH"
    return str(statement["type"]).upper()


def _token_entries(plan_text: str) -> List[TokenEntry]:
    return [
        {"start": match.start(), "end": match.end()}
        for match in re.finditer(r"\S+", plan_text)
    ]


def _char_span_to_token_span(
    token_entries: List[TokenEntry], span_start: int, span_end: int
) -> Optional[Tuple[int, int]]:
    start_index = -1
    end_index = -1
    for token_index, token in enumerate(token_entries):
        token_start = token["start"]
        token_end = token["end"]
        if token_end <= span_start:
            continue
        if token_start >= span_end:
            break
        if start_index < 0:
            start_index = token_index
        end_index = token_index + 1

    if start_index < 0 or end_index < 0:
        return None
    return (start_index, end_index)


def extract_decision_units(
    plan_text: str, task_complexity: str
) -> DecisionExtractionResult:
    try:
        parsed = parse_plan(plan_text)
    except ValueError as exc:
        return DecisionExtractionResult(
            units=[], skipped_count=1, skipped_reasons=[str(exc)]
        )

    lines = _line_entries(plan_text)
    token_entries = _token_entries(plan_text)
    if len(lines) < len(parsed):
        return DecisionExtractionResult(
            units=[],
            skipped_count=1,
            skipped_reasons=["Plan parsing/line mapping mismatch"],
        )

    units: List[DecisionUnit] = []
    defined_refs: List[str] = []
    pending_refs: Set[str] = set()
    used_tool_categories: List[str] = []
    skipped_reasons: List[str] = []

    for statement_index, statement in enumerate(parsed):
        line_entry = lines[statement_index]
        line_text = str(line_entry["stripped"])
        line_offset = line_entry["offset"] + line_entry["stripped_offset"]
        statement_type = str(statement["type"])
        if statement_type == "exec":
            skeleton = _statement_skeleton(statement)
            tool_token = str(statement["tool"])
            tool_start = line_text.find(tool_token)
            if tool_start >= 0:
                tool_span = _char_span_to_token_span(
                    token_entries,
                    line_offset + tool_start,
                    line_offset + tool_start + len(tool_token),
                )
                if tool_span is not None:
                    tool_prefix_signature = build_prefix_signature(
                        defined_refs=defined_refs,
                        pending_refs=pending_refs,
                        statement_skeleton=skeleton,
                        task_complexity=task_complexity,
                        used_tool_categories=used_tool_categories,
                    )
                    units.append(
                        DecisionUnit(
                            decision_id=f"stmt_{statement_index}_tool_choice",
                            decision_type="tool_choice",
                            token_span=tool_span,
                            prefix_signature=tool_prefix_signature,
                            original_choice={"tool": tool_token},
                            statement_index=statement_index,
                        )
                    )
                else:
                    skipped_reasons.append(
                        f"stmt_{statement_index}: tool token span not found for {tool_token}"
                    )
            else:
                skipped_reasons.append(
                    f"stmt_{statement_index}: tool span not found for {tool_token}"
                )

            wait_refs = [str(ref) for ref in statement.get("wait", [])]
            if wait_refs:
                wait_start = line_text.find("<WAIT>")
                exec_start = line_text.find("<EXEC>")
                if wait_start >= 0 and exec_start > wait_start:
                    wait_span = _char_span_to_token_span(
                        token_entries,
                        line_offset + wait_start,
                        line_offset + exec_start,
                    )
                    if wait_span is not None:
                        wait_prefix_signature = build_prefix_signature(
                            defined_refs=defined_refs,
                            pending_refs=pending_refs.union(wait_refs),
                            statement_skeleton=skeleton,
                            task_complexity=task_complexity,
                            used_tool_categories=used_tool_categories,
                        )
                        units.append(
                            DecisionUnit(
                                decision_id=f"stmt_{statement_index}_wait_control",
                                decision_type="wait_control",
                                token_span=wait_span,
                                prefix_signature=wait_prefix_signature,
                                original_choice={"wait_for": wait_refs},
                                statement_index=statement_index,
                            )
                        )
                    else:
                        skipped_reasons.append(
                            f"stmt_{statement_index}: wait token span not found"
                        )
                else:
                    skipped_reasons.append(
                        f"stmt_{statement_index}: wait span not found"
                    )

            pending_refs.update(wait_refs)
            ref = str(statement["ref"])
            if ref not in defined_refs:
                defined_refs.append(ref)
            update_tool_categories(used_tool_categories, tool_token)
            continue

        if statement_type == "sync":
            pending_refs.difference_update(
                str(ref) for ref in statement.get("refs", [])
            )
            continue

        if statement_type == "finish":
            pending_refs.difference_update(
                str(ref) for ref in statement.get("refs", [])
            )

    return DecisionExtractionResult(
        units=units,
        skipped_count=len(skipped_reasons),
        skipped_reasons=skipped_reasons,
    )
