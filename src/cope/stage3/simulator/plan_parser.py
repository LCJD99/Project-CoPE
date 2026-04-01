import re
from typing import Dict, List


EXEC_RE = re.compile(
    r"^\s*(?P<ref><REF_\d+>)\s*=\s*(?P<wait><WAIT>\s*(?:<REF_\d+>\s*)+)?"
    r"<EXEC>\s*(?P<tool><[A-Z0-9_]+>)\s*(?P<args>\(.*\))\s*$"
)
SYNC_RE = re.compile(r"^\s*<SYNC>\s*(?P<refs>(?:<REF_\d+>\s*)+)\s*$")
FINISH_RE = re.compile(r"^\s*<FINISH>\s*(?P<refs>(?:<REF_\d+>\s*)+)\s*$")
REF_RE = re.compile(r"<REF_\d+>")


def parse_plan(plan_text: str) -> List[Dict]:
    lines = [line.strip() for line in plan_text.strip().splitlines() if line.strip()]
    parsed: List[Dict] = []
    for line in lines:
        exec_match = EXEC_RE.match(line)
        if exec_match:
            wait_block = exec_match.group("wait")
            wait_refs = []
            if wait_block:
                wait_refs = REF_RE.findall(wait_block)
            parsed.append(
                {
                    "type": "exec",
                    "ref": exec_match.group("ref"),
                    "wait": wait_refs,
                    "tool": exec_match.group("tool"),
                    "args": exec_match.group("args"),
                }
            )
            continue

        sync_match = SYNC_RE.match(line)
        if sync_match:
            refs = REF_RE.findall(sync_match.group("refs"))
            parsed.append({"type": "sync", "refs": refs})
            continue

        finish_match = FINISH_RE.match(line)
        if finish_match:
            refs = REF_RE.findall(finish_match.group("refs"))
            parsed.append({"type": "finish", "refs": refs})
            continue

        raise ValueError(f"Invalid plan line: {line}")

    return parsed
