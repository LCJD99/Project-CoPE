from __future__ import annotations

from typing import Dict, List


STAGE2_CONTROL_TOKENS: List[str] = [
    "<EXEC>",
    "<FINISH>",
    "<WAIT>",
    "<EQ>",
    "<STATEMENT>",
    "<END_STATEMENT>",
]

STAGE2_REFERENCE_TOKENS: List[str] = [f"<REF_{i}>" for i in range(32)]

STAGE2_TOKENS: List[str] = STAGE2_CONTROL_TOKENS + STAGE2_REFERENCE_TOKENS

TOKEN_INIT_MAP: Dict[str, List[str]] = {
    "<EXEC>": ["execute", "run", "start", "launch"],
    "<FINISH>": ["finish", "complete", "return", "end"],
    "<WAIT>": ["wait", "pause", "hold", "pending"],
    "<EQ>": ["equals", "equal", "assign", "set"],
    "<STATEMENT>": ["statement", "line", "step", "instruction"],
    "<END_STATEMENT>": ["end", "close", "terminate", "complete"],
}
for i in range(32):
    TOKEN_INIT_MAP[f"<REF_{i}>"] = [str(i)]
