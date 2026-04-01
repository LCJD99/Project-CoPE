from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml


SUPPORTED_BASE_MODELS = {
    "/AI/HF_MODELS/Qwen2.5-3B",
    "/AI/HF_MODELS/Qwen2.5-7B",
}


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_base_model_path(path: str) -> None:
    if path not in SUPPORTED_BASE_MODELS:
        raise ValueError(
            f"Unsupported model.base_model '{path}'. "
            f"Expected one of: {sorted(SUPPORTED_BASE_MODELS)}"
        )


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | None, repo_root: Path | None = None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return str(p)
    root = repo_root or get_repo_root()
    return str((root / p).resolve())


def _git_commit_sha(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def emit_run_metadata(output_dir: str, resolved_config: Dict[str, Any]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    repo_root = get_repo_root()
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_sha(repo_root),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "resolved_config": resolved_config,
    }
    with open(out / "run_metadata.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)
