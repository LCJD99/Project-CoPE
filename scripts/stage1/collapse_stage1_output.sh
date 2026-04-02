#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

STAGE1_CONFIG="${STAGE1_CONFIG:-configs/stage1/stage1_qwen25_7b.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-checkpoints/01_stage1/memorization}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f "$STAGE1_CONFIG" ]]; then
  echo "Error: stage1 config not found: $STAGE1_CONFIG"
  exit 1
fi

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
  CHECKPOINT_DIR="$(python - <<'PY' "$STAGE1_CONFIG"
import sys
import yaml
from pathlib import Path

cfg_path = Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text())
out = Path(cfg["output_dir"]) / "final_model"
print(out.as_posix())
PY
)"
fi

if [[ -z "${REGISTRY_FILE:-}" ]]; then
  REGISTRY_FILE="$(python - <<'PY' "$STAGE1_CONFIG"
import sys
import yaml
from pathlib import Path

cfg_path = Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text())
print(cfg.get("tool_registry", "data/00_global/tool_registry.json"))
PY
)"
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "Error: stage1 checkpoint directory not found: $CHECKPOINT_DIR"
  echo "Expected final model path from config output_dir + /final_model."
  exit 1
fi

if [[ ! -f "$REGISTRY_FILE" ]]; then
  echo "Error: registry file not found: $REGISTRY_FILE"
  exit 1
fi

RUN_NAME="$(python - <<'PY' "$STAGE1_CONFIG"
import re
import sys
import yaml
from pathlib import Path

cfg_path = Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text())
raw = cfg.get("wandb_run_name") or cfg.get("experiment", {}).get("name") or "stage1"
safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(raw)).strip("_")
print(safe or "stage1")
PY
)"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${OUTPUT_BASE}/${STAMP}_${RUN_NAME}"
LATEST_LINK="${OUTPUT_BASE}/latest"

mkdir -p "$OUTPUT_BASE"

echo "Collapsing Stage1 checkpoint:"
echo "  config:      $STAGE1_CONFIG"
echo "  checkpoint:  $CHECKPOINT_DIR"
echo "  registry:    $REGISTRY_FILE"
echo "  output:      $OUTPUT_DIR"
echo "  latest_link: $LATEST_LINK"
echo "  device:      $DEVICE"

PYTHONPATH=. python -m src.cope.stage1.collapse \
  --checkpoint "$CHECKPOINT_DIR" \
  --config "$STAGE1_CONFIG" \
  --registry "$REGISTRY_FILE" \
  --output "$OUTPUT_DIR" \
  --device "$DEVICE"

OUTPUT_DIR_ABS="$(python - <<'PY' "$OUTPUT_DIR"
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
ln -sfn "$OUTPUT_DIR_ABS" "$LATEST_LINK"

echo ""
echo "Stage1 collapse completed."
echo "Stage2 config input path should use:"
echo "  stage1_checkpoint: checkpoints/01_stage1/memorization/latest"
