#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

SYSTEM_CONFIG="${SYSTEM_CONFIG:-data/raw/system.json}"
TRAIN_DATA="${TRAIN_DATA:-data/processed/train.jsonl}"
TOOL_REGISTRY="${TOOL_REGISTRY:-data/registry/tool_registry.json}"
TOKENIZER_DIR="${TOKENIZER_DIR:-checkpoints/tokenizer_expanded}"

echo "Checking Stage1 required inputs..."

[[ -f "$SYSTEM_CONFIG" ]] || { echo "Missing: $SYSTEM_CONFIG"; exit 1; }
[[ -f "$TRAIN_DATA" ]] || { echo "Missing: $TRAIN_DATA"; exit 1; }
[[ -f "$TOOL_REGISTRY" ]] || { echo "Missing: $TOOL_REGISTRY"; exit 1; }
[[ -f "$TOKENIZER_DIR/tokenizer_config.json" ]] || {
  echo "Missing: $TOKENIZER_DIR/tokenizer_config.json"
  echo "Run: bash scripts/stage1/prepare_stage1_assets.sh"
  exit 1
}

echo "OK system_config_path: $SYSTEM_CONFIG"
echo "OK train_data:        $TRAIN_DATA"
echo "OK tool_registry:     $TOOL_REGISTRY"
echo "OK tokenizer:         $TOKENIZER_DIR"
