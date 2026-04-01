#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

LEGACY_ROOT="${LEGACY_ROOT:-$PROJECT_ROOT/ref}"

SRC_SYSTEM="$LEGACY_ROOT/data/raw/system.json"
SRC_TRAIN="$LEGACY_ROOT/data/processed/train.jsonl"
SRC_REGISTRY="$LEGACY_ROOT/data/registry/tool_registry.json"
SRC_TOKENIZER_DIR="$LEGACY_ROOT/checkpoints/tokenizer_expanded"

DST_SYSTEM="data/raw/system.json"
DST_TRAIN="data/processed/train.jsonl"
DST_REGISTRY="data/registry/tool_registry.json"
DST_TOKENIZER_DIR="checkpoints/tokenizer_expanded"

mkdir -p data/raw data/processed data/registry checkpoints

if [[ ! -f "$SRC_SYSTEM" ]]; then
  echo "Error: missing $SRC_SYSTEM"
  exit 1
fi
if [[ ! -f "$SRC_TRAIN" ]]; then
  echo "Error: missing $SRC_TRAIN"
  exit 1
fi
if [[ ! -f "$SRC_REGISTRY" ]]; then
  echo "Error: missing $SRC_REGISTRY"
  exit 1
fi
if [[ ! -d "$SRC_TOKENIZER_DIR" ]]; then
  echo "Error: missing $SRC_TOKENIZER_DIR"
  exit 1
fi

cp "$SRC_SYSTEM" "$DST_SYSTEM"
cp "$SRC_TRAIN" "$DST_TRAIN"
cp "$SRC_REGISTRY" "$DST_REGISTRY"
rm -rf "$DST_TOKENIZER_DIR"
cp -r "$SRC_TOKENIZER_DIR" "$DST_TOKENIZER_DIR"

echo "Migrated Stage1 assets from legacy repo:"
echo "  system:    $DST_SYSTEM"
echo "  train:     $DST_TRAIN"
echo "  registry:  $DST_REGISTRY"
echo "  tokenizer: $DST_TOKENIZER_DIR"

echo ""
echo "Quick check against configs/stage1/stage1_qwen25_7b.yaml:"
test -f "$DST_SYSTEM" && echo "  OK system_config_path"
test -f "$DST_TRAIN" && echo "  OK train_data"
test -f "$DST_REGISTRY" && echo "  OK tool_registry"
test -f "$DST_TOKENIZER_DIR/tokenizer_config.json" && echo "  OK tokenizer"
