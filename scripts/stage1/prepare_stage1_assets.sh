#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

BASE_MODEL="${BASE_MODEL:-/AI/HF_MODELS/Qwen2.5-7B}"
REGISTRY="${REGISTRY:-data/00_global/tool_registry.json}"
TOKENIZER_OUT="${TOKENIZER_OUT:-checkpoints/01_stage1/tokenizer_expanded}"

if [[ ! -f "$REGISTRY" ]]; then
  echo "Error: registry not found: $REGISTRY"
  echo "Please generate it first (e.g. data preparation pipeline)."
  exit 1
fi

mkdir -p "$TOKENIZER_OUT"

python scripts/stage1/expand_tokenizer.py \
  --base_model "$BASE_MODEL" \
  --registry "$REGISTRY" \
  --output "$TOKENIZER_OUT" \
  --verify

echo ""
echo "Stage1 tokenizer assets prepared:"
echo "  base_model:   $BASE_MODEL"
echo "  registry:     $REGISTRY"
echo "  tokenizer_out:$TOKENIZER_OUT"
