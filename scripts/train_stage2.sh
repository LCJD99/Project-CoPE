#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/stage2/stage2_qwen25_7b.yaml}"
python train.py --stage stage2 --config "$CONFIG"
