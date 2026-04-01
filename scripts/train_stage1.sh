#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/stage1/stage1_qwen25_7b.yaml}"
python train.py --stage stage1 --config "$CONFIG"
