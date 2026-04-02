#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/stage3/stage3_qwen25_7b_lag_grpo.yaml}"
python train.py --stage stage3 --config "$CONFIG"
