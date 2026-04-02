# CoPE Training Repository

This repository contains the CoPE training mainline only:

- Stage 1: Configured Tool Token Alignment (V3 only)
- Stage 2: Structured Program Fine-Tuning
- Stage 3: Local-advantage-guided Policy Optimization (lag-grpo only)

## Entrypoints

Training (compatible interface):

```bash
python train.py --stage stage1 --config configs/stage1/stage1_qwen25_7b.yaml
python train.py --stage stage2 --config configs/stage2/stage2_qwen25_7b.yaml
python train.py --stage stage3 --config configs/stage3/stage3_qwen25_7b_lag_grpo.yaml
```

Stage1 tokenizer assets (expanded tokenizer):

```bash
bash scripts/stage1/prepare_stage1_assets.sh
# optional overrides:
# BASE_MODEL=/AI/HF_MODELS/Qwen2.5-3B REGISTRY=data/00_global/tool_registry.json TOKENIZER_OUT=checkpoints/01_stage1/tokenizer_expanded bash scripts/stage1/prepare_stage1_assets.sh
```

Stage1 required input paths (release contract):

- `data/00_global/system.json` -> `system_config_path`
- `data/01_stage1/train.jsonl` -> `train_data`
- `data/00_global/tool_registry.json` -> `tool_registry`
- `checkpoints/01_stage1/tokenizer_expanded` -> `tokenizer`

Validate these paths before training:

```bash
bash scripts/stage1/check_stage1_inputs.sh
```

Data/checkpoint path convention:

- `data/00_global`: global shared data (registry/system/profiling)
- `data/01_stage1`: Stage1-only training data
- `data/02_stage2`: Stage2-only training data
- `data/03_stage3`: Stage3-only RL data
- `checkpoints/01_stage1`: Stage1 assets and checkpoints (includes Stage1 tokenizer)
- `checkpoints/02_stage2`: Stage2 tokenizer/embeddings/checkpoints
- `checkpoints/03_stage3`: Stage3 checkpoints

Offline inference:

```bash
python infer.py --stage stage2 --config configs/infer/infer_stage2.yaml --input data/sample.json --output outputs/stage2_pred.json
python infer.py --stage stage3 --config configs/infer/infer_stage3.yaml --input data/sample.json --output outputs/stage3_pred.json
```
