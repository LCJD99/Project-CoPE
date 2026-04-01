# CoPE Training Repository

This repository contains the CoPE training mainline only:

- Stage 1: Configured Tool Token Alignment (V3 only)
- Stage 2: Structured Program Fine-Tuning
- Stage 3: Local-advantage-guided Policy Optimization (hierarchical GRPO only)

## Entrypoints

Training (compatible interface):

```bash
python train.py --stage stage1 --config configs/stage1/stage1_qwen25_7b.yaml
python train.py --stage stage2 --config configs/stage2/stage2_qwen25_7b.yaml
python train.py --stage stage3 --config configs/stage3/stage3_qwen25_7b_hgrpo.yaml
```

Stage1 tokenizer assets (expanded tokenizer):

```bash
bash scripts/stage1/prepare_stage1_assets.sh
# optional overrides:
# BASE_MODEL=/AI/HF_MODELS/Qwen2.5-3B REGISTRY=data/registry/tool_registry.json TOKENIZER_OUT=checkpoints/tokenizer_expanded bash scripts/stage1/prepare_stage1_assets.sh
```

Stage1 data migration from legacy repository (`ref` symlink by default):

```bash
bash scripts/stage1/migrate_data_from_legacy.sh
# optional:
# LEGACY_ROOT=/path/to/TOMAS-LLM bash scripts/stage1/migrate_data_from_legacy.sh
```

Offline inference:

```bash
python infer.py --stage stage2 --config configs/infer/infer_stage2.yaml --input data/sample.json --output outputs/stage2_pred.json
python infer.py --stage stage3 --config configs/infer/infer_stage3.yaml --input data/sample.json --output outputs/stage3_pred.json
```
