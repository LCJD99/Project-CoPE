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

Offline inference:

```bash
python infer.py --stage stage2 --config configs/infer/infer_stage2.yaml --input data/sample.json --output outputs/stage2_pred.json
python infer.py --stage stage3 --config configs/infer/infer_stage3.yaml --input data/sample.json --output outputs/stage3_pred.json
```
