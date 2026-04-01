# CoPE Train Repo Refactor Design

**Date:** 2026-04-01  
**Scope:** Repository refactor for training codebase (TOMAS-LLM legacy -> CoPE), preserving training behavior while enforcing clean research-repo contracts.

## 1. Confirmed Requirements

- Keep only training mainline + minimal necessary evaluation in main branch.
- Keep CLI compatibility for training entrypoint:
  - `python train.py --stage stage1|stage2|stage3 --config <yaml>`
- Follow `llm-research-repo-management` structure conventions.
- Stage 1 keeps only V3 implementation.
- Stage 3 keeps only `hierarchical_grpo` mode in main branch.
- Add one unified offline inference entrypoint (file-in/file-out), no serving API.
- Base model paths should support:
  - `/AI/HF_MODELS/Qwen2.5-3B`
  - `/AI/HF_MODELS/Qwen2.5-7B`

## 2. Non-Goals

- No baseline system migration into main training branch.
- No Stage1 V1/V2 maintenance.
- No Stage3 `global_grpo` branch in mainline.
- No online inference service.

## 3. Recommended Approach

Use **medium refactor**:

- Keep algorithmic behavior as-is for Stage1 V3, Stage2 SFT planning, Stage3 hierarchical GRPO.
- Re-organize code into clear stage packages + shared common package.
- Keep `train.py` as stable compatibility layer.
- Add `infer.py` as unified offline inference entrypoint.

This balances maintainability and migration risk.

## 4. Repository Contract

Target structure:

- `src/cope/stage1/`: Stage1 configured tool token alignment (V3 only)
- `src/cope/stage2/`: Stage2 structured program fine-tuning
- `src/cope/stage3/`: Stage3 local-advantage-guided policy optimization (hierarchical GRPO only)
- `src/cope/inference/`: stage-routed offline inference orchestration
- `src/cope/common/`: shared config/runtime/path/logging/io utilities
- `configs/stage1/`, `configs/stage2/`, `configs/stage3/`, `configs/infer/`
- `scripts/`: thin wrappers only (environment checks + CLI pass-through)
- `tests/`: minimal correctness and routing tests
- `docs/decisions/`: architecture decisions
- `docs/experiments/`: experiment conclusions and comparisons
- `data/`: small metadata/manifests/readmes only in git
- `outputs/`: disposable run artifacts
- `artifacts/`: exported reusable artifacts

## 5. Interface Contracts

### 5.1 Training Entry

Keep existing interface:

- `python train.py --stage stage1 --config ...`
- `python train.py --stage stage2 --config ...`
- `python train.py --stage stage3 --config ...`

Internal dispatch will move to `src/cope/*` modules.

### 5.2 Inference Entry

Add unified offline interface:

- `python infer.py --stage stage2 --config <yaml> --input <json|jsonl> --output <json>`
- `python infer.py --stage stage3 --config <yaml> --input <json|jsonl> --output <json>`

No interactive/server mode.

## 6. Configuration Contract

All stage configs should be normalized to:

- `experiment.name`
- `model.base_model`
- `paths.*` for data/registry/checkpoint/output
- `runtime.*` for seed/device/precision

Rules:

- Remove hardcoded absolute project-local paths from configs.
- Resolve relative paths against repository root.
- Stage3 validation enforces `training_mode: hierarchical_grpo`.

## 7. Data, Run, Artifact Lineage

Each meaningful run should emit:

- resolved config snapshot
- source revision marker
- runtime environment summary
- machine-readable metrics
- text log location
- checkpoint retention metadata

Checkpoint classes:

- recovery checkpoints (rolling cleanup)
- candidate best checkpoints
- release/export checkpoints (promoted explicitly)

## 8. Migration Strategy

Phase 1: runnable migration

- Copy/port Stage1 V3, Stage2, Stage3(hierarchical) into `src/cope/*`.
- Keep old behavior and external CLI.
- Add compatibility adapters only where needed.

Phase 2: cleanup and contract hardening

- Remove legacy entrypoints/modules outside new path.
- Normalize configs and script wrappers.
- Add run metadata emission and minimal tests.

## 9. Minimal Regression Gate

Must pass before sign-off:

- CLI routing to stage modules works for all three stages.
- Stage3 rejects non-hierarchical mode.
- Stage2 dataset load/tokenize smoke passes.
- Stage3 reward pipeline smoke passes.
- Inference entrypoint works for stage2/stage3 in offline mode.
- Run output includes resolved config and runtime metadata.

## 10. Risk Notes

- Import path breakage during module relocation.
- Hidden assumptions in scripts that rely on old directory layout.
- Mixed precision/device defaults may diverge if not centralized.
- Legacy artifacts in repo (wandb/checkpoints/models) can pollute git status if not isolated by policy.

## 11. Decision

Proceed with medium refactor and implement in-place inside current repository, preserving training CLI compatibility and adding a unified offline inference entrypoint.
