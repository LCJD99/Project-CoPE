# AGENTS.md

## 1. Project Scope

This repository is the **CoPE release training repo** and keeps only the mainline capabilities:

- Stage 1: Configured Tool Token Alignment
- Stage 2: Structured Program Fine-Tuning
- Stage 3: Local-advantage-guided Policy Optimization
- Offline inference

No baseline system and no legacy compatibility branches are part of the release scope.

Release positioning requirement:
- This repository may originate from internal refactoring work, but the release codebase must remain productized and self-contained. Do not expose refactoring lineage from local `ref` sources in tracked files, naming, or release metadata.

## 2. Unified Entrypoints

- Training entrypoint: `train.py`
  - `python train.py --stage stage1 --config <yaml>`
  - `python train.py --stage stage2 --config <yaml>`
  - `python train.py --stage stage3 --config <yaml>`
- Inference entrypoint: `infer.py`
  - `python infer.py --stage stage2 --config <yaml> --input <json|jsonl> --output <json>`
  - `python infer.py --stage stage3 --config <yaml> --input <json|jsonl> --output <json>`

## 3. Directory Naming Rules (Mandatory)

### 3.1 `data` directories

- `data/00_global`: globally shared static inputs (for example `system.json`, `tool_registry.json`, profiling files)
- `data/01_stage1`: Stage1 training data
- `data/02_stage2`: Stage2 training data
- `data/03_stage3`: Stage3 RL data

### 3.2 `checkpoints` directories

- `checkpoints/01_stage1`: Stage1 assets and checkpoints (including Stage1 tokenizer)
- `checkpoints/02_stage2`: Stage2 tokenizer/embedding/checkpoints
- `checkpoints/03_stage3`: Stage3 checkpoints

Notes:
- `tokenizer_expanded` explicitly belongs to Stage1: `checkpoints/01_stage1/tokenizer_expanded`.
- Stage2 uses its own tokenizer and must not reuse the Stage1 tokenizer directory.

## 4. Configuration Rules

Config files live under:

- `configs/stage1/*.yaml`
- `configs/stage2/*.yaml`
- `configs/stage3/*.yaml`
- `configs/infer/*.yaml`

Must-follow constraints:

- All `data/` and `checkpoints/` paths must follow the numbered directory convention above.
- `model.base_model` is restricted to:
  - `/AI/HF_MODELS/Qwen2.5-3B`
  - `/AI/HF_MODELS/Qwen2.5-7B`
- Stage3 main branch only allows: `training_mode: hierarchical_grpo`.

## 5. Stage1 Input Contract

For `configs/stage1/stage1_qwen25_7b.yaml`, required input paths are:

- `system_config_path`: `data/00_global/system.json`
- `train_data`: `data/01_stage1/train.jsonl`
- `tool_registry`: `data/00_global/tool_registry.json`
- `tokenizer`: `checkpoints/01_stage1/tokenizer_expanded`

Run before training:

```bash
bash scripts/stage1/check_stage1_inputs.sh
```

Prepare Stage1 tokenizer assets:

```bash
bash scripts/stage1/prepare_stage1_assets.sh
```

## 6. Code Organization

- Core implementation is under `src/cope/`:
  - `stage1/`, `stage2/`, `stage3/`, `inference/`, `common/`
- `scripts/` contains only thin wrappers and preflight checks, not core training logic.
- `docs/decisions/` stores architecture decisions.
- `docs/experiments/` stores experiment conclusions and retrospectives.

## 7. Release Cleanliness Rules

- Do not include “legacy migration” procedures as part of release workflow.
- Do not keep compatibility branches solely for old path layouts on the main branch.
- Do not commit large datasets, model weights, or wandb artifacts.
- `.agents/` and `ref` are local helper content and must not be included in release commits.

## 8. Branch and Commit Workflow

- `dev` is the integration branch.
- Feature branches are created from `dev`, merged back into `dev`, then deleted.
- Recommended commit message prefixes:
  - `feat(stageX): ...`
  - `fix(stageX): ...`
  - `chore(config): ...`
  - `docs: ...`
