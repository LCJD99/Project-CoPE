# CoPE Training Repository Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the repository into a clean, reproducible Stage1/Stage2/Stage3 training stack (Stage1 V3 only, Stage3 hierarchical GRPO only) while preserving `train.py` CLI compatibility and adding a unified offline `infer.py` entrypoint.

**Architecture:** Introduce a new package root `src/cope` with clear stage boundaries (`stage1`, `stage2`, `stage3`, `inference`, `common`), keep `train.py` as a compatibility router, and normalize configuration layout under `configs/`. Migrate only the required training/inference code from `ref` and drop non-goal subsystems (baseline, legacy stage trainers, global GRPO mode). Add minimal smoke tests that prove CLI routing, config validation, and core stage dataflow.

**Tech Stack:** Python 3, PyTorch, Transformers, PEFT, TRL, YAML configs, pytest

---

## Chunk 1: Bootstrap New Repository Skeleton

### Task 1: Create package directories and root files

**Files:**
- Create: `src/cope/__init__.py`
- Create: `src/cope/common/__init__.py`
- Create: `src/cope/stage1/__init__.py`
- Create: `src/cope/stage2/__init__.py`
- Create: `src/cope/stage3/__init__.py`
- Create: `src/cope/inference/__init__.py`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `README.md`

- [ ] **Step 1: Write failing repository structure test**

```python
def test_repo_has_core_packages():
    assert Path("src/cope/stage1").is_dir()
    assert Path("src/cope/stage2").is_dir()
    assert Path("src/cope/stage3").is_dir()
    assert Path("src/cope/inference").is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_repo_layout.py::test_repo_has_core_packages -v`  
Expected: FAIL because directories/files do not exist.

- [ ] **Step 3: Create minimal package skeleton**

```python
# src/cope/__init__.py
__all__ = ["common", "stage1", "stage2", "stage3", "inference"]
```

- [ ] **Step 4: Re-run test**

Run: `pytest tests/test_repo_layout.py::test_repo_has_core_packages -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cope requirements.txt .gitignore README.md tests/test_repo_layout.py
git commit -m "chore: bootstrap cope package skeleton"
```

## Chunk 2: Migrate Core Training Code

### Task 2: Stage1 V3 only migration

**Files:**
- Create: `src/cope/stage1/trainer.py`
- Create: `src/cope/stage1/model.py`
- Create: `src/cope/stage1/profile_encoder.py`
- Create: `src/cope/stage1/virtual_tokens.py`
- Create: `src/cope/stage1/collapse.py`
- Modify: `src/cope/stage1/__init__.py`
- Test: `tests/test_stage1_entry.py`

- [ ] **Step 1: Write failing stage1 routing test**

```python
def test_stage1_dispatch_only_v3(monkeypatch):
    ...
    assert called["train_stage1_v3"] is True
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/test_stage1_entry.py -v`  
Expected: FAIL because trainer module not wired.

- [ ] **Step 3: Port Stage1 V3 implementation from ref**

```python
def train_stage1(config_path: str, load_lora_from: str | None = None) -> None:
    return train_stage1_v3(config_path, load_lora_from=load_lora_from)
```

- [ ] **Step 4: Run stage1 tests**

Run: `pytest tests/test_stage1_entry.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cope/stage1 tests/test_stage1_entry.py
git commit -m "feat: migrate stage1 v3 trainer into cope package"
```

### Task 3: Stage2 migration

**Files:**
- Create: `src/cope/stage2/trainer.py`
- Create: `src/cope/stage2/model.py`
- Create: `src/cope/stage2/dataset.py`
- Modify: `src/cope/stage2/__init__.py`
- Test: `tests/test_stage2_smoke.py`

- [ ] **Step 1: Write failing stage2 dataset smoke test**

```python
def test_stage2_dataset_loads_minimal_sample(tmp_path):
    ...
    assert len(dataset) == 1
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/test_stage2_smoke.py::test_stage2_dataset_loads_minimal_sample -v`  
Expected: FAIL due missing module.

- [ ] **Step 3: Port Stage2 trainer/model/dataset**

```python
from src.cope.stage2.dataset import Stage2PlanningDataset
from src.cope.stage2.model import Stage2PlannerModel
```

- [ ] **Step 4: Run stage2 smoke tests**

Run: `pytest tests/test_stage2_smoke.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cope/stage2 tests/test_stage2_smoke.py
git commit -m "feat: migrate stage2 planning training pipeline"
```

### Task 4: Stage3 hierarchical-only migration

**Files:**
- Create: `src/cope/stage3/trainer.py`
- Create: `src/cope/stage3/dataset.py`
- Create: `src/cope/stage3/hierarchical/*.py`
- Create: `src/cope/stage3/simulator/*.py`
- Modify: `src/cope/stage3/__init__.py`
- Test: `tests/test_stage3_config.py`
- Test: `tests/test_stage3_reward_smoke.py`

- [ ] **Step 1: Write failing config guard test**

```python
def test_stage3_rejects_non_hierarchical_mode():
    with pytest.raises(ValueError):
        _resolve_training_mode({"training_mode": "global_grpo"})
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/test_stage3_config.py::test_stage3_rejects_non_hierarchical_mode -v`  
Expected: FAIL before mode guard update.

- [ ] **Step 3: Port Stage3 trainer and harden mode validation**

```python
if resolved_mode != "hierarchical_grpo":
    raise ValueError("main branch only supports hierarchical_grpo")
```

- [ ] **Step 4: Run stage3 tests**

Run: `pytest tests/test_stage3_config.py tests/test_stage3_reward_smoke.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cope/stage3 tests/test_stage3_config.py tests/test_stage3_reward_smoke.py
git commit -m "feat: migrate stage3 hierarchical grpo pipeline"
```

## Chunk 3: Entrypoints, Configs, and Path Contracts

### Task 5: Train entrypoint compatibility

**Files:**
- Create: `train.py`
- Create: `src/cope/common/cli.py`
- Test: `tests/test_train_cli_routing.py`

- [ ] **Step 1: Write failing CLI routing tests**

```python
def test_train_cli_routes_stage2(monkeypatch):
    ...
    assert stage_called == "stage2"
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_train_cli_routing.py -v`  
Expected: FAIL because `train.py` missing.

- [ ] **Step 3: Implement compatible `train.py` router**

```python
choices=["stage1", "stage2", "stage3"]
```

- [ ] **Step 4: Re-run tests**

Run: `pytest tests/test_train_cli_routing.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train.py src/cope/common/cli.py tests/test_train_cli_routing.py
git commit -m "feat: add compatible train entrypoint routed to cope stages"
```

### Task 6: Add unified offline inference entrypoint

**Files:**
- Create: `infer.py`
- Create: `src/cope/inference/runner.py`
- Create: `src/cope/inference/stage2_predictor.py`
- Create: `src/cope/inference/stage3_predictor.py`
- Test: `tests/test_infer_cli.py`

- [ ] **Step 1: Write failing inference CLI test**

```python
def test_infer_cli_accepts_stage2_and_stage3():
    assert parse_args(["--stage", "stage2", ...]).stage == "stage2"
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/test_infer_cli.py -v`  
Expected: FAIL before infer entrypoint exists.

- [ ] **Step 3: Implement offline file-in/file-out inference runner**

```python
def run_inference(stage: str, config_path: str, input_path: str, output_path: str) -> None:
    ...
```

- [ ] **Step 4: Re-run tests**

Run: `pytest tests/test_infer_cli.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add infer.py src/cope/inference tests/test_infer_cli.py
git commit -m "feat: add unified offline inference entrypoint"
```

### Task 7: Normalize configuration tree

**Files:**
- Create: `configs/stage1/stage1_qwen25_7b.yaml`
- Create: `configs/stage1/stage1_qwen25_3b.yaml`
- Create: `configs/stage2/stage2_qwen25_7b.yaml`
- Create: `configs/stage2/stage2_qwen25_3b.yaml`
- Create: `configs/stage3/stage3_qwen25_7b_hgrpo.yaml`
- Create: `configs/stage3/stage3_qwen25_3b_hgrpo.yaml`
- Create: `configs/infer/infer_stage2.yaml`
- Create: `configs/infer/infer_stage3.yaml`
- Create: `src/cope/common/config.py`
- Test: `tests/test_config_contract.py`

- [ ] **Step 1: Write failing config contract tests**

```python
def test_base_model_path_is_supported():
    assert config["model"]["base_model"].startswith("/AI/HF_MODELS/Qwen2.5-")
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_config_contract.py -v`  
Expected: FAIL before new config tree exists.

- [ ] **Step 3: Implement config loader + normalized configs**

```python
SUPPORTED_BASE_MODELS = {
    "/AI/HF_MODELS/Qwen2.5-3B",
    "/AI/HF_MODELS/Qwen2.5-7B",
}
```

- [ ] **Step 4: Re-run tests**

Run: `pytest tests/test_config_contract.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs src/cope/common/config.py tests/test_config_contract.py
git commit -m "feat: normalize stage configs and enforce model path contract"
```

## Chunk 4: Cleanup, Policies, and Verification

### Task 8: Data/artifact and run-lineage policy files

**Files:**
- Create: `docs/decisions/2026-04-01-repo-contract.md`
- Create: `docs/experiments/README.md`
- Create: `data/README.md`
- Create: `artifacts/README.md`
- Create: `outputs/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Add policy docs and ignore rules**

```text
Ignore: checkpoints/, wandb/, Qwen2.5-*/, logs/, outputs/*
Keep: outputs/.gitkeep
```

- [ ] **Step 2: Verify git status cleanliness**

Run: `git status --short`  
Expected: only policy/doc files changed, no large artifacts tracked.

- [ ] **Step 3: Commit**

```bash
git add docs data artifacts outputs .gitignore
git commit -m "docs: add repo contract and artifact lineage policy"
```

### Task 9: Final verification suite

**Files:**
- Modify: `tests/conftest.py` (if needed)
- Modify: `README.md`

- [ ] **Step 1: Run targeted smoke suite**

Run:

```bash
pytest tests/test_repo_layout.py \
  tests/test_train_cli_routing.py \
  tests/test_infer_cli.py \
  tests/test_stage2_smoke.py \
  tests/test_stage3_config.py \
  tests/test_stage3_reward_smoke.py \
  tests/test_config_contract.py -v
```

Expected: PASS.

- [ ] **Step 2: Run static sanity checks (optional but recommended)**

Run: `python -m compileall src train.py infer.py`  
Expected: no syntax errors.

- [ ] **Step 3: Update README quickstart**

```bash
python train.py --stage stage1 --config configs/stage1/stage1_qwen25_7b.yaml
python infer.py --stage stage2 --config configs/infer/infer_stage2.yaml --input data/sample.json --output outputs/pred.json
```

- [ ] **Step 4: Final commit**

```bash
git add README.md tests
git commit -m "test: add regression smokes and finalize training/inference refactor"
```
