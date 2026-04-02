.PHONY: stage1 stage1-tokenizer stage1-check stage1-train stage1-collapse

STAGE1_CONFIG ?= configs/stage1/stage1_qwen25_7b.yaml
CUDA_VISIBLE_DEVICES ?= 3
CUDA_HOME ?= /usr/local/cuda-12.8

STAGE1_REGISTRY ?= data/00_global/tool_registry.json
STAGE1_TOKENIZER_OUT ?= checkpoints/01_stage1/tokenizer_expanded

stage1: stage1-tokenizer stage1-check stage1-train stage1-collapse

stage1-tokenizer:
	BASE_MODEL=/AI/HF_MODELS/Qwen2.5-7B REGISTRY=$(STAGE1_REGISTRY) TOKENIZER_OUT=$(STAGE1_TOKENIZER_OUT) bash scripts/stage1/prepare_stage1_assets.sh

stage1-check:
	bash scripts/stage1/check_stage1_inputs.sh

stage1-train:
	CUDA_HOME=$(CUDA_HOME) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) python train.py --stage stage1 --config $(STAGE1_CONFIG)

stage1-collapse:
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) STAGE1_CONFIG=$(STAGE1_CONFIG) bash scripts/stage1/collapse_stage1_output.sh
