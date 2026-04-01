"""
Stage 2 Trainer: Multi-Step Planning with Three-Layer Embedding Architecture

This trainer implements Stage 2 planning model training with:
- Frozen base LLM embeddings (0-151664)
- Frozen Stage 1 tool embeddings (151665-153365)
- Trainable Stage 2 control token embeddings (153366-153401)
- Continued LoRA training from Stage 1
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

from src.cope.stage2.model import Stage2PlannerModel
from src.cope.stage2.dataset import Stage2PlanningDataset

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging disabled")


class Stage2MonitorCallback(TrainerCallback):
    """Lightweight callback to monitor Stage 2 training health."""

    def __init__(self, model):
        self.model = model

    def on_train_begin(self, args, state, control, **kwargs):
        """Verify embeddings are healthy after optimizer initialization."""
        has_nan = torch.isnan(self.model.stage2_embedding_module.weight).any().item()
        if has_nan:
            raise RuntimeError(
                "Stage 2 embeddings contain NaN after optimizer init! "
                "This should not happen with the dtype fixes applied."
            )
        print("Stage 2 embeddings verified healthy after optimizer init.")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Periodically check for NaN corruption during training."""
        has_nan = torch.isnan(self.model.stage2_embedding_module.weight).any().item()
        if has_nan:
            raise RuntimeError(
                f"Stage 2 embeddings corrupted to NaN at step {state.global_step}!"
            )


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_base_model_with_lora(
    base_model_name: str,
    stage1_checkpoint: str,
    lora_config_dict: Dict,
    device: str = "cuda",
) -> AutoModelForCausalLM:
    """
    Load base model and apply LoRA from Stage 1 checkpoint.

    Args:
        base_model_name: Base model name (Qwen2.5-7B)
        stage1_checkpoint: Path to Stage 1 checkpoint with LoRA weights
        lora_config_dict: LoRA configuration
        device: Device for model

    Returns:
        Model with LoRA loaded from Stage 1
    """
    print(f"Loading base model: {base_model_name}")

    # Load base model in bfloat16 (recommended for 4090, matches bf16 training)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    print(f"Base model loaded, dtype: {base_model.dtype}")

    # Load Stage 1 LoRA weights
    print(f"Loading Stage 1 LoRA weights from: {stage1_checkpoint}")

    # Check if Stage 1 checkpoint has adapter files
    adapter_config_path = Path(stage1_checkpoint) / "adapter_config.json"
    adapter_model_path = Path(stage1_checkpoint) / "adapter_model.safetensors"

    if not adapter_config_path.exists():
        print(
            f"Warning: adapter_config.json not found in {stage1_checkpoint}, "
            "initializing new LoRA"
        )
        # Create new LoRA
        lora_config = LoraConfig(
            r=lora_config_dict.get("r", 128),
            lora_alpha=lora_config_dict.get("alpha", 256),
            target_modules=lora_config_dict.get(
                "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
            lora_dropout=lora_config_dict.get("dropout", 0.1),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model_with_lora = get_peft_model(base_model, lora_config)
    else:
        # Load existing LoRA
        model_with_lora = PeftModel.from_pretrained(
            base_model, stage1_checkpoint, is_trainable=True
        )
        print("✓ Stage 1 LoRA weights loaded successfully")

    # Enable training
    model_with_lora.train()

    return model_with_lora


def create_stage2_model(
    config: Dict, device: str = "cuda"
) -> tuple[Stage2PlannerModel, AutoTokenizer]:
    """
    Create Stage 2 planning model with three-layer architecture.

    Args:
        config: Training configuration
        device: Device for model

    Returns:
        Tuple of (stage2_model, tokenizer)
    """
    print("\n" + "=" * 60)
    print("Creating Stage 2 Planning Model")
    print("=" * 60)

    # Get paths from config
    base_model_name = config["base_model"]
    stage1_checkpoint = config["stage1_checkpoint"]
    tokenizer_path = config.get(
        "tokenizer_path", "checkpoints/02_stage2/tokenizer_stage2"
    )
    stage2_embeddings_path = config.get(
        "stage2_embeddings_path", "checkpoints/02_stage2/stage2_initialized"
    )

    # Load extended tokenizer (with Stage 2 tokens)
    print(f"\nLoading extended tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # Verify vocabulary size
    expected_vocab_size = 153402
    if len(tokenizer) != expected_vocab_size:
        print(
            f"Warning: Expected vocabulary size {expected_vocab_size}, got {len(tokenizer)}"
        )

    # Setup base model with LoRA
    lora_config = {
        "r": config.get("lora_r", 128),
        "alpha": config.get("lora_alpha", 256),
        "dropout": config.get("lora_dropout", 0.1),
        "target_modules": config.get(
            "lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
    }

    base_model_with_lora = setup_base_model_with_lora(
        base_model_name, stage1_checkpoint, lora_config, device
    )

    # Create Stage 2 wrapper model
    print("\nCreating Stage 2 model wrapper...")
    stage2_model = Stage2PlannerModel(
        llm=base_model_with_lora,
        tokenizer=tokenizer,
        stage1_collapsed_checkpoint=stage1_checkpoint,
        stage2_embeddings_path=stage2_embeddings_path,
        device=device,
    )

    # Print trainable parameters
    print("\n" + "=" * 60)
    print("Trainable Parameters")
    print("=" * 60)
    stage2_model.print_trainable_parameters()

    return stage2_model, tokenizer


def create_datasets(
    config: Dict, tokenizer: AutoTokenizer
) -> tuple[Stage2PlanningDataset, Optional[Stage2PlanningDataset]]:
    """
    Create training and validation datasets.

    Args:
        config: Training configuration
        tokenizer: Extended tokenizer

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    print("\n" + "=" * 60)
    print("Loading Datasets")
    print("=" * 60)

    train_data_path = config["train_data"]
    val_data_path = config.get("val_data", None)
    max_length = config.get("max_length", 1024)
    val_split = config.get("val_split", 0.1)

    print(f"Training data: {train_data_path}")
    print(f"Max sequence length: {max_length}")

    # Load training dataset
    full_dataset = Stage2PlanningDataset(
        data_path=train_data_path, tokenizer=tokenizer, max_length=max_length
    )

    # Split into train/val if no separate validation data
    if val_data_path is None and val_split > 0:
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        print(f"Split dataset: {train_size} train, {val_size} val")
    elif val_data_path:
        train_dataset = full_dataset
        val_dataset = Stage2PlanningDataset(
            data_path=val_data_path, tokenizer=tokenizer, max_length=max_length
        )
        print(f"Separate validation data: {len(val_dataset)} examples")
    else:
        train_dataset = full_dataset
        val_dataset = None
        print("No validation split")

    return train_dataset, val_dataset


def train_stage2(config_path: str):
    """
    Main training function for Stage 2 planning.

    Args:
        config_path: Path to YAML configuration file
    """
    # Load configuration
    config = load_config(config_path)
    print("Loaded configuration from:", config_path)

    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"Random seed: {seed}")

    # Device
    device = config.get("device", "cuda")
    print(f"Device: {device}")

    # Create output directory
    output_dir = config["output_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.get("wandb_run_name", f"stage2_planning_{timestamp}")
    output_dir_full = os.path.join(output_dir, run_name)
    os.makedirs(output_dir_full, exist_ok=True)

    print(f"Output directory: {output_dir_full}")

    # Save config to output directory
    config_save_path = os.path.join(output_dir_full, "training_config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f)
    print(f"Saved config to: {config_save_path}")

    # Initialize wandb if available
    if WANDB_AVAILABLE and config.get("use_wandb", True):
        wandb.init(
            project=config.get("wandb_project", "tomas_stage2_planning"),
            name=run_name,
            config=config,
        )
        print("✓ WandB initialized")

    # Create model and tokenizer
    stage2_model, tokenizer = create_stage2_model(config, device=device)

    # Create datasets
    train_dataset, val_dataset = create_datasets(config, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir_full,
        num_train_epochs=config.get("num_epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 4),
        per_device_eval_batch_size=config.get("eval_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 5e-5),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_steps=config.get("warmup_steps", 100),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 200),
        eval_steps=config.get("eval_steps", 200) if val_dataset else None,
        save_strategy=config.get("save_strategy", "steps"),
        eval_strategy="steps"
        if val_dataset
        else "no",  # Changed from evaluation_strategy
        bf16=config.get("bf16", True),
        fp16=config.get("fp16", False),
        optim=config.get("optimizer", "adamw_torch"),
        save_total_limit=config.get("save_total_limit", 3),
        load_best_model_at_end=False,
        report_to="wandb"
        if WANDB_AVAILABLE and config.get("use_wandb", True)
        else "none",
        run_name=run_name,
        logging_dir=os.path.join(output_dir_full, "logs"),
        dataloader_num_workers=config.get("num_workers", 0),
        remove_unused_columns=False,  # Important: keep all dataset columns
    )

    print("\n" + "=" * 60)
    print("Training Arguments")
    print("=" * 60)
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(
        f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}"
    )
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Warmup steps: {training_args.warmup_steps}")
    print(f"  Save steps: {training_args.save_steps}")
    print(f"  Mixed precision (bf16): {training_args.bf16}")

    # Create trainer
    trainer = Trainer(
        model=stage2_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[Stage2MonitorCallback(stage2_model)],
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")

    train_result = trainer.train()

    # Save final model
    final_model_path = os.path.join(output_dir_full, "final_model")
    print(f"\nSaving final model to: {final_model_path}")

    # Save Stage 2 embeddings and heads
    os.makedirs(final_model_path, exist_ok=True)

    # Save Stage 2 trainable components
    stage2_emb_save_path = os.path.join(final_model_path, "stage2_embeddings.bin")
    stage2_head_save_path = os.path.join(final_model_path, "stage2_lm_head.bin")

    torch.save(
        stage2_model.stage2_embedding_module.weight.data.cpu(), stage2_emb_save_path
    )
    torch.save(
        stage2_model.stage2_lm_head_module.weight.data.cpu(), stage2_head_save_path
    )

    print(f"  Stage 2 embeddings: {stage2_emb_save_path}")
    print(f"  Stage 2 LM head: {stage2_head_save_path}")

    # Save LoRA weights
    lora_save_path = os.path.join(final_model_path, "lora_weights")
    stage2_model.llm.save_pretrained(lora_save_path)
    print(f"  LoRA weights: {lora_save_path}")

    # Save tokenizer
    tokenizer.save_pretrained(final_model_path)
    print(f"  Tokenizer: {final_model_path}")

    # Save training metrics
    metrics_path = os.path.join(output_dir_full, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(train_result.metrics, f, indent=2)
    print(f"  Metrics: {metrics_path}")

    # Finish wandb
    if WANDB_AVAILABLE and config.get("use_wandb", True):
        wandb.finish()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final model saved to: {final_model_path}")
    print(f"Total training time: {train_result.metrics.get('train_runtime', 0):.2f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Stage 2 Planning Model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )

    args = parser.parse_args()
    train_stage2(args.config)
