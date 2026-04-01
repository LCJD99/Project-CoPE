"""
Stage 1 Trainer V3: Tool Token Learning with Virtual Token Architecture
Uses custom virtual token embeddings and heads (no embedding expansion).
Based on ref/model/offline_rl.py design pattern.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType

from src.cope.stage1.profile_encoder import create_profile_encoder, load_system_config
from src.cope.stage1.model import ToolPlannerModel

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_tool_registry(registry_path: str) -> Dict:
    """Load tool registry JSON"""
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    return registry


def prepare_tool_data(
    registry: Dict,
    tokenizer,
    base_model,
    device: str = 'cuda',
    use_real_values: bool = True
) -> tuple:
    """
    Prepare tool semantic vectors and configuration vectors.
    
    Args:
        registry: Tool registry
        tokenizer: Tokenizer
        base_model: Base model (for computing semantic vectors)
        device: Device
        use_real_values: Whether to use real config values (True) or discrete levels (False)
    
    Returns:
        (tool_semantics, profiles, token_list)
        - tool_semantics: [num_tokens, hidden_dim]
        - profiles: [num_tokens, 5] real config values or discrete levels
        - token_list: token name list
    """
    print("Preparing tool semantic embeddings and profile vectors...")
    
    tokens = registry['tokens']
    token_list = sorted(tokens.keys())
    num_tokens = len(token_list)
    
    # Get hidden_size
    hidden_size = base_model.config.hidden_size
    
    # Initialize storage
    tool_semantics = torch.zeros(num_tokens, hidden_size, device=device)
    profiles = torch.zeros(num_tokens, 5, device=device)
    
    # Get original embedding layer
    input_embeddings = base_model.get_input_embeddings()
    
    with torch.no_grad():
        for idx, token_name in enumerate(token_list):
            token_info = tokens[token_name]
            
            # 1. Compute semantic vector (using semantic_description)
            desc_text = token_info.get('semantic_description', token_info['description'])
            
            # Tokenize description
            desc_input_ids = tokenizer(
                desc_text,
                return_tensors='pt',
                truncation=True,
                max_length=512
            )['input_ids'].to(device)
            
            # Get embeddings and average
            desc_embeddings = input_embeddings(desc_input_ids)
            semantic_vec = desc_embeddings.mean(dim=1).squeeze(0)  # [hidden_size]
            
            tool_semantics[idx] = semantic_vec
            
            # 2. Extract resource configuration vector
            resources = token_info.get('resources', {})
            resource_levels = token_info.get('resource_levels', {})
            
            if use_real_values:
                # Use real config values
                # Map input_size to numeric: small=1, medium=2, large=3
                size_map = {'small': 1, 'medium': 2, 'large': 3}
                tool_size = size_map.get(token_info.get('input_size', 'medium'), 2)
                
                cpu_core = resources.get('cpu_core', 2)
                cpu_mem = resources.get('cpu_mem_gb', 4.0)
                gpu_sm = resources.get('gpu_sm', 20)
                gpu_mem = resources.get('gpu_mem_gb', 2.0)
            else:
                # Use discrete levels: low=1, medium=2, high=3
                level_map = {'low': 1, 'medium': 2, 'med': 2, 'high': 3}
                
                tool_size = level_map.get(resource_levels.get('tool_size', 'medium'), 2)
                cpu_core = level_map.get(resource_levels.get('cpu_core', 'low'), 1)
                cpu_mem = level_map.get(resource_levels.get('cpu_mem', 'low'), 1)
                gpu_sm = level_map.get(resource_levels.get('gpu_sm', 'low'), 1)
                gpu_mem = level_map.get(resource_levels.get('gpu_mem', 'low'), 1)
            
            profiles[idx] = torch.tensor([tool_size, cpu_core, cpu_mem, gpu_sm, gpu_mem])
    
    print(f"Prepared {num_tokens} tool embeddings")
    print(f"Tool semantics shape: {tool_semantics.shape}")
    print(f"Profiles shape: {profiles.shape}")
    print(f"Profile value range: [{profiles.min():.1f}, {profiles.max():.1f}]")
    if use_real_values:
        print("Using real configuration values")
    else:
        print("Using discrete levels (1-3)")
    
    return tool_semantics, profiles, token_list


def setup_model_with_virtual_tokens(
    config: Dict,
    load_lora_from: Optional[str] = None
):
    """
    Setup model with virtual token architecture.
    
    Args:
        config: Configuration dict
        load_lora_from: Existing LoRA checkpoint path (optional)
    
    Returns:
        (model, tokenizer, new_token_start_idx)
    """
    print("=" * 60)
    print("Setting up model with Virtual Token Architecture")
    print("=" * 60)
    
    # 1. Load tool registry
    registry = load_tool_registry(config['tool_registry'])
    
    # 2. Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['tokenizer'],
        trust_remote_code=True
    )
    
    original_vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size: {original_vocab_size}")
    
    # Calculate new token start index
    token_list = sorted(registry['tokens'].keys())
    new_tokens = [f"<{token_name}>" for token_name in token_list]
    
    # Check if need to add new tokens
    first_new_token = '<IMG_CLS_SMALL_LOW_LOW_LOW_LOW>'
    if first_new_token not in tokenizer.get_vocab():
        print(f"Adding {len(new_tokens)} new tokens to tokenizer...")
        num_added = tokenizer.add_tokens(new_tokens)
        print(f"Added {num_added} tokens")
    
    new_token_start_idx = tokenizer.convert_tokens_to_ids(first_new_token)
    num_virtual_tokens = len(new_tokens)
    print(f"New token start index: {new_token_start_idx}")
    print(f"Number of virtual tokens: {num_virtual_tokens}")
    
    # 3. Load base model
    print("\nLoading base model...")
    base_llm = AutoModelForCausalLM.from_pretrained(
        config.get('base_model', config.get('initialized_model')),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.get('bf16', True) else torch.float16,
        device_map='auto'
    )
    
    # NOTE: Do NOT resize token embeddings - we use custom virtual token embeddings
    print("✓ Base LLM loaded (NOT resizing embeddings)")
    
    # 4. Prepare tool data
    device = next(base_llm.parameters()).device
    use_real_values = config.get('use_real_profile_values', True)
    tool_semantics, profiles, _ = prepare_tool_data(
        registry, tokenizer, base_llm, device=device, use_real_values=use_real_values
    )
    
    # 5. Load system config (for normalization)
    system_config_path = config.get('system_config_path', None)
    max_values = load_system_config(system_config_path) if use_real_values else None
    if max_values:
        print(f"\nSystem max configuration:")
        for k, v in max_values.items():
            print(f"  {k}: {v}")
    
    # 6. Create Profile Encoder (Hypernetwork)
    print("\nCreating Profile Encoder (Hypernetwork)...")
    profile_encoder = create_profile_encoder(
        encoder_type=config.get('profile_encoder_type', 'simple'),
        input_dim=5,
        output_dim=base_llm.config.hidden_size,
        hidden_dims=config.get('profile_encoder_hidden_dims', [128, 512]),
        activation=config.get('profile_encoder_activation', 'gelu'),
        dropout=config.get('profile_encoder_dropout', 0.0),
        zero_init=True,
        normalize=use_real_values,
        max_values=max_values,
        system_config_path=system_config_path
    )
    
    profile_encoder = profile_encoder.to(
        device=device,
        dtype=torch.bfloat16 if config.get('bf16', True) else torch.float16
    )
    
    print(f"Profile Encoder parameters: {sum(p.numel() for p in profile_encoder.parameters()):,}")
    
    # 7. Create ToolPlannerModel
    print("\nCreating ToolPlannerModel...")
    model = ToolPlannerModel(
        llm=base_llm,
        tokenizer=tokenizer,
        num_virtual_tokens=num_virtual_tokens,
        virtual_token_start_idx=new_token_start_idx,
        profile_encoder=profile_encoder,
        use_profile_encoding=True,
        device=device
    )
    
    # Initialize virtual tokens
    model.initialize_virtual_tokens(tool_semantics, profiles)
    print("✓ Virtual tokens initialized")
    
    # 8. Configure LoRA
    if config.get('use_lora', True):
        if load_lora_from:
            print(f"\nLoading existing LoRA from {load_lora_from}...")
            from peft import PeftModel
            model.llm = PeftModel.from_pretrained(model.llm, load_lora_from)
            print("✓ LoRA loaded")
        else:
            print("\nConfiguring new LoRA...")
            lora_config = LoraConfig(
                r=config.get('lora_r', 64),
                lora_alpha=config.get('lora_alpha', 32),
                target_modules=config.get('lora_target_modules', [
                    'q_proj', 'v_proj', 'k_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'
                ]),
                lora_dropout=config.get('lora_dropout', 0.1),
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            model.llm = get_peft_model(model.llm, lora_config)
            print("✓ LoRA configured")
        
        if hasattr(model.llm, 'print_trainable_parameters'):
            model.llm.print_trainable_parameters()
    
    # 9. Enable gradient checkpointing
    if config.get('gradient_checkpointing', False):
        model.llm.enable_input_require_grads()
        model.llm.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")
    
    # 10. Set trainable parameters
    print("\nSetting trainable parameters...")
    
    # Profile encoder trainable
    for param in model.profile_encoder.parameters():
        param.requires_grad = True
    
    # Virtual token embedding and head trainable
    for param in model.virtual_embedding.parameters():
        param.requires_grad = True
    for param in model.virtual_head.parameters():
        param.requires_grad = True
    
    encoder_params = sum(p.numel() for p in model.profile_encoder.parameters() if p.requires_grad)
    virtual_params = sum(p.numel() for p in model.modules_except_llm.parameters() if p.requires_grad)
    
    print(f"✓ Profile Encoder trainable params: {encoder_params:,}")
    print(f"✓ Virtual components trainable params: {virtual_params:,}")
    
    print("\n" + "=" * 60)
    print("Model setup complete!")
    print("=" * 60)
    
    return model, tokenizer, new_token_start_idx


class VirtualTokenTrainer(Trainer):
    """
    Custom Trainer for virtual token architecture.
    Manages different learning rates for different components.
    """
    
    def __init__(self, *args, hypernet_lr: float = 1e-3, use_lora: bool = True, **kwargs):
        self.hypernet_lr = hypernet_lr
        self.use_lora = use_lora
        super().__init__(*args, **kwargs)
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Custom save method: only save LoRA weights and modules_except_llm.
        Similar to ref/utils/utils.py save_model pattern.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving model to {output_dir}")
        
        if self.use_lora:
            # Save LoRA adapter weights
            print("  Saving LoRA adapter...")
            self.model.llm.save_pretrained(output_dir)
            
            # Save modules except LLM (virtual tokens + profile encoder)
            print("  Saving virtual components (modules_except_llm)...")
            torch.save(
                self.model.modules_except_llm.state_dict(),
                os.path.join(output_dir, 'modules_except_llm.bin')
            )
            
            # Save metadata
            print("  Saving metadata...")
            metadata = {
                'num_virtual_tokens': self.model.num_virtual_tokens,
                'virtual_token_start_idx': self.model.virtual_token_start_idx,
                'hidden_size': self.model.hidden_size,
                'vocab_size': self.model.vocab_size,
                'expanded_vocab_size': self.model.expanded_vocab_size,
                'use_profile_encoding': self.model.use_profile_encoding
            }
            torch.save(metadata, os.path.join(output_dir, 'metadata.bin'))
            
            print("✓ Model saved (LoRA + virtual components + metadata)")
        else:
            # LoRA is disabled, save whole model
            print("  Saving entire model state...")
            torch.save(self.model.state_dict(), os.path.join(output_dir, 'model.bin'))
            print("✓ Full model saved")
        
    
    def create_optimizer(self):
        """
        Create optimizer with different learning rates for:
        - LoRA parameters: config learning rate
        - Profile Encoder (Hypernetwork): higher learning rate
        - Virtual token components: config learning rate
        """
        if self.optimizer is None:
            # Group parameters
            hypernet_params = []
            lora_params = []
            virtual_params = []
            other_params = []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                if "profile_encoder" in name or "hypernet" in name:
                    hypernet_params.append(param)
                elif "lora" in name:
                    lora_params.append(param)
                elif "virtual_embedding" in name or "virtual_head" in name:
                    virtual_params.append(param)
                else:
                    other_params.append(param)
            
            # Build optimizer groups
            optimizer_grouped_parameters = []
            
            if lora_params:
                optimizer_grouped_parameters.append({
                    "params": lora_params,
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                })
                print(f"LoRA params: {sum(p.numel() for p in lora_params):,}, lr={self.args.learning_rate}")
            
            if hypernet_params:
                optimizer_grouped_parameters.append({
                    "params": hypernet_params,
                    "weight_decay": 0.0,  # No weight decay for hypernetwork
                    "lr": self.hypernet_lr,
                })
                print(f"Hypernet params: {sum(p.numel() for p in hypernet_params):,}, lr={self.hypernet_lr}")
            
            if virtual_params:
                optimizer_grouped_parameters.append({
                    "params": virtual_params,
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                })
                print(f"Virtual token params: {sum(p.numel() for p in virtual_params):,}, lr={self.args.learning_rate}")
            
            if other_params:
                optimizer_grouped_parameters.append({
                    "params": other_params,
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                })
                print(f"Other params: {sum(p.numel() for p in other_params):,}, lr={self.args.learning_rate}")
            
            # Create optimizer
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        
        return self.optimizer


def setup_training_args(config: Dict) -> TrainingArguments:
    """Setup TrainingArguments from config"""
    # Determine report_to
    report_to = config.get('report_to', 'none')
    if report_to == 'wandb' and not WANDB_AVAILABLE:
        print("Warning: wandb not available, setting report_to='none'")
        report_to = 'none'
    
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        
        # Training hyperparameters
        num_train_epochs=int(config.get('num_epochs', 3)),
        per_device_train_batch_size=int(config.get('batch_size', 4)),
        gradient_accumulation_steps=int(config.get('gradient_accumulation_steps', 8)),
        learning_rate=float(config.get('learning_rate', 2e-5)),
        weight_decay=float(config.get('weight_decay', 0.01)),
        warmup_ratio=float(config.get('warmup_ratio', 0.03)),
        max_grad_norm=float(config.get('max_grad_norm', 1.0)),
        
        # Optimizer
        optim=config.get('optim', 'adamw_torch'),
        adam_beta1=float(config.get('adam_beta1', 0.9)),
        adam_beta2=float(config.get('adam_beta2', 0.999)),
        adam_epsilon=float(config.get('adam_epsilon', 1e-8)),
        
        # Mixed precision
        fp16=config.get('fp16', False),
        bf16=config.get('bf16', True),
        
        # Logging
        logging_steps=int(config.get('logging_steps', 10)),
        logging_dir=os.path.join(config['output_dir'], 'logs'),
        report_to=report_to,
        
        # Checkpointing
        save_steps=int(config.get('save_steps', 500)),
        save_total_limit=int(config.get('save_total_limit', 3)),
        save_strategy='steps',
        
        # Other
        dataloader_num_workers=int(config.get('dataloader_num_workers', 4)),
        remove_unused_columns=bool(config.get('remove_unused_columns', False)),
        seed=int(config.get('seed', 42)),
    )
    
    return training_args


def train_stage1_v3(config_path: str, load_lora_from: Optional[str] = None):
    """
    Main training function with Virtual Token architecture.
    
    Args:
        config_path: Path to configuration YAML file
        load_lora_from: Path to existing LoRA checkpoint (optional)
    """
    # Load config
    config = load_config(config_path)
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Initialize wandb if enabled
    if config.get('report_to') == 'wandb' and WANDB_AVAILABLE:
        wandb.init(
            project=config.get('wandb_project', 'tomas-llm'),
            name=config.get('wandb_run_name', 'stage1-v3-virtual-tokens'),
            config=config
        )
    
    # Setup model with virtual tokens
    model, tokenizer, new_token_start_idx = setup_model_with_virtual_tokens(
        config, load_lora_from=load_lora_from
    )
    
    # Setup training arguments
    training_args = setup_training_args(config)
    
    # Load dataset
    from src.cope.stage1.dataset import ToolInstructionDataset, ToolInstructionDataCollator
    
    print(f"\nLoading training data from {config['train_data']}...")
    train_dataset = ToolInstructionDataset(
        data_path=config['train_data'],
        tokenizer=tokenizer,
        max_length=config.get('max_length', 512)
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Data collator
    data_collator = ToolInstructionDataCollator(tokenizer=tokenizer)
    
    # Custom callback
    from src.cope.stage1.evaluator import EvaluationCallback
    callbacks = [EvaluationCallback(tokenizer=tokenizer)]
    
    # Initialize Custom Trainer
    hypernet_lr = float(config.get('hypernet_lr', 1e-3))
    use_lora = config.get('use_lora', True)
    
    trainer = VirtualTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        hypernet_lr=hypernet_lr,
        use_lora=use_lora
    )
    
    # Train
    print("\n" + "=" * 60)
    if load_lora_from:
        print(f"Starting stage-2 training with LoRA from: {load_lora_from}")
    else:
        print("Starting stage-1 training with Virtual Tokens...")
    print("=" * 60)
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    output_dir = Path(config['output_dir']) / 'final_model'
    
    # Save using ToolPlannerModel's save method
    model.save_pretrained(str(output_dir), save_llm=True)
    tokenizer.save_pretrained(str(output_dir))
    
    # Save additional info
    model_info = {
        'new_token_start_idx': new_token_start_idx,
        'num_virtual_tokens': model.num_virtual_tokens,
        'hypernet_lr': hypernet_lr,
        'architecture': 'virtual_token_v3'
    }
    
    with open(output_dir / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nTraining complete! Model saved to {output_dir}")
    
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Stage 1 Model V3 (Virtual Tokens)')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--load_lora_from',
        type=str,
        default=None,
        help='Path to existing LoRA checkpoint for stage-2 training'
    )
    
    args = parser.parse_args()
    
    train_stage1_v3(args.config, load_lora_from=args.load_lora_from)
