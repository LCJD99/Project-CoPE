"""
Collapse Stage 1 Output: Pre-compute dynamic embeddings into fixed tensors.

This script loads a trained Stage 1 checkpoint and collapses the dynamic embeddings
(base_embedding + profile_encoder(profile)) into fixed tensors for Stage 2 use.

The collapsed output includes:
- Fixed virtual token embeddings (no profile encoder needed)
- Fixed LM head weights
- LoRA adapter weights (copied)
- Tokenizer files (copied)
- Token mappings and metadata

Usage:
    python -m src.cope.stage1.collapse \
        --checkpoint checkpoints/test/final_model \
        --config configs/stage1/stage1_qwen25_7b.yaml \
        --registry data/00_global/tool_registry.json \
        --output checkpoints/01_stage1/memorization/20260209_tool_learing_supernetwork
"""

import os
import json
import yaml
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

import torch
import torch.nn as nn


def load_metadata(checkpoint_dir: Path) -> Dict:
    """
    Load metadata from checkpoint.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Metadata dictionary
    """
    metadata_path = checkpoint_dir / "metadata.bin"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    metadata = torch.load(metadata_path, map_location="cpu")
    print(f"✓ Metadata loaded: {metadata['num_virtual_tokens']} virtual tokens")
    return metadata


def load_config(config_path: Path) -> Dict:
    """
    Load training configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded: {config_path.name}")
    return config


def load_registry(registry_path: Path) -> Dict:
    """
    Load tool registry JSON.

    Args:
        registry_path: Path to registry JSON file

    Returns:
        Registry dictionary
    """
    with open(registry_path, "r") as f:
        registry = json.load(f)
    print(f"✓ Registry loaded: {len(registry['tokens'])} tokens")
    return registry


def extract_profiles(registry: Dict) -> Tuple[torch.Tensor, List[str]]:
    """
    Extract profile vectors from registry in sorted order.

    Args:
        registry: Tool registry dictionary

    Returns:
        Tuple of (profiles tensor [num_tokens, 5], token_list)
    """
    tokens = registry["tokens"]
    token_list = sorted(tokens.keys())
    num_tokens = len(token_list)

    profiles = torch.zeros(num_tokens, 5)

    for idx, token_name in enumerate(token_list):
        token_info = tokens[token_name]
        resources = token_info["resources"]

        # Extract profile: [input_size, cpu_core, cpu_mem, gpu_sm, gpu_mem]
        # Map input_size to numeric: small=1, medium=2, large=3
        input_size_map = {"small": 1, "medium": 2, "large": 3}
        input_size = input_size_map[token_info["input_size"]]

        profiles[idx] = torch.tensor(
            [
                input_size,
                resources["cpu_core"],
                resources["cpu_mem_gb"],
                resources["gpu_sm"],
                resources["gpu_mem_gb"],
            ]
        )

    return profiles, token_list


def create_profile_encoder(
    config: Dict, hidden_size: int, device: str = "cpu"
) -> nn.Module:
    """
    Create profile encoder instance from config.

    Args:
        config: Training configuration
        hidden_size: Model hidden size
        device: Device to load to

    Returns:
        ProfileHyperNet instance
    """
    from src.cope.stage1.profile_encoder import ProfileHyperNet, load_system_config

    system_config_path = config.get("system_config_path", None)
    use_real_values = config.get("use_real_profile_values", True)
    max_values = load_system_config(system_config_path) if use_real_values else None

    profile_encoder = ProfileHyperNet(
        input_dim=5,
        hidden_dims=config.get("profile_encoder_hidden_dims", [128, 512]),
        output_dim=hidden_size,
        activation=config.get("profile_encoder_activation", "gelu"),
        dropout=config.get("profile_encoder_dropout", 0.0),
        zero_init=True,
        normalize=use_real_values,
        max_values=max_values,
        system_config_path=system_config_path,
    )

    return profile_encoder.to(device)


def load_modules(
    checkpoint_dir: Path, metadata: Dict, config: Dict, device: str = "cpu"
) -> Dict[str, nn.Module]:
    """
    Load trained modules from checkpoint.

    Args:
        checkpoint_dir: Path to checkpoint directory
        metadata: Metadata dictionary
        config: Training configuration
        device: Device to load to

    Returns:
        Dictionary with loaded modules
    """
    modules_path = checkpoint_dir / "modules_except_llm.bin"
    if not modules_path.exists():
        raise FileNotFoundError(f"Modules not found: {modules_path}")

    # Create module instances
    from src.cope.stage1.virtual_tokens import VirtualTokenEmbedding, VirtualTokenHead

    hidden_size = metadata["hidden_size"]
    num_virtual_tokens = metadata["num_virtual_tokens"]

    # Create profile encoder
    profile_encoder = create_profile_encoder(config, hidden_size, device)

    # Create virtual token modules
    virtual_embedding = VirtualTokenEmbedding(
        num_virtual_tokens=num_virtual_tokens,
        embed_dim=hidden_size,
        profile_encoder=profile_encoder,
        use_profile_encoding=True,
    ).to(device)

    virtual_head = VirtualTokenHead(
        num_virtual_tokens=num_virtual_tokens,
        hidden_size=hidden_size,
        profile_encoder=profile_encoder,
        use_profile_encoding=True,
    ).to(device)

    # Load state dict
    state_dict = torch.load(modules_path, map_location=device)

    # Create ModuleList as in training
    modules_except_llm = nn.ModuleList(
        [virtual_embedding, virtual_head, profile_encoder]
    )

    modules_except_llm.load_state_dict(state_dict)
    modules_except_llm.eval()

    total_params = sum(p.numel() for p in modules_except_llm.parameters())
    print(f"✓ Modules loaded: {total_params / 1e6:.1f}M parameters")

    return {
        "virtual_embedding": virtual_embedding,
        "virtual_head": virtual_head,
        "profile_encoder": profile_encoder,
    }


def compute_collapsed_embeddings(
    virtual_embedding: nn.Module, profiles: torch.Tensor, device: str = "cpu"
) -> torch.Tensor:
    """
    Compute collapsed embeddings: base_embedding + profile_encoder(profile).

    Args:
        virtual_embedding: VirtualTokenEmbedding module
        profiles: Profile vectors [num_tokens, 5]
        device: Device for computation

    Returns:
        Collapsed embeddings [num_tokens, hidden_size]
    """
    with torch.no_grad():
        # Get base embeddings
        base_embeddings = virtual_embedding.virtual_embeddings.weight.to(device)

        # Compute profile deltas
        profiles = profiles.to(device=device, dtype=base_embeddings.dtype)
        profile_deltas = virtual_embedding.profile_encoder(profiles)

        # Compute collapsed embeddings
        collapsed_embeddings = base_embeddings + profile_deltas

    return collapsed_embeddings


def compute_collapsed_lm_head(
    virtual_head: nn.Module, profiles: torch.Tensor, device: str = "cpu"
) -> torch.Tensor:
    """
    Compute collapsed LM head weights: base_weight + profile_encoder(profile).

    Args:
        virtual_head: VirtualTokenHead module
        profiles: Profile vectors [num_tokens, 5]
        device: Device for computation

    Returns:
        Collapsed LM head weights [num_tokens, hidden_size]
    """
    with torch.no_grad():
        # Get base weights
        base_weights = virtual_head.head.weight.to(device)

        # Compute profile deltas
        profiles = profiles.to(device=device, dtype=base_weights.dtype)
        profile_deltas = virtual_head.profile_encoder(profiles)

        # Compute collapsed weights
        collapsed_weights = base_weights + profile_deltas

    return collapsed_weights


def create_output_directory(output_dir: Path) -> Path:
    """
    Create output directory with auto-increment if exists.

    Args:
        output_dir: Desired output directory path

    Returns:
        Actual output directory path (may have _2, _3, etc. suffix)
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    # Directory exists, find next available suffix
    base_dir = output_dir
    suffix = 2
    while output_dir.exists():
        output_dir = Path(str(base_dir) + f"_{suffix}")
        suffix += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"⚠ Output directory exists, using: {output_dir.name}")
    return output_dir


def save_collapsed_tensors(
    output_dir: Path,
    collapsed_embeddings: torch.Tensor,
    collapsed_lm_head: torch.Tensor,
):
    """
    Save collapsed tensors to output directory.

    Args:
        output_dir: Output directory path
        collapsed_embeddings: Collapsed embeddings tensor
        collapsed_lm_head: Collapsed LM head tensor
    """
    # Save embeddings
    emb_path = output_dir / "collapsed_embeddings.bin"
    torch.save(collapsed_embeddings.cpu(), emb_path)
    emb_size_mb = emb_path.stat().st_size / (1024 * 1024)
    print(f"✓ collapsed_embeddings.bin ({emb_size_mb:.1f} MB)")

    # Save LM head
    head_path = output_dir / "collapsed_lm_head.bin"
    torch.save(collapsed_lm_head.cpu(), head_path)
    head_size_mb = head_path.stat().st_size / (1024 * 1024)
    print(f"✓ collapsed_lm_head.bin ({head_size_mb:.1f} MB)")


def save_token_mappings(output_dir: Path, token_list: List[str]):
    """
    Save token name to index mappings.

    Args:
        output_dir: Output directory path
        token_list: Sorted list of token names
    """
    mappings = {name: idx for idx, name in enumerate(token_list)}

    mapping_path = output_dir / "token_mappings.json"
    with open(mapping_path, "w") as f:
        json.dump(mappings, f, indent=2)

    print(f"✓ token_mappings.json")


def save_collapse_info(
    output_dir: Path,
    checkpoint_dir: Path,
    config_path: Path,
    config_name: str,
    metadata: Dict,
    collapse_date: str,
):
    """
    Save collapse metadata information.

    Args:
        output_dir: Output directory path
        checkpoint_dir: Source checkpoint directory
        config_path: Source config file path
        config_name: Config name (wandb_run_name)
        metadata: Loaded metadata
        collapse_date: Collapse date string (YYYYMMDD)
    """
    info = {
        "source_checkpoint": str(checkpoint_dir),
        "source_config": str(config_path),
        "config_name": config_name,
        "collapse_date": collapse_date,
        "collapse_timestamp": datetime.now().isoformat(),
        "num_virtual_tokens": metadata["num_virtual_tokens"],
        "virtual_token_start_idx": metadata["virtual_token_start_idx"],
        "hidden_size": metadata["hidden_size"],
        "vocab_size": metadata["vocab_size"],
        "expanded_vocab_size": metadata["expanded_vocab_size"],
        "embedding_shape": [metadata["num_virtual_tokens"], metadata["hidden_size"]],
        "lm_head_shape": [metadata["num_virtual_tokens"], metadata["hidden_size"]],
        "profile_encoder_discarded": True,
        "stage": "01_stage1_memorization",
    }

    info_path = output_dir / "collapse_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"✓ collapse_info.json")


def copy_lora_weights(checkpoint_dir: Path, output_dir: Path):
    """
    Copy LoRA adapter weights to output directory.

    Args:
        checkpoint_dir: Source checkpoint directory
        output_dir: Output directory path
    """
    # Copy adapter files
    adapter_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",  # Fallback if safetensors not available
    ]

    copied_count = 0
    total_size = 0

    for filename in adapter_files:
        src = checkpoint_dir / filename
        if src.exists():
            dst = output_dir / filename
            shutil.copy2(src, dst)
            total_size += dst.stat().st_size
            copied_count += 1

    if copied_count > 0:
        size_mb = total_size / (1024 * 1024)
        print(f"✓ LoRA weights copied ({size_mb:.1f} MB)")
    else:
        print("⚠ No LoRA weights found to copy")


def copy_tokenizer(checkpoint_dir: Path, output_dir: Path):
    """
    Copy tokenizer files to output directory.

    Args:
        checkpoint_dir: Source checkpoint directory
        output_dir: Output directory path
    """
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]

    copied_count = 0

    for filename in tokenizer_files:
        src = checkpoint_dir / filename
        if src.exists():
            dst = output_dir / filename
            shutil.copy2(src, dst)
            copied_count += 1

    print(f"✓ Tokenizer files copied ({copied_count} files)")


def main():
    parser = argparse.ArgumentParser(
        description="Collapse Stage 1 dynamic embeddings to fixed tensors"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint directory",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to training config YAML file"
    )
    parser.add_argument(
        "--registry", type=str, required=True, help="Path to tool registry JSON file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory path"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for computation (cpu/cuda)"
    )

    args = parser.parse_args()

    # Convert to Path objects
    checkpoint_dir = Path(args.checkpoint)
    config_path = Path(args.config)
    registry_path = Path(args.registry)
    output_dir = Path(args.output)

    print("=" * 60)
    print("  Stage 1 Collapsed Output Generation")
    print("=" * 60)
    print()

    # Validate input files
    print("[检查] 验证输入文件...")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    print(f"✓ Checkpoint: {checkpoint_dir}")
    print(f"✓ Config: {config_path}")
    print(f"✓ Registry: {registry_path}")
    print()

    # Load files
    print("[加载] 加载训练权重...")
    metadata = load_metadata(checkpoint_dir)
    config = load_config(config_path)
    registry = load_registry(registry_path)
    profiles, token_list = extract_profiles(registry)
    print()

    # Load modules
    modules = load_modules(checkpoint_dir, metadata, config, device=args.device)
    print()

    # Compute collapsed outputs
    print("[计算] 生成 collapsed embeddings...")
    collapsed_embeddings = compute_collapsed_embeddings(
        modules["virtual_embedding"], profiles, device=args.device
    )
    print(f"✓ Collapsed embeddings: {list(collapsed_embeddings.shape)}")

    collapsed_lm_head = compute_collapsed_lm_head(
        modules["virtual_head"], profiles, device=args.device
    )
    print(f"✓ Collapsed LM head: {list(collapsed_lm_head.shape)}")
    print()

    # Create output directory
    output_dir = create_output_directory(output_dir)

    # Extract config name and date
    config_name = config.get("wandb_run_name", "default")
    collapse_date = datetime.now().strftime("%Y%m%d")

    # Save outputs
    print(f"[保存] 保存到 {output_dir}/")
    save_collapsed_tensors(output_dir, collapsed_embeddings, collapsed_lm_head)
    save_token_mappings(output_dir, token_list)
    save_collapse_info(
        output_dir, checkpoint_dir, config_path, config_name, metadata, collapse_date
    )
    copy_lora_weights(checkpoint_dir, output_dir)
    copy_tokenizer(checkpoint_dir, output_dir)
    print()

    print("=" * 60)
    print("✓ Collapsed output generation completed!")
    print("=" * 60)
    print()
    print(f"Output directory: {output_dir}")
    print()


if __name__ == "__main__":
    main()
