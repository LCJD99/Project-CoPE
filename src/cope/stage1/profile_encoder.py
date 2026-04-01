"""
Profile Encoder (Hypernetwork)
将离散的资源配置向量映射为具有语义拓扑结构的 Embedding 偏移量
"""

import json
from pathlib import Path
import torch
import torch.nn as nn


def load_system_config(system_config_path: str = None) -> dict:
    """
    加载系统最大配置
    
    Args:
        system_config_path: system.json 文件路径，如为 None 则使用默认路径
    
    Returns:
        系统配置字典
    """
    if system_config_path is None:
        # 默认路径
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        system_config_path = project_root / 'data' / 'raw' / 'system.json'
    
    with open(system_config_path, 'r') as f:
        config = json.load(f)
    
    # 解析配置
    max_config = {
        'cpu_core': config['cpu core'],
        'cpu_mem_gb': float(config['cpu memory'].replace('GB', '')),
        'gpu_sm': config['gpu sm'],
        'gpu_mem_gb': float(config['gpu memory'].replace('GB', ''))
    }
    
    return max_config


class ProfileHyperNet(nn.Module):
    """
    Profile Encoder: 将资源配置向量编码为 Embedding 空间的偏移量
    
    核心思路:
        Token_Embedding = E_tool + MLP(Profile_Vector)
    
    通过相加而非拼接，在保留工具语义的基础上注入资源约束信息
    
    支持两种输入模式：
    1. 离散级别 (1=low, 2=medium, 3=high)
    2. 归一化的真实配置值 (0-1 范围)
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: list = [128, 512],
        output_dim: int = 4096,
        activation: str = 'gelu',
        dropout: float = 0.0,
        zero_init: bool = True,
        normalize: bool = True,
        max_values: dict = None,
        system_config_path: str = None
    ):
        """
        初始化 Profile Encoder
        
        Args:
            input_dim: 输入配置向量维度（默认5：tool_size, cpu_core, cpu_mem, gpu_sm, gpu_mem）
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度，应与模型的 hidden_size 一致
            activation: 激活函数类型 ('gelu', 'relu', 'silu')
            dropout: Dropout 概率
            zero_init: 是否对最后一层进行零初始化（确保初始状态不破坏基础语义）
            normalize: 是否对输入进行归一化
            max_values: 最大配置值字典 {'cpu_core': 32, 'cpu_mem_gb': 64, ...}
            system_config_path: system.json 路径，如果 max_values 为 None 则从此加载
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize = normalize
        
        # 加载最大配置值
        if normalize:
            if max_values is None:
                max_values = load_system_config(system_config_path)
            
            self.max_values = max_values
            # tool_size 的最大值设为 3 (small=1, medium=2, large=3)
            self.max_tool_size = 3.0
        else:
            self.max_values = None
            self.max_tool_size = 3.0
        
        # 选择激活函数
        if activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 构建 MLP 层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Zero Initialization - 关键设计
        # 确保初始状态下 Delta = 0，不破坏工具的基础语义 Embedding
        if zero_init:
            self._zero_init_output_layer()
    
    def _zero_init_output_layer(self):
        """将最后一层的权重和偏置初始化为零"""
        # 找到最后一个 Linear 层
        for module in reversed(list(self.net.modules())):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                break
    
    def normalize_profile(self, profile_vector: torch.Tensor) -> torch.Tensor:
        """
        将真实配置值归一化到 [0, 1] 范围
        
        Args:
            profile_vector: 真实配置值 [batch_size, 5]
                           [tool_size, cpu_core, cpu_mem_gb, gpu_sm, gpu_mem_gb]
        
        Returns:
            归一化后的向量 [batch_size, 5]
        """
        if not self.normalize or self.max_values is None:
            return profile_vector
        
        normalized = profile_vector.clone()
        
        # tool_size: 已经是 1-3，归一化到 [0, 1]
        normalized[:, 0] = (profile_vector[:, 0] - 1) / (self.max_tool_size - 1)
        
        # cpu_core
        normalized[:, 1] = profile_vector[:, 1] / self.max_values['cpu_core']
        
        # cpu_mem_gb
        normalized[:, 2] = profile_vector[:, 2] / self.max_values['cpu_mem_gb']
        
        # gpu_sm
        normalized[:, 3] = profile_vector[:, 3] / self.max_values['gpu_sm']
        
        # gpu_mem_gb
        normalized[:, 4] = profile_vector[:, 4] / self.max_values['gpu_mem_gb']
        
        # 防止超出范围（理论上不应该超过，但保险起见）
        normalized = torch.clamp(normalized, 0.0, 1.0)
        
        return normalized
    
    def forward(self, profile_vector: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            profile_vector: 资源配置向量 [batch_size, input_dim] 或 [num_tokens, input_dim]
                           如果 normalize=True，期望真实配置值
                           如果 normalize=False，期望离散级别 (1-3)
        
        Returns:
            embedding_delta: Embedding 空间的偏移量 [batch_size, output_dim]
        """
        # 归一化（如果启用）
        if self.normalize:
            profile_vector = self.normalize_profile(profile_vector)
        
        embedding = self.net(profile_vector)
        return embedding
    
    def get_embedding_shift(self, profile_configs: list) -> torch.Tensor:
        """
        获取批量配置的 Embedding 偏移
        
        Args:
            profile_configs: 配置列表，如 [[3, 3, 3, 3, 3], [1, 1, 1, 1, 1], ...]
        
        Returns:
            embedding_shifts: [num_configs, output_dim]
        """
        profile_tensor = torch.tensor(
            profile_configs,
            dtype=torch.float32,
            device=next(self.parameters()).device
        )
        return self.forward(profile_tensor)


class ProfileHyperNetV2(nn.Module):
    """
    Profile Encoder V2: 增强版，支持独立编码每个资源维度后融合
    
    适用于需要更细粒度控制的场景
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        per_dim_hidden: int = 64,
        fusion_hidden_dims: list = [512],
        output_dim: int = 4096,
        activation: str = 'gelu',
        dropout: float = 0.0,
        zero_init: bool = True
    ):
        """
        Args:
            input_dim: 输入维度数量
            per_dim_hidden: 每个维度独立编码的隐藏层大小
            fusion_hidden_dims: 融合层的隐藏层维度
            output_dim: 输出维度
            activation: 激活函数
            dropout: Dropout 概率
            zero_init: 是否零初始化输出层
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 激活函数
        if activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 为每个维度创建独立的编码器
        self.dim_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, per_dim_hidden),
                act_fn,
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )
            for _ in range(input_dim)
        ])
        
        # 融合层
        fusion_input_dim = input_dim * per_dim_hidden
        fusion_layers = []
        prev_dim = fusion_input_dim
        
        for hidden_dim in fusion_hidden_dims:
            fusion_layers.append(nn.Linear(prev_dim, hidden_dim))
            fusion_layers.append(act_fn)
            if dropout > 0:
                fusion_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        fusion_layers.append(nn.Linear(prev_dim, output_dim))
        self.fusion_net = nn.Sequential(*fusion_layers)
        
        if zero_init:
            self._zero_init_output_layer()
    
    def _zero_init_output_layer(self):
        """将融合网络的最后一层初始化为零"""
        for module in reversed(list(self.fusion_net.modules())):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                break
    
    def forward(self, profile_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            profile_vector: [batch_size, input_dim]
        
        Returns:
            embedding_delta: [batch_size, output_dim]
        """
        # 分离各维度并独立编码
        encoded_dims = []
        for i, encoder in enumerate(self.dim_encoders):
            dim_value = profile_vector[:, i:i+1]  # [batch_size, 1]
            encoded = encoder(dim_value)  # [batch_size, per_dim_hidden]
            encoded_dims.append(encoded)
        
        # 拼接所有编码
        concatenated = torch.cat(encoded_dims, dim=-1)  # [batch_size, input_dim * per_dim_hidden]
        
        # 融合并输出
        return self.fusion_net(concatenated)


def create_profile_encoder(
    encoder_type: str = 'simple',
    input_dim: int = 5,
    output_dim: int = 4096,
    **kwargs
) -> nn.Module:
    """
    工厂函数：创建 Profile Encoder
    
    Args:
        encoder_type: 编码器类型 ('simple', 'v2')
        input_dim: 输入维度
        output_dim: 输出维度
        **kwargs: 其他参数传递给具体的编码器
    
    Returns:
        Profile Encoder 实例
    """
    if encoder_type == 'simple':
        return ProfileHyperNet(
            input_dim=input_dim,
            output_dim=output_dim,
            **kwargs
        )
    elif encoder_type == 'v2':
        return ProfileHyperNetV2(
            input_dim=input_dim,
            output_dim=output_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


if __name__ == '__main__':
    # 测试代码
    print("Testing ProfileHyperNet...")
    
    # 创建编码器
    encoder = ProfileHyperNet(
        input_dim=5,
        hidden_dims=[128, 512],
        output_dim=4096,
        zero_init=True
    )
    
    # 测试输入
    profile_configs = [
        [3, 3, 3, 3, 3],  # 高配置
        [1, 1, 1, 1, 1],  # 低配置
        [2, 2, 2, 2, 2],  # 中配置
    ]
    
    profile_tensor = torch.tensor(profile_configs, dtype=torch.float32)
    
    # 前向传播
    deltas = encoder(profile_tensor)
    
    print(f"Input shape: {profile_tensor.shape}")
    print(f"Output shape: {deltas.shape}")
    print(f"Output norm (should be ~0 initially due to zero init): {deltas.norm(dim=-1)}")
    
    # 测试 V2 版本
    print("\nTesting ProfileHyperNetV2...")
    encoder_v2 = ProfileHyperNetV2(
        input_dim=5,
        per_dim_hidden=64,
        fusion_hidden_dims=[512],
        output_dim=4096,
        zero_init=True
    )
    
    deltas_v2 = encoder_v2(profile_tensor)
    print(f"V2 Output shape: {deltas_v2.shape}")
    print(f"V2 Output norm: {deltas_v2.norm(dim=-1)}")
