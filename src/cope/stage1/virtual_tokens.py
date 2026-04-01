"""
Virtual token embeddings for tool tokens.
Similar to ref/model/tokens.py but adapted for our use case.
"""
import torch
import torch.nn as nn
from typing import Optional


class VirtualTokenEmbedding(nn.Module):
    """
    Custom embedding layer for virtual tool tokens.
    This replaces the expanded embedding approach to avoid training base model embeddings.
    """
    
    def __init__(
        self,
        num_virtual_tokens: int,
        embed_dim: int,
        profile_encoder: Optional[nn.Module] = None,
        use_profile_encoding: bool = True
    ):
        """
        Args:
            num_virtual_tokens: Number of virtual tokens
            embed_dim: Embedding dimension (should match LLM hidden size)
            profile_encoder: Optional profile encoder (hypernetwork) for dynamic embeddings
            use_profile_encoding: Whether to use profile encoding
        """
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.embed_dim = embed_dim
        self.use_profile_encoding = use_profile_encoding
        
        # Base embeddings for virtual tokens (semantic part)
        self.virtual_embeddings = nn.Embedding(num_virtual_tokens, embed_dim)
        
        # Profile encoder for dynamic adjustment
        self.profile_encoder = profile_encoder
        
        # Buffers for tool semantics and profiles (set externally)
        self.register_buffer('tool_semantics', torch.zeros(num_virtual_tokens, embed_dim))
        self.register_buffer('profiles', torch.zeros(num_virtual_tokens, 5))
    
    def initialize_from_semantics(self, semantics: torch.Tensor):
        """
        Initialize virtual token embeddings from semantic vectors.
        
        Args:
            semantics: Tensor of shape [num_virtual_tokens, embed_dim]
        """
        assert semantics.shape[0] == self.num_virtual_tokens
        assert semantics.shape[1] == self.embed_dim
        
        with torch.no_grad():
            self.virtual_embeddings.weight.data.copy_(semantics)
            self.tool_semantics.copy_(semantics)
    
    def set_profiles(self, profiles: torch.Tensor):
        """
        Set profile vectors for all virtual tokens.
        
        Args:
            profiles: Tensor of shape [num_virtual_tokens, 5]
        """
        assert profiles.shape[0] == self.num_virtual_tokens
        self.profiles.copy_(profiles)
    
    def forward(self, token_ids: torch.Tensor, virtual_token_start_idx: int) -> torch.Tensor:
        """
        Forward pass. Returns embeddings for given token IDs.
        For virtual tokens, applies profile encoding if enabled.
        
        Args:
            token_ids: Token IDs of shape [batch_size, seq_len] or any shape
            virtual_token_start_idx: Starting index of virtual tokens in vocabulary
        
        Returns:
            Embeddings of shape [..., embed_dim]
        """
        original_shape = token_ids.shape
        token_ids_flat = token_ids.reshape(-1)
        
        # Identify virtual tokens
        is_virtual = token_ids_flat >= virtual_token_start_idx
        virtual_indices = (token_ids_flat[is_virtual] - virtual_token_start_idx).long()
        
        # Get base embeddings
        embeddings = self.virtual_embeddings(virtual_indices)
        
        # Apply profile encoding if enabled
        if self.use_profile_encoding and self.profile_encoder is not None:
            # Get profiles for these virtual tokens
            # Ensure profiles match dtype and device of embeddings
            token_profiles = self.profiles[virtual_indices].to(dtype=embeddings.dtype, device=embeddings.device)
            
            # Compute profile delta
            profile_delta = self.profile_encoder(token_profiles)
            
            # Add profile adjustment
            embeddings = embeddings + profile_delta
        
        # Create output tensor
        output = torch.zeros(
            token_ids_flat.shape[0], self.embed_dim,
            dtype=embeddings.dtype, device=embeddings.device
        )
        output[is_virtual] = embeddings
        
        return output.reshape(*original_shape, self.embed_dim)
    
    def get_all_embeddings(self) -> torch.Tensor:
        """
        Get embeddings for all virtual tokens (for efficient batch computation).
        
        Returns:
            Tensor of shape [num_virtual_tokens, embed_dim]
        """
        if self.use_profile_encoding and self.profile_encoder is not None:
            base_embeddings = self.virtual_embeddings.weight
            # Ensure profiles match dtype and device of base_embeddings
            profiles = self.profiles.to(dtype=base_embeddings.dtype, device=base_embeddings.device)
            profile_delta = self.profile_encoder(profiles)
            return base_embeddings + profile_delta
        else:
            return self.virtual_embeddings.weight


class VirtualTokenHead(nn.Module):
    """
    Custom output head for virtual tokens.
    This replaces the expanded lm_head approach.
    """
    
    def __init__(
        self,
        num_virtual_tokens: int,
        hidden_size: int,
        profile_encoder: Optional[nn.Module] = None,
        use_profile_encoding: bool = True
    ):
        """
        Args:
            num_virtual_tokens: Number of virtual tokens
            hidden_size: Hidden size of the model
            profile_encoder: Optional profile encoder (should be shared with embedding layer)
            use_profile_encoding: Whether to use profile encoding
        """
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.hidden_size = hidden_size
        self.use_profile_encoding = use_profile_encoding
        
        # Linear projection from hidden states to virtual token logits
        self.head = nn.Linear(hidden_size, num_virtual_tokens, bias=False)
        
        # Profile encoder (shared with embedding layer)
        self.profile_encoder = profile_encoder
        
        # Buffers for tool semantics and profiles
        self.register_buffer('tool_semantics', torch.zeros(num_virtual_tokens, hidden_size))
        self.register_buffer('profiles', torch.zeros(num_virtual_tokens, 5))
    
    def initialize_from_semantics(self, semantics: torch.Tensor):
        """
        Initialize head weights from semantic vectors.
        
        Args:
            semantics: Tensor of shape [num_virtual_tokens, hidden_size]
        """
        assert semantics.shape[0] == self.num_virtual_tokens
        assert semantics.shape[1] == self.hidden_size
        
        with torch.no_grad():
            # Initialize as transpose of semantics (for dot product)
            self.head.weight.data.copy_(semantics)
            self.tool_semantics.copy_(semantics)
    
    def set_profiles(self, profiles: torch.Tensor):
        """
        Set profile vectors for all virtual tokens.
        
        Args:
            profiles: Tensor of shape [num_virtual_tokens, 5]
        """
        assert profiles.shape[0] == self.num_virtual_tokens
        self.profiles.copy_(profiles)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for virtual tokens.
        
        Args:
            hidden_states: Hidden states of shape [..., hidden_size]
        
        Returns:
            Logits of shape [..., num_virtual_tokens]
        """
        if self.use_profile_encoding and self.profile_encoder is not None:
            # Compute dynamic token weights
            base_weights = self.head.weight.to(dtype=hidden_states.dtype, device=hidden_states.device)  # [num_virtual_tokens, hidden_size]
            # Ensure profiles match dtype and device of hidden_states
            profiles = self.profiles.to(dtype=hidden_states.dtype, device=hidden_states.device)
            profile_delta = self.profile_encoder(profiles)  # [num_virtual_tokens, hidden_size]
            
            dynamic_weights = base_weights + profile_delta
            
            # Compute logits as dot product
            logits = torch.matmul(hidden_states, dynamic_weights.T)
        else:
            # Standard linear projection
            logits = self.head(hidden_states)
        
        return logits
