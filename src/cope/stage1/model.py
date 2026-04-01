"""
Tool planner model that wraps LLM with virtual token embeddings and heads.
Similar to ref/model/offline_rl.py but adapted for our supervised learning approach.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.cope.stage1.virtual_tokens import VirtualTokenEmbedding, VirtualTokenHead
from src.cope.stage1.profile_encoder import ProfileHyperNet


class ToolPlannerModel(nn.Module):
    """
    Wrapper model for tool planning that manages:
    - Base LLM
    - Virtual token embeddings
    - Virtual token head
    - Profile encoder (hypernetwork)
    
    This design allows separate training of virtual components while freezing base LLM.
    """
    
    def __init__(
        self,
        llm: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        num_virtual_tokens: int,
        virtual_token_start_idx: int,
        profile_encoder: Optional[ProfileHyperNet] = None,
        use_profile_encoding: bool = True,
        device: str = 'cuda'
    ):
        """
        Args:
            llm: Base language model
            tokenizer: Tokenizer (with expanded vocabulary)
            num_virtual_tokens: Number of virtual tokens
            virtual_token_start_idx: Starting index of virtual tokens in vocabulary
            profile_encoder: Profile encoder for dynamic embeddings
            use_profile_encoding: Whether to use profile encoding
            device: Device for computation
        """
        super().__init__()
        
        self.llm = llm
        self.tokenizer = tokenizer
        self.num_virtual_tokens = num_virtual_tokens
        self.virtual_token_start_idx = virtual_token_start_idx
        self.use_profile_encoding = use_profile_encoding
        self.device = device
        
        # Get LLM dimensions
        self.hidden_size = llm.config.hidden_size
        self.vocab_size = llm.config.vocab_size  # Original vocab size
        self.expanded_vocab_size = len(tokenizer)  # Expanded vocab size
        
        # Get original embedding and lm_head
        self.base_embeddings = llm.get_input_embeddings()
        self.base_lm_head = llm.get_output_embeddings()
        
        # Profile encoder
        self.profile_encoder = profile_encoder
        
        # Virtual token embedding
        self.virtual_embedding = VirtualTokenEmbedding(
            num_virtual_tokens=num_virtual_tokens,
            embed_dim=self.hidden_size,
            profile_encoder=profile_encoder,
            use_profile_encoding=use_profile_encoding
        )
        
        # Virtual token head
        self.virtual_head = VirtualTokenHead(
            num_virtual_tokens=num_virtual_tokens,
            hidden_size=self.hidden_size,
            profile_encoder=profile_encoder,
            use_profile_encoding=use_profile_encoding
        )
        
        # Store modules except LLM for efficient saving/loading
        self.modules_except_llm = nn.ModuleList([
            self.virtual_embedding,
            self.virtual_head
        ])
        if self.profile_encoder is not None:
            self.modules_except_llm.append(self.profile_encoder)
    
    def initialize_virtual_tokens(
        self,
        tool_semantics: torch.Tensor,
        profiles: torch.Tensor
    ):
        """
        Initialize virtual token embeddings and head from semantic vectors.
        
        Args:
            tool_semantics: Semantic embeddings of shape [num_virtual_tokens, hidden_size]
            profiles: Profile vectors of shape [num_virtual_tokens, 5]
        """
        # Initialize embedding
        self.virtual_embedding.initialize_from_semantics(tool_semantics)
        self.virtual_embedding.set_profiles(profiles)
        
        # Initialize head
        self.virtual_head.initialize_from_semantics(tool_semantics)
        self.virtual_head.set_profiles(profiles)
    
    def update_profiles(self, profiles: torch.Tensor):
        """
        Update profile vectors (e.g., during training with different configurations).
        
        Args:
            profiles: Profile vectors of shape [num_virtual_tokens, 5]
        """
        self.virtual_embedding.set_profiles(profiles)
        self.virtual_head.set_profiles(profiles)
    
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed input tokens (both base vocab and virtual tokens).
        
        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
        
        Returns:
            Embeddings of shape [batch_size, seq_len, hidden_size]
        """
        # Separate base and virtual tokens
        is_virtual = input_ids >= self.virtual_token_start_idx
        
        # Clamp input_ids to valid range for base_embeddings to avoid out-of-bounds access
        # Virtual token positions will be replaced anyway, so clamping them is safe
        safe_input_ids = torch.clamp(input_ids, max=self.vocab_size - 1)
        
        # Get base embeddings (safe now, virtual tokens are clamped)
        base_embeds = self.base_embeddings(safe_input_ids)
        
        # Get virtual embeddings and replace where needed
        if is_virtual.any():
            virtual_embeds = self.virtual_embedding(input_ids, self.virtual_token_start_idx)
            # Replace base embeddings with virtual embeddings where applicable
            base_embeds = torch.where(
                is_virtual.unsqueeze(-1),
                virtual_embeds,
                base_embeds
            )
        
        return base_embeds
    
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for both base vocab and virtual tokens.
        
        Args:
            hidden_states: Hidden states of shape [batch_size, seq_len, hidden_size]
        
        Returns:
            Logits of shape [batch_size, seq_len, expanded_vocab_size]
        """
        # Get base logits (for original vocabulary)
        base_logits = self.base_lm_head(hidden_states)  # [..., vocab_size]
        
        # Only use logits up to virtual_token_start_idx (true base vocabulary)
        # This avoids including any padding/alignment tokens in base_lm_head
        base_logits = base_logits[..., :self.virtual_token_start_idx]
        
        # Get virtual token logits
        virtual_logits = self.virtual_head(hidden_states)  # [..., num_virtual_tokens]
        
        # Concatenate to form full vocabulary logits
        full_logits = torch.cat([base_logits, virtual_logits], dim=-1)
        
        return full_logits
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for language modeling [batch_size, seq_len]
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with 'loss' and 'logits'
        """
        # Embed tokens
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        
        # Compute logits
        logits = self.compute_logits(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            vocab_dim = shift_logits.size(-1)
            loss = loss_fct(
                shift_logits.view(-1, vocab_dim),
                shift_labels.view(-1)
            )
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate tokens (for inference).
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation arguments
        
        Returns:
            Generated token IDs
        """
        # Embed input tokens
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Use LLM's generate method (need to override to use our logits computation)
        # For now, do simple greedy decoding
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.llm(
                inputs_embeds=self.embed_tokens(generated),
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            hidden_states = outputs.hidden_states[-1]
            logits = self.compute_logits(hidden_states)
            
            # Get next token (greedy)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update attention mask if provided
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
                ], dim=1)
        
        return generated
    
    def save_pretrained(self, save_dir: str, save_llm: bool = False):
        """
        Save model components.
        
        Args:
            save_dir: Directory to save to
            save_llm: Whether to save the base LLM (if using LoRA, save adapter instead)
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Always save virtual components
        torch.save(
            self.modules_except_llm.state_dict(),
            os.path.join(save_dir, 'modules_except_llm.bin')
        )
        
        # Save LLM if requested
        if save_llm:
            # Check if using LoRA
            if hasattr(self.llm, 'save_pretrained'):
                # This will save LoRA adapters if PEFT is used
                self.llm.save_pretrained(save_dir)
            else:
                torch.save(
                    self.llm.state_dict(),
                    os.path.join(save_dir, 'llm.bin')
                )
        
        # Save metadata
        metadata = {
            'num_virtual_tokens': self.num_virtual_tokens,
            'virtual_token_start_idx': self.virtual_token_start_idx,
            'hidden_size': self.hidden_size,
            'vocab_size': self.vocab_size,
            'expanded_vocab_size': self.expanded_vocab_size,
            'use_profile_encoding': self.use_profile_encoding
        }
        torch.save(metadata, os.path.join(save_dir, 'metadata.bin'))
    
    @classmethod
    def load_pretrained(
        cls,
        load_dir: str,
        llm: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        profile_encoder: Optional[ProfileHyperNet] = None,
        device: str = 'cuda'
    ):
        """
        Load model from saved directory.
        
        Args:
            load_dir: Directory to load from
            llm: Base LLM (should be loaded separately)
            tokenizer: Tokenizer
            profile_encoder: Profile encoder instance
            device: Device to load to
        
        Returns:
            ToolPlannerModel instance
        """
        import os
        
        # Load metadata
        metadata = torch.load(os.path.join(load_dir, 'metadata.bin'), map_location=device)
        
        # Create model instance
        model = cls(
            llm=llm,
            tokenizer=tokenizer,
            num_virtual_tokens=metadata['num_virtual_tokens'],
            virtual_token_start_idx=metadata['virtual_token_start_idx'],
            profile_encoder=profile_encoder,
            use_profile_encoding=metadata['use_profile_encoding'],
            device=device
        )
        
        # Load virtual components
        state_dict = torch.load(
            os.path.join(load_dir, 'modules_except_llm.bin'),
            map_location=device
        )
        model.modules_except_llm.load_state_dict(state_dict)
        
        return model
