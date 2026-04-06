"""
Stage 2 Model with Three-Layer Embedding Architecture

This module implements the Stage 2 planning model with a three-layer design:
- Layer 1: Base LLM embeddings (frozen, 0-151664)
- Layer 2: Stage 1 collapsed tool embeddings (frozen, 151665-153365)
- Layer 3: Stage 2 control token embeddings (trainable, 153366-153403)

The model loads Stage 1 collapsed embeddings as fixed tensors and only trains
the new Stage 2 control tokens along with LoRA adapters.
"""

from contextlib import nullcontext
import os
from pathlib import Path
import re
from typing import Optional, Dict, Any, List, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutput
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.cope.stage2.tokens import TOKEN_INIT_MAP


_REF_TOKEN_RE = re.compile(r"<REF_(\d+)>")
_LHS_REF_RE = re.compile(
    r"^\s*(?:<STATEMENT>\s*)?<REF_(\d+)>\s*(?:(?:<EQ>)|=)"
)
_TOOL_TOKEN_RE = re.compile(r"<([A-Z0-9_]+)>")


def _get_stage2_trainable_dtype(_base_dtype: torch.dtype) -> torch.dtype:
    return torch.float32


def _semantic_init_stage2_tensors(
    llm: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_stage2_tokens: int,
    hidden_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_embeddings = llm.get_input_embeddings()
    output_embeddings = llm.get_output_embeddings()

    stage2_embeddings = torch.zeros(
        num_stage2_tokens,
        hidden_size,
        dtype=input_embeddings.weight.dtype,
        device=device,
    )
    stage2_lm_head = torch.zeros(
        num_stage2_tokens,
        hidden_size,
        dtype=(
            output_embeddings.weight.dtype
            if output_embeddings is not None
            else input_embeddings.weight.dtype
        ),
        device=device,
    )

    token_list = list(TOKEN_INIT_MAP.keys())
    with torch.no_grad():
        for idx, token_str in enumerate(token_list):
            if idx >= num_stage2_tokens:
                break
            semantic_word_embs = []
            for word in TOKEN_INIT_MAP[token_str]:
                token_ids = tokenizer(
                    word,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"].to(device)
                word_emb = input_embeddings(token_ids).mean(dim=1).squeeze(0)
                semantic_word_embs.append(word_emb)

            semantic_emb = torch.stack(semantic_word_embs).mean(dim=0)
            stage2_embeddings[idx] = semantic_emb
            stage2_lm_head[idx] = semantic_emb

    return stage2_embeddings, stage2_lm_head


def _enforce_first_token_structure(
    next_token_logits: torch.Tensor,
    step_idx: int,
    statement_token_id: Optional[int],
    ref0_token_id: Optional[int],
) -> None:
    if step_idx != 0:
        return

    preferred_token_id = statement_token_id
    if preferred_token_id is None:
        preferred_token_id = ref0_token_id
    if preferred_token_id is None:
        return
    if preferred_token_id < 0 or preferred_token_id >= next_token_logits.shape[-1]:
        return

    mask = torch.ones_like(next_token_logits, dtype=torch.bool)
    mask[:, preferred_token_id] = False
    next_token_logits.masked_fill_(mask, float("-inf"))


def infer_plan_constraint_state(plan_suffix: str) -> Dict[str, Any]:
    """
    Infer coarse FSM state from current generated plan suffix.

    Returns a dict with:
      - kind: one of start_stmt/stmt_lhs_ref/stmt_eq/stmt_wait_or_exec/wait_refs/finish_refs/expect_tool/in_args/stmt_end/none
      - defined_refs: set[int] refs defined by completed exec statements
      - finish_has_ref: whether current finish line already has >=1 ref token
      - wait_has_ref: whether current wait segment already has >=1 ref token
    """
    lines = plan_suffix.split("\n")
    if plan_suffix.endswith("\n"):
        current_line = ""
        completed_lines = [ln for ln in lines if ln.strip()]
    else:
        current_line = lines[-1] if lines else ""
        completed_lines = [ln for ln in lines[:-1] if ln.strip()]

    defined_refs: Set[int] = set()
    for line in completed_lines:
        m = _LHS_REF_RE.match(line)
        if m:
            defined_refs.add(int(m.group(1)))

    line = current_line.strip()
    if line == "":
        return {"kind": "start_stmt", "defined_refs": defined_refs}

    if line.startswith("<FINISH>"):
        finish_refs = _REF_TOKEN_RE.findall(line)
        return {
            "kind": "finish_refs",
            "defined_refs": defined_refs,
            "finish_has_ref": len(finish_refs) > 0,
        }

    statement_prefix = "<STATEMENT>"
    if line.startswith(statement_prefix):
        statement_body = line[len(statement_prefix) :].strip()
        if statement_body == "":
            return {"kind": "stmt_lhs_ref", "defined_refs": defined_refs}

        if "<EQ>" not in statement_body and "=" not in statement_body:
            if _REF_TOKEN_RE.search(statement_body):
                return {"kind": "stmt_eq", "defined_refs": defined_refs}
            return {"kind": "stmt_lhs_ref", "defined_refs": defined_refs}

        if "<EQ>" in statement_body:
            statement_rhs = statement_body.split("<EQ>", 1)[1].strip()
        else:
            statement_rhs = statement_body.split("=", 1)[1].strip()

        if statement_rhs == "":
            return {"kind": "stmt_wait_or_exec", "defined_refs": defined_refs}

        line = statement_rhs

    if "<WAIT>" in line and "<EXEC>" not in line:
        wait_tail = line.split("<WAIT>", 1)[1]
        wait_refs = [int(ref) for ref in _REF_TOKEN_RE.findall(wait_tail)]
        return {
            "kind": "wait_refs",
            "defined_refs": defined_refs,
            "wait_has_ref": len(wait_refs) > 0,
            "wait_used_refs": set(wait_refs),
        }

    if "<EXEC>" in line:
        after_exec = line.split("<EXEC>", 1)[1]
        if "(" not in after_exec:
            tool_match = _TOOL_TOKEN_RE.search(after_exec)
            if tool_match is not None:
                return {"kind": "expect_lparen", "defined_refs": defined_refs}
            return {"kind": "expect_tool", "defined_refs": defined_refs}

        after_lparen = after_exec.split("(", 1)[1]
        in_quote = False
        escaped = False
        paren_balance = 1
        for ch in after_lparen:
            if in_quote:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    in_quote = False
                continue

            if ch == '"':
                in_quote = True
            elif ch == "(":
                paren_balance += 1
            elif ch == ")":
                paren_balance -= 1
                if paren_balance <= 0:
                    break
        if in_quote or paren_balance > 0:
            return {"kind": "in_args", "defined_refs": defined_refs}
        if "<END_STATEMENT>" not in line and plan_suffix.strip().endswith(")"):
            return {"kind": "stmt_end", "defined_refs": defined_refs}

    return {"kind": "none", "defined_refs": defined_refs}


class Stage2EmbeddingLayer(nn.Module):
    """
    Three-layer embedding architecture for Stage 2.
    Routes tokens to appropriate embedding layers based on token ID ranges.
    """

    def __init__(
        self,
        base_embeddings: nn.Embedding,
        stage1_collapsed_embeddings: torch.Tensor,
        stage2_embeddings: nn.Embedding,
        stage1_start_idx: int = 151665,
        stage2_start_idx: int = 153366,
    ):
        """
        Args:
            base_embeddings: Base LLM embeddings (frozen)
            stage1_collapsed_embeddings: Stage 1 collapsed embeddings [1701, hidden_size] (frozen)
            stage2_embeddings: Stage 2 trainable embeddings [38, hidden_size]
            stage1_start_idx: Starting index of Stage 1 tokens (default: 151665)
            stage2_start_idx: Starting index of Stage 2 tokens (default: 153366)
        """
        super().__init__()

        # Layer 1: Base embeddings (frozen)
        self.base_embeddings = base_embeddings
        self.base_embeddings.weight.requires_grad = False

        # Get base embedding dtype
        base_emb_dtype = base_embeddings.weight.dtype

        # Layer 2: Stage 1 embeddings (frozen, from collapsed checkpoint)
        # CRITICAL: Ensure dtype matches base_embeddings
        if stage1_collapsed_embeddings.dtype != base_emb_dtype:
            stage1_collapsed_embeddings = stage1_collapsed_embeddings.to(
                dtype=base_emb_dtype
            )

        self.stage1_embeddings = nn.Embedding.from_pretrained(
            stage1_collapsed_embeddings, freeze=True
        )

        # Layer 3: Stage 2 embeddings (trainable)
        # NOTE: stage2_embeddings should already be in correct dtype and device
        # from Stage2PlannerModel initialization. Just assign it directly.
        self.stage2_embeddings = stage2_embeddings

        # Token range boundaries
        self.stage1_start_idx = stage1_start_idx
        self.stage2_start_idx = stage2_start_idx

        # Hidden size
        self.hidden_size = base_embeddings.weight.shape[1]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Route tokens to appropriate embedding layers.

        All embeddings are cast to a unified dtype (base_embeddings dtype) to
        prevent dtype mismatches that can cause NaN when mixing float16/float32.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]

        Returns:
            Embeddings [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape
        target_dtype = self.base_embeddings.weight.dtype

        # Create output tensor in unified dtype
        embeddings = torch.zeros(
            batch_size,
            seq_len,
            self.hidden_size,
            device=input_ids.device,
            dtype=target_dtype,
        )

        # Create masks for each token range
        base_mask = input_ids < self.stage1_start_idx
        stage1_mask = (input_ids >= self.stage1_start_idx) & (
            input_ids < self.stage2_start_idx
        )
        stage2_mask = input_ids >= self.stage2_start_idx

        # Fill embeddings from appropriate layers (cast to unified dtype)
        if base_mask.any():
            base_emb = self.base_embeddings(input_ids[base_mask])
            embeddings[base_mask] = base_emb.to(dtype=target_dtype)

        if stage1_mask.any():
            stage1_indices = input_ids[stage1_mask] - self.stage1_start_idx
            # Clamp indices to valid range to prevent crashes
            stage1_indices = stage1_indices.clamp(
                0, self.stage1_embeddings.num_embeddings - 1
            )
            stage1_emb = self.stage1_embeddings(stage1_indices)
            embeddings[stage1_mask] = stage1_emb.to(dtype=target_dtype)

        if stage2_mask.any():
            stage2_indices = input_ids[stage2_mask] - self.stage2_start_idx
            # Clamp indices to valid range to prevent crashes
            stage2_indices = stage2_indices.clamp(
                0, self.stage2_embeddings.num_embeddings - 1
            )
            stage2_emb = self.stage2_embeddings(stage2_indices)
            # CRITICAL: Cast to target_dtype to prevent dtype mismatch
            # Stage2 embeddings may be in float32 (promoted by optimizer/Trainer)
            # while output tensor is in float16/bfloat16
            embeddings[stage2_mask] = stage2_emb.to(dtype=target_dtype)

        return embeddings


class Stage2LMHead(nn.Module):
    """
    Three-layer LM head for Stage 2.
    Computes logits for all tokens by concatenating outputs from three layers.
    """

    def __init__(
        self,
        base_lm_head: nn.Module,
        stage1_collapsed_lm_head: torch.Tensor,
        stage2_lm_head: nn.Embedding,
        base_vocab_size: int = 151665,
        stage1_vocab_size: int = 1701,
        stage2_vocab_size: int = 38,
        base_lm_head_size: Optional[int] = None,
    ):
        """
        Args:
            base_lm_head: Base LLM head (frozen)
            stage1_collapsed_lm_head: Stage 1 collapsed head [1701, hidden_size] (frozen)
            stage2_lm_head: Stage 2 trainable head [38, hidden_size]
            base_vocab_size: Size of base vocabulary (to use from base_lm_head)
            stage1_vocab_size: Number of Stage 1 tool tokens
            stage2_vocab_size: Number of Stage 2 control tokens
            base_lm_head_size: Actual output size of base_lm_head (may be larger than base_vocab_size)
        """
        super().__init__()

        # Layer 1: Base LM head (frozen)
        self.base_lm_head = base_lm_head
        for param in self.base_lm_head.parameters():
            param.requires_grad = False

        # Get base LM head dtype
        base_lm_dtype = next(base_lm_head.parameters()).dtype

        # Layer 2: Stage 1 LM head (frozen, as tensor)
        # CRITICAL: Ensure dtype matches base LM head
        if stage1_collapsed_lm_head.dtype != base_lm_dtype:
            stage1_collapsed_lm_head = stage1_collapsed_lm_head.to(dtype=base_lm_dtype)
        self.register_buffer("stage1_lm_head", stage1_collapsed_lm_head)

        # Layer 3: Stage 2 LM head (trainable)
        # NOTE: stage2_lm_head should already be in correct dtype and device
        # from Stage2PlannerModel initialization. Just assign it directly.
        self.stage2_lm_head = stage2_lm_head

        self.base_vocab_size = base_vocab_size
        self.stage1_vocab_size = stage1_vocab_size
        self.stage2_vocab_size = stage2_vocab_size
        self.total_vocab_size = base_vocab_size + stage1_vocab_size + stage2_vocab_size

        # Track actual base LM head size for slicing
        self.base_lm_head_size = (
            base_lm_head_size if base_lm_head_size is not None else base_vocab_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for all vocabulary tokens.

        All computations are cast to a unified dtype to prevent NaN from
        dtype mismatches between frozen (float16) and trainable (float32) components.

        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]

        Returns:
            Logits [batch_size, seq_len, total_vocab_size]
        """
        # Use hidden_states dtype as the computation dtype
        compute_dtype = hidden_states.dtype

        # Base logits [batch_size, seq_len, base_lm_head_size]
        base_logits_full = self.base_lm_head(hidden_states)

        # Slice to match tokenizer's base vocab size
        # (base LM head may output more tokens than we need)
        base_logits = base_logits_full[..., : self.base_vocab_size]

        # Stage 1 tool logits [batch_size, seq_len, stage1_vocab_size]
        # Cast stage1_lm_head to compute_dtype to prevent dtype mismatch
        stage1_weight = self.stage1_lm_head.to(dtype=compute_dtype)
        stage1_logits = torch.matmul(hidden_states, stage1_weight.T)

        # Stage 2 control token logits [batch_size, seq_len, stage2_vocab_size]
        # Cast stage2_lm_head to compute_dtype to prevent dtype mismatch
        stage2_weight = self.stage2_lm_head.weight.to(dtype=compute_dtype)
        stage2_logits = torch.matmul(hidden_states, stage2_weight.T)

        # Concatenate all logits [batch_size, seq_len, total_vocab_size]
        full_logits = torch.cat([base_logits, stage1_logits, stage2_logits], dim=-1)

        return full_logits


class Stage2PlannerModel(nn.Module):
    """
    Stage 2 planning model wrapper.

    Manages:
    - Base LLM with LoRA (continued from Stage 1)
    - Three-layer embedding architecture
    - Three-layer LM head architecture
    - Frozen Stage 1 embeddings/heads
    - Trainable Stage 2 embeddings/heads
    """

    def __init__(
        self,
        llm: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        stage1_collapsed_checkpoint: str,
        stage2_embeddings_path: str,
        device: str = "cuda",
    ):
        """
        Args:
            llm: Base LLM (with or without LoRA)
            tokenizer: Extended tokenizer with Stage 2 tokens
            stage1_collapsed_checkpoint: Path to Stage 1 collapsed checkpoint
            stage2_embeddings_path: Path to initialized Stage 2 embeddings
            device: Device for computation
        """
        super().__init__()

        self.llm = llm
        self.tokenizer = tokenizer
        self.device = device
        self.uses_stage2_embedding_wrapper = True
        self.config = llm.config
        self.generation_config = getattr(llm, "generation_config", None)
        self.warnings_issued = getattr(llm, "warnings_issued", {})

        # Get dimensions
        self.hidden_size = llm.config.hidden_size
        self.num_refs = 32

        # CRITICAL: tokenizer size is the source of truth for label token ids.
        self.total_vocab_size = len(tokenizer)

        # Load Stage 1 collapsed embeddings (frozen)
        stage1_emb_path = Path(stage1_collapsed_checkpoint) / "collapsed_embeddings.bin"
        stage1_head_path = Path(stage1_collapsed_checkpoint) / "collapsed_lm_head.bin"

        stage1_collapsed_emb = torch.load(stage1_emb_path, map_location=device)
        stage1_collapsed_head = torch.load(stage1_head_path, map_location=device)

        # Infer Stage1 size from checkpoint assets instead of hardcoded values.
        self.stage1_vocab_size = int(stage1_collapsed_emb.shape[0])
        if int(stage1_collapsed_head.shape[0]) != self.stage1_vocab_size:
            raise ValueError(
                "Stage1 collapsed embedding/head size mismatch: "
                f"emb={stage1_collapsed_emb.shape[0]}, head={stage1_collapsed_head.shape[0]}"
            )

        # Infer Stage2 band from mandatory Stage2 control/reference tokens.
        stage2_anchor_tokens = [
            "<EXEC>",
            "<FINISH>",
            "<WAIT>",
            "<EQ>",
            "<STATEMENT>",
            "<END_STATEMENT>",
        ] + [f"<REF_{i}>" for i in range(self.num_refs)]
        stage2_token_ids: List[int] = []
        for token in stage2_anchor_tokens:
            token_id = int(self.tokenizer.convert_tokens_to_ids(token))
            if 0 <= token_id < self.total_vocab_size:
                stage2_token_ids.append(token_id)
        if not stage2_token_ids:
            raise ValueError("Failed to locate Stage2 control/reference tokens in tokenizer.")

        self.stage2_start_idx = min(stage2_token_ids)
        self.stage2_vocab_size = max(stage2_token_ids) - self.stage2_start_idx + 1
        self.stage1_start_idx = self.stage2_start_idx - self.stage1_vocab_size
        if self.stage1_start_idx < 0:
            raise ValueError(
                f"Invalid inferred stage1_start_idx={self.stage1_start_idx}, "
                f"stage2_start_idx={self.stage2_start_idx}, stage1_vocab_size={self.stage1_vocab_size}"
            )
        self.base_vocab_size = self.stage1_start_idx

        expected_total = (
            self.base_vocab_size + self.stage1_vocab_size + self.stage2_vocab_size
        )
        if expected_total != self.total_vocab_size:
            print(
                "WARNING: inferred vocab partition mismatch, using contiguous fallback. "
                f"inferred_total={expected_total}, tokenizer_total={self.total_vocab_size}"
            )
            self.base_vocab_size = (
                self.total_vocab_size - self.stage1_vocab_size - self.stage2_vocab_size
            )
            self.stage1_start_idx = self.base_vocab_size
            self.stage2_start_idx = self.base_vocab_size + self.stage1_vocab_size

        # Convert to match base LLM dtype (critical for dtype consistency)
        stage1_collapsed_emb = stage1_collapsed_emb.to(dtype=llm.dtype)
        stage1_collapsed_head = stage1_collapsed_head.to(dtype=llm.dtype)

        # Load Stage 2 initialized embeddings (trainable)
        stage2_emb_tensor = None
        stage2_head_tensor = None
        force_runtime_init = os.getenv("COPE_STAGE2_FORCE_RUNTIME_INIT", "0") == "1"
        stage2_path = Path(stage2_embeddings_path)
        if not force_runtime_init and (stage2_path / "stage2_embeddings.bin").exists():
            stage2_emb_path = stage2_path / "stage2_embeddings.bin"
            stage2_head_path = stage2_path / "stage2_lm_head.bin"
            stage2_emb_tensor = torch.load(stage2_emb_path, map_location=device)
            stage2_head_tensor = torch.load(stage2_head_path, map_location=device)
        elif (
            not force_runtime_init
            and (stage2_path / "stage2_embeddings_init.bin").exists()
        ):
            # Use initialized embeddings
            stage2_emb_path = stage2_path / "stage2_embeddings_init.bin"
            stage2_head_path = stage2_path / "stage2_lm_head_init.bin"
            stage2_emb_tensor = torch.load(stage2_emb_path, map_location=device)
            stage2_head_tensor = torch.load(stage2_head_path, map_location=device)
        if stage2_emb_tensor is None or stage2_head_tensor is None:
            print(
                "Stage2 embedding init files not found, performing in-memory semantic "
                "initialization for this training run."
            )
            stage2_emb_tensor, stage2_head_tensor = _semantic_init_stage2_tensors(
                llm=llm,
                tokenizer=tokenizer,
                num_stage2_tokens=self.stage2_vocab_size,
                hidden_size=self.hidden_size,
                device=device,
            )

        trainable_dtype = _get_stage2_trainable_dtype(llm.dtype)
        stage2_emb_tensor = stage2_emb_tensor.to(dtype=trainable_dtype)
        stage2_head_tensor = stage2_head_tensor.to(dtype=trainable_dtype)

        # Create Stage 2 embedding and head modules (trainable)
        # CRITICAL: Create embeddings in correct dtype to avoid NaN
        self.stage2_embedding_module = nn.Embedding(
            self.stage2_vocab_size, self.hidden_size
        )
        # Keep trainable embeddings in fp32 to avoid AMP GradScaler fp16 unscale errors.
        self.stage2_embedding_module = self.stage2_embedding_module.to(
            device=device, dtype=trainable_dtype
        )
        # Now copy the weights (both are same dtype, no conversion)
        self.stage2_embedding_module.weight.data.copy_(stage2_emb_tensor)

        self.stage2_lm_head_module = nn.Embedding(
            self.stage2_vocab_size, self.hidden_size
        )
        # Keep trainable LM head in fp32 for stable mixed-precision training.
        self.stage2_lm_head_module = self.stage2_lm_head_module.to(
            device=device, dtype=trainable_dtype
        )
        # Now copy the weights (both are same dtype, no conversion)
        self.stage2_lm_head_module.weight.data.copy_(stage2_head_tensor)

        # Get base embeddings and head from LLM
        base_embeddings = llm.get_input_embeddings()
        base_lm_head = llm.get_output_embeddings()

        # Check base LM head output size
        actual_base_lm_size = base_lm_head.weight.shape[0]

        # CRITICAL: Adjust base_vocab_size to match actual LM head output
        # We'll only use the first base_vocab_size logits from base LM head
        if actual_base_lm_size != self.base_vocab_size:
            print(
                f"WARNING: Base LM head size ({actual_base_lm_size}) != "
                f"calculated base vocab ({self.base_vocab_size}), will slice logits"
            )
            self.base_lm_head_size = actual_base_lm_size
        else:
            self.base_lm_head_size = self.base_vocab_size

        # Create three-layer embedding and head
        self.embedding_layer = Stage2EmbeddingLayer(
            base_embeddings=base_embeddings,
            stage1_collapsed_embeddings=stage1_collapsed_emb,
            stage2_embeddings=self.stage2_embedding_module,
            stage1_start_idx=self.stage1_start_idx,
            stage2_start_idx=self.stage2_start_idx,
        )

        self.lm_head = Stage2LMHead(
            base_lm_head=base_lm_head,
            stage1_collapsed_lm_head=stage1_collapsed_head,
            stage2_lm_head=self.stage2_lm_head_module,
            base_vocab_size=self.base_vocab_size,
            stage1_vocab_size=self.stage1_vocab_size,
            stage2_vocab_size=self.stage2_vocab_size,
            base_lm_head_size=self.base_lm_head_size,
        )

        print(
            f"Stage 2 model initialized: vocab={self.total_vocab_size} "
            f"(base={self.base_vocab_size} + s1={self.stage1_vocab_size} + s2={self.stage2_vocab_size}), "
            f"dtype={llm.dtype}"
        )
        self._init_decode_constraint_cache()
        self._apply_structure_constraints_for_logprobs = False
        self._restrict_decode_to_new_and_tools = True

    def set_logprob_structure_constraints(self, enabled: bool) -> None:
        self._apply_structure_constraints_for_logprobs = bool(enabled)

    def set_decode_vocab_restriction(self, enabled: bool) -> None:
        """
        Control decode-time vocab restriction.
        When enabled, generation only allows:
        - Stage1 tool/new tokens
        - Stage2 control/new tokens
        - BOS/EOS (if defined in tokenizer)
        """
        self._restrict_decode_to_new_and_tools = bool(enabled)

    def _init_decode_constraint_cache(self) -> None:
        self.exec_token_id = self._token_id_or_none("<EXEC>")
        self.finish_token_id = self._token_id_or_none("<FINISH>")
        self.wait_token_id = self._token_id_or_none("<WAIT>")
        self.eq_token_id = self._token_id_or_none("<EQ>")
        self.statement_token_id = self._token_id_or_none("<STATEMENT>")
        self.end_statement_token_id = self._token_id_or_none("<END_STATEMENT>")

        self.ref_token_ids: Dict[int, int] = {}
        for i in range(self.num_refs):
            token_id = self._token_id_or_none(f"<REF_{i}>")
            if token_id is not None:
                self.ref_token_ids[i] = token_id
        self.ref0_token_id = self.ref_token_ids.get(0)

        self.control_token_ids = {
            tid
            for tid in [
                self.exec_token_id,
                self.finish_token_id,
                self.wait_token_id,
                self.eq_token_id,
                self.statement_token_id,
                self.end_statement_token_id,
            ]
            if tid is not None
        }
        self.open_paren_token_ids: Set[int] = set()
        for token_id in range(self.total_vocab_size):
            token_text = self.tokenizer.convert_ids_to_tokens(token_id)
            if not isinstance(token_text, str):
                continue
            if "(" in token_text:
                self.open_paren_token_ids.add(int(token_id))

        # Decode whitelist: only Stage1/Stage2 new tokens, plus BOS/EOS.
        stage1_ids = set(range(self.stage1_start_idx, self.stage2_start_idx))
        stage2_ids = set(
            range(self.stage2_start_idx, self.stage2_start_idx + self.stage2_vocab_size)
        )
        decode_allowed_ids = set(stage1_ids) | set(stage2_ids)
        if self.tokenizer.bos_token_id is not None:
            decode_allowed_ids.add(int(self.tokenizer.bos_token_id))
        if self.tokenizer.eos_token_id is not None:
            decode_allowed_ids.add(int(self.tokenizer.eos_token_id))
        self.decode_allowed_token_ids = {
            tid for tid in decode_allowed_ids if 0 <= tid < self.total_vocab_size
        }

    def _token_id_or_none(self, token: str) -> Optional[int]:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            return None
        token_id = int(token_id)
        if token_id < 0 or token_id >= self.total_vocab_size:
            return None
        return token_id

    def _plan_suffix_from_generated_ids(self, token_ids: torch.Tensor) -> Optional[str]:
        text = self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=False)
        if "[PLAN_START]" in text:
            return text.split("[PLAN_START]", 1)[1]
        return text

    def _extract_generated_control_ids(self, generated_row: torch.Tensor) -> List[int]:
        """
        Extract generated suffix token ids (after prompt) by taking the trailing
        contiguous span restricted to decode-allowed ids.
        """
        ids = [int(x) for x in generated_row.tolist()]
        if not ids:
            return []
        start = len(ids)
        for idx in range(len(ids) - 1, -1, -1):
            if ids[idx] in self.decode_allowed_token_ids:
                start = idx
                continue
            break
        return ids[start:]

    def _compute_ebnf_allowed_next_ids(
        self,
        generated_row: torch.Tensor,
        eos_token_ids: Set[int],
    ) -> Set[int]:
        """
        State-machine constrained next-token set based on docs/EBNF.md:

          Program   ::= ExecStmt* <FINISH> ResultList
          ExecStmt  ::= <STATEMENT> Ref <EQ> PreCondition? MetaExec <END_STATEMENT>
          PreCond   ::= <WAIT> RefList
          MetaExec  ::= <EXEC> Tool
          RefList   ::= Ref+
          ResultList::= Ref+
        """
        seq = self._extract_generated_control_ids(generated_row)
        ref_ids = set(self.ref_token_ids.values())
        defined_ref_ids: Set[int] = set()
        stage1_tool_ids = set(range(self.stage1_start_idx, self.stage2_start_idx))

        phase = "PROGRAM"
        current_lhs_ref: Optional[int] = None
        wait_ref_count = 0

        for token_id in seq:
            # Program phase: zero or more statements, then FINISH.
            if phase == "PROGRAM":
                if (
                    self.statement_token_id is not None
                    and token_id == self.statement_token_id
                ):
                    phase = "STMT_LHS"
                    continue
                if (
                    self.finish_token_id is not None
                    and token_id == self.finish_token_id
                ):
                    phase = "FINISH_REFS"
                    continue
                return set(eos_token_ids)

            if phase == "STMT_LHS":
                if token_id in ref_ids:
                    current_lhs_ref = token_id
                    phase = "STMT_EQ"
                    continue
                return set(eos_token_ids)

            if phase == "STMT_EQ":
                if self.eq_token_id is not None and token_id == self.eq_token_id:
                    phase = "STMT_AFTER_EQ"
                    continue
                return set(eos_token_ids)

            if phase == "STMT_AFTER_EQ":
                if self.wait_token_id is not None and token_id == self.wait_token_id:
                    wait_ref_count = 0
                    phase = "STMT_WAIT_REFS"
                    continue
                if self.exec_token_id is not None and token_id == self.exec_token_id:
                    phase = "STMT_TOOL"
                    continue
                return set(eos_token_ids)

            if phase == "STMT_WAIT_REFS":
                if token_id in defined_ref_ids:
                    wait_ref_count += 1
                    continue
                if (
                    self.exec_token_id is not None
                    and token_id == self.exec_token_id
                    and wait_ref_count > 0
                ):
                    phase = "STMT_TOOL"
                    continue
                return set(eos_token_ids)

            if phase == "STMT_TOOL":
                if token_id in stage1_tool_ids:
                    if current_lhs_ref is not None:
                        defined_ref_ids.add(current_lhs_ref)
                    phase = "STMT_END"
                    continue
                return set(eos_token_ids)

            if phase == "STMT_END":
                if (
                    self.end_statement_token_id is not None
                    and token_id == self.end_statement_token_id
                ):
                    phase = "PROGRAM"
                    continue
                return set(eos_token_ids)

            if phase == "FINISH_REFS":
                if token_id in defined_ref_ids:
                    phase = "FINISH_REFS_NONEMPTY"
                    continue
                return set(eos_token_ids)

            if phase == "FINISH_REFS_NONEMPTY":
                if token_id in defined_ref_ids:
                    continue
                if token_id in eos_token_ids:
                    phase = "DONE"
                    continue
                return set(eos_token_ids)

            if phase == "DONE":
                if token_id in eos_token_ids:
                    continue
                return set(eos_token_ids)

        # Decide allowed next ids from current phase.
        if phase == "PROGRAM":
            allowed: Set[int] = set()
            if self.statement_token_id is not None:
                allowed.add(self.statement_token_id)
            if self.finish_token_id is not None:
                allowed.add(self.finish_token_id)
            return allowed

        if phase == "STMT_LHS":
            return set(ref_ids)

        if phase == "STMT_EQ":
            return {self.eq_token_id} if self.eq_token_id is not None else set()

        if phase == "STMT_AFTER_EQ":
            allowed = set()
            if self.wait_token_id is not None:
                allowed.add(self.wait_token_id)
            if self.exec_token_id is not None:
                allowed.add(self.exec_token_id)
            return allowed

        if phase == "STMT_WAIT_REFS":
            allowed = set(defined_ref_ids)
            if wait_ref_count > 0 and self.exec_token_id is not None:
                allowed.add(self.exec_token_id)
            return allowed

        if phase == "STMT_TOOL":
            return set(stage1_tool_ids)

        if phase == "STMT_END":
            return (
                {self.end_statement_token_id}
                if self.end_statement_token_id is not None
                else set()
            )

        if phase == "FINISH_REFS":
            return set(defined_ref_ids)

        if phase == "FINISH_REFS_NONEMPTY":
            return set(defined_ref_ids) | set(eos_token_ids)

        if phase == "DONE":
            return set(eos_token_ids)

        return set(eos_token_ids)

    def _apply_constraint_mask_for_row(
        self,
        row_logits: torch.Tensor,
        generated_row: torch.Tensor,
        eos_token_ids: Set[int],
    ) -> None:
        allowed_ids = self._compute_ebnf_allowed_next_ids(generated_row, eos_token_ids)
        if not allowed_ids:
            return
        valid_allowed = [idx for idx in allowed_ids if 0 <= idx < row_logits.shape[-1]]
        if not valid_allowed:
            return
        mask = torch.ones_like(row_logits, dtype=torch.bool)
        mask[valid_allowed] = False
        masked = row_logits.masked_fill(mask, float("-inf"))
        if torch.isfinite(masked).any():
            row_logits.copy_(masked)

    def _apply_constraint_mask_to_sequence_logits(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> None:
        if logits.size(1) < 2:
            return
        eos_token_ids: Set[int] = set()
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            eos_token_ids.add(int(eos_token_id))

        batch_size, seq_len = input_ids.shape
        for row_idx in range(batch_size):
            valid_len = seq_len
            if attention_mask is not None:
                valid_len = int(attention_mask[row_idx].sum().item())
            if valid_len < 2:
                continue
            row_input = input_ids[row_idx]
            for pos in range(valid_len - 1):
                prefix = row_input[: pos + 1]
                target_token_id = int(row_input[pos + 1].item())
                original_row_logits = logits[row_idx, pos].clone()
                target_logit = logits[row_idx, pos, target_token_id].clone()
                self._apply_constraint_mask_for_row(
                    logits[row_idx, pos],
                    prefix,
                    eos_token_ids,
                )
                if not torch.isfinite(logits[row_idx, pos]).any():
                    logits[row_idx, pos].copy_(original_row_logits)
                if not torch.isfinite(logits[row_idx, pos, target_token_id]):
                    logits[row_idx, pos, target_token_id] = target_logit

    def _apply_decode_vocab_mask(self, row_logits: torch.Tensor) -> None:
        """Restrict decode vocab to Stage1/Stage2 additions (+ BOS/EOS)."""
        allowed_ids = [
            idx for idx in self.decode_allowed_token_ids if 0 <= idx < row_logits.shape[-1]
        ]
        if not allowed_ids:
            return
        mask = torch.ones_like(row_logits, dtype=torch.bool)
        mask[allowed_ids] = False
        masked = row_logits.masked_fill(mask, float("-inf"))
        if torch.isfinite(masked).any():
            row_logits.copy_(masked)

    def __getattr__(self, name):
        """
        Fallback missing attributes to wrapped LLM for TRL/Transformers compatibility.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            llm = self.__dict__.get("llm")
            if llm is None:
                llm = self._modules.get("llm")
            if llm is not None and hasattr(llm, name):
                return getattr(llm, name)
            raise

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutput:
        """
        Forward pass for training/inference.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for language modeling [batch_size, seq_len]
            **kwargs: Additional arguments (e.g. num_items_in_batch from Trainer)

        Returns:
            Dictionary with 'loss' (if labels provided) and 'logits'
        """
        # Get embeddings through three-layer architecture
        inputs_embeds = self.embedding_layer(input_ids)

        # Forward through LLM transformer layers only (skip internal LM head)
        # For PEFT models: llm (PeftModel) -> llm.model (Qwen2ForCausalLM) -> llm.model.model (Qwen2Model)
        # For base models: llm (Qwen2ForCausalLM) -> llm.model (Qwen2Model)
        if hasattr(self.llm, "model"):
            # PEFT-wrapped: need to go two levels deep
            base_model = self.llm.model
            if hasattr(base_model, "model"):
                transformer = base_model.model  # Qwen2Model
            else:
                transformer = base_model
        else:
            # Non-PEFT: llm itself might be the CausalLM
            transformer = self.llm.model if hasattr(self.llm, "model") else self.llm

        transformer_outputs = transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        # Get last hidden state (first element of transformer output)
        hidden_states = transformer_outputs[0]

        # Compute logits through three-layer LM head
        logits = self.lm_head(hidden_states)
        if (
            self._apply_structure_constraints_for_logprobs
            and labels is None
            and not torch.is_grad_enabled()
        ):
            self._apply_constraint_mask_to_sequence_logits(
                logits=logits,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Validate label range against actual logits vocab size.
            logits_vocab_size = shift_logits.shape[-1]
            valid_mask = shift_labels != -100
            if valid_mask.any():
                valid_labels = shift_labels[valid_mask]
                max_label = valid_labels.max().item()
                if max_label >= logits_vocab_size:
                    # Clamp invalid labels to prevent crash
                    shift_labels = torch.clamp(
                        shift_labels, min=-100, max=logits_vocab_size - 1
                    )

            # Flatten tokens and compute CE in chunks to reduce peak memory.
            flat_logits = shift_logits.view(-1, logits_vocab_size)
            flat_labels = shift_labels.view(-1)
            flat_valid_mask = flat_labels != -100

            if not flat_valid_mask.any():
                # Preserve graph while yielding zero loss for empty-valid batches.
                loss = flat_logits.sum() * 0.0
            else:
                valid_logits = flat_logits[flat_valid_mask]
                valid_labels = flat_labels[flat_valid_mask]
                chunk_size = 4096
                loss_sum = valid_logits.new_zeros(())
                valid_count = int(valid_labels.numel())

                for start in range(0, valid_count, chunk_size):
                    end = min(start + chunk_size, valid_count)
                    loss_sum = loss_sum + F.cross_entropy(
                        valid_logits[start:end],
                        valid_labels[start:end],
                        reduction="sum",
                    )
                loss = loss_sum / max(valid_count, 1)

        return CausalLMOutput(loss=loss, logits=logits)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text using the model.

        Uses the same transformer-only forward path as forward() for consistency.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs
        """
        # Get transformer (same logic as forward())
        # For PEFT models: llm (PeftModel) -> llm.model (Qwen2ForCausalLM) -> llm.model.model (Qwen2Model)
        # For base models: llm (Qwen2ForCausalLM) -> llm.model (Qwen2Model)
        if hasattr(self.llm, "model"):
            # PEFT-wrapped: need to go two levels deep
            base_model = self.llm.model
            if hasattr(base_model, "model"):
                transformer = base_model.model  # Qwen2Model
            else:
                transformer = base_model
        else:
            # Non-PEFT: llm itself might be the CausalLM
            transformer = self.llm.model if hasattr(self.llm, "model") else self.llm

        do_sample = bool(kwargs.get("do_sample", False))
        temperature = float(kwargs.get("temperature", 1.0))
        top_p = float(kwargs.get("top_p", 1.0))
        top_k = int(kwargs.get("top_k", 0))
        repetition_penalty = float(kwargs.get("repetition_penalty", 1.0))
        num_return_sequences = int(kwargs.get("num_return_sequences", 1))
        pad_token_id = kwargs.get("pad_token_id", self.tokenizer.pad_token_id)
        eos_token_id = kwargs.get("eos_token_id", self.tokenizer.eos_token_id)

        if num_return_sequences < 1:
            raise ValueError("num_return_sequences must be >= 1")

        generated = input_ids.clone()
        if attention_mask is None:
            attention_mask = torch.ones_like(generated, device=generated.device)
        else:
            attention_mask = attention_mask.clone()

        if num_return_sequences > 1:
            generated = generated.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(
                num_return_sequences, dim=0
            )

        eos_token_ids = set()
        if eos_token_id is not None:
            if isinstance(eos_token_id, (list, tuple, set)):
                eos_token_ids = {int(x) for x in eos_token_id}
            else:
                eos_token_ids = {int(eos_token_id)}
        finished = torch.zeros(
            generated.shape[0], dtype=torch.bool, device=generated.device
        )

        for step_idx in range(max_new_tokens):
            # Get embeddings
            inputs_embeds = self.embedding_layer(generated)

            # Forward through transformer only (no redundant LM head)
            transformer_outputs = transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            hidden_states = transformer_outputs[0]

            # Get logits through three-layer LM head
            logits = self.lm_head(hidden_states)
            next_token_logits = logits[:, -1, :]
            greedy_next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            if self._restrict_decode_to_new_and_tools:
                for row_idx in range(generated.shape[0]):
                    if finished[row_idx]:
                        continue
                    self._apply_decode_vocab_mask(next_token_logits[row_idx])

            _enforce_first_token_structure(
                next_token_logits=next_token_logits,
                step_idx=step_idx,
                statement_token_id=self.statement_token_id,
                ref0_token_id=self.ref0_token_id,
            )

            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits,
                    generated,
                    repetition_penalty,
                )

            for row_idx in range(generated.shape[0]):
                if finished[row_idx]:
                    continue
                self._apply_constraint_mask_for_row(
                    next_token_logits[row_idx],
                    generated[row_idx],
                    eos_token_ids,
                )

            if do_sample:
                if temperature > 0 and temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    next_token_logits = self._apply_top_k(next_token_logits, top_k)
                if 0.0 < top_p < 1.0:
                    next_token_logits = self._apply_top_p(next_token_logits, top_p)

                if step_idx == 0 and eos_token_ids:
                    if self.ref0_token_id is None:
                        valid_eos_ids = [
                            token_id
                            for token_id in eos_token_ids
                            if 0 <= token_id < next_token_logits.shape[-1]
                        ]
                        if valid_eos_ids:
                            next_token_logits[:, valid_eos_ids] = float("-inf")

                probs = torch.softmax(next_token_logits, dim=-1)
                probs_sum = probs.sum(dim=-1, keepdim=True)
                invalid = torch.isnan(probs_sum) | (probs_sum <= 0)
                probs = probs / probs_sum.clamp_min(1e-8)
                sampled = torch.multinomial(probs, num_samples=1)
                next_token = torch.where(invalid, greedy_next_token, sampled)
            else:
                next_token = greedy_next_token

            if eos_token_ids:
                if pad_token_id is None:
                    pad_token_id = next(iter(eos_token_ids))
                pad_tensor = torch.full_like(next_token, int(pad_token_id))
                next_token = torch.where(finished.unsqueeze(1), pad_tensor, next_token)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)

            # Update attention mask if provided
            next_attn = (~finished).long().unsqueeze(1)
            attention_mask = torch.cat([attention_mask, next_attn], dim=1)

            # Check EOS status
            if eos_token_ids:
                just_finished = torch.zeros_like(finished)
                for token_id in eos_token_ids:
                    just_finished = just_finished | (next_token.squeeze(1) == token_id)
                finished = finished | just_finished
            if finished.all():
                break

        return generated

    def _apply_top_k(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        k = min(top_k, logits.shape[-1])
        if k <= 0:
            return logits
        values, _ = torch.topk(logits, k, dim=-1)
        threshold = values[:, -1].unsqueeze(-1)
        return logits.masked_fill(logits < threshold, float("-inf"))

    def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs > top_p
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False

        filtered_sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits.scatter_(
            dim=-1, index=sorted_indices, src=filtered_sorted_logits
        )
        return filtered_logits

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        if penalty <= 0:
            return logits
        adjusted = logits.clone()
        for row in range(generated.shape[0]):
            token_ids = torch.unique(generated[row])
            row_logits = adjusted[row, token_ids]
            row_logits = torch.where(
                row_logits < 0,
                row_logits * penalty,
                row_logits / penalty,
            )
            adjusted[row, token_ids] = row_logits
        return adjusted

    def get_trainable_parameters(self) -> Dict[str, Any]:
        """
        Get only trainable parameters (Stage 2 embeddings/heads + LoRA).

        Returns:
            Dictionary of trainable parameter lists
        """
        trainable_params = {}

        # Stage 2 embeddings (trainable)
        trainable_params["stage2_embedding"] = list(
            self.stage2_embedding_module.parameters()
        )
        trainable_params["stage2_lm_head"] = list(
            self.stage2_lm_head_module.parameters()
        )

        # LoRA parameters (if using PEFT)
        trainable_params["llm"] = [p for p in self.llm.parameters() if p.requires_grad]

        return trainable_params

    def disable_adapter(self):
        """
        Delegate adapter disabling to the underlying PEFT model when available.

        TRL uses this context manager for reference-policy logprobs when beta > 0.
        """
        if hasattr(self.llm, "disable_adapter"):
            return self.llm.disable_adapter()
        return nullcontext()

    def add_model_tags(self, tags):
        """Delegate TRL model-tag registration when supported by wrapped model."""
        if hasattr(self.llm, "add_model_tags"):
            return self.llm.add_model_tags(tags)
        return None

    def print_trainable_parameters(self):
        """Print statistics about trainable parameters."""
        total_params = 0
        trainable_params = 0

        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"  Trainable: {name:50s} {param.numel():>12,} params")

        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    def state_dict(self, *args, **kwargs):
        """
        Custom state_dict to exclude shared tensors that cause safetensors errors.

        Only include trainable Stage 2 components. Frozen components (base embeddings,
        Stage 1 embeddings/heads) are loaded from external checkpoints.
        """
        # Get the full state dict
        full_state_dict = super().state_dict(*args, **kwargs)

        # Filter to only include trainable components
        filtered_state_dict = {}

        # Include Stage 2 trainable embeddings and LM head
        for key in full_state_dict.keys():
            if key.startswith("stage2_embedding_module.") or key.startswith(
                "stage2_lm_head_module."
            ):
                filtered_state_dict[key] = full_state_dict[key]
            # Include LoRA weights (from llm)
            elif key.startswith("llm.") and "lora" in key.lower():
                filtered_state_dict[key] = full_state_dict[key]

        return filtered_state_dict
