from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minivllm.attention import attention
from minivllm.config import ModelConfig
from minivllm.inputs import ModelInput


class RMSNorm(nn.Module):
    def __init__(self, hidden: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accumulate in fp32 regardless of activation dtype -- matches HF, and in
        # fp16 the sum of 2048 squares is genuinely at risk of losing precision.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class RotaryEmbedding(nn.Module):
    """Llama/HF rotary convention: rotate halves, not interleaved pairs."""

    def __init__(self, head_dim: int, max_pos: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_pos, head_dim]
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # positions is per-token, which is exactly what a flattened varlen batch
        # gives us for free -- no per-sequence bookkeeping needed.
        cos = self.cos[positions].unsqueeze(1).to(q.dtype)  # [T, 1, d]
        sin = self.sin[positions].unsqueeze(1).to(q.dtype)
        return (
            q * cos + self._rotate_half(q) * sin,
            k * cos + self._rotate_half(k) * sin,
        )


class LlamaAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, rotary: RotaryEmbedding):
        super().__init__()
        self.cfg = cfg
        self.rotary = rotary
        d = cfg.head_dim
        self.q_size = cfg.num_heads * d
        self.kv_size = cfg.num_kv_heads * d
        # q, k and v fused into one GEMM: three separate matmuls would stream the
        # same activations through the bus three times.
        self.qkv_proj = nn.Linear(cfg.hidden_size, self.q_size + 2 * self.kv_size, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x, inp: ModelInput, kv_cache, block_size: int):
        cfg, d = self.cfg, self.cfg.head_dim
        q, k, v = self.qkv_proj(x).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, cfg.num_heads, d)
        k = k.view(-1, cfg.num_kv_heads, d)
        v = v.view(-1, cfg.num_kv_heads, d)

        q, k = self.rotary(q, k, inp.positions)
        o = attention(q, k, v, kv_cache, inp, block_size)
        return self.o_proj(o.reshape(-1, self.q_size))


class LlamaMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.inter = cfg.intermediate_size
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).split([self.inter, self.inter], dim=-1)
        return self.down_proj(F.silu(gate) * up)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, rotary: RotaryEmbedding):
        super().__init__()
        self.self_attn = LlamaAttention(cfg, rotary)
        self.mlp = LlamaMLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, inp: ModelInput, kv_cache, block_size: int):
        x = x + self.self_attn(self.input_layernorm(x), inp, kv_cache, block_size)
        return x + self.mlp(self.post_attention_layernorm(x))


class LlamaForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        rotary = RotaryEmbedding(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(cfg, rotary) for _ in range(cfg.num_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(
        self,
        inp: ModelInput,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        block_size: int = 16,
        all_logits: bool = False,
    ) -> torch.Tensor:
        x = self.embed_tokens(inp.input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(x, inp, kv_caches[i] if kv_caches is not None else None, block_size)
        x = self.norm(x)
        # Only the last token of each sequence is ever sampled, so during prefill
        # the other T-S rows would be a pure waste of a [hidden, 32000] GEMM.
        if not all_logits:
            x = x[inp.logits_indices]
        return self.lm_head(x)
