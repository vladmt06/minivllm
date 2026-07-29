from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@dataclass(frozen=True)
class ModelConfig:
    """Defaults are TinyLlama-1.1B-Chat-v1.0, verified against the HF config."""

    model_id: str = DEFAULT_MODEL
    hidden_size: int = 2048
    num_layers: int = 22
    num_heads: int = 32
    num_kv_heads: int = 4
    intermediate_size: int = 5632
    vocab_size: int = 32000
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    bos_token_id: int = 1
    eos_token_id: int = 2

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_kv_groups(self) -> int:
        """Query heads per KV head. 8 here — the reason decode reads 4 heads, not 32."""
        return self.num_heads // self.num_kv_heads

    def bytes_per_block(self, block_size: int, dtype: torch.dtype) -> int:
        """Bytes one block occupies across K, V and *all* layers.

        A block is the unit the allocator hands out, so it must account for every
        layer at once: you cannot own layer 3's page without owning layer 4's.
        TinyLlama @ block_size 16, fp16 -> 352 KiB.
        """
        itemsize = torch.empty((), dtype=dtype).element_size()
        return 2 * block_size * self.num_kv_heads * self.head_dim * itemsize * self.num_layers

    @classmethod
    def from_hf(cls, path: str | Path, model_id: str = DEFAULT_MODEL) -> ModelConfig:
        c = json.loads((Path(path) / "config.json").read_text())
        if c.get("tie_word_embeddings"):
            raise NotImplementedError("tied embeddings: loader expects a separate lm_head")
        if c.get("attention_bias") or c.get("rope_scaling"):
            raise NotImplementedError("attention bias / rope scaling not supported")
        return cls(
            model_id=model_id,
            hidden_size=c["hidden_size"],
            num_layers=c["num_hidden_layers"],
            num_heads=c["num_attention_heads"],
            num_kv_heads=c["num_key_value_heads"],
            intermediate_size=c["intermediate_size"],
            vocab_size=c["vocab_size"],
            max_position_embeddings=c["max_position_embeddings"],
            rope_theta=c.get("rope_theta", 10000.0),
            rms_norm_eps=c["rms_norm_eps"],
            bos_token_id=c.get("bos_token_id", 1),
            eos_token_id=c.get("eos_token_id", 2),
        )


@dataclass
class CacheConfig:
    block_size: int = 16
    kv_cache_gb: float = 4.0
    num_blocks: int | None = None  # explicit override; tests starve the pool with this
    watermark: float = 0.01  # fraction kept free so admission can't instantly thrash

    def resolve_num_blocks(self, model: ModelConfig, dtype: torch.dtype) -> int:
        if self.num_blocks is not None:
            return self.num_blocks
        return int(self.kv_cache_gb * 2**30 // model.bytes_per_block(self.block_size, dtype))


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 4096
    max_model_len: int = 2048

    # Program-aware serving (Continuum-style). Off by default: the scheduler is
    # request-level FCFS and nothing below runs, so existing behaviour is
    # untouched. On: priority is program-level FCFS and a suspended program's KV
    # cache is pinned for kv_ttl_steps steps across a tool-call pause.
    program_aware: bool = False
    kv_ttl_steps: int = 32

    # Defense knobs, all off by default (they only matter under program_aware).
    # - reserve_slots_on_suspend: a paused program keeps its max_num_seqs slot, so
    #   suspension opens no slot for another tenant to observe. Closes the batch
    #   channel outright, at the cost of the idle slot during every pause.
    # - reserved_blocks_per_tenant: caps blocks per tenant, so one tenant's pinning
    #   cannot move another's admission. Closes the memory channel, costs sharing.
    # - admission_period: admit only on a fixed clock, so the moment a request
    #   enters is decoupled from live pin state. Coarsens the attacker's time
    #   resolution (tunable security/latency trade) rather than closing outright.
    reserve_slots_on_suspend: bool = False
    reserved_blocks_per_tenant: int | None = None
    admission_period: int = 0
    # Randomised admission delay: each request is held 0..noise_admission_steps
    # steps before it may be admitted, so its start time no longer pins the pin
    # state. Blurs the attacker's timing (a softer, tunable cadence) at a latency
    # cost. 0 disables.
    noise_admission_steps: int = 0


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype(device: torch.device) -> torch.dtype:
    """fp32 on CPU so correctness tests can match HF exactly; fp16 on MPS for speed."""
    return torch.float16 if device.type == "mps" else torch.float32
