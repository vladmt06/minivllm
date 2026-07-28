from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from minivllm.config import DEFAULT_MODEL, ModelConfig
from minivllm.model.llama import LlamaForCausalLM


def model_path(model_id: str = DEFAULT_MODEL) -> Path:
    return Path(
        snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "tokenizer.model"])
    )


def _fuse(sd: dict[str, torch.Tensor], prefix: str, names: list[str]) -> torch.Tensor:
    return torch.cat([sd[f"{prefix}.{n}.weight"] for n in names], dim=0)


def convert_state_dict(hf: dict[str, torch.Tensor], cfg: ModelConfig) -> dict[str, torch.Tensor]:
    """HF checkpoint names -> ours, fusing the projections we run as one GEMM."""
    out: dict[str, torch.Tensor] = {
        "embed_tokens.weight": hf["model.embed_tokens.weight"],
        "norm.weight": hf["model.norm.weight"],
        "lm_head.weight": hf["lm_head.weight"],
    }
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}"
        out[f"layers.{i}.self_attn.qkv_proj.weight"] = _fuse(
            hf, f"{p}.self_attn", ["q_proj", "k_proj", "v_proj"]
        )
        out[f"layers.{i}.self_attn.o_proj.weight"] = hf[f"{p}.self_attn.o_proj.weight"]
        out[f"layers.{i}.mlp.gate_up_proj.weight"] = _fuse(
            hf, f"{p}.mlp", ["gate_proj", "up_proj"]
        )
        out[f"layers.{i}.mlp.down_proj.weight"] = hf[f"{p}.mlp.down_proj.weight"]
        out[f"layers.{i}.input_layernorm.weight"] = hf[f"{p}.input_layernorm.weight"]
        out[f"layers.{i}.post_attention_layernorm.weight"] = hf[
            f"{p}.post_attention_layernorm.weight"
        ]
    return out


def load_model(
    path: str | Path | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    model_id: str = DEFAULT_MODEL,
) -> tuple[LlamaForCausalLM, ModelConfig]:
    path = Path(path) if path is not None else model_path(model_id)
    cfg = ModelConfig.from_hf(path, model_id)

    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors under {path}")
    hf: dict[str, torch.Tensor] = {}
    for s in shards:
        hf.update(load_file(str(s)))

    sd = convert_state_dict(hf, cfg)
    for k, t in sd.items():
        t = t.to(dtype)
        # Checkpoint is bf16, which has fp32's exponent range; fp16 does not.
        # An overflow here would be silent and would only surface as garbage text.
        if not torch.isfinite(t).all():
            raise ValueError(f"{k}: non-finite after cast to {dtype}")
        sd[k] = t

    with torch.device("meta"):
        model = LlamaForCausalLM(cfg)
    model.to_empty(device=device)
    # Cast before loading: load_state_dict copies into the existing parameter, so
    # loading first would materialise the whole model in fp32 and only then halve it.
    model.to(dtype=dtype)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    # RoPE cos/sin are computed, not loaded; anything else missing is a real bug.
    missing = [m for m in missing if not m.endswith((".cos", ".sin"))]
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing} unexpected={unexpected}")

    # to_empty leaves the non-persistent RoPE buffers uninitialised. Rebuild them
    # after the dtype cast and keep them fp32 -- HF does the same and casts at use;
    # fp16 cos/sin measurably degrades long-position accuracy. The instance is
    # shared by every layer, so assigning once is enough.
    rotary = model.layers[0].self_attn.rotary
    ref = type(rotary)(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)
    rotary.cos = ref.cos.to(device)
    rotary.sin = ref.sin.to(device)

    return model.eval(), cfg
