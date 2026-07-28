"""Tier 1: does our model compute the same function as HuggingFace's?

No paging is involved here. This isolates RoPE convention, GQA broadcast, SwiGLU
ordering, norm placement and the fused-projection split -- so that when tier 3
fails later, we already know the model is not the culprit.
"""

from __future__ import annotations

import pytest
import torch

from minivllm.inputs import ModelInput
from tests.conftest import DEVICE, PROMPTS


def make_input(ids: torch.Tensor) -> ModelInput:
    n = ids.shape[0]
    return ModelInput(
        input_ids=ids,
        positions=torch.arange(n, device=ids.device),
        slot_mapping=torch.zeros(n, dtype=torch.long, device=ids.device),  # unused, no cache
        is_prefill=True,
        logits_indices=torch.tensor([n - 1], device=ids.device),
        cu_seqlens=torch.tensor([0, n], dtype=torch.int32, device=ids.device),
    )


@pytest.mark.parametrize("prompt", PROMPTS[:4])
def test_logits_match_hf(prompt, model, hf_model, tokenizer):
    ids = tokenizer(prompt, return_tensors="pt").input_ids[0].to(DEVICE)

    with torch.inference_mode():
        ours = model(make_input(ids), kv_caches=None, all_logits=True)
        theirs = hf_model(ids.unsqueeze(0)).logits[0]

    assert ours.shape == theirs.shape, (ours.shape, theirs.shape)
    err = (ours - theirs).abs().max().item()
    assert err < 1e-3, f"max abs logit diff {err:.3e}"

    # Tolerances can hide a wrong argmax; the ranking is what actually decodes.
    assert torch.equal(ours.argmax(-1), theirs.argmax(-1)), "argmax diverges"


def test_fused_projections_match_unfused(model, cfg):
    """The qkv/gate_up fusion must be a pure reshape of the checkpoint, not a re-order."""
    layer = model.layers[0]
    q, k, v = layer.self_attn.qkv_proj.weight.split(
        [cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, cfg.num_kv_heads * cfg.head_dim]
    )
    assert q.shape == (cfg.hidden_size, cfg.hidden_size)
    assert k.shape == v.shape == (cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size)

    gate, up = layer.mlp.gate_up_proj.weight.split([cfg.intermediate_size] * 2)
    assert gate.shape == up.shape == (cfg.intermediate_size, cfg.hidden_size)
