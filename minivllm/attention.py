from __future__ import annotations

import torch
import torch.nn.functional as F

from minivllm.inputs import ModelInput


def write_kv(
    k: torch.Tensor,  # [T, n_kv, d]
    v: torch.Tensor,
    k_cache: torch.Tensor,  # [num_blocks, block_size, n_kv, d]
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,  # int64 [T]
) -> None:
    """Scatter this step's K/V into the paged cache. This is the page table in action."""
    n_kv, d = k.shape[1], k.shape[2]
    k_cache.view(-1, n_kv, d).index_copy_(0, slot_mapping, k)
    v_cache.view(-1, n_kv, d).index_copy_(0, slot_mapping, v)


def prefill_attention(
    q: torch.Tensor,  # [T, n_heads, d]
    k: torch.Tensor,  # [T, n_kv, d]
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,  # int32 [S+1]
) -> torch.Tensor:
    """Causal attention over freshly-computed K/V, one sequence at a time.

    The Python loop is deliberate. The batched alternative is a block-diagonal
    mask over all T tokens, which is O(T^2) memory for a matrix that is almost
    entirely -inf. Real vLLM calls varlen flash attention; on Metal we have
    neither that nor a way to write one, and S is small (bounded by the
    prefill token budget), while each SDPA call is large.
    """
    bounds = cu_seqlens.tolist()
    out = torch.empty_like(q)
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        # SDPA wants [B, H, L, D]; we hold [L, H, D].
        o = F.scaled_dot_product_attention(
            q[s:e].transpose(0, 1).unsqueeze(0),
            k[s:e].transpose(0, 1).unsqueeze(0),
            v[s:e].transpose(0, 1).unsqueeze(0),
            is_causal=True,
            enable_gqa=True,
        )
        out[s:e] = o.squeeze(0).transpose(0, 1)
    return out


def decode_attention(
    q: torch.Tensor,  # [B, n_heads, d]  — exactly one query token per sequence
    k_cache: torch.Tensor,  # [num_blocks, block_size, n_kv, d]
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,  # int32 [B, max_nb]
    context_lens: torch.Tensor,  # int32 [B]
    block_size: int,
) -> torch.Tensor:
    """Attention over KV scattered across physical blocks.

    Blocks belonging to one sequence are non-contiguous in memory, so we walk
    the block table to build slot indices and gather. Materialising that gather
    is exactly what vLLM's fused kernel avoids -- it walks blocks in registers
    and accumulates with an online softmax. Here the gather is a real
    [B, L, n_kv, d] tensor, so decode costs extra bandwidth versus a contiguous
    cache. That cost is the price of not fragmenting, and it buys a much larger
    concurrent batch; see plan section 9.
    """
    B, n_kv, d = block_tables.shape[0], k_cache.shape[2], k_cache.shape[3]
    max_nb = block_tables.shape[1]

    # block id -> the block_size slot ids it owns, flattened to [B, L]
    offsets = torch.arange(block_size, device=q.device, dtype=block_tables.dtype)
    slots = (block_tables[:, :, None] * block_size + offsets).view(B, max_nb * block_size)
    slots = slots.long()

    keys = k_cache.view(-1, n_kv, d)[slots]  # [B, L, n_kv, d]
    vals = v_cache.view(-1, n_kv, d)[slots]

    # Slots past context_len hold either stale KV or another sequence's data.
    # A bool mask (True == attend) is used rather than an additive -inf mask so
    # that a fully-masked row can never produce NaN.
    L = slots.shape[1]
    keep = torch.arange(L, device=q.device)[None, :] < context_lens[:, None].to(q.device)

    o = F.scaled_dot_product_attention(
        q.unsqueeze(2),  # [B, n_heads, 1, d]
        keys.permute(0, 2, 1, 3),  # [B, n_kv, L, d]
        vals.permute(0, 2, 1, 3),
        attn_mask=keep[:, None, None, :],  # [B, 1, 1, L]
        enable_gqa=True,
    )
    return o.squeeze(2)  # [B, n_heads, d]


def attention(
    q: torch.Tensor,  # [T, n_heads, d]
    k: torch.Tensor,  # [T, n_kv, d]
    v: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
    inp: ModelInput,
    block_size: int,
) -> torch.Tensor:
    """Dispatch to the prefill or decode path, writing K/V to the cache first.

    kv_cache is None only in the M1 correctness harness, which runs a single
    full sequence and therefore needs no cache at all.
    """
    if kv_cache is not None:
        write_kv(k, v, kv_cache[0], kv_cache[1], inp.slot_mapping)

    if inp.is_prefill:
        assert inp.cu_seqlens is not None
        return prefill_attention(q, k, v, inp.cu_seqlens)

    assert kv_cache is not None and inp.block_tables is not None
    return decode_attention(
        q, kv_cache[0], kv_cache[1], inp.block_tables, inp.context_lens, block_size
    )
