"""M0 gate: verify MPS supports the ops the paged attention path depends on.

Fail here rather than at M3. Every check has a documented fallback in the plan;
what we cannot afford is discovering a gap after the scheduler is written.
"""

import sys

import torch
import torch.nn.functional as F

from minivllm.config import ModelConfig

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        try:
            note = fn() or ""
            RESULTS.append((name, True, note))
        except Exception as e:  # noqa: BLE001 - probe reports, never raises
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        return fn

    return deco


def main() -> int:
    if not torch.backends.mps.is_available():
        print("MPS unavailable — everything falls back to CPU. Not fatal, just slow.")
        return 1

    dev = torch.device("mps")
    cfg = ModelConfig()
    nkv, d, bs = cfg.num_kv_heads, cfg.head_dim, 16

    @check("index_copy_ on 3-D flat KV view")
    def _():
        # The scatter behind write_kv: k_cache.view(-1, nkv, d).index_copy_(0, slots, k)
        cache = torch.zeros(8 * bs, nkv, d, device=dev, dtype=torch.float16)
        slots = torch.tensor([0, 17, 33, 127], device=dev)
        src = torch.randn(4, nkv, d, device=dev, dtype=torch.float16)
        cache.index_copy_(0, slots, src)
        assert torch.equal(cache[17], src[1]), "scatter landed in the wrong slot"

    @check("advanced-index gather [B,L] -> [B,L,nkv,d]")
    def _():
        flat = torch.randn(8 * bs, nkv, d, device=dev, dtype=torch.float16)
        idx = torch.randint(0, 8 * bs, (4, 32), device=dev)
        out = flat[idx]
        assert out.shape == (4, 32, nkv, d), out.shape

    @check("SDPA enable_gqa=True")
    def _():
        q = torch.randn(2, cfg.num_heads, 1, d, device=dev, dtype=torch.float16)
        k = torch.randn(2, nkv, 64, d, device=dev, dtype=torch.float16)
        v = torch.randn_like(k)
        o = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        assert o.shape == q.shape, o.shape
        # Cross-check against manual repeat_interleave: enable_gqa must not be a no-op lie.
        ref = F.scaled_dot_product_attention(
            q, k.repeat_interleave(8, 1), v.repeat_interleave(8, 1)
        )
        err = (o.float() - ref.float()).abs().max().item()
        assert err < 2e-2, f"enable_gqa disagrees with repeat_interleave: {err}"
        return f"matches repeat_interleave, max err {err:.2e}"

    @check("SDPA is_causal=True")
    def _():
        q = torch.randn(1, cfg.num_heads, 37, d, device=dev, dtype=torch.float16)
        k = torch.randn(1, nkv, 37, d, device=dev, dtype=torch.float16)
        F.scaled_dot_product_attention(q, k, torch.randn_like(k), is_causal=True, enable_gqa=True)

    @check("SDPA with additive float mask")
    def _():
        q = torch.randn(2, cfg.num_heads, 1, d, device=dev, dtype=torch.float16)
        k = torch.randn(2, nkv, 48, d, device=dev, dtype=torch.float16)
        m = torch.zeros(2, 1, 1, 48, device=dev, dtype=torch.float16)
        m[:, :, :, 30:] = float("-inf")
        o = F.scaled_dot_product_attention(q, k, torch.randn_like(k), attn_mask=m, enable_gqa=True)
        assert not o.isnan().any(), "masked SDPA produced NaN"

    @check("fp16 matmul")
    def _():
        a = torch.randn(512, 2048, device=dev, dtype=torch.float16)
        assert not (a @ a.T.contiguous()).isnan().any()

    @check("bf16 matmul")
    def _():
        a = torch.randn(512, 2048, device=dev, dtype=torch.bfloat16)
        assert not (a @ a.T.contiguous()).isnan().any()

    @check("bf16 -> fp16 cast is finite for N(0,1)")
    def _():
        a = torch.randn(4096, device=dev, dtype=torch.bfloat16)
        assert a.to(torch.float16).isfinite().all()

    @check("KV pool allocation (4 GiB)")
    def _():
        nb = int(4 * 2**30 // cfg.bytes_per_block(bs, torch.float16))
        # One layer only — allocating all 22 here would just measure the allocator.
        t = torch.empty(nb, bs, nkv, d, device=dev, dtype=torch.float16)
        gb = t.numel() * 2 / 2**30
        del t
        return f"{nb} blocks, {gb:.2f} GiB/layer for K"

    width = max(len(n) for n, _, _ in RESULTS)
    failed = 0
    for name, ok, note in RESULTS:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name:<{width}}  {note}")
        failed += not ok

    print()
    if failed:
        print(f"{failed} probe(s) failed — see plan §10 for the fallback each one implies.")
    else:
        print("All probes passed. Paged attention can use the fast path on MPS.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
