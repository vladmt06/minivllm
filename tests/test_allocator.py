from __future__ import annotations

import random

import pytest
import torch

from minivllm.config import CacheConfig, ModelConfig
from minivllm.memory.allocator import BlockAllocator, OutOfBlocks
from minivllm.memory.kv_cache import KVCache


def test_conservation():
    a = BlockAllocator(16)
    assert a.num_free == 16
    blocks = a.allocate(10)
    assert a.num_free == 6 and a.num_used == 10
    assert len(set(blocks)) == 10, "allocated the same block twice"
    a.free(blocks)
    assert a.num_free == 16
    a.check_invariants()


def test_exhaustion_is_signalled_not_silent():
    a = BlockAllocator(4)
    a.allocate(4)
    assert not a.can_allocate(1)
    with pytest.raises(OutOfBlocks):
        a.allocate(1)
    # The failed allocation must not have consumed anything.
    assert a.num_free == 0
    a.check_invariants()


def test_watermark_reserves_headroom():
    a = BlockAllocator(100)
    a.allocate(98)
    assert a.can_allocate(2, watermark=0)
    assert not a.can_allocate(2, watermark=1)


def test_double_free_raises():
    a = BlockAllocator(8)
    b = a.allocate(2)
    a.free(b)
    with pytest.raises(RuntimeError, match="double free"):
        a.free(b)


def test_sharing_defers_reclaim():
    """A shared block must survive its first owner -- the basis of copy-on-write."""
    a = BlockAllocator(8)
    (blk,) = a.allocate(1)
    a.incref(blk)
    assert a.refcount(blk) == 2 and a.is_shared(blk)

    a.free([blk])
    assert a.num_free == 7, "block reclaimed while still referenced"
    assert not a.is_shared(blk)

    a.free([blk])
    assert a.num_free == 8
    a.check_invariants()


def test_cannot_share_unallocated():
    a = BlockAllocator(4)
    with pytest.raises(RuntimeError, match="unallocated"):
        a.incref(0)


def test_fuzz_never_leaks_or_aliases():
    """Random alloc/free must never leak a block or hand the same one out twice."""
    rng = random.Random(0)
    a = BlockAllocator(64)
    held: list[list[int]] = []

    for _ in range(2000):
        if held and (rng.random() < 0.5 or a.num_free == 0):
            a.free(held.pop(rng.randrange(len(held))))
        else:
            n = rng.randint(1, 8)
            if a.can_allocate(n):
                held.append(a.allocate(n))

        live = [b for group in held for b in group]
        assert len(live) == len(set(live)), "same block handed to two owners"
        assert a.num_used == len(live)
        a.check_invariants()

    for g in held:
        a.free(g)
    assert a.num_free == 64


def test_pool_sizing_matches_hand_arithmetic():
    m = ModelConfig()
    # 2 (K,V) * 16 tok * 4 kv heads * 64 dim * 2 B * 22 layers
    assert m.bytes_per_block(16, torch.float16) == 2 * 16 * 4 * 64 * 2 * 22 == 360_448

    c = CacheConfig(block_size=16, kv_cache_gb=4.0)
    assert c.resolve_num_blocks(m, torch.float16) == 11_915
    assert CacheConfig(num_blocks=64).resolve_num_blocks(m, torch.float16) == 64


def test_cache_tensor_shapes():
    m = ModelConfig()
    kv = KVCache(m, CacheConfig(num_blocks=8), device="cpu", dtype=torch.float16)
    assert len(kv.caches) == m.num_layers
    for k, v in kv.caches:
        assert k.shape == v.shape == (8, 16, m.num_kv_heads, m.head_dim)
    assert kv.num_slots == 128
    # Sanity: the per-layer tensors really do sum to the advertised block size.
    per_block = sum(k[0].numel() + v[0].numel() for k, v in kv.caches) * 2
    assert per_block == kv.bytes_per_block
