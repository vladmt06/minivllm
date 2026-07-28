from __future__ import annotations

import torch

from minivllm.config import CacheConfig, ModelConfig


class KVCache:
    """The physical block pool, as tensors.

    One (K, V) pair per layer, each [num_blocks, block_size, num_kv_heads, head_dim].
    A logical block id indexes the same position in every layer at once, which is
    why the allocator hands out one id rather than one per layer.
    """

    def __init__(
        self,
        model: ModelConfig,
        cache: CacheConfig,
        device: torch.device | str,
        dtype: torch.dtype,
    ):
        self.block_size = cache.block_size
        self.num_blocks = cache.resolve_num_blocks(model, dtype)
        self.device = torch.device(device)
        self.dtype = dtype

        shape = (self.num_blocks, cache.block_size, model.num_kv_heads, model.head_dim)
        self.caches: list[tuple[torch.Tensor, torch.Tensor]] = [
            (
                torch.zeros(shape, device=device, dtype=dtype),
                torch.zeros(shape, device=device, dtype=dtype),
            )
            for _ in range(model.num_layers)
        ]

        self.bytes_per_block = model.bytes_per_block(cache.block_size, dtype)
        self.total_bytes = self.num_blocks * self.bytes_per_block

    @property
    def num_slots(self) -> int:
        """Total cacheable tokens across the whole pool."""
        return self.num_blocks * self.block_size

    def __repr__(self) -> str:
        return (
            f"KVCache({self.num_blocks} blocks x {self.block_size} tok = "
            f"{self.num_slots} slots, {self.total_bytes / 2**30:.2f} GiB, "
            f"{self.bytes_per_block / 1024:.0f} KiB/block)"
        )
