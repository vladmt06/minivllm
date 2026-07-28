from __future__ import annotations

from collections import deque


class OutOfBlocks(RuntimeError):
    """Raised on over-allocation. Callers must consult can_allocate first --
    the scheduler treats exhaustion as a signal to preempt, not as an error."""


class BlockAllocator:
    """Physical block pool: a free list plus a refcount per block.

    Refcounts are not exercised by the core scheduler -- nothing shares a block
    yet. They exist because prefix sharing is copy-on-write over exactly this
    structure, and retrofitting refcounts into an allocator that has already
    been threaded through the scheduler is far more invasive than carrying
    them from the start.
    """

    def __init__(self, num_blocks: int):
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        self.num_blocks = num_blocks
        self._free: deque[int] = deque(range(num_blocks))
        self._ref = [0] * num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    def can_allocate(self, n: int, watermark: int = 0) -> bool:
        return self.num_free - n >= watermark

    def allocate(self, n: int) -> list[int]:
        if n > self.num_free:
            raise OutOfBlocks(f"requested {n}, only {self.num_free} free")
        out = []
        for _ in range(n):
            b = self._free.popleft()
            self._ref[b] = 1
            out.append(b)
        return out

    def free(self, blocks: list[int]) -> None:
        for b in blocks:
            if self._ref[b] == 0:
                raise RuntimeError(f"double free of block {b}")
            self._ref[b] -= 1
            if self._ref[b] == 0:
                self._free.append(b)

    def incref(self, block: int) -> None:
        """Share a block with another sequence (prefix reuse)."""
        if self._ref[block] == 0:
            raise RuntimeError(f"cannot share unallocated block {block}")
        self._ref[block] += 1

    def refcount(self, block: int) -> int:
        return self._ref[block]

    def is_shared(self, block: int) -> bool:
        return self._ref[block] > 1

    def check_invariants(self) -> None:
        """free list <=> zero refcount, with no duplicates. Used by tests and asserts."""
        free = set(self._free)
        if len(free) != len(self._free):
            raise AssertionError("duplicate block in free list")
        for b in range(self.num_blocks):
            if (self._ref[b] == 0) != (b in free):
                raise AssertionError(f"block {b}: ref={self._ref[b]} free={b in free}")
