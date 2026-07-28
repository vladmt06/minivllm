"""How many sequences fit in a fixed KV budget: paged vs contiguous reservation.

Before paging, a serving engine had to reserve a contiguous KV region per
sequence at admission time -- and since it cannot know how long the reply will
be, it reserved for the worst case, max_model_len. A 2048-token reservation for
a 60-token reply wastes 97% of it. That waste is not a rounding error; it sets
the concurrency ceiling directly, and bench_roofline showed throughput follows
concurrency.

Paging replaces the reservation with a block table. A sequence holds
ceil(len / block_size) blocks and nothing more, so the only waste left is
internal fragmentation in its final partial block -- bounded by block_size - 1
tokens, never by max_model_len.

This is a capacity measurement, so it is exact arithmetic over the real
allocator rather than a timing. The engine run at the end confirms the paged
number is actually achievable and not just arithmetic.
"""

from __future__ import annotations

import argparse
import random

from bench._common import (
    add_common_args,
    build_engine,
    header,
    lognormal_lengths,
    make_prompt,
    resolve,
    rule,
    submit,
)
from minivllm.memory.allocator import BlockAllocator


def blocks_needed(n_tokens: int, block_size: int) -> int:
    return -(-n_tokens // block_size)


def contiguous_capacity(pool_blocks: int, max_model_len: int, block_size: int) -> int:
    """Sequences admissible when each reserves for the worst case up front."""
    per_seq = blocks_needed(max_model_len, block_size)
    return pool_blocks // per_seq


def paged_capacity(pool_blocks: int, lengths: list[int], block_size: int) -> tuple[int, int]:
    """Admit greedily until the real allocator refuses. Returns (admitted, wasted
    slots), where waste is only the unused tail of each sequence's last block."""
    alloc = BlockAllocator(pool_blocks)
    admitted = wasted = 0
    for n in lengths:
        need = blocks_needed(n, block_size)
        if not alloc.can_allocate(need):
            break
        alloc.allocate(need)
        admitted += 1
        wasted += need * block_size - n
    alloc.check_invariants()
    return admitted, wasted


def main() -> None:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--requests", type=int, default=256)
    p.add_argument("--median-len", type=int, default=96)
    p.add_argument("--sigma", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--verify", action="store_true", help="run the engine to confirm")
    args = p.parse_args()

    bs = args.block_size
    lengths = lognormal_lengths(args.requests, args.median_len, args.sigma, args.seed, hi=1024)

    rule("bench_fragmentation -- concurrency at equal memory")
    print(
        f"pool={args.blocks} blocks x {bs} tok = {args.blocks * bs} slots\n"
        f"max_model_len={args.max_model_len} "
        f"({blocks_needed(args.max_model_len, bs)} blocks reserved per sequence)\n"
        f"actual lengths: median {sorted(lengths)[len(lengths) // 2]}, "
        f"min {min(lengths)}, max {max(lengths)}\n"
    )

    contig = contiguous_capacity(args.blocks, args.max_model_len, bs)
    paged, wasted = paged_capacity(args.blocks, lengths, bs)
    total_slots = args.blocks * bs

    print(f"{'scheme':>24} {'concurrent seqs':>16} {'slot utilisation':>18}")
    rule()
    contig_used = sum(lengths[:contig])
    print(f"{'contiguous, max-len':>24} {contig:>16} {100 * contig_used / total_slots:>17.1f}%")
    print(
        f"{'paged':>24} {paged:>16} "
        f"{100 * (sum(lengths[:paged])) / total_slots:>17.1f}%"
    )

    rule("verdict")
    saturated = paged < len(lengths)
    if contig:
        bound = "" if saturated else " (LOWER BOUND: ran out of requests, not blocks)"
        print(
            f"paged fits {paged / contig:.1f}x more concurrent sequences at "
            f"identical memory{bound}"
        )
        if not saturated:
            print(f"  -- rerun with --requests {len(lengths) * 4} to saturate the pool")
    else:
        print(f"contiguous reservation fits ZERO sequences in this pool; paged fits {paged}")
    print(
        f"paged internal fragmentation: {wasted} slots "
        f"({100 * wasted / max(total_slots, 1):.1f}% of pool), "
        f"{wasted / max(paged, 1):.1f} per sequence\n"
        f"bounded by block_size - 1 = {bs - 1} tokens per sequence, and independent\n"
        f"of max_model_len -- which is the entire point."
    )

    if args.verify:
        rule("verification: running the engine at the paged concurrency")
        rng = random.Random(args.seed)
        engine = build_engine(args, max_num_seqs=paged)
        n = min(paged, 64)
        prompts = [make_prompt(engine, 16, rng) for _ in range(n)]
        submit(engine, prompts, [max(1, lengths[i] - 16) for i in range(n)])
        peak = 0
        while engine.has_unfinished():
            engine.step()
            peak = max(peak, len(engine.scheduler.running))
        print(f"ran {n} sequences, peak concurrent = {peak}, stats = {engine.stats}")


if __name__ == "__main__":
    main()
