"""Static vs continuous batching under realistic, skewed output lengths.

Static batching fixes a batch and runs it to completion. As its members finish
at different times the batch shrinks, and the freed capacity sits idle until the
slowest sequence in the chunk is done -- so a single long reply holds an
otherwise-empty batch open. Continuous batching admits waiting work into those
slots on the very next step.

bench_roofline establishes that throughput follows batch size. This one measures
what that costs when you let the batch drain, which is the only reason
continuous batching exists.

Output lengths are lognormal on purpose. Under uniform lengths every sequence in
a chunk finishes at nearly the same moment, there is almost nothing to reclaim,
and static batching looks fine -- which is exactly the measurement error that
makes people believe it is.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass

from bench._common import (
    add_common_args,
    build_engine,
    drain_timed,
    header,
    lognormal_lengths,
    make_prompt,
    percentile,
    resolve,
    rule,
    submit,
    sync,
)


@dataclass
class Result:
    label: str
    wall: float
    tokens: int
    latencies: list[float]

    @property
    def throughput(self) -> float:
        return self.tokens / self.wall


def run_continuous(engine, prompts, lengths, max_seqs) -> Result:
    engine.reset()
    engine.scheduler.max_num_seqs = max_seqs

    sync(engine.device)
    t0 = time.perf_counter()
    seqs = submit(engine, prompts, lengths)
    done = drain_timed(engine, t0)
    sync(engine.device)
    wall = time.perf_counter() - t0

    return Result("continuous", wall, sum(s.num_output_tokens for s in seqs), [c.finished_at - t0 for c in done])


def run_static(engine, prompts, lengths, max_seqs) -> Result:
    """Same engine, but requests are released in fixed chunks and each chunk must
    drain before the next is admitted. That single restriction is the whole
    difference between the two policies."""
    engine.reset()
    engine.scheduler.max_num_seqs = max_seqs

    sync(engine.device)
    t0 = time.perf_counter()
    latencies, tokens = [], 0
    for i in range(0, len(prompts), max_seqs):
        chunk_p, chunk_l = prompts[i : i + max_seqs], lengths[i : i + max_seqs]
        seqs = submit(engine, chunk_p, chunk_l)
        latencies += [c.finished_at - t0 for c in drain_timed(engine, t0)]
        tokens += sum(s.num_output_tokens for s in seqs)
    sync(engine.device)
    return Result("static", time.perf_counter() - t0, tokens, latencies)


def main() -> None:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--max-seqs", type=int, default=16)
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--median-len", type=int, default=48)
    p.add_argument("--sigma", type=float, default=0.9, help="output-length skew")
    args = p.parse_args()

    rng = random.Random(args.seed)
    device, dtype = resolve(args)
    engine = build_engine(args, max_num_seqs=args.max_seqs)

    lengths = lognormal_lengths(args.requests, args.median_len, args.sigma, args.seed)
    prompts = [make_prompt(engine, args.prompt_len, rng) for _ in range(args.requests)]

    rule("bench_batching -- static vs continuous")
    header(engine, device, dtype)
    print(
        f"{args.requests} requests, batch slots={args.max_seqs}, prompt={args.prompt_len} tok\n"
        f"output lengths: median {sorted(lengths)[len(lengths) // 2]}, "
        f"min {min(lengths)}, max {max(lengths)}, total {sum(lengths)}\n"
    )

    results = [
        run_static(engine, prompts, lengths, args.max_seqs),
        run_continuous(engine, prompts, lengths, args.max_seqs),
    ]

    print(f"{'policy':>11} {'wall s':>8} {'tok/s':>9} {'p50 s':>8} {'p99 s':>8}")
    rule()
    for r in results:
        print(
            f"{r.label:>11} {r.wall:>8.2f} {r.throughput:>9.1f} "
            f"{percentile(r.latencies, 50):>8.2f} {percentile(r.latencies, 99):>8.2f}"
        )

    static, cont = results
    rule("verdict")
    speedup = cont.throughput / static.throughput
    p99 = percentile(static.latencies, 99) / max(percentile(cont.latencies, 99), 1e-9)
    print(f"continuous batching: throughput x{speedup:.2f}, p99 latency x{p99:.2f} better")
    if speedup > 1.05:
        print(
            "The gap is idle batch slots. Static holds capacity open until its\n"
            "slowest member finishes; continuous refills it the next step. The\n"
            "skew in output lengths sets how wide the gap gets -- rerun with\n"
            "--sigma 0.1 and it should mostly close."
        )
    else:
        print(
            "No meaningful win. Either the length distribution is too flat to\n"
            "leave slots idle, or the request count is too small for refill to\n"
            "matter. Report this rather than tuning until it agrees."
        )


if __name__ == "__main__":
    main()
