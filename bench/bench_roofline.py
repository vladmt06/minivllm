"""Is decode bandwidth-bound? The benchmark the whole thesis rests on.

One decode step reads every weight in the model to produce ONE token per
sequence. At batch 1 that is 2.2 GB of traffic for one token; at batch 64 it is
the same 2.2 GB for sixty-four. If decode is bound by memory bandwidth rather
than arithmetic, per-step latency should stay roughly flat as the batch grows,
and throughput should climb close to linearly.

FALSIFICATION: if latency instead rises linearly with batch size starting from
batch 1, decode is compute-bound here and the argument for paging and
continuous batching collapses -- both exist only to keep a bandwidth-bound
device fed. This script prints that verdict either way.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

from bench._common import (
    PEAK_BANDWIDTH_GB_S,
    add_common_args,
    build_engine,
    header,
    kv_bytes_per_token,
    make_prompt,
    model_bytes,
    resolve,
    rule,
    submit,
    sync,
)


def measure(engine, batch: int, prompt_len: int, steps: int, warmup: int, rng) -> dict:
    engine.reset()
    prompts = [make_prompt(engine, prompt_len, rng) for _ in range(batch)]
    submit(engine, prompts, [steps + warmup + 8] * batch)

    # Prefill everything first; those steps are a different regime and are not
    # what this benchmark is about.
    guard = 0
    while engine.scheduler.waiting or not engine.scheduler.running:
        engine.step()
        guard += 1
        assert guard < 1000, "prefill never completed"
    assert len(engine.scheduler.running) == batch

    latencies = []
    for _ in range(warmup + steps):
        sync(engine.device)
        t0 = time.perf_counter()
        engine.step()
        sync(engine.device)
        latencies.append(time.perf_counter() - t0)
    latencies = latencies[warmup:]

    median = statistics.median(latencies)
    context = prompt_len + warmup + steps // 2  # mean context over the window
    kv_read = batch * context * kv_bytes_per_token(engine.cfg, engine.kv_cache.dtype)
    moved = model_bytes(engine) + kv_read

    return {
        "batch": batch,
        "ms": median * 1e3,
        "tok_s": batch / median,
        "gb_s": moved / median / 1e9,
        "kv_frac": kv_read / moved,
    }


def main() -> None:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--warmup", type=int, default=6)
    args = p.parse_args()

    rng = random.Random(args.seed)
    device, dtype = resolve(args)
    engine = build_engine(args, max_num_seqs=max(args.batches))

    rule("bench_roofline -- decode latency vs batch size")
    header(engine, device, dtype)
    print(f"prompt_len={args.prompt_len} steps={args.steps} warmup={args.warmup}\n")
    print(f"{'batch':>6} {'ms/step':>9} {'tok/s':>9} {'GB/s':>8} {'% peak':>7} {'KV share':>9}")
    rule()

    rows = []
    for b in args.batches:
        r = measure(engine, b, args.prompt_len, args.steps, args.warmup, rng)
        rows.append(r)
        print(
            f"{r['batch']:>6} {r['ms']:>9.1f} {r['tok_s']:>9.1f} {r['gb_s']:>8.1f} "
            f"{100 * r['gb_s'] / PEAK_BANDWIDTH_GB_S:>6.0f}% {100 * r['kv_frac']:>8.1f}%"
        )

    rule("verdict")
    first, last = rows[0], rows[-1]
    latency_growth = last["ms"] / first["ms"]
    throughput_gain = last["tok_s"] / first["tok_s"]
    batch_growth = last["batch"] / first["batch"]

    print(
        f"batch x{batch_growth:.0f}: latency x{latency_growth:.2f}, "
        f"throughput x{throughput_gain:.1f}"
    )
    if latency_growth > 0.75 * batch_growth:
        print(
            "FALSIFIED: latency scales with batch size, so decode is compute-bound\n"
            "on this device. Paging and continuous batching cannot buy throughput\n"
            "that the arithmetic units are not leaving on the table."
        )
    else:
        print(
            f"CONFIRMED: {batch_growth:.0f}x the batch costs only {latency_growth:.2f}x the\n"
            f"time, because the {model_bytes(engine) / 1e9:.1f} GB of weights is read once per\n"
            "step regardless. Decode is bandwidth-bound, so throughput is a\n"
            "question of how many sequences you can keep resident -- which is\n"
            "what the paged cache raises and continuous batching sustains."
        )

    # The GB/s column counts weights plus ONE read of each live KV byte. If that
    # were the whole story it would stay flat. It does not, and the gap is the
    # most honest number this benchmark produces.
    if last["gb_s"] < 0.8 * first["gb_s"]:
        implied = first["gb_s"] / last["gb_s"]
        print(
            f"\nCaveat, and it is the real cost of this implementation: achieved\n"
            f"bandwidth falls {first['gb_s']:.0f} -> {last['gb_s']:.0f} GB/s across the sweep. Since the\n"
            f"counted traffic is nearly constant, actual traffic at batch {last['batch']} is about\n"
            f"{implied:.1f}x what is counted. That is the decode gather: with no fused\n"
            "paged-attention kernel on Metal, every KV byte is read from the pool,\n"
            "written into a materialised [B, L, n_kv, d] tensor, then read back by\n"
            "SDPA -- three touches where vLLM's CUDA kernel does one, in registers,\n"
            "with an online softmax. Paging still wins overall (see\n"
            "bench_fragmentation), but it is winning on capacity while paying here."
        )


if __name__ == "__main__":
    main()
