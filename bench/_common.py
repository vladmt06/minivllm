from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass

import torch

from minivllm.config import CacheConfig, ModelConfig, SchedulerConfig, resolve_device
from minivllm.core.engine import LLMEngine
from minivllm.core.sequence import SamplingParams, Sequence

# M3 Pro, 150 GB/s LPDDR5 across the package. Peak, not achievable -- treat the
# fraction of it we reach as the interesting number, not the shortfall.
PEAK_BANDWIDTH_GB_S = 150.0


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--device", default="auto", help="auto | mps | cpu")
    p.add_argument("--dtype", default=None, choices=[None, "float16", "float32", "bfloat16"])
    p.add_argument("--blocks", type=int, default=4096, help="KV pool size in blocks")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    return p


def resolve(args) -> tuple[torch.device, torch.dtype]:
    device = resolve_device(args.device)
    if args.dtype:
        dtype = getattr(torch, args.dtype)
    else:
        # Benchmarks are a performance claim, so they run in the dtype you would
        # actually serve in. Correctness lives in the tests, at CPU/fp32.
        dtype = torch.float16 if device.type == "mps" else torch.float32
    return device, dtype


def sync(device: torch.device) -> None:
    """MPS dispatch is asynchronous; without this every measurement below would
    be timing the enqueue, not the work."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def build_engine(args, blocks: int | None = None, max_num_seqs: int = 256) -> LLMEngine:
    device, dtype = resolve(args)
    return LLMEngine.from_pretrained(
        device=device,
        dtype=dtype,
        cache_config=CacheConfig(block_size=args.block_size, num_blocks=blocks or args.blocks),
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=8192, max_model_len=2048
        ),
    )


def model_bytes(engine: LLMEngine) -> int:
    """Weight bytes touched by one forward pass. Counted, not assumed."""
    return sum(p.numel() * p.element_size() for p in engine.model.parameters())


def kv_bytes_per_token(cfg: ModelConfig, dtype: torch.dtype) -> int:
    itemsize = torch.empty((), dtype=dtype).element_size()
    return 2 * cfg.num_kv_heads * cfg.head_dim * itemsize * cfg.num_layers


def lognormal_lengths(n: int, median: int, sigma: float, seed: int, lo=8, hi=512) -> list[int]:
    """Real traffic is heavy-tailed: most replies are short, a few are very long.
    That skew is precisely what static batching handles badly, so a uniform
    length distribution would quietly stack the comparison in its favour."""
    rng = random.Random(seed)
    import math

    return [max(lo, min(hi, int(median * math.exp(rng.gauss(0, sigma))))) for _ in range(n)]


@dataclass
class Completion:
    seq: Sequence
    finished_at: float


def drain_timed(engine: LLMEngine, t0: float | None = None) -> list[Completion]:
    """Run to completion, recording when each request actually finished, so we
    can report latency percentiles rather than only aggregate throughput."""
    t0 = time.perf_counter() if t0 is None else t0
    done: list[Completion] = []
    while engine.has_unfinished():
        finished = engine.step()
        if finished:
            sync(engine.device)
            now = time.perf_counter()
            done.extend(Completion(s, now) for s in finished)
    return done


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q / 100.0 * (len(ordered) - 1))))
    return ordered[idx]


def make_prompt(engine: LLMEngine, n_tokens: int, rng: random.Random) -> list[int]:
    """Random in-vocabulary tokens. Content is irrelevant to a bandwidth
    measurement -- length is the only thing the memory system sees."""
    lo, hi = 1000, min(engine.cfg.vocab_size, 30000)
    return [engine.cfg.bos_token_id] + [rng.randrange(lo, hi) for _ in range(n_tokens - 1)]


def submit(engine: LLMEngine, prompts: list[list[int]], lengths: list[int]) -> list[Sequence]:
    return [
        engine.add_request(p, SamplingParams(max_tokens=n, ignore_eos=True), arrival=float(i))
        for i, (p, n) in enumerate(zip(prompts, lengths))
    ]


def rule(title: str = "", width: int = 78) -> None:
    if title:
        print(f"\n{title}\n{'-' * width}")
    else:
        print("-" * width)


def header(engine: LLMEngine, device: torch.device, dtype: torch.dtype) -> None:
    gb = model_bytes(engine) / 1e9
    print(f"device={device.type} dtype={str(dtype).split('.')[-1]}  weights={gb:.2f} GB")
    print(f"{engine.kv_cache}")
