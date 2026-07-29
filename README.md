# minivllm

A miniature vLLM for TinyLlama-1.1B: **paged KV cache** and **continuous
batching**, written from scratch on top of PyTorch. Runs on Apple Silicon (MPS).

The model is not the point — it is 130 lines and exists so there is something to
serve. The point is the serving layer, and the claim it exists to demonstrate:

> LLM inference is a memory-bandwidth scheduling problem wearing an AI costume.

Decode reads all 2.2 GB of weights to produce **one** token per sequence. At
batch 64 it reads the same 2.2 GB to produce **64**. Weight traffic amortises
across the batch, so decode throughput scales with batch size until KV memory
runs out. Everything else follows: paging removes the fragmentation that caps
concurrency, and continuous batching keeps you at that cap instead of draining
to the slowest request.

Neither idea makes a single forward pass faster. Both exist to keep the memory
bus busy with useful work. `bench/` measures that, and says so when it fails to.

## Results

Measured on an M3 Pro (18 GPU cores, 36 GB unified, ~150 GB/s peak), fp16/MPS.

**Decode is bandwidth-bound** (`bench_roofline`) — 64× the batch for 2.97× the
latency:

| batch | ms/step | tok/s | GB/s | % peak |
|---:|---:|---:|---:|---:|
| 1 | 23.1 | 43 | 95.3 | 64% |
| 8 | 27.6 | 290 | 80.8 | 54% |
| 32 | 43.6 | 733 | 52.8 | 35% |
| 64 | 68.6 | 932 | 35.1 | 23% |

**Paging raises the concurrency ceiling 13.3×** at identical memory
(`bench_fragmentation`, 4096-block pool, lognormal lengths). Reserving
`max_model_len` per sequence fits 32 sequences at 7.4% slot utilisation; paged
fits 425 at 95.1%. Internal fragmentation is 7.2 slots per sequence — bounded by
`block_size - 1`, and independent of `max_model_len`, which is the whole idea.

**Continuous batching is worth 1.80× throughput and 1.84× better p99**
(`bench_batching`, 64 requests, 16 slots, lognormal outputs). Re-run with
`--sigma 0.05` and the gap closes to 0.97× — the win comes from length skew, and
the benchmark predicts its own disappearance.

### What this implementation pays

Achieved bandwidth *falls* from 95 to 35 GB/s across the roofline sweep, so real
traffic at batch 64 is ~2.7× what a weights-plus-KV model counts. That is the
decode gather. Metal has no fused paged-attention kernel and no Triton, so
attention materialises a `[B, L, n_kv, d]` tensor: every KV byte is read from the
pool, written to the gather, and read back by SDPA — three touches where vLLM's
CUDA kernel does one, in registers, with an online softmax.

Paging still wins overall, but it wins on *capacity* while losing on kernel
efficiency. An isolated paged-vs-contiguous attention microbenchmark would show
paging losing, and that result would be real. Flash-decoding is the fix; it is
out of scope here.

## Layout

| Path | Responsibility |
|---|---|
| `minivllm/config.py` | `ModelConfig`, `CacheConfig`, `SchedulerConfig`, pool arithmetic |
| `minivllm/model/` | loader (safetensors → fused QKV/gate-up) and the Llama forward pass |
| `minivllm/memory/` | block allocator (free list + refcounts), KV pool tensors |
| `minivllm/attention.py` | `write_kv`, `prefill_attention`, `decode_attention` |
| `minivllm/inputs.py` | `build_model_input` — the slot mapping, i.e. the page table |
| `minivllm/core/sequence.py` | `Sequence`, and the `num_computed` invariant everything keys off |
| `minivllm/core/scheduler.py` | waiting/running queues, admission, recompute preemption |
| `minivllm/core/engine.py` | `add_request`, `step`, `run` |

The two load-bearing lines:

```python
# the page table                      (minivllm/core/sequence.py)
slot = block_table[i // block_size] * block_size + (i % block_size)

# what makes batching continuous      (minivllm/core/scheduler.py)
free_finished()   # blocks return in the step a request finishes, not at a batch boundary
```

## Usage

```python
from minivllm.core.engine import LLMEngine
from minivllm.core.sequence import SamplingParams

engine = LLMEngine.from_pretrained()          # MPS/fp16 by default
engine.add_request(prompt_ids, SamplingParams(max_tokens=64))
while engine.has_unfinished():
    for seq in engine.step():
        print(seq.output_ids)
```

## Verification

```bash
uv run python -m minivllm.probe    # MPS op-support gate
uv run pytest tests/ -q            # 57 tests; -m "not slow" skips the weight-loading tiers
uv run python -m bench.bench_roofline
uv run python -m bench.bench_batching
uv run python -m bench.bench_fragmentation --requests 2000
```

Correctness runs **CPU/fp32** throughout; MPS fp16 will not reproduce
HuggingFace fp32 bit-for-bit and chasing that gap teaches nothing about paging.
Performance runs MPS/fp16, separately. Three tiers, each catching what the
previous cannot:

1. **Logits vs HF** — isolates the model (RoPE, GQA, SwiGLU, norm placement).
2. **Greedy vs HF** — isolates sampling and the decode loop. Token IDs, not text.
3. **Concurrent + starved pool** — isolates block tables, slot mapping, eviction
   and recompute.

Tier 3 is the one that matters. Tiers 1 and 2 cannot see a broken page table,
because a wrong slot still reads *some* plausible KV and still produces fluent
text. `tests/test_e2e.py` runs eight prompts 8-way concurrent in a pool sized to
half the batch's demand, and requires output token-identical to
`hf.generate(do_sample=False)` **while the scheduler is actively evicting and
recomputing**. It asserts a preemption actually fired, so it cannot pass by
never exercising the path.

## Scope

**In:** model, loader, block allocator, paged attention, continuous-batching
scheduler with recompute preemption, engine, sampler, correctness harness,
three benchmarks.

**Out** (refcounts and block-table indirection are already in place so these stay
additive): HTTP server, flash-decoding online softmax, prefix sharing /
copy-on-write, swap-to-CPU preemption, chunked prefill.

---

# Security: program-aware scheduling is a timing side channel

A defensive-security result built on the engine above. Two synthetic tenants on
one machine; no external target.

**Every documented LLM-serving timing attack leaks cache _content_** — a prefix
cache hit speeds up TTFT and reveals what was cached (*Early Bird*, InputSnatch).
Every defense protects content too: PrefixWall tags blocks with an owner, SafeKV
isolates sensitive prompts. **Program-aware agentic serving** — Continuum's
program-level FCFS plus KV cache pinned with a TTL across tool-call pauses —
introduces a leak nobody has attacked: **the scheduler's own state**.

A co-resident attacker who can only time its **own** admission latency recovers a
victim's program timeline. When the victim pauses for a tool call it leaves the
running batch, freeing a concurrency slot; the attacker's probe is admitted
exactly then. Admissions cluster into bursts — one per tool call — and each
burst's width is the tool-call duration, which fingerprints the tool.

```
uv run python -m sidechannel.run_experiment
```

Undefended, from admission timestamps alone (four-tool agent, `sidechannel/`):

| | result |
|---|---|
| turns recovered | 4 / 4 (exact) |
| tool-call duration MAE | 0.0 steps |
| tool identification | **100%** |
| cross-tenant shared cache blocks | **0** |

That last row is the point: the cache is fully isolated and the channel is still
wide open, because **the leak is in the scheduler, not the cache**. User-level
cache isolation — the standard mitigation — changes nothing.

The fix has to live in the scheduler too, and it costs:

| defense | tool accuracy | cost |
|---|---|---|
| none | 100% | — |
| **slot reservation** (hold a paused program's slot) | 0% (attacker starved) | idle capacity during every pause |
| **admission cadence** (period > tool durations) | ≤ chance | every tenant's admission latency |

The claim is deliberately narrow: this is a property of program-aware scheduling
(program-level priority + TTL pinning), demonstrated on a faithful reference
implementation, invisible to every content-level defense in the literature.
Single-threaded determinism makes the mechanism exact; replication on production
vLLM is future work.

Run the tests: `uv run pytest tests/test_program.py tests/test_sidechannel.py -q`
(the leak is checked across five seeds). Program-aware mode is a `SchedulerConfig`
flag, off by default — the engine above is byte-for-byte unchanged when it is off.

**Full study:** [`sidechannel/REPORT.md`](sidechannel/REPORT.md) — threat model,
channel capacity in bits with degradation curves, the defense Pareto frontier,
multi-victim breakpoint, related work, and limitations. It hardens the
demonstration into a rigorous study: the channel is shown to survive a real
TinyLlama forward on MPS with the attacker on a separate thread (100% tool
identification, ~23 ms pause-duration error), and a randomised-admission-delay
defense closes it while preserving 74% of benign throughput. Regenerate every
number and figure with `uv run python -m sidechannel.run_all`.
