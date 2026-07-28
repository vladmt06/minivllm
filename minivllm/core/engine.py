from __future__ import annotations

from pathlib import Path

import torch

from minivllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    default_dtype,
    resolve_device,
)
from minivllm.core.sampler import sample
from minivllm.core.scheduler import Scheduler, SchedulerOutputs
from minivllm.core.sequence import SamplingParams, Sequence, SequenceStatus
from minivllm.inputs import build_model_input
from minivllm.memory.kv_cache import KVCache
from minivllm.model.llama import LlamaForCausalLM
from minivllm.model.loader import load_model


class LLMEngine:
    """Model + paged cache + scheduler, driven one step at a time.

    `step()` is the whole engine: ask the scheduler what to run, flatten it into
    a ModelInput, one forward, sample, advance each sequence, reclaim whatever
    finished. Requests join at any step and leave at any step, which is the
    entire point -- nothing here waits for a batch boundary because there are no
    batch boundaries.
    """

    def __init__(
        self,
        model: LlamaForCausalLM,
        model_config: ModelConfig,
        cache_config: CacheConfig | None = None,
        scheduler_config: SchedulerConfig | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int | None = None,
    ):
        cache_config = cache_config or CacheConfig()
        scheduler_config = scheduler_config or SchedulerConfig()

        self.model = model
        self.cfg = model_config
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config
        self.device = torch.device(device)
        self.block_size = cache_config.block_size
        self.max_model_len = min(
            scheduler_config.max_model_len, model_config.max_position_embeddings
        )

        self.kv_cache = KVCache(model_config, cache_config, self.device, dtype)
        self.scheduler = Scheduler(cache_config, scheduler_config, self.kv_cache.num_blocks)

        self.generator: torch.Generator | None = None
        if seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(seed)

        self.num_steps = 0
        self.num_generated_tokens = 0

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path | None = None,
        device: torch.device | str = "auto",
        dtype: torch.dtype | None = None,
        cache_config: CacheConfig | None = None,
        scheduler_config: SchedulerConfig | None = None,
        seed: int | None = None,
    ) -> LLMEngine:
        dev = resolve_device(device) if isinstance(device, str) else torch.device(device)
        dt = dtype if dtype is not None else default_dtype(dev)
        model, cfg = load_model(path, device=dev, dtype=dt)
        return cls(model, cfg, cache_config, scheduler_config, dev, dt, seed)

    # -- requests ------------------------------------------------------------

    def add_request(
        self,
        prompt_ids: list[int],
        params: SamplingParams | None = None,
        arrival: float | None = None,
    ) -> Sequence:
        params = params or SamplingParams()
        if len(prompt_ids) >= self.max_model_len:
            raise ValueError(
                f"prompt of {len(prompt_ids)} tokens exceeds max_model_len {self.max_model_len}"
            )
        kw = {} if arrival is None else {"arrival": arrival}
        seq = Sequence(prompt_ids=list(prompt_ids), params=params, **kw)
        self.scheduler.add(seq)
        return seq

    def abort(self, seq_id: int) -> bool:
        return self.scheduler.abort(seq_id)

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    # -- the loop ------------------------------------------------------------

    def step(self) -> list[Sequence]:
        """Advance every scheduled sequence by one token. Returns those that
        finished in this step, whose blocks are already back in the pool."""
        out: SchedulerOutputs = self.scheduler.schedule()
        if out.is_empty:
            return []

        inp = build_model_input(out.scheduled, self.block_size, self.device, out.is_prefill)
        with torch.inference_mode():
            logits = self.model(inp, self.kv_cache.caches, self.block_size)

        tokens = sample(logits, out.scheduled, self.generator)
        for seq, token in zip(out.scheduled, tokens):
            # The forward just wrote KV for everything it was given; only now is
            # the newly sampled token uncomputed. This pair of lines is the
            # num_computed invariant that Sequence documents.
            seq.num_computed = seq.num_tokens
            seq.append_token(token)
            if not seq.maybe_finish(self.cfg.eos_token_id):
                if seq.num_tokens >= self.max_model_len:
                    seq.status = SequenceStatus.FINISHED_LENGTH

        self.num_steps += 1
        self.num_generated_tokens += len(out.scheduled)
        return self.scheduler.free_finished()

    def run(self, max_steps: int = 100_000) -> list[Sequence]:
        """Drive to completion. Returns every finished sequence, in finish order."""
        finished: list[Sequence] = []
        for _ in range(max_steps):
            if not self.has_unfinished():
                break
            done = self.step()
            if not done and not self.has_unfinished():
                break
            finished.extend(done)
        else:
            raise RuntimeError(f"did not drain within {max_steps} steps")
        return finished

    def generate(
        self,
        prompts: list[list[int]],
        params: SamplingParams | None = None,
    ) -> list[list[int]]:
        """Convenience: submit every prompt at once, drain, return output token
        IDs in submission order."""
        seqs = [self.add_request(p, params, arrival=float(i)) for i, p in enumerate(prompts)]
        self.run()
        return [s.output_ids for s in seqs]

    def reset(self) -> None:
        """Drop all queues and hand every block back, keeping the loaded weights
        and the allocated cache tensors. Benchmarks sweep a parameter across
        many runs; reloading 2.2 GB of weights between them would dominate.

        Stale KV is left in the pool on purpose -- it is unreachable once the
        block tables are gone, and zeroing it would cost a full pool write that
        the real engine never pays.
        """
        self.scheduler = Scheduler(
            self.cache_config, self.scheduler_config, self.kv_cache.num_blocks
        )
        self.num_steps = 0
        self.num_generated_tokens = 0

    # -- diagnostics ---------------------------------------------------------

    @property
    def stats(self) -> dict[str, int | float]:
        s = self.scheduler
        return {
            "steps": self.num_steps,
            "prefill_steps": s.num_prefill_steps,
            "decode_steps": s.num_decode_steps,
            "preemptions": s.num_preemptions,
            "generated_tokens": self.num_generated_tokens,
            "blocks_total": s.allocator.num_blocks,
            "blocks_free": s.allocator.num_free,
            "kv_cache_gib": self.kv_cache.total_bytes / 2**30,
        }
