"""Tiers 2 and 3: the engine end to end, against HuggingFace.

Tier 2 (M5) isolates sampling and the decode loop -- greedy output must be
token-identical to `hf.generate(do_sample=False)`, IDs not text, because text
comparison hides a tokenizer round-trip that can mask a wrong token.

Tier 3 (M6) is the milestone that matters. The same prompts run 8-way
concurrent in a pool deliberately too small to hold them, so the scheduler is
forced to evict and recompute. Recompute preemption is only correct if it is
invisible: if the output moves by a single token, the block tables, the slot
mapping or the num_computed handshake is wrong. Tiers 1 and 2 cannot see any of
those, because a subtly wrong page table still reads plausible KV and still
produces fluent text.
"""

from __future__ import annotations

import pytest
import torch

from minivllm.config import CacheConfig, SchedulerConfig
from minivllm.core.engine import LLMEngine
from minivllm.core.sequence import SamplingParams
from tests.conftest import DEVICE, DTYPE, PROMPTS

BLOCK = 16
MAX_TOKENS = 16

pytestmark = pytest.mark.slow


def blocks_for(n_tokens: int) -> int:
    return -(-(n_tokens + MAX_TOKENS) // BLOCK)


@pytest.fixture(scope="session")
def prompt_ids(tokenizer):
    return [tokenizer(p).input_ids for p in PROMPTS]


@pytest.fixture(scope="session")
def hf_greedy(hf_model, prompt_ids):
    """Reference token IDs from HuggingFace, one prompt at a time."""
    out = []
    for ids in prompt_ids:
        t = torch.tensor([ids], device=DEVICE)
        with torch.inference_mode():
            gen = hf_model.generate(
                t,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
                num_beams=1,
                pad_token_id=hf_model.config.eos_token_id,
            )
        out.append(gen[0, t.shape[1] :].tolist())
    return out


def make_engine(model, cfg, num_blocks: int, max_num_seqs: int = 64) -> LLMEngine:
    return LLMEngine(
        model,
        cfg,
        CacheConfig(block_size=BLOCK, num_blocks=num_blocks),
        SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=4096),
        device=DEVICE,
        dtype=DTYPE,
    )


# -- tier 2 ------------------------------------------------------------------


def test_greedy_matches_hf(model, cfg, prompt_ids, hf_greedy):
    engine = make_engine(model, cfg, num_blocks=512)
    got = engine.generate(prompt_ids, SamplingParams(max_tokens=MAX_TOKENS))

    for i, (ours, theirs) in enumerate(zip(got, hf_greedy)):
        assert ours == theirs, f"prompt {i} diverged:\n  ours   {ours}\n  theirs {theirs}"
    assert engine.stats["preemptions"] == 0, "a roomy pool should never preempt"


def test_single_request_matches_batched(model, cfg, prompt_ids, hf_greedy):
    """One at a time must equal all at once. Catches state leaking between
    sequences that share a batch -- the failure a single-sequence test misses."""
    engine = make_engine(model, cfg, num_blocks=512)
    for ids, expected in zip(prompt_ids, hf_greedy):
        assert engine.generate([ids], SamplingParams(max_tokens=MAX_TOKENS))[0] == expected


# -- tier 3: the M6 gate -----------------------------------------------------


def test_identical_output_under_a_starved_pool(model, cfg, prompt_ids, hf_greedy):
    """Size the pool from real demand rather than a round number: these prompts
    are short, and a 64-block pool would hold all eight comfortably and quietly
    test nothing. Half of what the batch needs guarantees contention, while
    still fitting the largest single sequence so admission can always progress.
    """
    need = [blocks_for(len(ids)) for ids in prompt_ids]
    pool = max(max(need), sum(need) // 2)

    engine = make_engine(model, cfg, num_blocks=pool)
    got = engine.generate(prompt_ids, SamplingParams(max_tokens=MAX_TOKENS))

    assert engine.stats["preemptions"] > 0, (
        f"pool of {pool} blocks (demand {sum(need)}) never forced an eviction; "
        "this test would pass without exercising recompute at all"
    )
    for i, (ours, theirs) in enumerate(zip(got, hf_greedy)):
        assert ours == theirs, (
            f"prompt {i} changed under preemption -- recompute is not output-invariant:"
            f"\n  ours   {ours}\n  theirs {theirs}"
        )
    assert engine.scheduler.allocator.num_free == pool, "leaked blocks"


def test_requests_arriving_mid_flight_do_not_disturb_running_ones(model, cfg, prompt_ids, hf_greedy):
    """Continuous batching's actual claim: a request joining an in-flight batch
    changes throughput, never output."""
    engine = make_engine(model, cfg, num_blocks=512)
    params = SamplingParams(max_tokens=MAX_TOKENS)

    seqs = [engine.add_request(prompt_ids[0], params, arrival=0.0)]
    pending = list(enumerate(prompt_ids[1:], start=1))

    step = 0
    while engine.has_unfinished():
        # Drip a new request into the middle of the running batch.
        if pending and step % 3 == 1:
            i, ids = pending.pop(0)
            seqs.append(engine.add_request(ids, params, arrival=float(i)))
        engine.step()
        step += 1
    for i, ids in pending:
        seqs.append(engine.add_request(ids, params, arrival=float(i)))
    engine.run()

    for i, seq in enumerate(seqs):
        assert seq.output_ids == hf_greedy[i], f"prompt {i} disturbed by a mid-flight arrival"
