"""M4 gate: the scheduler, driven by a mock model.

No weights here. The scheduler's contract is entirely about block accounting and
queue discipline, and running a real forward pass to test it would only make the
suite slow enough that nobody runs it. What the mock reproduces exactly is the
`num_computed` handshake the engine performs -- the forward writes this step's
KV, then the sampled token is appended -- because every off-by-one in the
scheduler is expressed in those two lines.
"""

from __future__ import annotations

import random

import pytest

from minivllm.config import CacheConfig, SchedulerConfig
from minivllm.core.scheduler import Scheduler, SequenceTooLong
from minivllm.core.sequence import Sequence, SamplingParams, SequenceStatus

BLOCK = 16
MOCK_TOKEN = 999
NO_EOS = -1


def make_scheduler(num_blocks=8, block_size=BLOCK, max_num_seqs=64, max_batched=4096):
    return Scheduler(
        CacheConfig(block_size=block_size, num_blocks=num_blocks, watermark=0.01),
        SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_batched),
        num_blocks,
    )


def make_seq(prompt_len=20, max_tokens=32, arrival=0.0, seq_id=None):
    kw = {} if seq_id is None else {"seq_id": seq_id}
    return Sequence(
        prompt_ids=[1] * prompt_len,
        params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        arrival=arrival,
        **kw,
    )


def run_step(sched: Scheduler, eos: int = NO_EOS):
    """One engine step, minus the model. Returns the SchedulerOutputs."""
    out = sched.schedule()
    for seq in out.scheduled:
        # The forward just wrote KV for every token it was given.
        seq.num_computed = seq.num_tokens
        seq.append_token(MOCK_TOKEN)
        seq.maybe_finish(eos)
    sched.free_finished()
    sched.allocator.check_invariants()
    return out


def drain(sched: Scheduler, max_steps=10_000) -> int:
    steps = 0
    while sched.has_unfinished() and steps < max_steps:
        if run_step(sched).is_empty:
            break
        steps += 1
    assert steps < max_steps, "scheduler failed to make progress"
    return steps


# -- accounting --------------------------------------------------------------


def test_blocks_return_to_the_pool():
    sched = make_scheduler(num_blocks=64)
    for i in range(6):
        sched.add(make_seq(prompt_len=20, max_tokens=16, arrival=i))

    drain(sched)

    assert not sched.has_unfinished()
    assert sched.allocator.num_free == 64, "leaked blocks"
    sched.allocator.check_invariants()


def test_no_leak_under_a_thousand_randomised_steps():
    """Random arrivals, lengths and pool pressure. The invariant that must hold
    at every single step is block conservation, not any particular schedule."""
    rng = random.Random(0)
    sched = make_scheduler(num_blocks=48)
    live: list[Sequence] = []
    arrival = 0.0

    for _ in range(1000):
        if rng.random() < 0.15:
            arrival += 1.0
            seq = make_seq(
                prompt_len=rng.randint(1, 40),
                max_tokens=rng.randint(1, 40),
                arrival=arrival,
            )
            live.append(seq)
            sched.add(seq)

        run_step(sched)

        held = sum(len(s.block_table) for s in sched.waiting) + sum(
            len(s.block_table) for s in sched.running
        )
        assert held == sched.allocator.num_used, "block table and allocator disagree"

    drain(sched)
    assert sched.allocator.num_free == 48
    for seq in live:
        assert seq.is_finished
        assert seq.block_table == []


def test_finished_sequences_free_blocks_in_the_same_step():
    """The defining property of continuous batching: memory comes back the
    instant a request finishes, not when the batch drains.

    Note max_tokens=2 rather than 1 -- with 1 the sequence finishes during
    prefill, before there is any decode step to observe.
    """
    sched = make_scheduler(num_blocks=64)
    short = make_seq(prompt_len=16, max_tokens=2, arrival=0)
    long = make_seq(prompt_len=16, max_tokens=40, arrival=1)
    sched.add(short)
    sched.add(long)

    run_step(sched)  # prefill both
    assert not short.is_finished
    run_step(sched)  # short hits max_tokens here

    assert short.is_finished and not long.is_finished
    assert short.block_table == []
    # The pool holds exactly what the still-running sequence holds -- no lag.
    assert sched.allocator.num_used == len(long.block_table)


def test_waiting_request_is_admitted_the_step_after_memory_frees():
    """Continuous batching, demonstrated rather than asserted: B does not fit
    while A is running, and is admitted on the very next step once A finishes.
    Static batching would make B wait for the whole batch to drain."""
    sched = make_scheduler(num_blocks=2)  # 32 slots -- room for one of these
    a = make_seq(prompt_len=16, max_tokens=2, arrival=0)
    b = make_seq(prompt_len=32, max_tokens=2, arrival=1)  # needs both blocks
    sched.add(a)
    sched.add(b)

    out = run_step(sched)
    assert out.scheduled == [a], "B should not have fit alongside A"
    assert b.status is SequenceStatus.WAITING

    run_step(sched)  # A finishes here and releases both its blocks
    assert a.is_finished

    out = run_step(sched)
    assert out.is_prefill and out.scheduled == [b], "B waited a step longer than it had to"


# -- discipline --------------------------------------------------------------


def test_steps_are_never_mixed():
    sched = make_scheduler(num_blocks=64)
    for i in range(4):
        sched.add(make_seq(prompt_len=10, max_tokens=8, arrival=i))

    saw_prefill = saw_decode = False
    while sched.has_unfinished():
        out = run_step(sched)
        if out.is_empty:
            break
        # A prefill step admits from waiting; a decode step feeds exactly one
        # token per sequence. Nothing may do both.
        if out.is_prefill:
            saw_prefill = True
        else:
            saw_decode = True
            assert all(s.num_uncomputed == 1 for s in out.scheduled)
    assert saw_prefill and saw_decode


def test_prefill_respects_max_num_seqs():
    sched = make_scheduler(num_blocks=256, max_num_seqs=3)
    for i in range(10):
        sched.add(make_seq(prompt_len=10, max_tokens=20, arrival=i))

    out = sched.schedule()
    assert len(out.scheduled) == 3


def test_fcfs_the_oldest_is_never_the_victim():
    """Under pressure the newest running sequence is evicted, so the oldest
    request keeps its progress and the queue cannot starve from the front."""
    sched = make_scheduler(num_blocks=10)
    seqs = [make_seq(prompt_len=20, max_tokens=40, arrival=i, seq_id=i) for i in range(4)]
    for s in seqs:
        sched.add(s)

    for _ in range(200):
        if run_step(sched).is_empty:
            break

    assert sched.num_preemptions > 0, "pool was not starved; test proves nothing"
    assert seqs[0].num_preemptions == 0, "oldest sequence was preempted"
    assert seqs[0].num_output_tokens >= max(s.num_output_tokens for s in seqs)


def test_preemption_preserves_output_and_requeues_at_the_front():
    sched = make_scheduler(num_blocks=10)
    seqs = [make_seq(prompt_len=20, max_tokens=40, arrival=i, seq_id=i) for i in range(4)]
    for s in seqs:
        sched.add(s)

    victim = None
    for _ in range(200):
        out = run_step(sched)
        if out.preempted:
            victim = out.preempted[0]
            break
        if out.is_empty:
            break

    assert victim is not None, "no preemption occurred"
    assert victim.status is SequenceStatus.WAITING
    assert victim.block_table == []
    assert victim.num_computed == 0, "must re-prefill from scratch"
    assert victim.num_output_tokens > 0, "output was discarded; recompute cannot restore it"
    assert sched.waiting[0] is victim, "preempted sequence lost its place in line"


def test_preempted_sequence_still_finishes_with_the_right_length():
    """Preemption trades compute for memory. It must not change what comes out --
    here, only the token count is checkable; test_e2e checks the tokens."""
    sched = make_scheduler(num_blocks=10)
    seqs = [make_seq(prompt_len=20, max_tokens=25, arrival=i, seq_id=i) for i in range(4)]
    for s in seqs:
        sched.add(s)

    drain(sched)

    assert sched.num_preemptions > 0
    for s in seqs:
        assert s.num_output_tokens == 25, f"{s!r} produced the wrong number of tokens"
    assert sched.allocator.num_free == 10


# -- failure modes -----------------------------------------------------------


def test_request_larger_than_the_pool_fails_loudly():
    """Silently queueing it forever would look like a hang, not a bug."""
    sched = make_scheduler(num_blocks=4)  # 64 slots
    sched.add(make_seq(prompt_len=200, max_tokens=1))
    with pytest.raises(SequenceTooLong, match="needs"):
        sched.schedule()


def test_abort_releases_blocks():
    sched = make_scheduler(num_blocks=64)
    seq = make_seq(prompt_len=20, max_tokens=40)
    sched.add(seq)
    run_step(sched)
    assert seq.block_table

    assert sched.abort(seq.seq_id)
    assert seq.status is SequenceStatus.FINISHED_ABORTED
    assert seq.block_table == []
    assert sched.allocator.num_free == 64
    assert not sched.has_unfinished()
    assert not sched.abort(seq.seq_id), "aborting twice should report nothing to do"


def test_empty_scheduler_is_idle_not_busy():
    sched = make_scheduler()
    assert not sched.has_unfinished()
    assert sched.schedule().is_empty
