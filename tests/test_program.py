"""Program-aware serving: multi-turn advancement, TTL pinning, program-FCFS.

The mock forward makes these deterministic and weightless -- the scheduler's
block accounting is the entire contract, exactly as in test_scheduler.py.
"""

from __future__ import annotations

from minivllm.config import CacheConfig, SchedulerConfig
from minivllm.core.program import Program, ProgramRunner, Tool, TurnSpec
from minivllm.core.scheduler import Scheduler
from minivllm.core.sequence import SequenceStatus

BLOCK = 16


def make_runner(num_blocks=64, ttl=64, reserved=None, period=0, seed=0):
    sched = Scheduler(
        CacheConfig(block_size=BLOCK, num_blocks=num_blocks, watermark=0.0),
        SchedulerConfig(
            program_aware=True,
            kv_ttl_steps=ttl,
            reserved_blocks_per_tenant=reserved,
            admission_period=period,
            max_num_seqs=256,
            max_num_batched_tokens=8192,
        ),
        num_blocks,
    )
    return ProgramRunner(sched, seed=seed)


def drain(runner, max_steps=5000):
    n = 0
    while runner.has_work() and n < max_steps:
        runner.step()
        runner.sched.allocator.check_invariants()
        n += 1
    assert n < max_steps, "runner failed to drain"
    return n


# -- multi-turn advancement --------------------------------------------------


def test_program_runs_every_turn_and_frees_all_blocks():
    r = make_runner()
    tool = Tool("search", duration_mean=5)
    prog = Program(turns=[TurnSpec(32, tool), TurnSpec(32, tool), TurnSpec(24)], prompt_len=16)
    r.submit(prog)

    drain(r)

    assert prog.turn_idx == 2, "did not reach the last turn"
    assert len(prog.ground_truth) == 2, "should have logged two tool calls"
    assert r.sched.allocator.num_free == 64, "leaked blocks"
    assert r.sched.num_pins == 2


# -- TTL pinning -------------------------------------------------------------


def test_pin_holds_blocks_during_pause_then_releases_on_resume():
    r = make_runner(ttl=1000)
    tool = Tool("db", duration_mean=8)
    prog = Program(turns=[TurnSpec(48, tool), TurnSpec(16)], prompt_len=16)
    r.submit(prog)

    # Advance until the tool call pins the cache.
    while r.sched.num_pins == 0:
        r.step()
    assert prog.state.name == "TOOL_CALL"
    pinned = r.sched.pinned_blocks
    assert pinned > 0, "a paused program must hold its blocks"
    assert r.sched.suspended and r.sched.suspended[0].status is SequenceStatus.SUSPENDED

    # Free pool is reduced by exactly the pin for the whole pause -- the leak.
    free_during = r.sched.allocator.num_free
    assert free_during == 64 - pinned

    drain(r)
    assert r.sched.allocator.num_free == 64


def test_ttl_expiry_evicts_an_overrunning_tool_and_forces_recompute():
    r = make_runner(ttl=4)
    tool = Tool("slow_tool", duration_mean=40)  # far longer than the TTL
    prog = Program(turns=[TurnSpec(32, tool), TurnSpec(16)], prompt_len=16)
    r.submit(prog)

    drain(r)

    assert r.sched.num_pin_evictions == 1, "the overrunning pin should have been evicted"
    # The program still completes -- eviction costs a recompute, it is not a failure.
    assert prog.turn_idx == 1
    assert r.sched.allocator.num_free == 64


def test_pin_reduces_free_pool_visible_to_other_tenants():
    """The whole premise of the side channel, in one assertion: one program's
    pause changes how much memory another program can see."""
    r = make_runner(num_blocks=16, ttl=1000)
    victim = Program(turns=[TurnSpec(48, Tool("t", 20)), TurnSpec(16)], prompt_len=16, arrival=0)
    r.submit(victim)
    while r.sched.num_pins == 0:
        r.step()

    free_with_pin = r.sched.allocator.num_free
    pinned = r.sched.pinned_blocks
    assert pinned >= 1
    # If the pin were not held, all those blocks would be free.
    assert free_with_pin == 16 - pinned
    drain(r)


# -- program-level FCFS ------------------------------------------------------


def test_program_fcfs_prioritises_an_early_programs_later_turn():
    """An early program's SECOND turn must outrank a late program's FIRST turn --
    that is program-level FCFS, and request-level FCFS would get it wrong because
    the late first turn has the earlier request arrival."""
    r = make_runner(num_blocks=64)
    early = Program(turns=[TurnSpec(16, Tool("t", 2)), TurnSpec(16)], prompt_len=16, arrival=0.0)
    r.submit(early)
    # Run until early pauses for its tool call.
    while early.state.name != "TOOL_CALL":
        r.step()

    # A brand-new program arrives now -- later program arrival, but its request is
    # created after early's pause, so request-arrival order would favour it.
    late = Program(turns=[TurnSpec(16)], prompt_len=16, arrival=100.0)
    r.submit(late)

    # When early resumes, its turn-2 sequence must sort ahead of late's turn.
    early_seq = None
    while early.turn_idx == 0:
        r.step()
    early_seq = early.seq
    late_seq = late.seq
    assert early_seq is not None and late_seq is not None
    assert early_seq.priority < late_seq.priority, "program-FCFS violated"
    drain(r)


def test_standalone_priority_is_unchanged():
    """A sequence with no program keeps the exact old (arrival, arrival, seq_id)
    ordering, so the non-program scheduler path is untouched."""
    from minivllm.core.sequence import Sequence

    s = Sequence(prompt_ids=[1, 2, 3], arrival=5.0)
    assert s.priority == (5.0, 5.0, s.seq_id)
