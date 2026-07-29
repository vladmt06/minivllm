from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from random import Random

from minivllm.core.scheduler import Scheduler
from minivllm.core.sequence import SamplingParams, Sequence

# The mock forward produces this token id. Content is irrelevant to a scheduler
# side channel -- only lengths and timing move the pool, and those are exact.
_MOCK_TOKEN = 7
_prog_counter = itertools.count()


@dataclass(frozen=True)
class Tool:
    """An external tool a program calls between turns. `duration` is how many
    engine steps the call blocks the program -- and, because the KV cache is
    pinned for exactly that long, it is what a co-tenant reads off the pool. A
    tool is fingerprinted by its duration distribution."""

    name: str
    duration_mean: int
    duration_jitter: int = 0

    def sample_duration(self, rng: Random) -> int:
        j = self.duration_jitter
        return max(1, self.duration_mean + (rng.randint(-j, j) if j else 0))


@dataclass(frozen=True)
class TurnSpec:
    gen_len: int  # tokens this turn generates before it calls a tool / finishes
    tool: Tool | None = None  # None on the final turn
    result_len: int = 8  # tool-result tokens prepended to the next turn's prefill


class ProgramState(Enum):
    ACTING = auto()  # a turn is decoding
    TOOL_CALL = auto()  # paused; KV pinned, waiting for the tool
    DONE = auto()


@dataclass
class Program:
    """A multi-turn agent session. Turns are separated by tool calls, during
    which the accumulated context is pinned rather than recomputed."""

    turns: list[TurnSpec]
    prompt_len: int = 16
    arrival: float = 0.0
    program_id: int = field(default_factory=lambda: next(_prog_counter))
    tenant_id: int | None = None

    state: ProgramState = ProgramState.ACTING
    turn_idx: int = 0
    seq: Sequence | None = None
    resume_at: int | None = None  # step the tool returns
    ground_truth: list[tuple[str, int, int]] = field(default_factory=list)  # (kind, start, end)

    def _new_seq(self, prompt_len: int, gen_len: int, block_table=None, num_computed=0) -> Sequence:
        seq = Sequence(
            prompt_ids=[_MOCK_TOKEN] * prompt_len,
            params=SamplingParams(max_tokens=gen_len, ignore_eos=True),
            program_id=self.program_id,
            program_arrival=self.arrival,
            tenant_id=self.tenant_id if self.tenant_id is not None else self.program_id,
        )
        if block_table:
            seq.block_table = block_table
            seq.num_computed = num_computed
        return seq

    def first_turn(self) -> Sequence:
        t = self.turns[0]
        self.seq = self._new_seq(self.prompt_len, t.gen_len)
        return self.seq


class ProgramRunner:
    """Drives programs against a Scheduler with a mock forward.

    This is the deterministic, single-threaded multi-tenant engine the side
    channel needs: no weights, no wall clock in the loop, so admission latency is
    an exact step count and the channel capacity is exactly measurable. Attacker
    probes are modelled as trivial one-turn programs, so victim and attacker share
    one code path and the scheduler cannot tell them apart -- which is the point.
    """

    def __init__(self, scheduler: Scheduler, seed: int = 0):
        self.sched = scheduler
        self.rng = Random(seed)
        self.programs: dict[int, Program] = {}
        self.first_run_step: dict[int, int] = {}  # seq_id -> step it was first admitted
        self._pending_first_run: set[int] = set()

    def submit(self, program: Program) -> Program:
        self.programs[program.program_id] = program
        program.state = ProgramState.ACTING
        seq = program.first_turn()
        self._pending_first_run.add(seq.seq_id)
        self.sched.add(seq)
        return program

    def has_work(self) -> bool:
        return (
            self.sched.has_unfinished()
            or any(p.state == ProgramState.TOOL_CALL for p in self.programs.values())
        )

    def step(self) -> "StepRecord":
        out = self.sched.schedule()

        # Admission: a sequence is admitted the first prefill step it is scheduled.
        newly = []
        for seq in out.scheduled:
            if seq.seq_id in self._pending_first_run:
                self._pending_first_run.discard(seq.seq_id)
                self.first_run_step[seq.seq_id] = self.sched.step_counter
                newly.append(seq)

        # Mock forward: identical bookkeeping to LLMEngine.step, minus the model.
        for seq in out.scheduled:
            seq.num_computed = seq.num_tokens
            seq.append_token(_MOCK_TOKEN)
            seq.maybe_finish(eos_token_id=-1)  # only max_tokens can fire

        # Turn boundaries: a finished turn either pauses (tool call, pin the KV) or
        # ends the program. Do this BEFORE free_finished so a paused turn is pulled
        # out of running and its blocks are not reclaimed.
        for seq in out.scheduled:
            if seq.is_finished:
                self._on_turn_end(seq)

        self.sched.free_finished()
        self._resume_due()

        return StepRecord(
            step=self.sched.step_counter,
            free_blocks=self.sched.allocator.num_free,
            pinned_blocks=self.sched.pinned_blocks,
            num_running=len(self.sched.running),
            num_suspended=len(self.sched.suspended),
            newly_admitted=[s.seq_id for s in newly],
        )

    def _on_turn_end(self, seq: Sequence) -> None:
        prog = self.programs.get(seq.program_id) if seq.program_id is not None else None
        if prog is None:
            return
        has_next = prog.turn_idx + 1 < len(prog.turns)
        tool = prog.turns[prog.turn_idx].tool
        if has_next and tool is not None:
            self.sched.suspend(seq)  # pin KV for the tool call
            dur = tool.sample_duration(self.rng)
            prog.state = ProgramState.TOOL_CALL
            prog.resume_at = self.sched.step_counter + dur
            prog.ground_truth.append((tool.name, self.sched.step_counter, prog.resume_at))
        else:
            prog.state = ProgramState.DONE  # last turn: let free_finished reclaim it

    def _resume_due(self) -> None:
        for prog in self.programs.values():
            if prog.state != ProgramState.TOOL_CALL or prog.resume_at is None:
                continue
            if self.sched.step_counter < prog.resume_at:
                continue
            prev = prog.seq
            assert prev is not None
            alive = self.sched.resume(prev)
            prog.turn_idx += 1
            t = prog.turns[prog.turn_idx]
            result = prog.turns[prog.turn_idx - 1].result_len
            if alive:
                # Prefix cache survived: reuse the pinned blocks, prefill only the
                # tool-result tokens.
                new_len = prev.num_tokens + result
                seq = prog._new_seq(new_len, t.gen_len, block_table=prev.block_table,
                                    num_computed=prev.num_tokens)
            else:
                # TTL evicted the pin: recompute the whole context from scratch.
                seq = prog._new_seq(prev.num_tokens + result, t.gen_len)
            prog.seq = seq
            prog.state = ProgramState.ACTING
            prog.resume_at = None
            self._pending_first_run.add(seq.seq_id)
            self.sched.add(seq)


@dataclass
class StepRecord:
    step: int
    free_blocks: int
    pinned_blocks: int
    num_running: int
    num_suspended: int
    newly_admitted: list[int]
