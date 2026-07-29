from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from minivllm.config import CacheConfig, SchedulerConfig
from minivllm.core.sequence import Sequence, SequenceStatus
from minivllm.memory.allocator import BlockAllocator


class SequenceTooLong(ValueError):
    """A request that cannot fit in an empty pool. Raised rather than queued,
    because the alternative is a scheduler that spins forever making no
    progress and reports nothing."""


@dataclass
class SchedulerOutputs:
    scheduled: list[Sequence]
    is_prefill: bool
    preempted: list[Sequence] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


class Scheduler:
    """Continuous batching: two queues and a block budget.

    Each step is prefill-only or decode-only, which is the classic vLLM shape --
    mixing them is chunked prefill, deliberately out of scope. Admission is
    greedy and FCFS; when memory runs out mid-decode the *newest* running
    sequence is evicted so that the oldest keeps its progress.

    The property that makes this "continuous" rather than "static" batching is
    narrow and easy to miss: finished sequences release their blocks in the same
    step they finish, so a waiting request can be admitted on the very next one.
    Nothing waits for the slowest member of a batch to drain.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
        num_blocks: int,
    ):
        self.block_size = cache_config.block_size
        self.allocator = BlockAllocator(num_blocks)
        # Reserve a sliver so admission cannot instantly force the eviction of
        # what it just admitted. Rounds to 0 on the tiny pools tests use, which
        # is what makes those tests thrash on purpose.
        self.watermark = int(cache_config.watermark * num_blocks)
        self.max_num_seqs = scheduler_config.max_num_seqs
        self.max_num_batched_tokens = scheduler_config.max_num_batched_tokens
        self.max_model_len = scheduler_config.max_model_len

        # Program-aware serving + the two defense knobs. All inert unless the flag
        # is set, so the request-FCFS path below is unchanged.
        self.program_aware = scheduler_config.program_aware
        self.kv_ttl_steps = scheduler_config.kv_ttl_steps
        self.reserve_slots_on_suspend = scheduler_config.reserve_slots_on_suspend
        self.reserved_blocks_per_tenant = scheduler_config.reserved_blocks_per_tenant
        self.admission_period = scheduler_config.admission_period

        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        # Suspended turns: a program paused on a tool call, KV cache pinned. These
        # hold their block_table (excluded from num_free) and are exempt from
        # preemption. Their memory shadow on the free pool is the side channel.
        self.suspended: list[Sequence] = []
        self._pin_expiry: dict[int, int] = {}  # seq_id -> step at which TTL evicts

        # A monotonic step clock. Program pins and the admission-cadence defense
        # are both timed against it.
        self.step_counter = 0

        # Diagnostics. A preemption test that never moves this counter is not
        # testing preemption, so it is part of the contract, not a nicety.
        self.num_preemptions = 0
        self.num_pins = 0
        self.num_pin_evictions = 0
        self.num_prefill_steps = 0
        self.num_decode_steps = 0

    # -- queue management ----------------------------------------------------

    def add(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    @property
    def num_free_blocks(self) -> int:
        return self.allocator.num_free

    def abort(self, seq_id: int) -> bool:
        for queue in (self.waiting, self.running):
            for seq in list(queue):
                if seq.seq_id == seq_id:
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self._release(seq)
                    queue.remove(seq)  # type: ignore[arg-type]
                    return True
        return False

    # -- the step ------------------------------------------------------------

    def schedule(self) -> SchedulerOutputs:
        self.step_counter += 1
        if self.program_aware:
            # Evict pins whose TTL lapsed *before* admitting, so the reclaimed
            # blocks are available this step -- the same immediacy free_finished
            # gives finished sequences.
            self._evict_expired_pins()

        prefill = self._schedule_prefill()
        if not prefill.is_empty:
            self.num_prefill_steps += 1
            return prefill
        decode = self._schedule_decode()
        if not decode.is_empty:
            self.num_decode_steps += 1
        return decode

    def _admission_open(self) -> bool:
        """Admission-cadence defense: when set, prefill admission may only happen
        on a fixed clock, so the moment a request enters is independent of the
        live pin state an attacker is trying to read."""
        if not self.program_aware or self.admission_period <= 0:
            return True
        return self.step_counter % self.admission_period == 0

    def _schedule_prefill(self) -> SchedulerOutputs:
        if self.program_aware:
            return self._schedule_prefill_program_aware()

        scheduled: list[Sequence] = []
        budget = self.max_num_batched_tokens
        capacity = self.allocator.num_blocks - self.watermark

        while self.waiting:
            seq = self.waiting[0]
            need = seq.num_blocks_needed(self.block_size) - len(seq.block_table)

            if need > capacity:
                raise SequenceTooLong(
                    f"{seq!r} needs {need} blocks; pool holds {self.allocator.num_blocks} "
                    f"(usable {capacity}). Raise kv_cache_gb or shorten the request."
                )
            if len(self.running) + len(scheduled) >= self.max_num_seqs:
                break
            if not self.allocator.can_allocate(need, self.watermark):
                break
            # The budget is a soft cap: a lone prompt bigger than the whole
            # budget still has to run, or it would stall the queue forever.
            # Chunked prefill is the real fix and is out of scope.
            if seq.num_uncomputed > budget and scheduled:
                break

            self.waiting.popleft()
            seq.block_table.extend(self.allocator.allocate(need))
            seq.status = SequenceStatus.RUNNING
            scheduled.append(seq)
            budget -= seq.num_uncomputed

        if scheduled:
            self.running.extend(scheduled)
        return SchedulerOutputs(scheduled, is_prefill=True)

    def _schedule_prefill_program_aware(self) -> SchedulerOutputs:
        """Admit in program-level FCFS order, honouring the two defenses.

        Differs from the default path only in that (a) candidates are considered
        in priority order rather than strict queue order, (b) an over-quota tenant
        is skipped rather than blocking the queue behind it (the reservation
        defense isolates, it must not deadlock), and (c) admission may be gated to
        a fixed cadence.
        """
        if not self._admission_open():
            return SchedulerOutputs([], is_prefill=True)

        scheduled: list[Sequence] = []
        budget = self.max_num_batched_tokens
        capacity = self.allocator.num_blocks - self.watermark
        held = self._blocks_by_tenant()
        # Slot-reservation defense: count paused programs against the concurrency
        # cap, so a suspension never opens a slot another tenant can observe.
        slot_floor = len(self.suspended) if self.reserve_slots_on_suspend else 0

        candidates = sorted(self.waiting, key=lambda s: s.priority)
        for seq in candidates:
            need = seq.num_blocks_needed(self.block_size) - len(seq.block_table)
            if need > capacity:
                raise SequenceTooLong(
                    f"{seq!r} needs {need} blocks; pool holds {self.allocator.num_blocks} "
                    f"(usable {capacity})."
                )
            if len(self.running) + len(scheduled) + slot_floor >= self.max_num_seqs:
                break
            if not self.allocator.can_allocate(need, self.watermark):
                break
            if seq.num_uncomputed > budget and scheduled:
                break
            if self._over_quota(seq, need, held):
                continue  # skip, do not block lower-priority tenants behind it

            self.waiting.remove(seq)
            seq.block_table.extend(self.allocator.allocate(need))
            seq.status = SequenceStatus.RUNNING
            scheduled.append(seq)
            budget -= seq.num_uncomputed
            held[self._tenant_of(seq)] = held.get(self._tenant_of(seq), 0) + need

        if scheduled:
            self.running.extend(scheduled)
        return SchedulerOutputs(scheduled, is_prefill=True)

    def _schedule_decode(self) -> SchedulerOutputs:
        if not self.running:
            return SchedulerOutputs([], is_prefill=False)

        # FCFS is the priority order, so arrival time decides who survives. For a
        # standalone request Sequence.priority is exactly (arrival, arrival,
        # seq_id), so this path is unchanged when program_aware is off; under it,
        # the leading key becomes the program's arrival (program-level FCFS).
        self.running.sort(key=lambda s: s.priority)
        queue, ok, preempted = self.running, [], []
        self.running = []

        while queue:
            seq: Sequence | None = queue.pop(0)  # oldest = highest priority
            assert seq is not None
            while seq.needs_new_block(self.block_size) and self.allocator.num_free == 0:
                # Evict the newest running sequence. If this one *is* the newest,
                # it evicts itself and goes back to the front of the queue.
                victim = queue.pop() if queue else seq
                self._preempt(victim)
                preempted.append(victim)
                if victim is seq:
                    seq = None
                    break
            if seq is None:
                continue
            if seq.needs_new_block(self.block_size):
                seq.block_table.extend(self.allocator.allocate(1))
            ok.append(seq)

        self.running = ok
        return SchedulerOutputs(ok, is_prefill=False, preempted=preempted)

    def _preempt(self, seq: Sequence) -> None:
        """Recompute preemption: drop the KV, keep the tokens.

        output_ids survives, so re-prefill regenerates exactly the state that was
        discarded. That is what makes preemption invisible in the output and
        purely a compute-for-memory trade. tests/test_e2e.py enforces it.
        """
        self._release(seq)
        seq.reset_for_recompute()
        self.waiting.appendleft(seq)
        self.num_preemptions += 1

    def _release(self, seq: Sequence) -> None:
        if seq.block_table:
            self.allocator.free(seq.block_table)
            seq.block_table = []

    def free_finished(self) -> list[Sequence]:
        """Reclaim finished sequences' blocks *now*, not at the end of the batch.

        This one line is the difference between static and continuous batching.
        """
        done = [s for s in self.running if s.is_finished]
        for seq in done:
            self._release(seq)
        if done:
            self.running = [s for s in self.running if not s.is_finished]
        return done

    # -- program-aware: tool-call pauses and KV pinning ----------------------

    def suspend(self, seq: Sequence) -> None:
        """Pin a program's KV cache for a tool-call pause.

        The turn has stopped generating but its context must survive so the next
        turn reuses it instead of recomputing. The blocks stay allocated (still
        counted against num_free) and out of both queues, so nothing preempts
        them -- and their absence from the free pool is exactly what a co-tenant
        can feel. The TTL bounds the pin: if the tool overruns, tick evicts.
        """
        if seq in self.running:
            self.running.remove(seq)
        seq.status = SequenceStatus.SUSPENDED
        self.suspended.append(seq)
        self._pin_expiry[seq.seq_id] = self.step_counter + self.kv_ttl_steps
        self.num_pins += 1

    def resume(self, seq: Sequence) -> bool:
        """Bring a paused program back. Returns True if its pinned cache survived
        (blocks intact, only the tool-result tokens need prefilling); False if the
        TTL evicted it first, in which case the caller must recompute from zero."""
        if seq in self.suspended:
            self.suspended.remove(seq)
            self._pin_expiry.pop(seq.seq_id, None)
            seq.status = SequenceStatus.WAITING
            return True
        return False  # already TTL-evicted

    def _evict_expired_pins(self) -> list[Sequence]:
        """Free pins whose tool call outran the TTL. Their blocks return to the
        pool and the program will have to re-prefill on resume."""
        expired = [s for s in self.suspended if self._pin_expiry[s.seq_id] <= self.step_counter]
        for seq in expired:
            self.suspended.remove(seq)
            self._pin_expiry.pop(seq.seq_id, None)
            self._release(seq)
            seq.reset_for_recompute()
            self.num_pin_evictions += 1
        return expired

    @property
    def pinned_blocks(self) -> int:
        return sum(len(s.block_table) for s in self.suspended)

    # -- per-tenant accounting (reservation defense) -------------------------

    @staticmethod
    def _tenant_of(seq: Sequence) -> int:
        if seq.tenant_id is not None:
            return seq.tenant_id
        if seq.program_id is not None:
            return seq.program_id
        return -seq.seq_id - 1  # standalone: unique, negative to avoid collision

    def _blocks_by_tenant(self) -> dict[int, int]:
        held: dict[int, int] = {}
        for seq in (*self.running, *self.suspended):
            t = self._tenant_of(seq)
            held[t] = held.get(t, 0) + len(seq.block_table)
        return held

    def _over_quota(self, seq: Sequence, need: int, held: dict[int, int]) -> bool:
        if self.reserved_blocks_per_tenant is None:
            return False
        t = self._tenant_of(seq)
        return held.get(t, 0) + len(seq.block_table) + need > self.reserved_blocks_per_tenant
