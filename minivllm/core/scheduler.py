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

        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

        # Diagnostics. A preemption test that never moves this counter is not
        # testing preemption, so it is part of the contract, not a nicety.
        self.num_preemptions = 0
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
        prefill = self._schedule_prefill()
        if not prefill.is_empty:
            self.num_prefill_steps += 1
            return prefill
        decode = self._schedule_decode()
        if not decode.is_empty:
            self.num_decode_steps += 1
        return decode

    def _schedule_prefill(self) -> SchedulerOutputs:
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

    def _schedule_decode(self) -> SchedulerOutputs:
        if not self.running:
            return SchedulerOutputs([], is_prefill=False)

        # FCFS is the priority order, so arrival time decides who survives.
        self.running.sort(key=lambda s: (s.arrival, s.seq_id))
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
