from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum, auto


@dataclass(frozen=True)
class SamplingParams:
    """Greedy is `temperature == 0.0`, and the sampler branches on it explicitly
    rather than dividing by something tiny. Exact reproducibility is what the
    tier-2 and tier-3 correctness tests compare against HuggingFace."""

    max_tokens: int = 32
    temperature: float = 0.0
    top_k: int = 0  # 0 disables
    top_p: float = 1.0  # 1.0 disables
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    SUSPENDED = auto()  # program paused for a tool call; KV pinned, not decoding
    FINISHED_STOPPED = auto()  # emitted EOS
    FINISHED_LENGTH = auto()  # hit max_tokens
    FINISHED_ABORTED = auto()

    @property
    def is_finished(self) -> bool:
        return self in _FINISHED


_FINISHED = frozenset(
    {
        SequenceStatus.FINISHED_STOPPED,
        SequenceStatus.FINISHED_LENGTH,
        SequenceStatus.FINISHED_ABORTED,
    }
)

_seq_counter = itertools.count()


@dataclass
class Sequence:
    """One request, plus the page table that maps its tokens to physical blocks.

    `num_computed` -- tokens whose KV already sits in the cache -- is the
    invariant everything else keys off, so state it exactly:

        At the START of a decode step, num_computed == num_tokens - 1: the token
        sampled last step has been appended but its KV has not been written yet.
        The forward writes it, and num_computed becomes num_tokens. Prefill
        covers [num_computed, num_tokens), which for a fresh request is the
        whole prompt.

    Preemption sets num_computed = 0 and frees every block, but KEEPS
    output_ids. Re-admission therefore re-prefills prompt_ids + output_ids and
    lands in exactly the state it left, which is what makes recompute
    preemption output-invariant rather than merely plausible. tests/test_e2e.py
    is the enforcement.
    """

    prompt_ids: list[int]
    params: SamplingParams = SamplingParams()
    seq_id: int = field(default_factory=lambda: next(_seq_counter))
    arrival: float = field(default_factory=time.monotonic)

    output_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    block_table: list[int] = field(default_factory=list)
    num_computed: int = 0

    # Diagnostics: how often this sequence lost its blocks. Benchmarks report it,
    # and a preemption test that never increments it is not testing anything.
    num_preemptions: int = 0

    # Program-aware serving (default None = a plain standalone request). When set,
    # this turn belongs to a multi-turn program, and program_arrival — the arrival
    # of the *program*, not this turn — is the scheduler's priority key. All turns
    # of an early program thus outrank a late program's, which is program-level
    # FCFS and also the thing the side channel reads.
    program_id: int | None = None
    program_arrival: float | None = None

    # Billing/isolation identity. The per-tenant reservation defense caps blocks
    # by this key, so an attacker's many probes must share one. None = each
    # sequence is its own tenant.
    tenant_id: int | None = None

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise ValueError("empty prompt")

    @property
    def priority(self) -> tuple[float, float, int]:
        """Scheduler ordering. Falls back to request arrival when standalone, so
        the non-program path is byte-for-byte the old (arrival, seq_id) order."""
        base = self.program_arrival if self.program_arrival is not None else self.arrival
        return (base, self.arrival, self.seq_id)

    # -- sizes ---------------------------------------------------------------

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def num_uncomputed(self) -> int:
        return self.num_tokens - self.num_computed

    # -- tokens --------------------------------------------------------------

    @property
    def token_ids(self) -> list[int]:
        """Concatenation, rebuilt per call. Decode wants only the final token --
        use last_token_id there and keep the step O(1) rather than O(context)."""
        return self.prompt_ids + self.output_ids

    @property
    def last_token_id(self) -> int:
        return self.output_ids[-1] if self.output_ids else self.prompt_ids[-1]

    def token_at(self, i: int) -> int:
        n = self.num_prompt_tokens
        return self.prompt_ids[i] if i < n else self.output_ids[i - n]

    # -- blocks --------------------------------------------------------------

    def num_blocks_needed(self, block_size: int) -> int:
        return -(-self.num_tokens // block_size)  # ceil

    def needs_new_block(self, block_size: int) -> bool:
        """Whether the next write has nowhere to go.

        The original plan wrote this as `num_tokens % block_size == 0`, which is
        equivalent only while the num_computed invariant holds exactly. Deriving
        it from the block table instead means a scheduler bug shows up as a
        failed allocation rather than as a silent write into another sequence's
        page.
        """
        return self.num_blocks_needed(block_size) > len(self.block_table)

    def slot(self, i: int, block_size: int) -> int:
        """Physical KV slot of this sequence's i-th token. The page table, in one line."""
        return self.block_table[i // block_size] * block_size + (i % block_size)

    # -- transitions ---------------------------------------------------------

    def append_token(self, token_id: int) -> None:
        self.output_ids.append(token_id)

    def reset_for_recompute(self) -> None:
        """Blocks are freed by the caller (only the scheduler owns the allocator)."""
        self.block_table = []
        self.num_computed = 0
        self.status = SequenceStatus.WAITING
        self.num_preemptions += 1

    def maybe_finish(self, eos_token_id: int) -> bool:
        """Apply stop conditions to the just-appended token. EOS is checked first
        so a sequence that ends exactly at max_tokens reports STOPPED, matching
        HuggingFace's ordering."""
        if not self.params.ignore_eos and self.output_ids and self.output_ids[-1] == eos_token_id:
            self.status = SequenceStatus.FINISHED_STOPPED
        elif self.num_output_tokens >= self.params.max_tokens:
            self.status = SequenceStatus.FINISHED_LENGTH
        return self.status.is_finished

    @property
    def is_finished(self) -> bool:
        return self.status.is_finished

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, {self.status.name}, "
            f"{self.num_prompt_tokens}+{self.num_output_tokens} tok, "
            f"computed={self.num_computed}, blocks={len(self.block_table)})"
        )
