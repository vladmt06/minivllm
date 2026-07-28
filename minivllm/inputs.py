from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence as Seq

import torch

if TYPE_CHECKING:
    from minivllm.core.sequence import Sequence


@dataclass
class ModelInput:
    """One forward pass worth of work, flattened.

    Every token of every scheduled sequence is concatenated into a single
    dimension: there is no batch axis and no padding. Padding would cost both
    compute and — the part that actually matters here — memory bandwidth.
    """

    input_ids: torch.Tensor  # int64 [T]
    positions: torch.Tensor  # int64 [T]  index within its OWN sequence; drives RoPE
    slot_mapping: torch.Tensor  # int64 [T]  physical KV slot per token
    is_prefill: bool
    logits_indices: torch.Tensor  # int64 [S]  index into T of each seq's last token

    # Prefill only: [S+1] cumulative sequence lengths, for varlen slicing.
    cu_seqlens: torch.Tensor | None = None

    # Decode only.
    block_tables: torch.Tensor | None = None  # int32 [B, max_nb]
    context_lens: torch.Tensor | None = None  # int32 [B]

    @property
    def num_tokens(self) -> int:
        return self.input_ids.shape[0]

    @property
    def num_seqs(self) -> int:
        return self.logits_indices.shape[0]


def _t(values: list[int], dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    # One tensor construction per field. Building these element-wise on-device
    # would issue thousands of tiny copies for what is a few KB of indices.
    return torch.tensor(values, dtype=dtype, device=device)


def build_model_input(
    seqs: Seq["Sequence"],
    block_size: int,
    device: torch.device | str = "cpu",
    is_prefill: bool = False,
) -> ModelInput:
    """Flatten scheduled sequences into one forward pass, resolving every token
    to a physical KV slot.

    This is the page table. `Sequence.slot` does the arithmetic

        block_table[i // block_size] * block_size + (i % block_size)

    and the only hard part is which `i` to ask for. Prefill covers each
    sequence's uncomputed suffix `[num_computed, num_tokens)`; decode covers
    exactly one token, at `num_tokens - 1`. Getting that single index wrong does
    not crash -- it reads a neighbouring sequence's KV and produces fluent,
    wrong text -- so both branches assert the invariant they rely on, and
    tests/test_attention.py pins the arithmetic against a dense reference.
    """
    if not seqs:
        raise ValueError("no sequences to build input from")

    input_ids: list[int] = []
    positions: list[int] = []
    slot_mapping: list[int] = []
    logits_indices: list[int] = []

    if is_prefill:
        cu: list[int] = [0]
        for seq in seqs:
            assert seq.num_uncomputed > 0, f"{seq!r} has nothing to prefill"
            assert len(seq.block_table) >= seq.num_blocks_needed(block_size), (
                f"{seq!r} is under-allocated for {seq.num_tokens} tokens"
            )
            for i in range(seq.num_computed, seq.num_tokens):
                input_ids.append(seq.token_at(i))
                positions.append(i)
                slot_mapping.append(seq.slot(i, block_size))
            cu.append(len(input_ids))
            logits_indices.append(len(input_ids) - 1)

        return ModelInput(
            input_ids=_t(input_ids, torch.long, device),
            positions=_t(positions, torch.long, device),
            slot_mapping=_t(slot_mapping, torch.long, device),
            is_prefill=True,
            logits_indices=_t(logits_indices, torch.long, device),
            cu_seqlens=_t(cu, torch.int32, device),
        )

    # Decode: one query token per sequence, whose KV is about to be written and
    # which therefore attends to a context of exactly num_tokens.
    context_lens: list[int] = []
    max_nb = max(len(seq.block_table) for seq in seqs)
    block_tables: list[list[int]] = []

    for b, seq in enumerate(seqs):
        assert seq.num_uncomputed == 1, (
            f"{seq!r} has {seq.num_uncomputed} uncomputed tokens; decode feeds exactly 1"
        )
        i = seq.num_tokens - 1
        input_ids.append(seq.last_token_id)
        positions.append(i)
        slot_mapping.append(seq.slot(i, block_size))
        context_lens.append(seq.num_tokens)
        logits_indices.append(b)
        # Pad short rows with block 0. Those slots gather real data belonging to
        # another sequence, but they sit at flattened offsets >= context_len and
        # the mask in decode_attention drops them. Padding is bandwidth, never
        # correctness -- which is why max_nb is the batch max and not a constant:
        # the gather is [B, max_nb * block_size] and every wasted column is read.
        block_tables.append(seq.block_table + [0] * (max_nb - len(seq.block_table)))

    return ModelInput(
        input_ids=_t(input_ids, torch.long, device),
        positions=_t(positions, torch.long, device),
        slot_mapping=_t(slot_mapping, torch.long, device),
        is_prefill=False,
        logits_indices=_t(logits_indices, torch.long, device),
        block_tables=torch.tensor(block_tables, dtype=torch.int32, device=device),
        context_lens=_t(context_lens, torch.int32, device),
    )
