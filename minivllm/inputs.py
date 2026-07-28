from __future__ import annotations

from dataclasses import dataclass

import torch


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
