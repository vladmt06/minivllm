from __future__ import annotations

import torch

from minivllm.core.sequence import Sequence
from typing import Sequence as Seq


def _top_k(logits: torch.Tensor, ks: list[int]) -> torch.Tensor:
    """Mask everything below each row's k-th largest logit. k == 0 disables."""
    vocab = logits.shape[-1]
    eff = [k if 0 < k < vocab else vocab for k in ks]
    kmax = max(eff)
    if kmax >= vocab:
        return logits

    vals, _ = logits.topk(kmax, dim=-1)  # [S, kmax], descending
    kt = torch.tensor(eff, device=logits.device).clamp(max=kmax)
    threshold = vals.gather(1, (kt - 1).unsqueeze(1))  # [S, 1]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _top_p(logits: torch.Tensor, ps: list[float]) -> torch.Tensor:
    """Nucleus: keep the smallest prefix of the sorted distribution whose mass
    reaches p. The comparison uses the cumulative mass *excluding* each token,
    so the single most likely token always survives even when p is tiny."""
    if all(p >= 1.0 for p in ps):
        return logits

    ordered, idx = logits.sort(dim=-1, descending=True)
    probs = ordered.softmax(dim=-1)
    p = torch.tensor(ps, device=logits.device).unsqueeze(1)
    drop = (probs.cumsum(dim=-1) - probs) >= p
    return torch.empty_like(logits).scatter_(1, idx, ordered.masked_fill(drop, float("-inf")))


def sample(
    logits: torch.Tensor,  # [S, vocab] -- one row per scheduled sequence
    seqs: Seq[Sequence],
    generator: torch.Generator | None = None,
) -> list[int]:
    """Pick one token per sequence.

    Greedy is a separate branch rather than a very small temperature. Dividing
    by 1e-6 and sampling would *usually* return the argmax, and "usually" is not
    a basis for a test that compares token IDs against HuggingFace.

    Sampling runs in fp32 even when the model ran in fp16: softmax over 32000
    logits is exactly where half precision stops being free.
    """
    assert logits.shape[0] == len(seqs), (logits.shape, len(seqs))
    tokens = [0] * len(seqs)

    greedy = [i for i, s in enumerate(seqs) if s.params.is_greedy]
    random_ = [i for i, s in enumerate(seqs) if not s.params.is_greedy]

    if greedy:
        picked = logits[greedy].float().argmax(dim=-1)
        for j, i in enumerate(greedy):
            tokens[i] = int(picked[j])

    if random_:
        rows = logits[random_].float()
        params = [seqs[i].params for i in random_]

        temp = torch.tensor([p.temperature for p in params], device=rows.device).unsqueeze(1)
        rows = rows / temp
        rows = _top_k(rows, [p.top_k for p in params])
        rows = _top_p(rows, [p.top_p for p in params])

        picked = torch.multinomial(rows.softmax(dim=-1), 1, generator=generator).squeeze(1)
        for j, i in enumerate(random_):
            tokens[i] = int(picked[j])

    return tokens
