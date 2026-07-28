"""Sampler unit tests.

The e2e tiers only ever exercise the greedy path, because that is the only one
HuggingFace can be compared against exactly. Everything else needs pinning here
or it is untested.
"""

from __future__ import annotations

import torch

from minivllm.core.sampler import sample
from minivllm.core.sequence import SamplingParams, Sequence

VOCAB = 64


def seqs_with(*params: SamplingParams) -> list[Sequence]:
    return [Sequence(prompt_ids=[1], params=p) for p in params]


def test_greedy_is_argmax():
    logits = torch.randn(4, VOCAB)
    got = sample(logits, seqs_with(*[SamplingParams()] * 4))
    assert got == logits.argmax(-1).tolist()


def test_greedy_ignores_temperature_scaling_entirely():
    """Greedy must be a branch, not a limit. Scaling logits cannot move it."""
    logits = torch.randn(1, VOCAB)
    a = sample(logits, seqs_with(SamplingParams()))
    b = sample(logits * 1000, seqs_with(SamplingParams()))
    assert a == b == [int(logits.argmax())]


def test_top_k_one_collapses_to_argmax():
    logits = torch.randn(3, VOCAB)
    params = SamplingParams(temperature=1.0, top_k=1)
    got = sample(logits, seqs_with(params, params, params))
    assert got == logits.argmax(-1).tolist()


def test_tiny_top_p_keeps_only_the_most_likely_token():
    """The nucleus must never be empty, however small p is."""
    logits = torch.randn(3, VOCAB)
    params = SamplingParams(temperature=1.0, top_p=1e-9)
    got = sample(logits, seqs_with(params, params, params))
    assert got == logits.argmax(-1).tolist()


def test_top_k_restricts_the_support():
    torch.manual_seed(0)
    logits = torch.randn(1, VOCAB)
    allowed = set(logits.topk(5, dim=-1).indices[0].tolist())
    params = SamplingParams(temperature=1.0, top_k=5)

    gen = torch.Generator().manual_seed(0)
    for _ in range(200):
        (tok,) = sample(logits, seqs_with(params), generator=gen)
        assert tok in allowed


def test_sampling_is_reproducible_from_a_generator():
    logits = torch.randn(2, VOCAB)
    params = SamplingParams(temperature=0.8, top_p=0.9)

    def run(seed):
        return sample(
            logits, seqs_with(params, params), generator=torch.Generator().manual_seed(seed)
        )

    assert run(0) == run(0)


def test_mixed_batch_routes_each_sequence_to_its_own_policy():
    """Greedy and sampled sequences share a batch, and the results must land in
    the caller's original order -- the split-and-recombine is easy to get wrong."""
    torch.manual_seed(0)
    logits = torch.randn(4, VOCAB)
    greedy, rand = SamplingParams(), SamplingParams(temperature=1.0, top_k=1)
    got = sample(logits, seqs_with(greedy, rand, greedy, rand))

    # top_k=1 is deterministic too, so every row must equal its own argmax.
    assert got == logits.argmax(-1).tolist()


def test_row_count_must_match_sequence_count():
    logits = torch.randn(3, VOCAB)
    try:
        sample(logits, seqs_with(SamplingParams(), SamplingParams()))
    except AssertionError:
        return
    raise AssertionError("mismatched batch went undetected")
