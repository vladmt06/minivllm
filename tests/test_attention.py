"""M3 gate: does the paged path compute the same attention as a dense one?

Tiers 1 and 2 cannot see a broken page table. A wrong slot still reads *some*
plausible KV, so the model keeps producing fluent text and the logits test keeps
passing. This is the only place the mapping itself is pinned.

Two properties make these tests bite, and both are easy to lose in a rewrite:

  1. Block tables are deliberately NOT the identity [0, 1, 2, ...]. Under an
     identity table, `block_table[i // bs] * bs + i % bs == i`, so every
     slot-mapping bug cancels out and the suite passes while paging is broken.
  2. The pool is pre-filled with noise, never zeros. A stale or out-of-bounds
     read then produces a visibly wrong answer instead of a near-miss.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from minivllm.attention import decode_attention, prefill_attention, write_kv
from minivllm.core.sequence import Sequence
from minivllm.inputs import build_model_input

BLOCK = 16
N_HEADS = 32
N_KV = 4
D = 64

DEVICE = torch.device("cpu")
DTYPE = torch.float32


def make_cache(num_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A pool full of noise, so reading the wrong slot is loud."""
    shape = (num_blocks, BLOCK, N_KV, D)
    return torch.randn(shape, dtype=DTYPE), torch.randn(shape, dtype=DTYPE)


def slots_for(block_table: list[int], n: int, start: int = 0) -> torch.Tensor:
    return torch.tensor(
        [block_table[i // BLOCK] * BLOCK + i % BLOCK for i in range(start, n)],
        dtype=torch.long,
    )


def scatter(k_cache, v_cache, k, v, block_table: list[int]) -> None:
    write_kv(k, v, k_cache, v_cache, slots_for(block_table, k.shape[0]))


def dense_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reference: one query token attending over a contiguous [L, n_kv, d] cache."""
    o = F.scaled_dot_product_attention(
        q[None, :, None, :],  # [1, H, 1, d]
        k.transpose(0, 1)[None],  # [1, n_kv, L, d]
        v.transpose(0, 1)[None],
        enable_gqa=True,
    )
    return o[:, :, 0, :]  # [1, H, d]


# -- the page table itself ---------------------------------------------------


def test_slot_arithmetic_at_block_boundaries():
    seq = Sequence(prompt_ids=[0] * 40, block_table=[7, 2, 11])
    assert seq.slot(0, BLOCK) == 7 * 16 + 0
    assert seq.slot(15, BLOCK) == 7 * 16 + 15  # last of block 0
    assert seq.slot(16, BLOCK) == 2 * 16 + 0  # first of block 1 -- the discontinuity
    assert seq.slot(31, BLOCK) == 2 * 16 + 15
    assert seq.slot(32, BLOCK) == 11 * 16 + 0
    assert seq.slot(36, BLOCK) == 11 * 16 + 4


@pytest.mark.parametrize("n,expected", [(1, 1), (15, 1), (16, 1), (17, 2), (32, 2), (37, 3)])
def test_blocks_needed_is_ceil(n, expected):
    assert Sequence(prompt_ids=[0] * n).num_blocks_needed(BLOCK) == expected


def test_write_then_gather_round_trips():
    """Isolates write_kv from attention: what went in must come back out, in order."""
    torch.manual_seed(0)
    L, table = 37, [7, 2, 11]
    k, v = torch.randn(L, N_KV, D), torch.randn(L, N_KV, D)
    k_cache, v_cache = make_cache(16)
    scatter(k_cache, v_cache, k, v, table)

    got_k = k_cache.view(-1, N_KV, D)[slots_for(table, L)]
    got_v = v_cache.view(-1, N_KV, D)[slots_for(table, L)]
    assert torch.equal(got_k, k)
    assert torch.equal(got_v, v)


def test_writes_stay_inside_owned_blocks():
    """A sequence must not touch a block it does not own -- the failure mode that
    corrupts a *neighbour* rather than itself."""
    torch.manual_seed(0)
    L, table = 37, [7, 2, 11]
    k_cache, v_cache = make_cache(16)
    before = k_cache.clone()
    scatter(k_cache, v_cache, torch.randn(L, N_KV, D), torch.randn(L, N_KV, D), table)

    untouched = [b for b in range(16) if b not in table]
    assert torch.equal(k_cache[untouched], before[untouched])
    # Block 11 holds only 5 of the sequence's tokens; slots 5..15 must be pristine.
    assert torch.equal(k_cache[11, 5:], before[11, 5:])


# -- decode: paged == dense --------------------------------------------------


@pytest.mark.parametrize("L", [1, 15, 16, 17, 32, 37, 48, 63])
def test_paged_decode_matches_dense(L):
    """The gate. 37 and 63 are the non-block-aligned cases the plan calls out;
    16/32/48 are the exact boundaries where an off-by-one flips a block."""
    torch.manual_seed(L)
    nb = -(-L // BLOCK)
    table = [7, 2, 11, 5][:nb]  # scrambled on purpose

    k, v = torch.randn(L, N_KV, D), torch.randn(L, N_KV, D)
    q = torch.randn(1, N_HEADS, D)
    k_cache, v_cache = make_cache(16)
    scatter(k_cache, v_cache, k, v, table)

    got = decode_attention(
        q,
        k_cache,
        v_cache,
        torch.tensor([table], dtype=torch.int32),
        torch.tensor([L], dtype=torch.int32),
        BLOCK,
    )
    assert torch.allclose(got, dense_decode(q[0], k, v), atol=1e-5)


def test_decode_batch_with_ragged_contexts():
    """Different lengths *and* different block counts in one batch, so the mask
    and the row padding are both exercised."""
    torch.manual_seed(1)
    lens = [37, 5, 16, 48]
    tables = [[7, 2, 11], [3], [9], [1, 12, 4]]
    k_cache, v_cache = make_cache(16)

    kvs, qs = [], []
    for L, table in zip(lens, tables):
        k, v = torch.randn(L, N_KV, D), torch.randn(L, N_KV, D)
        scatter(k_cache, v_cache, k, v, table)
        kvs.append((k, v))
        qs.append(torch.randn(N_HEADS, D))

    max_nb = max(len(t) for t in tables)
    padded = [t + [0] * (max_nb - len(t)) for t in tables]
    got = decode_attention(
        torch.stack(qs),
        k_cache,
        v_cache,
        torch.tensor(padded, dtype=torch.int32),
        torch.tensor(lens, dtype=torch.int32),
        BLOCK,
    )

    for b, (k, v) in enumerate(kvs):
        assert torch.allclose(got[b : b + 1], dense_decode(qs[b], k, v), atol=1e-5), f"seq {b}"


def test_test_has_teeth():
    """Guard against the guard: reading a block the sequence does not own must
    change the answer, or every test above is vacuous.

    Note what does *not* work as a corruption here: permuting owned blocks.
    Decode attention is a set operation over the context -- one query, softmax
    over all keys, no causal mask -- and RoPE was baked into K before it was
    cached, so position is carried by the values themselves rather than by their
    order. run([2, 7, 11]) matches run([7, 2, 11]) to within float reassociation
    -- the softmax reduction runs in a different order, so it is not bit-equal.

    That is a real property, not a bug, and it narrows what these tests can
    catch: the block table must point at the right *data*, and the slot
    arithmetic must be right, but block ordering within a sequence is genuinely
    free. Prefill is where order matters, because `is_causal=True` reintroduces
    it -- see test_prefill_matches_dense_causal.
    """
    torch.manual_seed(2)
    L = 37
    k, v = torch.randn(L, N_KV, D), torch.randn(L, N_KV, D)
    q = torch.randn(1, N_HEADS, D)
    k_cache, v_cache = make_cache(16)
    scatter(k_cache, v_cache, k, v, [7, 2, 11])

    def run(table, ctx=L):
        return decode_attention(
            q,
            k_cache,
            v_cache,
            torch.tensor([table], dtype=torch.int32),
            torch.tensor([ctx], dtype=torch.int32),
            BLOCK,
        )

    ref = run([7, 2, 11])
    assert not torch.allclose(run([7, 2, 5]), ref, atol=1e-5), "read a foreign block undetected"
    assert not torch.allclose(run([7, 2, 11], ctx=36), ref, atol=1e-5), "mask has no effect"
    assert torch.allclose(run([2, 7, 11]), ref, atol=1e-6), "documented invariance broke"


def test_mask_ignores_slots_past_context():
    """Trailing slots in the final block hold another epoch's noise. Growing the
    context must be the only thing that changes the answer."""
    torch.manual_seed(3)
    L, table = 37, [7, 2, 11]
    k, v = torch.randn(L, N_KV, D), torch.randn(L, N_KV, D)
    q = torch.randn(1, N_HEADS, D)
    k_cache, v_cache = make_cache(16)
    scatter(k_cache, v_cache, k, v, table)

    bt = torch.tensor([table], dtype=torch.int32)
    cl = torch.tensor([L], dtype=torch.int32)
    got = decode_attention(q, k_cache, v_cache, bt, cl, BLOCK)

    # Scribble over the unused tail of the last block; the answer must not move.
    k_cache[11, 5:] = torch.randn_like(k_cache[11, 5:])
    v_cache[11, 5:] = torch.randn_like(v_cache[11, 5:])
    again = decode_attention(q, k_cache, v_cache, bt, cl, BLOCK)
    assert torch.equal(got, again)


# -- prefill -----------------------------------------------------------------


def test_prefill_matches_dense_causal():
    torch.manual_seed(4)
    lens = [37, 5, 16]
    q = torch.randn(sum(lens), N_HEADS, D)
    k = torch.randn(sum(lens), N_KV, D)
    v = torch.randn(sum(lens), N_KV, D)
    cu = torch.tensor([0, 37, 42, 58], dtype=torch.int32)

    got = prefill_attention(q, k, v, cu)

    start = 0
    for L in lens:
        sl = slice(start, start + L)
        want = F.scaled_dot_product_attention(
            q[sl].transpose(0, 1)[None],
            k[sl].transpose(0, 1)[None],
            v[sl].transpose(0, 1)[None],
            is_causal=True,
            enable_gqa=True,
        )[0].transpose(0, 1)
        assert torch.allclose(got[sl], want, atol=1e-5), f"seq at {start}"
        start += L


# -- build_model_input -------------------------------------------------------


def test_prefill_input_maps_every_token():
    seq = Sequence(prompt_ids=list(range(37)), block_table=[7, 2, 11])
    inp = build_model_input([seq], BLOCK, DEVICE, is_prefill=True)

    assert inp.num_tokens == 37
    assert torch.equal(inp.input_ids, torch.arange(37))
    assert torch.equal(inp.positions, torch.arange(37))
    assert torch.equal(inp.slot_mapping, slots_for([7, 2, 11], 37))
    assert torch.equal(inp.cu_seqlens, torch.tensor([0, 37], dtype=torch.int32))
    assert torch.equal(inp.logits_indices, torch.tensor([36]))


def test_prefill_input_concatenates_without_padding():
    seqs = [
        Sequence(prompt_ids=list(range(37)), block_table=[7, 2, 11]),
        Sequence(prompt_ids=list(range(100, 105)), block_table=[3]),
    ]
    inp = build_model_input(seqs, BLOCK, DEVICE, is_prefill=True)

    assert inp.num_tokens == 42, "a padded batch would be 2 * 37"
    assert torch.equal(inp.cu_seqlens, torch.tensor([0, 37, 42], dtype=torch.int32))
    assert torch.equal(inp.logits_indices, torch.tensor([36, 41]))
    # Position resets per sequence -- it drives RoPE, not the batch offset.
    assert inp.positions[37].item() == 0


def test_decode_input_targets_the_last_token_only():
    seq = Sequence(prompt_ids=list(range(37)), block_table=[7, 2, 11], num_computed=36)
    seq.output_ids = []  # 37 tokens, 36 computed -> exactly one to feed
    inp = build_model_input([seq], BLOCK, DEVICE, is_prefill=False)

    assert inp.num_tokens == 1
    assert inp.positions.item() == 36
    assert inp.slot_mapping.item() == 11 * 16 + 4
    assert inp.context_lens.item() == 37
    assert torch.equal(inp.block_tables, torch.tensor([[7, 2, 11]], dtype=torch.int32))


def test_decode_block_tables_pad_to_batch_max():
    """Width is the batch max, not a constant: decode_attention gathers
    [B, max_nb * block_size] and reads every padded column it is given."""
    seqs = [
        Sequence(prompt_ids=[0] * 37, block_table=[7, 2, 11], num_computed=36),
        Sequence(prompt_ids=[0] * 5, block_table=[3], num_computed=4),
    ]
    inp = build_model_input(seqs, BLOCK, DEVICE, is_prefill=False)

    assert inp.block_tables.shape == (2, 3)
    assert inp.block_tables[1].tolist() == [3, 0, 0]
    assert torch.equal(inp.context_lens, torch.tensor([37, 5], dtype=torch.int32))


def test_decode_rejects_a_broken_invariant():
    """Feeding more than one uncomputed token down the decode path would silently
    write one slot and drop the rest."""
    seq = Sequence(prompt_ids=[0] * 37, block_table=[7, 2, 11], num_computed=10)
    with pytest.raises(AssertionError, match="uncomputed"):
        build_model_input([seq], BLOCK, DEVICE, is_prefill=False)


def test_prefill_rejects_under_allocation():
    seq = Sequence(prompt_ids=[0] * 37, block_table=[7, 2])  # needs 3
    with pytest.raises(AssertionError, match="under-allocated"):
        build_model_input([seq], BLOCK, DEVICE, is_prefill=True)
