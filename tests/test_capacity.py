"""Pin the capacity math against distributions with known answers, so the bits
numbers in the report can be trusted.
"""

from __future__ import annotations

import math

from sidechannel.capacity import (
    merge,
    mutual_information_bits,
    normalized_capacity,
    per_tool_prf,
    true_entropy_bits,
)

TOOLS = ["web_search", "db_query", "calc", "code_exec"]


def perfect(n=10):
    return {(t, t): n for t in TOOLS}


def blind(n=10):
    # Guesser always says the same tool, regardless of truth -> no information.
    return {(t, "calc"): n for t in TOOLS}


def test_perfect_channel_carries_full_entropy():
    c = perfect()
    assert math.isclose(true_entropy_bits(c), 2.0)  # log2(4)
    assert math.isclose(mutual_information_bits(c), 2.0, abs_tol=1e-9)
    assert math.isclose(normalized_capacity(c), 1.0, abs_tol=1e-9)


def test_blind_guesser_carries_zero():
    c = blind()
    assert math.isclose(mutual_information_bits(c), 0.0, abs_tol=1e-9)
    assert math.isclose(normalized_capacity(c), 0.0, abs_tol=1e-9)


def test_half_confused_is_between():
    # Two tools perfectly separated, two others swapped 50/50: partial information.
    c = {
        ("web_search", "web_search"): 10,
        ("db_query", "db_query"): 10,
        ("calc", "calc"): 5,
        ("calc", "code_exec"): 5,
        ("code_exec", "code_exec"): 5,
        ("code_exec", "calc"): 5,
    }
    cap = normalized_capacity(c)
    assert 0.0 < cap < 1.0


def test_perfect_prf_is_all_ones():
    prf = per_tool_prf(perfect())
    for tool in TOOLS:
        assert prf[tool] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_merge_sums_counts():
    m = merge([{("a", "a"): 1}, {("a", "a"): 2, ("b", "b"): 1}])
    assert m == {("a", "a"): 3, ("b", "b"): 1}
