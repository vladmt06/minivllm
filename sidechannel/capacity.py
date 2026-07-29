"""Channel capacity: how many bits about the victim leak per tool call.

A single "100%" is a toy number. The honest measure is the mutual information
between the tool the victim actually ran and the tool the attacker guessed,
aggregated over many trials. A perfect channel carries log2(n_tools) bits; a
blind guesser carries 0. Everything here is pure arithmetic over confusion
counts, so test_capacity.py can pin it against known distributions before any of
it is trusted.
"""

from __future__ import annotations

import math
from collections import Counter

Confusion = dict[tuple[str, str], int]  # (true tool, guessed tool) -> count


def merge(confusions) -> Confusion:
    out: Confusion = {}
    for c in confusions:
        for k, v in c.items():
            out[k] = out.get(k, 0) + v
    return out


def _entropy_bits(counts) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c:
            p = c / total
            h -= p * math.log2(p)
    return h


def mutual_information_bits(confusion: Confusion) -> float:
    """I(True; Guess) in bits: the information the attacker's guess carries about
    the victim's actual tool."""
    total = sum(confusion.values())
    if total == 0:
        return 0.0
    p_true: Counter[str] = Counter()
    p_guess: Counter[str] = Counter()
    for (t, g), c in confusion.items():
        p_true[t] += c
        p_guess[g] += c

    mi = 0.0
    for (t, g), c in confusion.items():
        if not c:
            continue
        p_tg = c / total
        denom = (p_true[t] / total) * (p_guess[g] / total)
        mi += p_tg * math.log2(p_tg / denom)
    return max(0.0, mi)  # clamp tiny negative from float error


def true_entropy_bits(confusion: Confusion) -> float:
    """H(True): the bits available to leak, given how often each tool was run."""
    p_true: Counter[str] = Counter()
    for (t, _), c in confusion.items():
        p_true[t] += c
    return _entropy_bits(list(p_true.values()))


def normalized_capacity(confusion: Confusion) -> float:
    """Fraction of the victim's tool identity the channel recovers: I/H(True),
    in [0, 1]. 1.0 = the tool is fully determined by the guess."""
    h = true_entropy_bits(confusion)
    return mutual_information_bits(confusion) / h if h > 0 else 0.0


def per_tool_prf(confusion: Confusion) -> dict[str, dict[str, float]]:
    """Precision, recall, F1 per tool -- which tools are reliably fingerprinted
    and which blur together."""
    tools = {t for t, _ in confusion} | {g for _, g in confusion}
    out: dict[str, dict[str, float]] = {}
    for tool in sorted(tools):
        tp = confusion.get((tool, tool), 0)
        fp = sum(c for (t, g), c in confusion.items() if g == tool and t != tool)
        fn = sum(c for (t, g), c in confusion.items() if t == tool and g != tool)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[tool] = {"precision": prec, "recall": rec, "f1": f1}
    return out


def ci95(values: list[float]) -> tuple[float, float, float]:
    """(median, lo, hi) with a simple percentile 95% interval. Non-parametric, so
    it does not assume the metric is normal across seeds."""
    if not values:
        return (float("nan"),) * 3
    xs = sorted(values)
    n = len(xs)
    med = xs[n // 2]
    lo = xs[max(0, int(0.025 * (n - 1)))]
    hi = xs[min(n - 1, int(round(0.975 * (n - 1))))]
    return med, lo, hi
