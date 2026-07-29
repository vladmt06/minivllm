"""Turn admission timestamps into a reconstructed victim program timeline.

Input is only the attacker's own admission steps. Output is the recovered tool
calls: their count (= turns), their durations, and a tool label per call from a
nearest-duration match against the public taxonomy. Everything is scored against
the hidden ground truth so the channel's accuracy is a number, not a vibe.
"""

from __future__ import annotations

from dataclasses import dataclass

from sidechannel.victim import TOOL_TAXONOMY


@dataclass
class Burst:
    start: int
    end: int

    @property
    def width(self) -> int:
        return self.end - self.start + 1


def infer_gap(admit_steps) -> float:
    """The grouping gap an adaptive attacker would use.

    Admissions are bimodal: tightly spaced *within* a tool-call pause (~one step,
    or the cadence period under the cadence defense) and widely spaced *between*
    pauses (the victim's generation phase). Any threshold between the two clusters
    groups a pause correctly. The attacker finds it by locating the largest
    multiplicative jump in its own sorted inter-admission gaps -- no knowledge of
    the victim or the step time required. This is what makes the reconstruction
    survive both clean integer steps and noisy wall-clock milliseconds.
    """
    steps = sorted(set(admit_steps))
    if len(steps) < 3:
        return 2
    diffs = sorted(b - a for a, b in zip(steps, steps[1:]))
    split_i, best_ratio = 0, 1.0
    for i in range(1, len(diffs)):
        prev = diffs[i - 1] or 0.5
        ratio = diffs[i] / prev
        if ratio > best_ratio:
            best_ratio, split_i = ratio, i
    if best_ratio >= 3.0:  # a clear gap between intra- and inter-pause spacing
        return (diffs[split_i - 1] + diffs[split_i]) / 2.0
    return diffs[-1] + 1  # unimodal: no separable pauses


def find_bursts(admit_steps: list[int], gap: int | None = None) -> list[Burst]:
    """Group admissions into bursts. A gap longer than `gap` steps ends a burst:
    during a victim ACTING phase the batch is full and admissions stop, so the
    silence between bursts is the victim generating. gap=None adapts to cadence."""
    if not admit_steps:
        return []
    if gap is None:
        gap = infer_gap(admit_steps)
    steps = sorted(admit_steps)
    bursts = [Burst(steps[0], steps[0])]
    for s in steps[1:]:
        if s - bursts[-1].end <= gap:
            bursts[-1].end = s
        else:
            bursts.append(Burst(s, s))
    return bursts


def classify(width: int, taxonomy=TOOL_TAXONOMY) -> str:
    """Nearest-mean-duration tool. The attacker knows the taxonomy (it is public
    system behaviour), just not which tool the victim ran."""
    return min(taxonomy, key=lambda name: abs(taxonomy[name].duration_mean - width))


def max_tool_width(taxonomy=TOOL_TAXONOMY) -> int:
    """No tool runs longer than this. A burst wider than it is not a tool call --
    it is the victim's session ending (its slot frees permanently) or noise, and
    the attacker, knowing the public taxonomy, discards it."""
    return 2 * max(t.duration_mean for t in taxonomy.values())


def session_bursts(admit_steps, gap: int | None = None, taxonomy=TOOL_TAXONOMY) -> list[Burst]:
    """Bursts during the victim's live session: every burst up to the first one
    too wide to be a tool (the departure flood), which is dropped along with
    everything after it. `taxonomy` sets the width units (steps or ms)."""
    cutoff = max_tool_width(taxonomy)
    kept: list[Burst] = []
    resolved_gap = infer_gap(admit_steps) if gap is None else gap
    for b in find_bursts(admit_steps, resolved_gap):
        if b.width > cutoff:
            break  # victim departed; ignore this burst and all that follow
        kept.append(b)
    return kept


@dataclass
class Score:
    n_truth: int
    n_recovered: int
    turn_count_exact: bool
    duration_mae: float
    tool_accuracy: float
    confusion: dict[tuple[str, str], int]  # (true, guessed) -> count

    def summary(self) -> str:
        return (
            f"turns {self.n_recovered}/{self.n_truth}"
            f"{' (exact)' if self.turn_count_exact else ''}, "
            f"duration MAE {self.duration_mae:.1f} steps, "
            f"tool accuracy {100 * self.tool_accuracy:.0f}%"
        )


def score(
    admit_steps,
    ground_truth: list[tuple[str, float, float]],  # (tool, start, end)
    gap: int | None = None,
    taxonomy=TOOL_TAXONOMY,
) -> Score:
    bursts = session_bursts(admit_steps, gap, taxonomy)
    n_truth = len(ground_truth)

    # Align bursts to truth in time order (both are chronological).
    paired = list(zip(ground_truth, bursts))
    abs_err, correct = [], 0
    confusion: dict[tuple[str, str], int] = {}
    for (tool, start, end), burst in paired:
        true_dur = end - start
        abs_err.append(abs(burst.width - true_dur))
        guess = classify(burst.width, taxonomy)
        confusion[(tool, guess)] = confusion.get((tool, guess), 0) + 1
        correct += guess == tool

    return Score(
        n_truth=n_truth,
        n_recovered=len(bursts),
        turn_count_exact=len(bursts) == n_truth,
        duration_mae=sum(abs_err) / len(abs_err) if abs_err else float("nan"),
        tool_accuracy=correct / n_truth if n_truth else 0.0,
        confusion=confusion,
    )


def chance_accuracy(taxonomy=TOOL_TAXONOMY) -> float:
    """A blind guesser picking the most common tool. The defense must push the
    attacker down to roughly this."""
    return 1.0 / len(taxonomy)
