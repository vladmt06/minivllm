"""H2: characterise the channel statistically, on the fast deterministic engine.

For each operating point we run many seeds, aggregate the confusion, and report
channel capacity in bits with a confidence interval -- not a single 100%. The
sweeps show how the leak degrades as the attacker probes less often, as its
timing gets noisier, and as benign tenants compete for the freed slot, and they
locate the floors where the channel dies.

    uv run python -m sidechannel.sweeps          # writes results/sweeps.json
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from sidechannel import defense as D
from sidechannel.capacity import (
    ci95,
    merge,
    mutual_information_bits,
    normalized_capacity,
    per_tool_prf,
)
from sidechannel.harness import Scenario, run
from sidechannel.reconstruct import score

WORKLOAD = ["web_search", "db_query", "calc", "code_exec"]
RESULTS = Path(__file__).resolve().parent.parent / "results"
# Mutual information has a positive finite-sample bias, so use enough seeds that
# the bits curves are stable; tool accuracy and normalized capacity are the
# primary metrics and degrade monotonically regardless.
N_SEEDS = 40


def _trials(base: Scenario, n_seeds: int = N_SEEDS) -> dict:
    """Run n_seeds of one operating point; aggregate into a capacity summary."""
    scores = []
    for seed in range(n_seeds):
        r = run(replace(base, seed=seed), D.NONE)
        scores.append(score(r.admit_steps, r.ground_truth))
    conf = merge(s.confusion for s in scores)
    acc_med, acc_lo, acc_hi = ci95([s.tool_accuracy for s in scores])
    return {
        "bits_per_call": mutual_information_bits(conf),
        "normalized_capacity": normalized_capacity(conf),
        "tool_accuracy_median": acc_med,
        "tool_accuracy_lo": acc_lo,
        "tool_accuracy_hi": acc_hi,
        "turn_exact_rate": sum(s.turn_count_exact for s in scores) / len(scores),
        "duration_mae_median": ci95([s.duration_mae for s in scores if s.n_recovered])[0],
        "n_seeds": n_seeds,
    }


def _sweep(base: Scenario, field: str, values: list) -> list[dict]:
    out = []
    for v in values:
        summary = _trials(replace(base, **{field: v}))
        summary[field] = v
        out.append(summary)
        print(
            f"  {field}={v:<4} bits/call={summary['bits_per_call']:.2f} "
            f"cap={summary['normalized_capacity']:.2f} "
            f"tool_acc={summary['tool_accuracy_median']:.0%} "
            f"[{summary['tool_accuracy_lo']:.0%},{summary['tool_accuracy_hi']:.0%}] "
            f"turns_exact={summary['turn_exact_rate']:.0%}"
        )
    return out


def run_all() -> dict:
    base = Scenario(tool_sequence=WORKLOAD)
    print("probe_period (attacker rate):")
    probe = _sweep(base, "probe_period", [1, 2, 4, 8, 16, 32])
    print("admit_jitter (timing noise):")
    jitter = _sweep(base, "admit_jitter", [0, 1, 2, 4, 8, 16])
    print("num_benign (slot contention):")
    benign = _sweep(base, "num_benign", [0, 1, 2, 4, 8])

    # A confusion matrix at the baseline operating point, for the report figure.
    conf = merge(
        score(run(replace(base, seed=s), D.NONE).admit_steps,
              run(replace(base, seed=s), D.NONE).ground_truth).confusion
        for s in range(N_SEEDS)
    )
    prf = per_tool_prf(conf)

    return {
        "workload": WORKLOAD,
        "n_seeds": N_SEEDS,
        "max_bits": mutual_information_bits({(t, t): 1 for t in WORKLOAD}),
        "sweeps": {"probe_period": probe, "admit_jitter": jitter, "num_benign": benign},
        "baseline_confusion": {f"{t}->{g}": c for (t, g), c in conf.items()},
        "baseline_prf": prf,
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    data = run_all()
    (RESULTS / "sweeps.json").write_text(json.dumps(data, indent=2))
    print(f"\nwrote {RESULTS / 'sweeps.json'}")


if __name__ == "__main__":
    main()
