"""H4: every defense on one security-vs-utility plane.

For each defense we measure how much of the channel it closes (bits/tool call,
averaged over seeds) and what it costs on two common axes: idle slot-steps
(throughput withheld) and p99 admission latency (delay every tenant pays). The
frontier is the set of defenses no other beats on both security and cost.

    uv run python -m sidechannel.pareto          # writes results/pareto.json
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from statistics import median

from sidechannel import defense as D
from sidechannel.capacity import merge, mutual_information_bits, normalized_capacity
from sidechannel.harness import Scenario, run
from sidechannel.reconstruct import score

WORKLOAD = ["web_search", "db_query", "calc", "code_exec"]
RESULTS = Path(__file__).resolve().parent.parent / "results"
N_SEEDS = 40

DEFENSES = [
    D.NONE,
    D.SLOT_RESERVATION,
    D.cadence(4), D.cadence(8), D.cadence(16), D.cadence(32),
    D.noise(4), D.noise(8), D.noise(16), D.noise(32),
    D.block_cap(64),  # the wrong tool for a batch-slot channel; measured, not assumed
]


def evaluate(defense: D.Defense) -> dict:
    # Benign co-tenants that would use a freed slot for real work -- without them
    # a defense that merely denies the attacker the slot looks free, because
    # nothing legitimate wanted it. Their presence is what gives slot-reservation
    # a throughput cost and makes the frontier a real trade.
    base = Scenario(tool_sequence=WORKLOAD, num_benign=3)
    scores, benign, p99 = [], [], []
    for seed in range(N_SEEDS):
        r = run(replace(base, seed=seed), defense)
        scores.append(score(r.admit_steps, r.ground_truth))
        benign.append(r.benign_admissions)
        p99.append(r.p99_admission_latency)
    conf = merge(s.confusion for s in scores)
    return {
        "defense": defense.name,
        "bits_per_call": mutual_information_bits(conf),
        "normalized_capacity": normalized_capacity(conf),
        "tool_accuracy": sum(s.tool_accuracy for s in scores) / len(scores),
        "benign_throughput": median(benign),  # higher is better (useful work done)
        "p99_admission_latency": median(p99),  # lower is better
    }


def is_dominated(a: dict, rows: list[dict]) -> bool:
    """a is dominated if some b is at least as good on security AND both costs,
    and strictly better on at least one. Costs: less benign throughput is worse,
    more latency is worse."""
    for b in rows:
        if b is a:
            continue
        no_worse = (
            b["bits_per_call"] <= a["bits_per_call"]
            and b["benign_throughput"] >= a["benign_throughput"]
            and b["p99_admission_latency"] <= a["p99_admission_latency"]
        )
        strictly = (
            b["bits_per_call"] < a["bits_per_call"]
            or b["benign_throughput"] > a["benign_throughput"]
            or b["p99_admission_latency"] < a["p99_admission_latency"]
        )
        if no_worse and strictly:
            return True
    return False


def run_all() -> dict:
    rows = [evaluate(d) for d in DEFENSES]
    for r in rows:
        r["on_frontier"] = not is_dominated(r, rows)
    baseline = next(r for r in rows if r["defense"] == "undefended")
    return {"n_seeds": N_SEEDS, "baseline_bits": baseline["bits_per_call"], "defenses": rows}


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    data = run_all()
    (RESULTS / "pareto.json").write_text(json.dumps(data, indent=2))

    base_tput = next(r["benign_throughput"] for r in data["defenses"] if r["defense"] == "undefended")
    print(f"{'defense':>22} {'bits/call':>9} {'cap':>5} {'benign-tput':>11} "
          f"{'p99 lat':>8} {'frontier':>9}")
    print("-" * 74)
    for r in sorted(data["defenses"], key=lambda x: -x["bits_per_call"]):
        tput_pct = 100 * r["benign_throughput"] / base_tput if base_tput else 0
        print(f"{r['defense']:>22} {r['bits_per_call']:>9.2f} {r['normalized_capacity']:>5.2f} "
              f"{tput_pct:>10.0f}% {r['p99_admission_latency']:>8.0f} "
              f"{'  *' if r['on_frontier'] else '':>9}")
    print(f"\n* on the security/utility Pareto frontier. baseline leaks "
          f"{data['baseline_bits']:.2f} bits/call; benign-tput is % of undefended useful work.")
    print("block-cap is the wrong tool for a batch-slot channel: it leaves the leak")
    print("open (a memory-quota defense against a concurrency-slot channel), which the")
    print("numbers show rather than assume.")
    print(f"\nwrote {RESULTS / 'pareto.json'}")


if __name__ == "__main__":
    main()
