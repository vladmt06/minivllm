"""End-to-end: the leak, the negative result, and the defenses with their cost.

    uv run python -m sidechannel.run_experiment
"""

from __future__ import annotations

from sidechannel import defense as D
from sidechannel.harness import Scenario, run
from sidechannel.reconstruct import chance_accuracy, classify, find_bursts, score

# A five-turn agent: four tool calls the attacker will try to recover blind.
VICTIM_TOOLS = ["web_search", "db_query", "calc", "code_exec"]


def rule(title: str = "", w: int = 74) -> None:
    print(f"\n{title}\n{'-' * w}" if title else "-" * w)


def show_reconstruction(r) -> None:
    s = score(r.admit_steps, r.ground_truth)
    bursts = find_bursts(r.admit_steps)
    print(f"  ground truth : {[t for t, _, _ in r.ground_truth]}")
    print(f"  recovered    : {[classify(b.width) for b in bursts]}")
    print(f"  {s.summary()}")


def main() -> None:
    sc = Scenario(tool_sequence=VICTIM_TOOLS)

    rule("1. THE LEAK  (program-aware serving, no defense)")
    base = run(sc, D.NONE)
    s = score(base.admit_steps, base.ground_truth)
    print(f"attacker saw only its own {base.attacker_admissions} admission timestamps.")
    show_reconstruction(base)

    rule("2. NEGATIVE RESULT  (user-level cache isolation)")
    print("minivllm shares no KV cache across tenants, so cache isolation is already")
    print("fully in force. If the leak were about cache content, it would be closed.")
    iso = run(sc, D.CACHE_ISOLATION)
    s_iso = score(iso.admit_steps, iso.ground_truth)
    print(f"cross-tenant shared cache blocks: {iso.cross_tenant_shared_blocks}  (zero = full isolation)")
    print(f"reconstruction still: {s_iso.summary()}")
    print("=> the leak is NOT cache content. It is scheduler timing.")

    rule("3. SCHEDULER DEFENSES  (and what they cost)")
    chance = chance_accuracy()
    print(f"chance-level tool accuracy = {100 * chance:.0f}%  (4 tools)\n")
    print(f"{'defense':>22} {'tool acc':>9} {'turns':>7} {'dur MAE':>8} {'atk adm':>8} {'cost':>22}")
    rule()

    def row(res, cost):
        s = score(res.admit_steps, res.ground_truth)
        turns = f"{s.n_recovered}/{s.n_truth}"
        print(
            f"{res.defense:>22} {100 * s.tool_accuracy:>7.0f}% {turns:>7} "
            f"{s.duration_mae:>8.1f} {res.attacker_admissions:>8} {cost:>22}"
        )

    row(base, "baseline")
    slots = run(sc, D.SLOT_RESERVATION)
    row(slots, f"{slots.wasted_slot_steps} idle slot-steps")
    for period in (4, 8, 16, 32):
        res = run(sc, D.cadence(period))
        row(res, f"p50 admit {res.median_admission_latency:.0f} steps")

    rule("verdict")
    print("slot-reservation closes the channel outright (attacker starved of the")
    print("free slot), paying a fixed idle-capacity cost. admission-cadence trades")
    print("temporal resolution for latency: as the period grows past a tool's")
    print("duration, the attacker can no longer separate tool calls -- at the cost")
    print("of every tenant's admission latency. Both live in the scheduler, which")
    print("is the only place this leak can be closed.")


if __name__ == "__main__":
    main()
