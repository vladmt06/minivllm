"""H3: can the attacker separate several co-resident victims?

A single admission prober yields the *union* of every victim's pause windows.
When victims have disjoint tool-duration signatures and their pauses do not
overlap, each burst belongs to one victim and is attributed by matching its width
to that victim's tools. Two things defeat this, and both are the honest point of
the experiment: overlapping pauses merge into one burst (miscount), and shared
tool durations make attribution ambiguous.

So this partly works, and degrades with overlap. main() measures the breakpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

from minivllm.config import CacheConfig, SchedulerConfig
from minivllm.core.program import Program, ProgramRunner, Tool, TurnSpec
from minivllm.core.scheduler import Scheduler
from sidechannel.attacker import ClosedLoopProber
from sidechannel.reconstruct import classify, session_bursts
from sidechannel.victim import TOOL_TAXONOMY, build_victim

BLOCK = 16


def _union_taxonomy(tool_lists: list[list[str]]) -> dict[str, Tool]:
    names = {n for lst in tool_lists for n in lst}
    return {n: TOOL_TAXONOMY[n] for n in names}


def _owner_map(tool_lists: list[list[str]]) -> dict[str, int]:
    owner: dict[str, int] = {}
    for vid, lst in enumerate(tool_lists):
        for n in lst:
            owner.setdefault(n, vid)  # first owner wins on a shared tool (a real limit)
    return owner


@dataclass
class VictimScore:
    truth: list[str]
    recovered: list[str]
    turn_count_exact: bool
    tool_accuracy: float


def run_multivictim(
    tool_lists: list[list[str]],
    stagger: int = 0,  # steps between victim submissions; larger = less pause overlap
    max_num_seqs: int = 8,
    gen_len: int = 20,
    steps: int = 2500,
    seed: int = 0,
) -> dict:
    n = len(tool_lists)
    sched = Scheduler(
        CacheConfig(block_size=BLOCK, num_blocks=8000, watermark=0.0),
        SchedulerConfig(program_aware=True, kv_ttl_steps=10**9, max_num_seqs=max_num_seqs,
                        max_num_batched_tokens=999_999),
        8000,
    )
    runner = ProgramRunner(sched, seed=seed)
    for i in range(max_num_seqs - n):
        runner.submit(Program([TurnSpec(10**7)], prompt_len=16, arrival=0.0, tenant_id=100 + i))
    for _ in range(20):
        runner.step()

    prober = ClosedLoopProber(runner)
    prober.start()

    victims: list[Program] = []
    to_submit = list(enumerate(tool_lists))
    for _ in range(steps):
        # Stagger victim arrivals in real step-time so their pause phases differ.
        while to_submit and to_submit[0][0] * stagger <= (sched.step_counter - 20):
            vid, tools = to_submit.pop(0)
            v = build_victim(tools, gen_len=gen_len, arrival=0.1 + 0.001 * vid, tenant_id=1000 + vid)
            runner.submit(v)
            victims.append(v)
        runner.step()
        prober.observe()
        if victims and len(victims) == n and all(v.state.name == "DONE" for v in victims):
            break

    taxonomy = _union_taxonomy(tool_lists)
    owner = _owner_map(tool_lists)
    bursts = session_bursts(prober.admit_steps, taxonomy=taxonomy)

    recovered: dict[int, list[str]] = {vid: [] for vid in range(n)}
    for b in bursts:
        tool = classify(b.width, taxonomy)
        recovered[owner[tool]].append(tool)

    scores = {}
    for vid, v in enumerate(victims):
        truth = [t for t, _, _ in v.ground_truth]
        got = recovered[vid]
        correct = sum(1 for a, b in zip(truth, got) if a == b)
        scores[vid] = VictimScore(truth, got, len(got) == len(truth),
                                  correct / len(truth) if truth else 0.0)
    total_truth = sum(len(v.ground_truth) for v in victims)
    return {
        "scores": scores,
        "n_bursts": len(bursts),
        "total_truth_calls": total_truth,
        "union_count_error": abs(len(bursts) - total_truth),
    }


def _mean_acc(res) -> float:
    s = res["scores"]
    return sum(v.tool_accuracy for v in s.values()) / len(s) if s else 0.0


def summarize(specs, seeds=range(6)) -> dict:
    """Aggregate over seeds: per-victim attribution accuracy vs the number of
    concurrent victims, and the aggregate (union) tool-call count accuracy."""
    rows = []
    for spec in specs:
        accs, union_ok = [], []
        for s in seeds:
            r = run_multivictim(spec, stagger=0, seed=s)
            accs.append(_mean_acc(r))
            union_ok.append(r["union_count_error"] <= 1)
        rows.append({
            "n_victims": len(spec),
            "attribution_accuracy": sum(accs) / len(accs),
            "union_count_within_1_rate": sum(union_ok) / len(union_ok),
        })
    return {"rows": rows}


def main() -> None:
    A, B, C = ["web_search", "db_query"], ["calc", "code_exec"], ["web_search", "code_exec"]
    chance = 1 / len(TOOL_TAXONOMY)

    print("A single admission prober yields the UNION of all victims' pauses. The")
    print("question is whether the attacker can attribute pauses to individual")
    print(f"victims. Chance tool accuracy = {chance:.0%}.\n")

    print("== 2 synchronised victims (one recovered stream, two owners) ==")
    r = run_multivictim([A, B], stagger=0, seed=1)
    for vid, s in r["scores"].items():
        print(f"  victim {vid}: truth {s.truth} -> attributed {s.recovered}  {s.tool_accuracy:.0%}")
    print(f"  aggregate: recovered {r['n_bursts']} pauses vs {r['total_truth_calls']} true "
          f"(off by {r['union_count_error']})")

    print("\n== the breakpoint: attribution vs concurrent-victim count ==")
    data = summarize([[A], [A, B], [A, B, C]])
    print(f"  {'victims':>8} {'per-victim tool acc':>20} {'union count OK':>15}")
    for row in data["rows"]:
        print(f"  {row['n_victims']:>8} {row['attribution_accuracy']:>19.0%} "
              f"{row['union_count_within_1_rate']:>14.0%}")
    print(f"\n  One victim: fully recovered. Two or more synchronised victims: attribution")
    print(f"  collapses toward chance ({chance:.0%}) -- the single prober cannot tell whose")
    print(f"  pause it saw. It still counts aggregate activity. Per-victim separation with a")
    print(f"  slot-count prober is left as future work; this documents the honest limit.")


if __name__ == "__main__":
    main()
