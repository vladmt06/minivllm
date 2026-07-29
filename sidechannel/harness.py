"""Run one victim-vs-attacker scenario and collect observables + utility.

Deterministic and weightless: background fillers hold all but one concurrency
slot, the victim takes the last slot while acting and vacates it during tool
pauses, and the attacker probes admission. Everything the attacker gets is in
Result.admit_steps; everything the defender pays is in the utility fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from minivllm.config import CacheConfig, SchedulerConfig
from minivllm.core.program import Program, ProgramRunner, TurnSpec
from minivllm.core.scheduler import Scheduler
from sidechannel.attacker import ClosedLoopProber
from sidechannel.defense import NONE, Defense
from sidechannel.victim import build_victim


@dataclass
class Scenario:
    tool_sequence: list[str]
    num_blocks: int = 4000  # large: batch slots, not memory, are the binding constraint
    block_size: int = 16
    max_num_seqs: int = 8
    gen_len: int = 25
    steps: int = 800
    warmup: int = 20
    seed: int = 1
    kv_ttl_steps: int = 10_000

    # Sweep knobs for the statistical characterization (H2).
    probe_period: int = 1  # steps between the attacker's probes; larger = slower prober
    admit_jitter: int = 0  # +/- steps of measurement noise added to observed admissions
    num_benign: int = 0  # benign tenants also grabbing freed slots (contention)


@dataclass
class Result:
    defense: str
    admit_steps: list[int]
    ground_truth: list[tuple[str, int, int]]
    victim_jct: int
    wasted_slot_steps: int  # capacity withheld from other tenants (slot defense)
    latencies: list[int]  # attacker probe admission latency (= what any tenant pays)
    cross_tenant_shared_blocks: int
    total_steps: int

    @property
    def attacker_admissions(self) -> int:
        return len(self.admit_steps)

    @property
    def median_admission_latency(self) -> float:
        return median(self.latencies) if self.latencies else float("nan")


def run(scenario: Scenario, defense: Defense = NONE) -> Result:
    cfg = SchedulerConfig(
        program_aware=True,
        kv_ttl_steps=scenario.kv_ttl_steps,
        max_num_seqs=scenario.max_num_seqs,
        max_num_batched_tokens=999_999,
        **defense.kwargs,
    )
    sched = Scheduler(
        CacheConfig(block_size=scenario.block_size, num_blocks=scenario.num_blocks, watermark=0.0),
        cfg,
        scenario.num_blocks,
    )
    runner = ProgramRunner(sched, seed=scenario.seed)

    # Background load: fill all but one slot with effectively-endless tenants.
    for i in range(scenario.max_num_seqs - 1):
        runner.submit(Program([TurnSpec(10**7)], prompt_len=16, arrival=0.0, tenant_id=100 + i))
    for _ in range(scenario.warmup):
        runner.step()

    victim = build_victim(scenario.tool_sequence, gen_len=scenario.gen_len)
    runner.submit(victim)
    victim_submit = sched.step_counter

    prober = ClosedLoopProber(runner, probe_period=scenario.probe_period)
    prober.start()
    # Benign co-tenants also compete for a freed slot, so the attacker wins only a
    # fraction of the pause windows -- realistic contention, and a sweep axis.
    benign = [ClosedLoopProber(runner, tenant_id=200 + i) for i in range(scenario.num_benign)]
    for b in benign:
        b.start()

    wasted = 0
    victim_done: int | None = None
    for _ in range(scenario.steps):
        runner.step()
        prober.observe()
        for b in benign:
            b.observe()
        if sched.reserve_slots_on_suspend:
            # Slots held for paused programs while the attacker is demanding one:
            # capacity that could have been served but was withheld.
            wasted += len(sched.suspended)
        if victim.state.name == "DONE" and not sched.suspended and victim_done is None:
            victim_done = sched.step_counter

    victim_jct = (victim_done if victim_done is not None else sched.step_counter) - victim_submit
    shared = sum(
        1 for b in range(sched.allocator.num_blocks) if sched.allocator.refcount(b) > 1
    )
    # Evaluate leakage over the victim's session only. After it departs its slot
    # frees permanently and the attacker just floods an empty seat -- that says
    # nothing about the program and would flatter or distort every defense. The
    # session boundary is the operator's ground truth, not attacker oracle input.
    end = victim_done if victim_done is not None else sched.step_counter
    session_admits = [s for s in prober.admit_steps if s <= end]
    if scenario.admit_jitter:
        # The attacker's own timing is noisy. Perturb each observed admission and
        # re-sort, modelling measurement jitter without needing the wall clock.
        import random as _random

        rng = _random.Random(scenario.seed ^ 0x5A5A)
        j = scenario.admit_jitter
        session_admits = sorted(s + rng.randint(-j, j) for s in session_admits)
    return Result(
        defense=defense.name,
        admit_steps=session_admits,
        ground_truth=list(victim.ground_truth),
        victim_jct=victim_jct,
        wasted_slot_steps=wasted,
        latencies=list(prober.latencies),
        cross_tenant_shared_blocks=shared,
        total_steps=sched.step_counter,
    )
