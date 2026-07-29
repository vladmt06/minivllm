"""The attacker tenant: a closed-loop admission prober.

The attacker's entire capability is submitting its own requests and timing when
they start running -- exactly what any co-resident tenant can do, no privilege.
It keeps one probe outstanding at a time (a real prober, not an unbounded flood
that would just measure its own queue). A probe with gen_len 1 vacates its slot
almost immediately, so a D-step pause admits ~D probes: the burst width is the
tool-call duration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from minivllm.core.program import Program, ProgramRunner, TurnSpec

ATTACKER_TENANT = 2


@dataclass
class ClosedLoopProber:
    runner: ProgramRunner
    tenant_id: int = ATTACKER_TENANT
    probe_period: int = 1  # steps to wait after an admission before re-arming
    admit_steps: list[int] = field(default_factory=list)  # the only observable
    latencies: list[int] = field(default_factory=list)
    _submit_step: int = 0
    _outstanding: int | None = None
    _rearm_at: int = 0

    def _arm(self) -> None:
        step = self.runner.sched.step_counter
        p = Program(
            turns=[TurnSpec(gen_len=1)],
            prompt_len=16,
            arrival=1e6 + step,  # lowest priority: the attacker never jumps a real tenant
            tenant_id=self.tenant_id,
        )
        self.runner.submit(p)
        assert p.seq is not None
        self._outstanding = p.seq.seq_id
        self._submit_step = step

    def start(self) -> None:
        self._arm()

    def observe(self) -> None:
        """Call once per engine step, after runner.step(). Records an admission
        the step it happens; re-arms after probe_period steps (probe_period=1 is
        the maximal-rate attacker, larger periods model a slower prober)."""
        step = self.runner.sched.step_counter
        if self._outstanding is None:
            if step >= self._rearm_at:
                self._arm()
            return
        run = self.runner.first_run_step.get(self._outstanding)
        if run == step:
            self.admit_steps.append(run)
            self.latencies.append(run - self._submit_step)
            self._outstanding = None
            self._rearm_at = step + self.probe_period
