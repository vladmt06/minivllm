"""Defenses, as scheduler configuration.

The point of the experiment is that none of these are cache defenses -- they are
all scheduler defenses, because the leak is in the scheduler. Each carries a
utility cost the harness measures, so "closed the channel" is always paired with
"and here is what it cost."
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Defense:
    name: str
    kwargs: dict = field(default_factory=dict)  # extra SchedulerConfig fields


NONE = Defense("undefended")

# The standard content-level mitigation, for the negative result. minivllm shares
# no KV across tenants already, so "user-level cache isolation" is the undefended
# system -- same config, relabelled to make the point explicit in output.
CACHE_ISOLATION = Defense("cache-isolation (content only)")

# Scheduler-level defenses.
SLOT_RESERVATION = Defense("slot-reservation", {"reserve_slots_on_suspend": True})


def cadence(period: int) -> Defense:
    return Defense(f"admission-cadence-{period}", {"admission_period": period})


def block_cap(n: int) -> Defense:
    return Defense(f"block-cap-{n}", {"reserved_blocks_per_tenant": n})
