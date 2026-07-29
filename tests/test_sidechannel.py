"""The side channel, as a test: the leak is real and robust, cache isolation does
not close it, and the scheduler defenses do -- at a measured cost.

Deterministic and weightless (mock forward), so these run in milliseconds and the
numbers are exact, not statistical.
"""

from __future__ import annotations

import pytest

from sidechannel import defense as D
from sidechannel.harness import Scenario, run
from sidechannel.reconstruct import chance_accuracy, score

WORKLOAD = ["web_search", "db_query", "calc", "code_exec"]


# -- the leak is real and not a lucky seed -----------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 42])
def test_undefended_leak_recovers_the_program(seed):
    r = run(Scenario(tool_sequence=WORKLOAD, seed=seed), D.NONE)
    s = score(r.admit_steps, r.ground_truth)
    assert s.turn_count_exact, f"seed {seed}: recovered {s.n_recovered} of {s.n_truth} tool calls"
    assert s.tool_accuracy >= 0.75, f"seed {seed}: only {s.tool_accuracy:.0%} tools identified"
    assert s.duration_mae <= 2.0, f"seed {seed}: duration MAE {s.duration_mae}"


def test_attacker_only_ever_sees_its_own_admissions():
    """Sanity on the threat model: the observable is nothing but the attacker's
    own admission timestamps -- no privileged read of the victim."""
    r = run(Scenario(tool_sequence=WORKLOAD), D.NONE)
    assert r.admit_steps, "attacker recorded no admissions at all"
    assert all(isinstance(s, int) for s in r.admit_steps)


# -- the negative result: it is not a cache-content leak ---------------------


def test_cache_isolation_does_not_close_it():
    r = run(Scenario(tool_sequence=WORKLOAD), D.CACHE_ISOLATION)
    assert r.cross_tenant_shared_blocks == 0, "tenants must not share cache blocks"
    s = score(r.admit_steps, r.ground_truth)
    # Full cache isolation, yet the program is still fully recovered.
    assert s.tool_accuracy >= 0.75
    assert s.turn_count_exact


# -- the scheduler defenses close it, at a cost ------------------------------


def test_slot_reservation_starves_the_attacker():
    r = run(Scenario(tool_sequence=WORKLOAD), D.SLOT_RESERVATION)
    # No free slot ever appears, so the attacker is admitted zero times during the
    # victim's session and recovers nothing.
    assert r.attacker_admissions == 0
    assert score(r.admit_steps, r.ground_truth).tool_accuracy == 0.0
    # And it is not free: capacity is withheld while programs are paused.
    assert r.wasted_slot_steps > 0, "a defense with no cost is suspicious; measure it"


def test_coarse_cadence_destroys_tool_fingerprinting():
    """A cadence coarser than the tools' durations leaves the attacker unable to
    tell tools apart -- at best it can guess, i.e. chance."""
    fine = run(Scenario(tool_sequence=WORKLOAD), D.cadence(4))
    coarse = run(Scenario(tool_sequence=WORKLOAD), D.cadence(32))

    acc_coarse = score(coarse.admit_steps, coarse.ground_truth).tool_accuracy
    assert acc_coarse <= chance_accuracy() + 1e-9, f"coarse cadence still leaked {acc_coarse:.0%}"
    # And the cost is latency: coarser admission is slower admission.
    assert coarse.median_admission_latency > fine.median_admission_latency


# -- statistical characterization (H2) ---------------------------------------


def test_capacity_degrades_as_the_attacker_probes_slower():
    fast = run(Scenario(tool_sequence=WORKLOAD, probe_period=1), D.NONE)
    slow = run(Scenario(tool_sequence=WORKLOAD, probe_period=32), D.NONE)
    assert score(fast.admit_steps, fast.ground_truth).tool_accuracy == 1.0
    # Probing far slower than the tool durations, the attacker resolves nothing.
    assert score(slow.admit_steps, slow.ground_truth).tool_accuracy <= chance_accuracy() + 1e-9


def test_capacity_degrades_under_slot_contention():
    alone = run(Scenario(tool_sequence=WORKLOAD, num_benign=0), D.NONE)
    crowded = run(Scenario(tool_sequence=WORKLOAD, num_benign=8), D.NONE)
    assert score(alone.admit_steps, alone.ground_truth).tool_accuracy > (
        score(crowded.admit_steps, crowded.ground_truth).tool_accuracy
    )


def test_timing_jitter_reduces_admission_precision():
    clean = run(Scenario(tool_sequence=WORKLOAD, admit_jitter=0), D.NONE)
    noisy = run(Scenario(tool_sequence=WORKLOAD, admit_jitter=16), D.NONE)
    assert score(clean.admit_steps, clean.ground_truth).tool_accuracy == 1.0
    assert score(noisy.admit_steps, noisy.ground_truth).tool_accuracy <= chance_accuracy() + 1e-9


def test_defense_is_a_tradeoff_not_a_free_lunch():
    """Undefended leaks fully and admits fast; the defense that closes it makes
    the attacker (and every tenant) wait or lose capacity. Pin the direction."""
    base = run(Scenario(tool_sequence=WORKLOAD), D.NONE)
    slots = run(Scenario(tool_sequence=WORKLOAD), D.SLOT_RESERVATION)

    assert score(base.admit_steps, base.ground_truth).tool_accuracy == 1.0
    assert score(slots.admit_steps, slots.ground_truth).tool_accuracy == 0.0
    assert slots.wasted_slot_steps > 0
