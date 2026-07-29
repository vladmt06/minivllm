"""Regenerate every result and figure in the report, from scratch, seeded.

    uv run python -m sidechannel.run_all

Writes results/{sweeps,pareto,multivictim,realtime}.json and results/*.png. The
deterministic experiments are exact given the seeds; the one real-model run
(realtime.json) carries wall-clock noise by design and will vary slightly.
"""

from __future__ import annotations

import json
from pathlib import Path

from sidechannel import figures, multivictim, pareto, sweeps

RESULTS = Path(__file__).resolve().parent.parent / "results"

A, B, C = ["web_search", "db_query"], ["calc", "code_exec"], ["web_search", "code_exec"]


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    print("[1/5] statistical sweeps (deterministic) ...")
    (RESULTS / "sweeps.json").write_text(json.dumps(sweeps.run_all(), indent=2))

    print("[2/5] defense Pareto (deterministic) ...")
    (RESULTS / "pareto.json").write_text(json.dumps(pareto.run_all(), indent=2))

    print("[3/5] multi-victim breakpoint (deterministic) ...")
    mv = multivictim.summarize([[A], [A, B], [A, B, C]])
    (RESULTS / "multivictim.json").write_text(json.dumps(mv, indent=2))

    print("[4/5] real-model wall-clock run (threaded, ~20 s) ...")
    from sidechannel.realtime import capture

    (RESULTS / "realtime.json").write_text(json.dumps(capture(A + B), indent=2))

    print("[5/5] rendering figures ...")
    made = figures.render_all()
    print("done. figures:", ", ".join(made))
    print(f"all outputs in {RESULTS}")


if __name__ == "__main__":
    main()
