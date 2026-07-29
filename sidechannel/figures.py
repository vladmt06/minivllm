"""Render the report figures from results/*.json.

Matplotlib PNGs, so the artifact is portable and the report renders anywhere. The
palette is Okabe-Ito -- a published colorblind-safe categorical set -- so identity
never rides on a hue a CVD reader cannot separate. Forms follow the data's job:
degradation is change-over-x (lines), the confusion is identity-vs-identity (a
matrix), the Pareto is a trade (scatter), realtime is events-over-time.

    uv run python -m sidechannel.figures      # reads results/, writes results/*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Okabe-Ito colorblind-safe categorical palette (fixed order, never cycled).
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]
INK, MUTED, GRID = "#1a1a1a", "#666666", "#dddddd"


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for lbl in (ax.xaxis.label, ax.yaxis.label):
        lbl.set_color(INK)
        lbl.set_fontsize(10)
    ax.title.set_color(INK)
    ax.title.set_fontsize(11)


def _load(name):
    return json.loads((RESULTS / name).read_text())


# -- degradation: capacity vs each sweep axis --------------------------------


def fig_degradation(sweeps: dict) -> None:
    axes_spec = [
        ("probe_period", "probe period (steps between probes)", "attacker probes slower →"),
        ("admit_jitter", "timing jitter (± steps)", "noisier attacker clock →"),
        ("num_benign", "competing benign tenants", "more slot contention →"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    maxbits = sweeps["max_bits"]
    for ax, (field, xlabel, sub) in zip(axs, axes_spec):
        rows = sweeps["sweeps"][field]
        xs = [r[field] for r in rows]
        bits = [r["bits_per_call"] for r in rows]
        acc = [r["tool_accuracy_median"] for r in rows]
        ax.axhline(maxbits, color=MUTED, lw=1, ls=":", label="max (log₂ 4)")
        ax.plot(xs, bits, "-o", color=OKABE[0], lw=2, ms=5, label="bits / tool call")
        ax.plot(xs, [a * maxbits for a in acc], "-s", color=OKABE[1], lw=2, ms=4,
                label="tool accuracy × max")
        ax.set_xlabel(f"{xlabel}\n{sub}")
        ax.set_ylim(-0.1, maxbits + 0.2)
        _style(ax)
    axs[0].set_ylabel("channel capacity (bits)")
    axs[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("The leak degrades gracefully on every axis, with a floor",
                 color=INK, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_degradation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# -- confusion matrix --------------------------------------------------------


def fig_confusion(sweeps: dict) -> None:
    tools = sweeps["workload"]
    idx = {t: i for i, t in enumerate(tools)}
    n = len(tools)
    mat = [[0] * n for _ in range(n)]
    for key, c in sweeps["baseline_confusion"].items():
        t, g = key.split("->")
        mat[idx[t]][idx[g]] = c
    row_tot = [sum(r) or 1 for r in mat]
    norm = [[mat[i][j] / row_tot[i] for j in range(n)] for i in range(n)]

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(tools, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(tools, fontsize=8)
    ax.set_xlabel("attacker's guess"); ax.set_ylabel("true tool")
    for i in range(n):
        for j in range(n):
            v = norm[i][j]
            if v > 0.01:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        color="white" if v > 0.5 else INK, fontsize=8)
    ax.set_title("Baseline tool confusion (row-normalised)")
    ax.title.set_color(INK); ax.title.set_fontsize(11)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# -- defense Pareto ----------------------------------------------------------


def fig_pareto(pareto: dict) -> None:
    rows = pareto["defenses"]
    base_tput = next(r["benign_throughput"] for r in rows if r["defense"] == "undefended") or 1
    pts = []
    for r in rows:
        pts.append((100 * r["benign_throughput"] / base_tput, r["bits_per_call"],
                    r["defense"].replace("admission-", ""), r["on_frontier"]))

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for x, y, _, on in pts:
        ax.scatter(x, y, s=110 if on else 45,
                   color=OKABE[3] if on else MUTED,
                   edgecolor=INK if on else "none", zorder=4 if on else 2,
                   marker="D" if on else "o")

    # Stagger labels vertically with leader lines so coincident points stay legible.
    # Fan each y-cluster (the bits≈0 defenses, and the bits≈1.2 no-op corner).
    order = sorted(range(len(pts)), key=lambda i: (round(pts[i][1], 1), pts[i][0]))
    low_slot = high_slot = 0
    for i in order:
        x, y, name, on = pts[i]
        if y < 0.3:
            dy = 14 + 16 * (low_slot % 4)
            low_slot += 1
        else:
            dy = 10 + 16 * (high_slot % 2)
            high_slot += 1
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(8, dy),
                    fontsize=7.5, color=INK if on else MUTED,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.7))
    ax.set_xlabel("benign tenant throughput (% of undefended)  → better")
    ax.set_ylabel("channel capacity leaked (bits / call)  ↓ better")
    ax.set_title("Defense trade-off: security vs utility  (diamonds = Pareto frontier)")
    _style(ax)
    ax.set_xlim(-6, 112)
    ax.set_ylim(-0.15, pareto["baseline_bits"] + 0.35)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_pareto.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# -- realtime timeline (the wall-clock money shot) ---------------------------


def fig_realtime(rt: dict) -> None:
    tools = rt["tools"]
    palette = {name: OKABE[i % len(OKABE)] for i, name in enumerate(sorted(tools))}
    fig, ax = plt.subplots(figsize=(9, 2.8))
    for name, s, e in rt["truth"]:
        ax.axvspan(s / 1000, e / 1000, color=palette[name], alpha=0.25)
        ax.text((s + e) / 2000, 1.06, name, ha="center", va="bottom", fontsize=8,
                color=palette[name])
    admits = [a / 1000 for a in rt["admits_ms"]]
    ax.plot(admits, [0.5] * len(admits), "|", color=INK, ms=14, mew=1.2)
    ax.set_yticks([])
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("wall-clock time (s)")
    ax.set_title("Real TinyLlama on MPS: attacker admissions (ticks) cluster inside "
                 "the victim's tool-call pauses (shaded)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(MUTED); ax.tick_params(colors=MUTED, labelsize=9)
    ax.title.set_color(INK); ax.title.set_fontsize(10); ax.xaxis.label.set_color(INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_realtime.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# -- multi-victim breakpoint -------------------------------------------------


def fig_multivictim(mv: dict) -> None:
    rows = mv["rows"]
    xs = [r["n_victims"] for r in rows]
    acc = [r["attribution_accuracy"] for r in rows]
    chance = mv["chance"]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.axhline(chance, color=MUTED, ls=":", lw=1, label="chance")
    ax.plot(xs, acc, "-o", color=OKABE[0], lw=2, ms=6, label="per-victim tool accuracy")
    for x, a in zip(xs, acc):
        ax.annotate(f"{a:.0%}", (x, a), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs)
    ax.set_xlabel("concurrent victims")
    ax.set_ylabel("per-victim attribution accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Separation breaks with concurrency\n(single-prober attacker)")
    _style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_multivictim.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_all() -> list[str]:
    made = []
    if (RESULTS / "sweeps.json").exists():
        s = _load("sweeps.json")
        fig_degradation(s); fig_confusion(s)
        made += ["fig_degradation.png", "fig_confusion.png"]
    if (RESULTS / "pareto.json").exists():
        fig_pareto(_load("pareto.json")); made.append("fig_pareto.png")
    if (RESULTS / "realtime.json").exists():
        fig_realtime(_load("realtime.json")); made.append("fig_realtime.png")
    if (RESULTS / "multivictim.json").exists():
        fig_multivictim(_load("multivictim.json")); made.append("fig_multivictim.png")
    return made


def main() -> None:
    made = render_all()
    print("rendered:", ", ".join(made) if made else "nothing (run the experiments first)")


if __name__ == "__main__":
    main()
