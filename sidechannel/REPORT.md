# A scheduler-state timing side channel in program-aware LLM serving

**A defensive-security study on a reference engine (minivllm).** Two synthetic
tenants on one machine; no external target, no real victim. All numbers reproduce
with `uv run python -m sidechannel.run_all`.

## 1. Summary

Program-aware agentic serving — a scheduler that knows a request belongs to a
multi-turn *program*, pins its KV cache across tool-call pauses, and prioritises
at program granularity ([Continuum](https://arxiv.org/abs/2511.02230)) — opens a
timing side channel that no published defense addresses. A co-resident attacker
who can only time **its own** admission latency reconstructs a victim program's
private control flow: how many turns it ran, when it called a tool, and *which*
tool, from the pause duration alone.

The leak is **not** in the KV cache. It is in the scheduler. Every documented
LLM-serving timing attack to date leaks cache *content* (a prefix cache-hit
speeds up TTFT), and every deployed defense protects content. We show a channel
that is fully open under complete cache isolation, because the signal is the
scheduler's admission timing, not any cached bytes.

Headline numbers (this engine, four-tool agent):

- **Undefended capacity 1.84 bits per tool call** out of a 2-bit maximum
  (log₂ 4) — 0.92 of the channel — with 100% tool identification and exact turn
  counts, from admission timestamps alone.
- **The channel survives real wall-clock timing**: on a live TinyLlama forward on
  Apple MPS, with the attacker on a separate thread, tool identification stays at
  100% and pause-duration error is ~23 ms against a ~57 ms serving step
  (`fig_realtime.png`).
- **A scheduler-level defense closes it at a stated cost.** Randomised admission
  delay drives capacity to 0 bits while preserving 74% of benign-tenant
  throughput — dominating both slot-reservation (0% throughput) and admission
  cadence (down to 5%).

## 2. Threat model

- **Setting.** A shared serving replica runs program-aware scheduling: program
  IDs, program-level FCFS, and KV cache pinned with a TTL during tool-call pauses.
  Multiple tenants are co-resident on the one replica.
- **Attacker.** An ordinary tenant. It submits its own requests and times its own
  admission latency (submit → first token). It has **no** privileged access, sees
  no other tenant's tokens, and shares **no** cache with anyone. It knows the
  public tool taxonomy (tool → typical duration), a realistic assumption for a
  known agent product.
- **Victim.** Another tenant running a multi-turn agent program: generate, call a
  tool (pause), generate, call another tool, … The tool identities, durations,
  and turn count are private.
- **Goal.** Reconstruct the victim's program timeline — turn count, tool-call
  timing, and per-call tool identity — from the attacker's admission latency.

## 3. Mechanism

The binding resource is the concurrency limit (`max_num_seqs`), the standard hard
cap on sequences in the running batch. Program-aware serving **suspends** a
program during a tool call: it leaves the running batch (its slot frees) while its
KV cache stays pinned in memory. So:

> victim actively generating ⟺ batch full ⟺ attacker's probe waits
> victim paused for a tool call ⟺ one slot free ⟺ attacker's probe is admitted

The attacker runs a closed-loop prober: submit a one-token request, time its
admission, repeat. Its admissions therefore **burst** exactly during the victim's
tool-call pauses. Each burst's **width is the tool-call duration**, which
fingerprints the tool. Between bursts, the silence is the victim generating.
`fig_realtime.png` shows this directly: admission ticks fall inside the shaded
true tool-call windows, and the burst widths track the tool durations.

Reconstruction (`reconstruct.py`): group admissions into bursts (the grouping gap
is learned from the attacker's own inter-admission spacing, so it is robust to a
noisy clock and to the cadence defense); classify each burst by nearest tool
duration; count bursts for turns. The session is bounded at the victim's
departure — after it leaves, its slot frees permanently and the attacker just
floods an empty seat, which says nothing about the program.

## 4. The leak is not cache content (negative result)

minivllm shares no KV blocks across tenants: cross-tenant shared-block count is
**0** in every run. User-level cache isolation — the standard mitigation, the
thing PrefixWall and SafeKV implement — is therefore already fully in force, and
the channel is still wide open at 100% tool accuracy. This is the crux: a defense
that scrubs, tags, or partitions *cache contents* cannot touch a leak carried by
*admission timing*.

## 5. How much leaks, and when it dies (`fig_degradation.png`, `fig_confusion.png`)

Capacity is the mutual information between the victim's true tool and the
attacker's guess, in bits per tool call (`capacity.py`, pinned by
`tests/test_capacity.py`: a perfect channel is 2.0 bits, a blind guesser 0.0).
Aggregated over 40 seeds, the channel degrades gracefully with a floor on every
axis:

- **Probe rate.** At full rate, 1.84 bits. It falls to 0 once the attacker probes
  slower (period ≥ 16 steps) than the tool durations it is trying to resolve.
- **Timing noise.** Robust to a few steps of jitter; gone by ±16 steps. (Bits
  carries the usual small-sample positive bias at the noisy end; tool accuracy and
  turn-exact rate, the clean monotone metrics, go to zero.)
- **Contention.** With 8 benign tenants also grabbing the freed slot, capacity
  falls to 0.63 bits — the attacker wins only a fraction of the pause windows.

The confusion matrix (`fig_confusion.png`) shows which tools blur: three of the
four tools are identified perfectly, and the residual error is concentrated in the
shortest tool (web_search), which is the most sensitive to burst-boundary noise
and is occasionally over-measured.

## 6. Multi-victim separation breaks (`fig_multivictim.png`)

A single prober yields the *union* of all victims' pauses. Attributing pauses to
individual victims by their tool-duration signatures works perfectly for one
victim (100%) but collapses toward chance (25% for four tools) with two or more
synchronised victims: **100% → 29% → 19%** for one, two, three concurrent
victims. The attacker still counts aggregate tool-call activity, but cannot say
whose. Per-victim separation with a free-slot-*count* prober (measuring how many
slots are simultaneously free) is left as future work; this is the honest limit of
the simple attacker.

## 7. Defenses and their cost (`fig_pareto.png`)

Because the leak is in the scheduler, so is the fix. We evaluate four defenses on
one plane — capacity leaked vs. two real costs: benign-tenant throughput withheld,
and p99 admission latency added (`pareto.py`, 40 seeds, with benign co-tenants
present so a defense that only denies the attacker a slot is not scored as free).

| Defense | bits/call | benign throughput | p99 latency | verdict |
|---|---|---|---|---|
| none / block-cap | 1.21 | 100% | baseline | block-cap is a memory-quota defense; it does nothing to a slot channel |
| **admission noise** | **0.00** | **74%** | ~baseline | **best trade — the frontier choice** |
| slot reservation | 0.00 | 0% | lowest | closes it, but starves benign tenants of the freed slot |
| admission cadence | 0.00 | 5–21% | up to ~9× | closes it, but wrecks throughput *and* latency |

The recommendation falls out of the numbers rather than taste: **randomised
admission delay** decouples a request's start time from the live pin state,
closing the channel while preserving most legitimate throughput. Reserving the
slot or coarsening admission also close it, but at much higher utility cost.
Block-cap is included to show, not assume, that a defense aimed at the wrong
resource (memory, not concurrency slots) leaves the channel fully open.

## 8. Related work

- **Cache-content timing attacks** — *The Early Bird Catches the Leak*
  ([arXiv:2409.20002](https://arxiv.org/pdf/2409.20002)), InputSnatch. All leak
  *what was cached* via prefix hit-vs-miss. We verified Early Bird presents no
  scheduler-state attack and no agentic scheduling.
- **Cache-content defenses** — PrefixWall
  ([arXiv:2603.10726](https://arxiv.org/html/2603.10726v2)) tags blocks with an
  owner; SafeKV isolates sensitive prompts; cache salting namespaces the cache.
  All protect *content*; none address scheduler timing.
- **Program-aware serving** — Continuum
  ([arXiv:2511.02230](https://arxiv.org/abs/2511.02230)): program-level FCFS + KV
  TTL-pinning across tool calls. This is the serving design whose scheduler state
  the channel reads. The efficiency mechanism and the leak are the same mechanism.

To our knowledge the scheduler-state channel introduced by program-aware serving
is unattacked in the literature.

## 9. Limitations

- **Reference engine, not production vLLM.** This demonstrates the mechanism and
  the defense on a faithful but small serving engine we control. It is not a break
  of a deployed system; real-vLLM replication is future work.
- **Determinism is the instrument, not the claim.** The statistics run on a
  deterministic mock forward for exactness and volume; §1's wall-clock result on a
  real TinyLlama forward is what shows the channel is not a determinism artifact.
- **Single-victim attacker.** Multi-victim attribution is weak (§6).
- **Known taxonomy.** The attacker is assumed to know tool→duration. Without it,
  it still recovers turn counts and pause timing, just not tool labels.

## 10. Responsible framing

This is a defensive characterisation of a serving-architecture design, run
end-to-end on the author's own machine with synthetic tenants. It targets no
deployed service and exfiltrates no real user's data. The contribution is the
mechanism, its quantification, and — the point of publishing it — a scheduler-level
defense that closes it at a measured, modest cost.

## 11. Reproduce

```bash
uv run python -m sidechannel.run_all         # all JSON + all figures (~1 min)
uv run python -m sidechannel.realtime        # the live wall-clock run on its own
uv run pytest tests/test_sidechannel.py tests/test_capacity.py tests/test_program.py -q
```

Figures land in `results/`. Program-aware scheduling is a `SchedulerConfig` flag,
off by default; with it off the engine is byte-for-byte the original miniature
vLLM and all of its correctness tests still pass.
