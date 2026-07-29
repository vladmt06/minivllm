"""H1 realism check: does the channel survive real wall-clock timing?

The deterministic harness proves the *mechanism* with zero noise. This one proves
it is not an artifact of noiseless integer steps: a real TinyLlama forward on MPS
gives every step a real, variable duration; tool calls pause for real
milliseconds; and the attacker is a genuine separate thread timing its own
submit -> first-token latency across a real thread boundary.

Threading rule, because PyTorch MPS is not thread-safe: the model is touched only
by the serving thread. The attacker thread submits requests to a queue and waits
on an Event. It never calls the model.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue

import torch

from minivllm.config import CacheConfig, SchedulerConfig
from minivllm.core.engine import LLMEngine
from minivllm.core.program import Program, Tool, TurnSpec
from minivllm.core.sampler import sample
from minivllm.core.sequence import Sequence, SequenceStatus
from minivllm.inputs import build_model_input
from sidechannel.reconstruct import classify, score

# Tool durations in MILLISECONDS now, well separated so the fingerprint has room
# to survive step-quantised timing (~one MPS decode step per admission).
REALTIME_TOOLS: dict[str, Tool] = {
    "calc": Tool("calc", duration_mean=80, duration_jitter=15),
    "web_search": Tool("web_search", duration_mean=350, duration_jitter=40),
    "db_query": Tool("db_query", duration_mean=800, duration_jitter=80),
    "code_exec": Tool("code_exec", duration_mean=1600, duration_jitter=150),
}

ATTACKER_TENANT = 2


@dataclass
class _Probe:
    pid: int
    submit_t: float
    event: threading.Event = field(default_factory=threading.Event)
    admit_t: float | None = None


class RealtimeServer:
    """Owns the model, KV cache and program-aware scheduler; runs the step loop.

    Reuses LLMEngine for everything heavy (loaded weights, KV pool, scheduler) and
    only orchestrates program transitions with a wall-clock clock so tool pauses
    are measured in real milliseconds.
    """

    def __init__(self, max_num_seqs: int = 6, num_blocks: int = 4000, seed: int = 0):
        self.engine = LLMEngine.from_pretrained(
            cache_config=CacheConfig(block_size=16, num_blocks=num_blocks),
            scheduler_config=SchedulerConfig(
                program_aware=True,
                kv_ttl_steps=10**9,  # wall-clock resume drives pins here, not the step TTL
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=999_999,
            ),
            seed=seed,
        )
        self.sched = self.engine.scheduler
        self.max_num_seqs = max_num_seqs

        self.programs: dict[int, Program] = {}
        self._resume_at: dict[int, float] = {}  # program_id -> wall-clock ms to resume
        self._pending_admit: dict[int, _Probe] = {}  # seq_id -> probe awaiting admission
        self.victim_departed_ms: float | None = None  # wall-clock the victim finished

        self.submit_q: Queue[_Probe] = Queue()
        self._stop = threading.Event()
        self.admit_ms: list[float] = []  # attacker observable, filled by the prober

    # -- request construction (content is irrelevant to a timing channel) ----

    def _mock_prompt(self, n: int) -> list[int]:
        return [self.engine.cfg.bos_token_id] + [7] * (n - 1)

    def submit_victim(self, program: Program) -> None:
        self.programs[program.program_id] = program
        seq = program.first_turn()
        seq.prompt_ids = self._mock_prompt(len(seq.prompt_ids))
        self.sched.add(seq)

    def submit_background(self, n: int) -> None:
        for i in range(n):
            p = Program([TurnSpec(10**7)], prompt_len=16, arrival=0.0, tenant_id=100 + i)
            seq = p.first_turn()
            seq.prompt_ids = self._mock_prompt(16)
            self.sched.add(seq)

    # -- the serving loop ----------------------------------------------------

    def _drain_submissions(self) -> None:
        while True:
            try:
                probe = self.submit_q.get_nowait()
            except Empty:
                return
            seq = Sequence(
                prompt_ids=self._mock_prompt(16),
                params=_one_token_params(),
                arrival=1e6 + self.sched.step_counter,
                tenant_id=ATTACKER_TENANT,
            )
            self._pending_admit[seq.seq_id] = probe
            self.sched.add(seq)

    def _now_ms(self) -> float:
        return time.perf_counter() * 1000.0

    def step_once(self) -> None:
        self._drain_submissions()
        self._resume_due(self._now_ms())

        out = self.sched.schedule()
        if out.is_empty:
            return
        inp = build_model_input(out.scheduled, self.engine.block_size, self.engine.device,
                                out.is_prefill)
        with torch.inference_mode():
            logits = self.engine.model(inp, self.engine.kv_cache.caches, self.engine.block_size)
        tokens = sample(logits, out.scheduled, self.engine.generator)

        # Record admissions the instant they happen, in wall-clock ms.
        now = self._now_ms()
        for seq in out.scheduled:
            probe = self._pending_admit.pop(seq.seq_id, None)
            if probe is not None:
                probe.admit_t = now
                probe.event.set()

        for seq, token in zip(out.scheduled, tokens):
            seq.num_computed = seq.num_tokens
            seq.append_token(token)
            seq.maybe_finish(self.engine.cfg.eos_token_id)

        for seq in list(out.scheduled):  # snapshot: suspend() mutates running (= out.scheduled)
            if seq.is_finished:
                self._on_turn_end(seq, now)
        self.sched.free_finished()

    def _on_turn_end(self, seq: Sequence, now: float) -> None:
        prog = self.programs.get(seq.program_id) if seq.program_id is not None else None
        if prog is None:
            return
        has_next = prog.turn_idx + 1 < len(prog.turns)
        tool = prog.turns[prog.turn_idx].tool
        if has_next and tool is not None:
            self.sched.suspend(seq)
            dur = tool.sample_duration(_rng())  # milliseconds
            self._resume_at[prog.program_id] = now + dur
            prog.ground_truth.append((tool.name, int(now), int(now + dur)))
            prog.state = prog.state.__class__.TOOL_CALL
        else:
            prog.state = prog.state.__class__.DONE
            if prog.tenant_id == 1:
                self.victim_departed_ms = now  # session ends; slot frees permanently

    def _resume_due(self, now: float) -> None:
        for pid, prog in self.programs.items():
            if prog.state.name != "TOOL_CALL":
                continue
            if now < self._resume_at.get(pid, 0.0):
                continue
            prev = prog.seq
            assert prev is not None
            alive = self.sched.resume(prev)
            prog.turn_idx += 1
            t = prog.turns[prog.turn_idx]
            result = prog.turns[prog.turn_idx - 1].result_len
            if alive:
                seq = prog._new_seq(prev.num_tokens + result, t.gen_len,
                                    block_table=prev.block_table, num_computed=prev.num_tokens)
            else:
                seq = prog._new_seq(prev.num_tokens + result, t.gen_len)
            seq.prompt_ids = self._mock_prompt(len(seq.prompt_ids))
            prog.seq = seq
            prog.state = prog.state.__class__.ACTING
            self._resume_at.pop(pid, None)
            self.sched.add(seq)

    def victim_done(self) -> bool:
        return all(p.state.name == "DONE" for p in self.programs.values() if p.tenant_id == 1)

    def run_serving(self) -> None:
        while not self._stop.is_set():
            self.step_once()


def _one_token_params():
    from minivllm.core.sequence import SamplingParams

    return SamplingParams(max_tokens=1, ignore_eos=True)


_RNG = None


def _rng():
    global _RNG
    if _RNG is None:
        import random

        _RNG = random.Random(0)
    return _RNG


class AttackerThread(threading.Thread):
    """Closed-loop prober on its own thread. Submits one probe, waits for its
    admission Event, records the wall-clock latency, repeats. Never touches the
    model -- only the queue and the returned timestamp."""

    def __init__(self, server: RealtimeServer):
        super().__init__(daemon=True)
        self.server = server
        self.admit_ms: list[float] = []
        self.latencies_ms: list[float] = []
        self._stop = threading.Event()

    def run(self) -> None:
        pid = 0
        while not self._stop.is_set():
            probe = _Probe(pid=pid, submit_t=time.perf_counter() * 1000.0)
            self.server.submit_q.put(probe)
            if probe.event.wait(timeout=5.0) and probe.admit_t is not None:
                self.admit_ms.append(probe.admit_t)
                self.latencies_ms.append(probe.admit_t - probe.submit_t)
            pid += 1

    def stop(self) -> None:
        self._stop.set()


def build_victim(tool_sequence: list[str], gen_len: int = 24) -> Program:
    turns = [TurnSpec(gen_len=gen_len, tool=REALTIME_TOOLS[n]) for n in tool_sequence]
    turns.append(TurnSpec(gen_len=gen_len))
    return Program(turns=turns, prompt_len=16, arrival=0.1, tenant_id=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", nargs="+", default=["web_search", "db_query", "calc", "code_exec"])
    ap.add_argument("--max-num-seqs", type=int, default=6)
    ap.add_argument("--drain-ms", type=int, default=800)
    args = ap.parse_args()

    server = RealtimeServer(max_num_seqs=args.max_num_seqs)
    server.submit_background(args.max_num_seqs - 1)
    # warm the model / fill the batch before the victim arrives
    for _ in range(30):
        server.step_once()

    victim = build_victim(args.tools)
    server.submit_victim(victim)

    serving = threading.Thread(target=server.run_serving, daemon=True)
    attacker = AttackerThread(server)
    t0 = time.perf_counter() * 1000.0
    serving.start()
    attacker.start()

    # Let it run until the victim's program completes, plus a drain tail.
    while not server.victim_done():
        time.sleep(0.02)
    time.sleep(args.drain_ms / 1000.0)
    attacker.stop()
    server._stop.set()
    serving.join(timeout=2.0)

    # Score over the victim's session only: admissions before it departed and its
    # slot freed permanently. Departure is the operator's ground truth, exactly as
    # in the deterministic harness.
    end = (server.victim_departed_ms or (t0 + 1e18)) - t0
    admits = [a - t0 for a in attacker.admit_ms if a - t0 <= end]
    truth = [(name, s - t0, e - t0) for name, s, e in victim.ground_truth]

    step_ms = _median_step_ms(attacker.latencies_ms)
    print(f"real-model wall-clock run: {len(attacker.admit_ms)} probe admissions, "
          f"median probe latency {step_ms:.0f} ms")
    print(f"victim tool calls (ground truth): {[t for t, _, _ in truth]}")

    s = score(admits, truth, taxonomy=REALTIME_TOOLS, gap=None)
    from sidechannel.reconstruct import session_bursts

    bursts = session_bursts(admits, taxonomy=REALTIME_TOOLS)
    print(f"recovered tools: {[classify(b.width, REALTIME_TOOLS) for b in bursts]}")
    print(f"{s.summary()}")
    print("\nSIGNAL SURVIVES WALL-CLOCK" if s.tool_accuracy >= 0.5 and s.turn_count_exact
          else "\nsignal degraded under wall-clock -- see numbers")


def _median_step_ms(latencies: list[float]) -> float:
    if not latencies:
        return float("nan")
    from statistics import median

    return median(latencies)


if __name__ == "__main__":
    main()
