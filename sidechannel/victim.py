"""The victim tenant: a synthetic agentic program with a tool taxonomy.

Tools are separated by their duration so that a recovered pause width maps back
to a tool. Real agent tools genuinely differ this way -- a calculator returns in
milliseconds, a web search or a code sandbox in seconds -- so duration is a real
fingerprint, not a contrivance. Where durations overlap, the confusion matrix in
reconstruct.py reports the limit honestly rather than hiding it.
"""

from __future__ import annotations

from minivllm.core.program import Program, Tool, TurnSpec

# name -> (mean duration in steps, jitter). Chosen distinct-but-not-trivially so;
# db_query and code_exec are close enough to confuse under jitter, which is the
# honest case.
TOOL_TAXONOMY: dict[str, Tool] = {
    "calc": Tool("calc", duration_mean=4, duration_jitter=1),
    "web_search": Tool("web_search", duration_mean=10, duration_jitter=2),
    "db_query": Tool("db_query", duration_mean=18, duration_jitter=2),
    "code_exec": Tool("code_exec", duration_mean=26, duration_jitter=3),
}


def build_victim(
    tool_sequence: list[str],
    gen_len: int = 25,
    prompt_len: int = 16,
    arrival: float = 0.1,
    tenant_id: int = 1,
) -> Program:
    """A program that runs one turn per tool call plus a final turn.

    tool_sequence=["web_search", "db_query"] -> 3 turns: generate, call web_search,
    generate, call db_query, generate, finish.
    """
    turns: list[TurnSpec] = []
    for name in tool_sequence:
        if name not in TOOL_TAXONOMY:
            raise KeyError(f"unknown tool {name!r}; known: {sorted(TOOL_TAXONOMY)}")
        turns.append(TurnSpec(gen_len=gen_len, tool=TOOL_TAXONOMY[name]))
    turns.append(TurnSpec(gen_len=gen_len))  # final turn, no tool
    return Program(turns=turns, prompt_len=prompt_len, arrival=arrival, tenant_id=tenant_id)
