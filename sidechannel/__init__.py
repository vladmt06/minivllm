"""Program-aware scheduling as a timing side channel.

A defensive-security demonstration on minivllm: two synthetic tenants the user
controls, showing that Continuum-style program-aware serving (program-level FCFS
+ TTL-pinned KV cache across tool calls) leaks a co-tenant's program timeline
through admission latency alone -- a leak no content-level defense touches,
because it lives in the scheduler.
"""
