"""ET-MCP adapter for tau2-bench.

Wraps tau2's HalfDuplexAgent interface around a planner+executor split that
shares an ET-MCP trace store. Four protocol variants are exposed for the
coordination ablation in §6 of the paper:

- no_coord:        planner + executor with no shared state
- push_scratchpad: shared dict written by planner, read by executor (CA-MCP style)
- message_passing: planner -> executor explicit message, no shared store
- et_mcp:          planner pulls from a typed trace store on demand
"""
