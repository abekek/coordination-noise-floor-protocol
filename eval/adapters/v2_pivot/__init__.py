"""v2-pivot: custom Anthropic-native harness for tau2 task data.

Bypasses tau2's broken litellm/orchestrator path. Reuses tau2's domain DB,
tools, task definitions, and scorer. Implements three coordination
protocols at the orchestrator-interception layer:
- no_coord: vanilla assistant ↔ user loop
- pull (ET-MCP): peer events queried via an explicit trace.query tool
- intercept: orchestrator injects peer-warning context into tool RESPONSES
  automatically; agent never queries explicitly

The intercept protocol is the v2 architectural pivot — moves ET-MCP from
"a tool the agent must use" to "a transparent layer the framework imposes."
"""
