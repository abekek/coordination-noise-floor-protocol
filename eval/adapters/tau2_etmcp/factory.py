"""Register ETMCPAgent with the tau2 registry.

Importing this module triggers the registration as a side effect; the CLI
flag becomes `--agent et_mcp_agent` and selecting the coordination protocol
is done via `--agent-llm-args '{"coord_protocol": "et_mcp"}'`.
"""

from __future__ import annotations

import logging

from tau2.registry import registry

from .et_mcp_agent import ETMCPAgent


def create_et_mcp_agent(tools, domain_policy, **kwargs):
    """Factory called by tau2 when --agent et_mcp_agent is selected.

    Args:
        tools: Environment tools the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: tau2 passes:
            - llm (str): model name
            - llm_args (dict): inference args; we also use it to carry
              `coord_protocol` and `query_top_k` (popped before passing to
              LLM).
            - task: the current Task instance (optional)
    """
    llm_args = dict(kwargs.get("llm_args") or {})
    coord_protocol = llm_args.pop("coord_protocol", "et_mcp")
    query_top_k = int(llm_args.pop("query_top_k", 3))

    task = kwargs.get("task")
    task_id = getattr(task, "id", None) or getattr(task, "task_id", None) or "unknown"

    return ETMCPAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=llm_args or None,
        coord_protocol=coord_protocol,
        query_top_k=query_top_k,
        task_id=task_id,
    )


def register() -> None:
    """Register et_mcp_agent. Idempotent."""
    try:
        existing = registry.get_agent_factory("et_mcp_agent")
        if existing is not None:
            return
    except Exception:
        pass
    registry.register_agent_factory(create_et_mcp_agent, "et_mcp_agent")
    logging.getLogger(__name__).info("Registered et_mcp_agent factory.")


# Register on import — tau2 CLI looks up agents at run() time, so importing
# this module before invoking the CLI is sufficient.
register()
