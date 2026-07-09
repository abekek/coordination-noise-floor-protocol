"""Tool registry — what the agent loop can call.

Tools are async callables with a name + description + JSON Schema for
their input. Benchmarks register their per-task tool set with a
ToolRegistry; the agent loop hands the registry's `anthropic_format()`
to the LLM as the `tools` parameter and routes the LLM's `tool_use`
blocks back through `invoke()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]

    async def __call__(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass
class ToolRegistry:
    tools: dict[str, Tool]

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def anthropic_format(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self.tools.values()
        ]

    async def invoke(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            tool = self.tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool {name!r}") from exc
        return await tool(**args)
