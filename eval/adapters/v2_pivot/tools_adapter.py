"""Convert tau2 Tool objects to Anthropic Messages-API tool definitions.

tau2's `Tool.openai_schema` returns an OpenAI function-tool dict. Anthropic
uses a slightly different shape: top-level {name, description, input_schema}
instead of the wrapped {type: function, function: {name, description,
parameters}}.

We also strip pydantic's `title` keys (Anthropic accepts them but they're
noise) and pass `$defs` through unchanged (Anthropic supports $defs).
"""

from __future__ import annotations

import copy
from typing import Any

from tau2.environment.tool import Tool


def tau2_tool_to_anthropic(tool: Tool) -> dict[str, Any]:
    """Convert a tau2 Tool's OpenAI schema into Anthropic tool format."""
    oai = tool.openai_schema
    fn = oai["function"]
    input_schema = copy.deepcopy(fn.get("parameters", {}))
    # Strip pydantic noise
    input_schema.pop("title", None)
    return {
        "name": fn["name"],
        "description": fn.get("description", "").strip(),
        "input_schema": input_schema,
    }


def all_tools_to_anthropic(tau2_tools: list[Tool]) -> list[dict[str, Any]]:
    return [tau2_tool_to_anthropic(t) for t in tau2_tools]
