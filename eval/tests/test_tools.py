"""Tests for the ToolRegistry."""

from __future__ import annotations

import pytest

from harness.tools import Tool, ToolRegistry


class _EchoTool:
    name = "echo"
    description = "echoes its input"
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def __call__(self, **kwargs):
        return {"echoed": kwargs["text"]}


class _BoomTool:
    name = "boom"
    description = "always errors"
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, **kwargs):
        raise RuntimeError("boom")


class TestAnthropicFormat:
    def test_emits_tool_definitions(self):
        r = ToolRegistry(tools={"echo": _EchoTool()})
        out = r.anthropic_format()
        assert len(out) == 1
        assert out[0]["name"] == "echo"
        assert out[0]["description"] == "echoes its input"
        assert out[0]["input_schema"]["type"] == "object"


class TestInvoke:
    async def test_invoke_routes_to_tool(self):
        r = ToolRegistry(tools={"echo": _EchoTool()})
        result = await r.invoke("echo", {"text": "hi"})
        assert result == {"echoed": "hi"}

    async def test_invoke_unknown_tool_raises(self):
        r = ToolRegistry(tools={"echo": _EchoTool()})
        with pytest.raises(KeyError):
            await r.invoke("missing", {})

    async def test_invoke_propagates_tool_error(self):
        r = ToolRegistry(tools={"boom": _BoomTool()})
        with pytest.raises(RuntimeError, match="boom"):
            await r.invoke("boom", {})


class TestRegister:
    def test_register_adds_tool(self):
        r = ToolRegistry(tools={})
        r.register(_EchoTool())
        assert "echo" in r.tools
