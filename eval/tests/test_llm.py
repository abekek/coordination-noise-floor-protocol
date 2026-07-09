"""Tests for AnthropicClient. The anthropic SDK is mocked end-to-end."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from harness.llm import AnthropicClient, LLMResponse


def _make_response(text: str = "hi", tool_uses: list[dict] | None = None,
                   input_tokens: int = 100, output_tokens: int = 20,
                   cache_read: int = 0) -> SimpleNamespace:
    content = []
    if text:
        content.append(SimpleNamespace(type="text", text=text))
    for tu in tool_uses or []:
        content.append(SimpleNamespace(
            type="tool_use", id=tu.get("id", "toolu_1"),
            name=tu["name"], input=tu["input"],
        ))
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=0,
    )
    return SimpleNamespace(content=content, usage=usage, stop_reason="end_turn")


@pytest.fixture
def mocked_anthropic(mocker):
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock()
    mocker.patch("harness.llm.AsyncAnthropic", return_value=client)
    return client


class TestCall:
    async def test_returns_text_only(self, mocked_anthropic):
        mocked_anthropic.messages.create.return_value = _make_response(text="hello")
        client = AnthropicClient()
        resp = await client.call(system="sys", messages=[{"role": "user", "content": "hi"}])
        assert isinstance(resp, LLMResponse)
        assert resp.text == "hello"
        assert resp.tool_uses == []
        assert resp.input_tokens == 100
        assert resp.output_tokens == 20

    async def test_returns_tool_uses(self, mocked_anthropic):
        mocked_anthropic.messages.create.return_value = _make_response(
            text="",
            tool_uses=[{"id": "toolu_a", "name": "search", "input": {"q": "x"}}],
        )
        client = AnthropicClient()
        resp = await client.call(system="sys", messages=[{"role": "user", "content": "hi"}])
        assert len(resp.tool_uses) == 1
        assert resp.tool_uses[0].name == "search"
        assert resp.tool_uses[0].input == {"q": "x"}

    async def test_cache_read_counted(self, mocked_anthropic):
        mocked_anthropic.messages.create.return_value = _make_response(
            text="ok", input_tokens=50, cache_read=200,
        )
        client = AnthropicClient()
        resp = await client.call(system="sys", messages=[{"role": "user", "content": "hi"}])
        assert resp.cache_read_tokens == 200

    async def test_system_prompt_gets_cache_control(self, mocked_anthropic):
        mocked_anthropic.messages.create.return_value = _make_response()
        client = AnthropicClient()
        await client.call(system="sys", messages=[{"role": "user", "content": "hi"}])
        kwargs = mocked_anthropic.messages.create.call_args.kwargs
        assert isinstance(kwargs["system"], list)
        assert kwargs["system"][-1].get("cache_control") == {"type": "ephemeral"}

    async def test_tools_get_cache_control(self, mocked_anthropic):
        mocked_anthropic.messages.create.return_value = _make_response()
        client = AnthropicClient()
        tools = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
        await client.call(system="sys",
                          messages=[{"role": "user", "content": "hi"}],
                          tools=tools)
        kwargs = mocked_anthropic.messages.create.call_args.kwargs
        assert kwargs["tools"][-1].get("cache_control") == {"type": "ephemeral"}


class TestRetry:
    async def test_retries_on_transient_failure(self, mocked_anthropic):
        from anthropic import APIError
        first_call = APIError("503", request=None, body=None)
        mocked_anthropic.messages.create.side_effect = [
            first_call,
            _make_response(text="recovered"),
        ]
        client = AnthropicClient(max_retries=2)
        resp = await client.call(system="sys",
                                 messages=[{"role": "user", "content": "hi"}])
        assert resp.text == "recovered"
        assert mocked_anthropic.messages.create.call_count == 2
