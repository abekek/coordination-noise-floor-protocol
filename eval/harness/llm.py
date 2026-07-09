"""Async Anthropic SDK wrapper.

Features:
- Cache control on last system-prompt block and last tool definition
  so per-trial-stable content is cached.
- Tenacity retry on transient errors (APIError, RateLimitError) with
  exponential backoff.
- LLMResponse normalises the SDK's content blocks into clean text +
  parsed tool_use blocks for the agent loop.

Environment variables:
  ANTHROPIC_API_KEY — required for Anthropic API access
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anthropic import APIError, AsyncAnthropic, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_uses: list[ToolUseBlock]
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    raw_response: Any


@dataclass
class AnthropicClient:
    model: str = "claude-haiku-4-5"
    max_retries: int = 3

    def __post_init__(self) -> None:
        self._client = AsyncAnthropic(max_retries=0)

    async def call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        return await self._call_with_retry(
            system=system,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def _call_with_retry(self, **kwargs: Any) -> LLMResponse:
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=2, min=2, max=20),
            retry=retry_if_exception_type((APIError, RateLimitError)),
            reraise=True,
        )
        async def _do_call() -> LLMResponse:
            return await self._raw_call(**kwargs)

        return await _do_call()

    async def _raw_call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        system_blocks = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
        tool_blocks: list[dict[str, Any]] | None = None
        if tools:
            tool_blocks = [dict(t) for t in tools]
            tool_blocks[-1]["cache_control"] = {"type": "ephemeral"}

        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_blocks,
            "messages": messages,
        }
        if tool_blocks is not None:
            request["tools"] = tool_blocks

        response = await self._client.messages.create(**request)

        text_parts: list[str] = []
        tool_uses: list[ToolUseBlock] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
            elif getattr(block, "type", None) == "tool_use":
                tool_uses.append(
                    ToolUseBlock(
                        id=block.id,
                        name=block.name,
                        input=dict(block.input),
                    )
                )

        return LLMResponse(
            text="".join(text_parts),
            tool_uses=tool_uses,
            input_tokens=int(response.usage.input_tokens),
            output_tokens=int(response.usage.output_tokens),
            cache_read_tokens=int(getattr(response.usage, "cache_read_input_tokens", 0) or 0),
            raw_response=response,
        )
