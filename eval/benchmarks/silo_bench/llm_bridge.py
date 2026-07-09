"""Bridge Silo-Bench's call_llm interface to our AnthropicClient.

Silo-Bench calls:
    call_llm(api_base, api_key, model, messages) -> {"content", "input_tokens", "output_tokens"}

`api_base` and `api_key` are Silo-Bench's OpenAI-compat fields — we ignore
them and let AnthropicClient pick up ANTHROPIC_API_KEY from the environment.

`messages` arrive as OpenAI-format dicts [{"role": ..., "content": ...}].
We split out the leading system messages and pass the remainder as the
conversation to Anthropic's Messages API.

Our AnthropicClient is async; we wrap it in asyncio.run() so that the
sync Silo-Bench engine can call it without modification.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from harness.llm import AnthropicClient, LLMResponse

# Module-level client cache keyed by model string so repeated calls
# within one benchmark run reuse the same client object.
_clients: dict[str, AnthropicClient] = {}
_last_call_at: float | None = None
_last_input_tokens: int | None = None


def _maybe_throttle() -> None:
    """Optionally space Silo LLM calls to respect strict per-minute quotas."""
    global _last_call_at
    fixed_cooldown = float(os.environ.get("SILO_LLM_CALL_COOLDOWN_SECONDS", "0") or 0)
    target_tpm = float(os.environ.get("SILO_LLM_INPUT_TOKENS_PER_MINUTE", "0") or 0)
    token_cooldown = 0.0
    if target_tpm > 0 and _last_input_tokens is not None:
        token_cooldown = (_last_input_tokens / target_tpm) * 60.0
    cooldown = max(fixed_cooldown, token_cooldown)
    if cooldown <= 0:
        _last_call_at = time.monotonic()
        return

    now = time.monotonic()
    if _last_call_at is not None:
        elapsed = now - _last_call_at
        remaining = cooldown - elapsed
        if remaining > 0:
            print(f"[llm-cooldown] sleeping {remaining:.1f}s before next model call")
            time.sleep(remaining)
            now = time.monotonic()
    _last_call_at = now


def _get_client(model: str) -> AnthropicClient:
    if model not in _clients:
        _clients[model] = AnthropicClient(model=model)
    return _clients[model]


def call_llm(
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Sync wrapper matching Silo-Bench's expected call_llm signature.

    Args:
        api_base: Ignored — we use Anthropic's SDK endpoint.
        api_key:  Ignored — AnthropicClient reads ANTHROPIC_API_KEY from env.
        model:    Anthropic model ID (e.g. "claude-haiku-4-5-20251001").
        messages: OpenAI-format list of {"role": ..., "content": ...}.

    Returns:
        {"content": str, "input_tokens": int, "output_tokens": int}
    """
    # Partition leading system messages; concatenate into a single system string.
    system_parts: list[str] = []
    user_messages: list[dict[str, Any]] = []
    for msg in messages:
        if msg["role"] == "system" and not user_messages:
            system_parts.append(str(msg["content"]))
        else:
            user_messages.append({"role": msg["role"], "content": msg["content"]})

    system = "\n\n".join(system_parts) if system_parts else "You are a helpful assistant."

    client = _get_client(model)
    _maybe_throttle()

    async def _run() -> LLMResponse:
        return await client.call(
            system=system,
            messages=user_messages,
        )

    response = asyncio.run(_run())
    global _last_input_tokens
    _last_input_tokens = response.input_tokens
    return {
        "content": response.text,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
    }
