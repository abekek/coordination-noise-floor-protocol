"""Silo-Bench LLM utility — patched to route through the project's AnthropicClient.

The original implementation called OpenAI-compatible APIs directly.
We replace call_llm with our bridge so that benchmark runs use
harness.llm.AnthropicClient (Anthropic SDK, with caching + retry).
"""

from __future__ import annotations

from benchmarks.silo_bench.llm_bridge import call_llm  # noqa: F401

__all__ = ["call_llm"]
