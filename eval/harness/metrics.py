"""Metric implementations for the evaluation harness.

The redundant-call detector is the novel metric in §6 of the paper. Its
definition is documented in `docs/specs/2026-05-23-et-mcp-design.md` §6.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_STRIPPED_KEYS = {"retry_count", "request_id", "timestamp", "trace_id"}


def normalize_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop retry-only fields, lowercase + strip strings, sort keys."""
    out: dict[str, Any] = {}
    for k in sorted(payload.keys()):
        if k in _STRIPPED_KEYS:
            continue
        v = payload[k]
        if isinstance(v, str):
            v = v.strip().lower()
        elif isinstance(v, dict):
            v = normalize_input(v)
        out[k] = v
    return out


@dataclass
class ToolCall:
    tool_name: str
    input: dict[str, Any]
    errored: bool


def _key(tool_name: str, payload: dict[str, Any]) -> tuple[str, str]:
    # Hashable representation
    import json
    return (tool_name.strip().lower(),
            json.dumps(normalize_input(payload), sort_keys=True))


def redundant_call_rate(
    calls: list[ToolCall],
    failed_paths: list[dict[str, Any]],
) -> float:
    """Fraction of calls whose (tool_name, normalized_input) matches either
    a prior errored call in the same trace OR any FAILED_PATH event's
    described call.
    """
    if not calls:
        return 0.0

    forbidden: set[tuple[str, str]] = set()
    for fp in failed_paths:
        if "tool_name" in fp and "input" in fp:
            forbidden.add(_key(fp["tool_name"], fp["input"]))

    redundant = 0
    seen_errored: set[tuple[str, str]] = set()
    for call in calls:
        k = _key(call.tool_name, call.input)
        if k in seen_errored or k in forbidden:
            redundant += 1
        if call.errored:
            seen_errored.add(k)
    return redundant / len(calls)


def disagreement_rate_placeholder() -> float:
    """Operationalized per-benchmark in Phase 2. Stub returns 0.0."""
    return 0.0


def compute_all_metrics(transcript, score, failed_paths: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Bundle metrics for one trial.

    `transcript` is a harness.transcript.Transcript.
    `score` is a benchmarks.base.BenchmarkResult.
    `failed_paths` is an optional list of {tool_name, input} dicts extracted
    from the trace store (ET-MCP baselines). Tool calls whose (name, input)
    matches one of these entries count as redundant.
    Returns a dict ready to embed in the JSONL line.
    """
    all_tool_calls = [
        ToolCall(tool_name=tc.tool_name, input=tc.input, errored=tc.errored)
        for tc in transcript.tool_calls()
    ]
    return {
        "llm_calls": transcript.llm_calls_total(),
        "tool_calls": len(all_tool_calls),
        "input_tokens": transcript.input_tokens_total(),
        "output_tokens": transcript.output_tokens_total(),
        "cache_read_tokens": transcript.cache_read_tokens_total(),
        "redundant_call_rate": redundant_call_rate(all_tool_calls, failed_paths=failed_paths or []),
        "disagreement_rate": None,  # Phase 2 follow-up: per-benchmark operationalization
        "trace_event_count": 0,     # Set by ET-MCP baseline-aware caller; 0 default
        "hit_max_handoffs": transcript.hit_max_handoffs,
    }
