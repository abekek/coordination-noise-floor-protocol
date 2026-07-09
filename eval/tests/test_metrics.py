"""Tests for the redundant-call detector."""

from __future__ import annotations

import pytest

from harness.metrics import (
    ToolCall,
    normalize_input,
    redundant_call_rate,
)


class TestNormalization:
    def test_strips_retry_count(self):
        a = normalize_input({"q": "NYC", "retry_count": 2})
        b = normalize_input({"q": "NYC", "retry_count": 5})
        assert a == b

    def test_strips_request_id(self):
        a = normalize_input({"q": "NYC", "request_id": "r1"})
        b = normalize_input({"q": "NYC", "request_id": "r2"})
        assert a == b

    def test_strips_timestamp(self):
        a = normalize_input({"q": "NYC", "timestamp": "2026-05-23T..."})
        b = normalize_input({"q": "NYC", "timestamp": "2026-05-24T..."})
        assert a == b

    def test_whitespace_and_case_normalized(self):
        a = normalize_input({"q": "  New York  "})
        b = normalize_input({"q": "new york"})
        assert a == b


class TestRedundantCallRate:
    def test_zero_when_no_repeats(self):
        calls = [
            ToolCall(tool_name="search", input={"q": "a"}, errored=False),
            ToolCall(tool_name="search", input={"q": "b"}, errored=False),
        ]
        assert redundant_call_rate(calls, failed_paths=[]) == 0.0

    def test_counts_repeat_of_previously_errored_call(self):
        calls = [
            ToolCall(tool_name="search", input={"q": "a"}, errored=True),
            ToolCall(tool_name="search", input={"q": "a"}, errored=False),
            ToolCall(tool_name="search", input={"q": "b"}, errored=False),
        ]
        # 1 redundant out of 3 calls = 1/3
        assert redundant_call_rate(calls, failed_paths=[]) == pytest.approx(1 / 3)

    def test_counts_call_matching_failed_path(self):
        calls = [
            ToolCall(tool_name="book", input={"airline": "X"}, errored=False),
        ]
        # The failed_path mentions a call to `book(airline=X)`
        fps = [{"tool_name": "book", "input": {"airline": "X"}}]
        assert redundant_call_rate(calls, failed_paths=fps) == 1.0


class TestComputeAllMetricsWithFailedPaths:
    def test_failed_paths_contribute_to_redundant_rate(self):
        """compute_all_metrics should accept failed_paths and pass them through
        to redundant_call_rate so paths in the trace store count as forbidden."""
        from harness.metrics import compute_all_metrics
        from harness.transcript import AgentStepResult, Transcript
        from harness.transcript import ToolCall as HarnessToolCall

        class _FakeScore:
            completed = True

        transcript = Transcript()
        transcript.append(AgentStepResult(
            agent_id="executor", final_text="",
            tool_calls=[
                HarnessToolCall(
                    tool_name="book_flight", input={"flight_id": "F2"},
                    output={"success": False, "error": "unavailable"},
                    errored=False, latency_ms=1.0,
                ),
            ],
            llm_calls=1, input_tokens=100, output_tokens=20,
            cache_read_tokens=0, raw_messages=[], hit_max_iterations=False,
        ))
        transcript.finalize("done")

        # Without failed_paths: only within-trial errored repetition counts.
        # Single call, no prior failure → 0 redundant.
        metrics_empty = compute_all_metrics(transcript, _FakeScore(), failed_paths=[])
        assert metrics_empty["redundant_call_rate"] == 0.0

        # With failed_paths including this exact call: the call IS redundant
        # (it attempts a path that was already known to fail).
        fps = [{"tool_name": "book_flight", "input": {"flight_id": "F2"}}]
        metrics_with = compute_all_metrics(transcript, _FakeScore(), failed_paths=fps)
        assert metrics_with["redundant_call_rate"] == 1.0

    def test_failed_paths_default_to_empty(self):
        """compute_all_metrics(transcript, score) without failed_paths still
        works (backward compatibility)."""
        from harness.metrics import compute_all_metrics
        from harness.transcript import Transcript

        class _FakeScore:
            completed = True

        transcript = Transcript()
        metrics = compute_all_metrics(transcript, _FakeScore())
        assert metrics["redundant_call_rate"] == 0.0
