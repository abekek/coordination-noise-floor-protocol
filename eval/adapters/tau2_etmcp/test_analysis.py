"""Tests for the trial-metric extractor + paired comparison.

Synthetic SimulationRun dicts; no tau2 import. Covers:

- redundant-call detection (chronological-only rule)
- pass^k computation across N trials per task
- Cliff's delta sign convention
- paired Wilcoxon pairing by (task_id, trial)
- Holm-Bonferroni correction
"""

from __future__ import annotations

import pathlib
import sys


def _setup() -> None:
    repo = pathlib.Path(__file__).resolve().parents[3]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


_setup()

from eval.adapters.tau2_etmcp.analysis import (  # noqa: E402
    PairedComparison,
    TrialMetrics,
    _cliffs_delta,
    _cliffs_magnitude,
    aggregate_cell,
    compute_trial_metrics,
    holm_correct,
    paired_comparison,
)


def _mk_msg_assistant_call(tc_id: str, name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {"id": tc_id, "name": name, "arguments": args, "requestor": "assistant"}
        ],
    }


def _mk_msg_tool_result(tc_id: str, error: bool, content: str = "x") -> dict:
    return {"role": "tool", "tool_call_id": tc_id, "error": error, "content": content}


def test_redundant_repeat_after_failure_counted() -> None:
    msgs = [
        _mk_msg_assistant_call("a1", "search", {"q": "x"}),
        _mk_msg_tool_result("a1", error=True),
        # Same call again after failure → redundant
        _mk_msg_assistant_call("a2", "search", {"q": "x"}),
        _mk_msg_tool_result("a2", error=True),
    ]
    sim = {"task_id": "t1", "trial": 1, "messages": msgs, "reward_info": {"reward": 0}}
    m = compute_trial_metrics(sim, protocol="et_mcp")
    assert m.n_tool_calls == 2
    assert m.n_errored_calls == 2
    assert m.n_redundant_calls == 1, m
    print("PASS test_redundant_repeat_after_failure_counted")


def test_first_failure_never_redundant() -> None:
    msgs = [
        _mk_msg_assistant_call("a1", "x", {"k": 1}),
        _mk_msg_tool_result("a1", error=True),
    ]
    sim = {"task_id": "t", "trial": 1, "messages": msgs, "reward_info": {"reward": 0}}
    m = compute_trial_metrics(sim, protocol="et_mcp")
    assert m.n_redundant_calls == 0
    print("PASS test_first_failure_never_redundant")


def test_success_after_failure_not_redundant() -> None:
    msgs = [
        _mk_msg_assistant_call("a1", "x", {"k": 1}),
        _mk_msg_tool_result("a1", error=True),
        _mk_msg_assistant_call("a2", "x", {"k": 1}),
        _mk_msg_tool_result("a2", error=False),
    ]
    sim = {"task_id": "t", "trial": 1, "messages": msgs, "reward_info": {"reward": 1}}
    m = compute_trial_metrics(sim, protocol="et_mcp")
    assert m.n_redundant_calls == 0
    print("PASS test_success_after_failure_not_redundant")


def test_normalization_order_independent() -> None:
    msgs = [
        _mk_msg_assistant_call("a1", "x", {"a": 1, "b": 2}),
        _mk_msg_tool_result("a1", error=True),
        # Same args, different order → counts as same key
        _mk_msg_assistant_call("a2", "x", {"b": 2, "a": 1}),
        _mk_msg_tool_result("a2", error=True),
    ]
    sim = {"task_id": "t", "trial": 1, "messages": msgs, "reward_info": {"reward": 0}}
    m = compute_trial_metrics(sim, protocol="et_mcp")
    assert m.n_redundant_calls == 1
    print("PASS test_normalization_order_independent")


def _trial(task_id: str, trial: int, success: bool, redundant_rate: float = 0.0):
    return TrialMetrics(
        task_id=task_id,
        trial=trial,
        protocol="et_mcp",
        reward=1.0 if success else 0.0,
        success=success,
        n_tool_calls=10,
        n_errored_calls=1,
        n_redundant_calls=int(redundant_rate * 10),
        redundant_rate=redundant_rate,
        agent_cost=0.01,
        user_cost=0.005,
        termination_reason="ok",
    )


def test_pass_at_k_all_success_yields_1() -> None:
    metrics = [
        _trial("t1", 1, True), _trial("t1", 2, True), _trial("t1", 3, True),
        _trial("t2", 1, True), _trial("t2", 2, True), _trial("t2", 3, True),
    ]
    agg = aggregate_cell(metrics)
    assert agg.pass_at_k[1] == 1.0
    assert agg.pass_at_k[2] == 1.0
    assert agg.pass_at_k[3] == 1.0
    print("PASS test_pass_at_k_all_success_yields_1")


def test_pass_at_k_one_failure_drops_higher_k() -> None:
    metrics = [
        _trial("t1", 1, True), _trial("t1", 2, True), _trial("t1", 3, True),
        _trial("t2", 1, True), _trial("t2", 2, False), _trial("t2", 3, True),
    ]
    agg = aggregate_cell(metrics)
    assert agg.pass_at_k[1] == 1.0      # both t1, t2 pass first trial
    assert agg.pass_at_k[2] == 0.5      # t2 fails on trial 2
    assert agg.pass_at_k[3] == 0.5      # still 1/2
    print("PASS test_pass_at_k_one_failure_drops_higher_k")


def test_cliffs_delta_sign() -> None:
    # a strictly larger than b
    assert _cliffs_delta([3, 3, 3], [1, 1, 1]) == 1.0
    assert _cliffs_delta([1, 1, 1], [3, 3, 3]) == -1.0
    assert _cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0
    assert _cliffs_magnitude(0.1) == "negligible"
    assert _cliffs_magnitude(0.2) == "small"
    assert _cliffs_magnitude(0.4) == "medium"
    assert _cliffs_magnitude(0.5) == "large"
    print("PASS test_cliffs_delta_sign")


def test_paired_comparison_pairs_by_task_trial() -> None:
    a = [_trial("t1", 1, True, 0.0), _trial("t1", 2, True, 0.0), _trial("t2", 1, True, 0.0)]
    b = [_trial("t1", 1, True, 0.5), _trial("t1", 2, True, 0.5), _trial("t2", 1, True, 0.5)]
    cmp = paired_comparison(a, b, "redundant_rate")
    assert cmp.n_pairs == 3
    assert cmp.mean_a == 0.0
    assert cmp.mean_b == 0.5
    assert cmp.direction == "a < b"
    assert cmp.cliffs_delta == -1.0
    print("PASS test_paired_comparison_pairs_by_task_trial")


def test_holm_correction_increases_pvals() -> None:
    # 3 p-values; the largest should map to itself (×1), smallest gets ×3
    p_in = [0.01, 0.04, 0.02]
    adj = holm_correct(p_in)
    # Sorted: 0.01 (×3 = 0.03), 0.02 (×2 = 0.04 enforced ≥ 0.03), 0.04 (×1)
    assert adj[0] == 0.03
    assert adj[2] == 0.04
    assert adj[1] == 0.04
    print("PASS test_holm_correction_increases_pvals")


def test_holm_handles_none() -> None:
    adj = holm_correct([None, 0.01, None])
    assert adj[0] is None
    assert adj[2] is None
    assert adj[1] == 0.01
    print("PASS test_holm_handles_none")


if __name__ == "__main__":
    tests = [
        test_redundant_repeat_after_failure_counted,
        test_first_failure_never_redundant,
        test_success_after_failure_not_redundant,
        test_normalization_order_independent,
        test_pass_at_k_all_success_yields_1,
        test_pass_at_k_one_failure_drops_higher_k,
        test_cliffs_delta_sign,
        test_paired_comparison_pairs_by_task_trial,
        test_holm_correction_increases_pvals,
        test_holm_handles_none,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    if failures:
        sys.exit(1)
    print(f"\nAll {len(tests)} test functions passed.")
