"""Tests for the oracle positive-control protocol.

The oracle condition (reviewer-requested, KDD'26 workshop) injects the
task's golden action sequence through pull's activation semantics. These
tests verify the two properties the measurement story depends on:

1. trial-0 inertness — with an empty store the augmenter is a byte-level
   no-op, so oracle stays configuration-equivalent with the other arms
   at trial 0 and cannot disturb the floor measurement;
2. activation content — with a non-empty store the golden actions are
   rendered into an <oracle_guidance> block appended to the system
   prompt, and the tool-response path stays unmodified (writer-only
   hook, matched payload with pull).

No LLM calls. Run from the tau2 venv:
    eval/benchmarks/tau2_bench/_vendor/.venv/bin/python \\
        -m pytest eval/adapters/v2_pivot/test_oracle.py -v
or directly:
    eval/benchmarks/tau2_bench/_vendor/.venv/bin/python \\
        eval/adapters/v2_pivot/test_oracle.py
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field


def _setup_path() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_setup_path()


from eval.adapters.v2_pivot.protocols import (  # noqa: E402
    PROTOCOLS,
    PER_TASK_AUGMENTERS,
    make_oracle_augmenter,
    make_pull_writer_hook,
    oracle_guidance_block,
)
from eval.adapters.v2_pivot.trace_store import TraceStore  # noqa: E402


# ---- duck-typed tau2 Task shims --------------------------------------

@dataclass
class _Action:
    name: str
    arguments: dict
    requestor: str = "assistant"


@dataclass
class _EvalCriteria:
    actions: list = field(default_factory=list)


@dataclass
class _Task:
    id: str = "t1"
    evaluation_criteria: object = None


def _task_with_golden() -> _Task:
    return _Task(
        id="42",
        evaluation_criteria=_EvalCriteria(actions=[
            _Action("cancel_order", {"order_id": "#W123"}),
            _Action("refund_payment", {"order_id": "#W123", "amount": 10.5}),
            _Action("confirm_identity", {"user_id": "u9"}, requestor="user"),
        ]),
    )


SYSTEM = "You are a customer service agent.\n<policy>...</policy>"


# ---- tests ------------------------------------------------------------

def test_registry() -> None:
    assert "oracle" in PROTOCOLS
    assert "oracle" in PER_TASK_AUGMENTERS
    # Matched payload with pull on the hook side: writer-only hook.
    _, hook_factory = PROTOCOLS["oracle"]
    assert hook_factory is make_pull_writer_hook


def test_trial0_inert_empty_store() -> None:
    """Empty store → byte-identical system prompt (floor undisturbed)."""
    aug = make_oracle_augmenter(_task_with_golden())
    store = TraceStore(task_id="42")
    out = aug(SYSTEM, store)
    assert out == SYSTEM


def test_activation_injects_golden_actions() -> None:
    aug = make_oracle_augmenter(_task_with_golden())
    store = TraceStore(task_id="42")
    store.write(trial=0, tool_name="cancel_order",
                tool_args={"order_id": "#W123"},
                error_text="trial 0 reward=0.0",
                event_type="FAILED_TRIAL_ACTION")
    out = aug(SYSTEM, store)
    assert out.startswith(SYSTEM)
    assert "<oracle_guidance>" in out and "</oracle_guidance>" in out
    assert 'cancel_order({"order_id":"#W123"})' in out
    assert 'refund_payment' in out
    assert "(user) confirm_identity" in out


def test_no_golden_actions_is_noop_even_when_active() -> None:
    """Tasks that score 1.0 unconditionally get no guidance block."""
    aug = make_oracle_augmenter(_Task(id="7", evaluation_criteria=None))
    store = TraceStore(task_id="7")
    store.write(trial=0, tool_name="x", tool_args={}, error_text="e",
                event_type="FAILED_TRIAL_ACTION")
    assert aug(SYSTEM, store) == SYSTEM
    assert oracle_guidance_block(_Task(id="7")) == ""


def test_hook_leaves_tool_results_unmodified() -> None:
    """Oracle rides the pull surface: reader never touches responses."""
    store = TraceStore(task_id="42")
    store.write(trial=0, tool_name="get_order", tool_args={"o": 1},
                error_text="boom", event_type="FAILED_TRIAL_ACTION")
    hook = make_pull_writer_hook(store)
    out = hook("get_order", {"o": 1}, "RESULT", False, {"trial": 1})
    assert out == "RESULT"


def _main() -> int:
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {exc}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_main())
