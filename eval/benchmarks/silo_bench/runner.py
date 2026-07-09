"""SiloBenchRunner — drives Silo-Bench tasks and emits TrialResult JSONL.

Wraps the vendored engine's init_case + run_round + evaluate sequence.
Output format matches our existing harness.results JSONL schema so the
`analysis report` CLI works on Silo-Bench results identically to other
benchmarks.

Signature notes (confirmed against _vendor/src/engine.py):
    init_case(task_file, protocol, model, api_base, api_key,
              max_rounds, workspace) -> str  (case_dir path)
    run_round(case_dir: str) -> bool         (protocol read from metadata.json)
    evaluate(case_dir: str) -> dict          (protocol read from metadata.json)

evaluate() returns {"metrics": {"S_success_rate", "P_partial_correctness",
                                "C_token_consumption", "D_communication_density"}}

Token counts live in metadata.json under execution.total_input_tokens /
execution.total_output_tokens — accumulated there by run_round.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ulid import ULID

# ---------------------------------------------------------------------------
# Vendor path bootstrap
# ---------------------------------------------------------------------------

_VENDOR_ROOT = Path(__file__).parent / "_vendor"
_VENDOR_SRC = _VENDOR_ROOT / "src"

import sys  # noqa: E402

for _p in (str(_VENDOR_ROOT), str(_VENDOR_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TASKS_DIR = _VENDOR_ROOT / "benchmarks"
_PROTOCOLS = ("msg", "broadcast", "sfs", "etmcp")

# Default model: prefer ANTHROPIC_MODEL env var, fall back to Haiku.
_DEFAULT_MODEL = os.environ.get(
    "ANTHROPIC_MODEL", "claude-haiku-4-5"
)

# Per-million-token pricing (USD). Keep in sync with eval.harness.runner.
_DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-haiku-4-5-0": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-1": {"input": 15.00, "output": 75.00},
}

_MTOK = 1_000_000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SiloTrialSpec:
    task_file: str   # filename like "I-01_n5.json"
    protocol: str    # "msg" | "broadcast" | "sfs" | "etmcp"
    seed: int


@dataclass
class SiloTrialResult:
    spec: SiloTrialSpec
    trial_id: str
    completed: bool        # S == 1.0
    s_score: float         # full success rate (mean over agents)
    p_score: float         # partial correctness
    c_score: float         # token consumption (normalised)
    d_score: float         # communication density
    total_input_tokens: int
    total_output_tokens: int
    wall_time_s: float
    error: str | None


class CostBudgetExceeded(Exception):
    """Raised after a trial pushes the cumulative estimated spend over budget."""

    def __init__(self, total_cost: float, ceiling: float, trials_completed: int) -> None:
        super().__init__(
            f"Cost budget exceeded: ${total_cost:.2f} > ${ceiling:.2f} "
            f"after {trials_completed} trials"
        )
        self.total_cost = total_cost
        self.ceiling = ceiling
        self.trials_completed = trials_completed


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def list_tasks(n: int = 5) -> list[str]:
    """Return sorted vendored task filenames for agent count *n*."""
    return sorted(p.name for p in _TASKS_DIR.glob(f"*_n{n}.json"))


def trial_key(task_file: str, protocol: str, seed: int) -> tuple[str, str, int]:
    """Stable identity for a Silo-Bench trial."""
    return (task_file.replace(".json", ""), protocol, seed)


def existing_trial_keys(results_dir: Path) -> set[tuple[str, str, int]]:
    """Read usable trial keys from an existing trials.jsonl.

    Zero-token setup/authentication failures do not represent real experiments,
    so leave them retryable.
    """
    keys: set[tuple[str, str, int]] = set()
    path = Path(results_dir) / "trials.jsonl"
    if not path.exists():
        return keys
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            spec = row["spec"]
            metrics = row.get("metrics", {})
            total_tokens = (
                int(metrics.get("input_tokens", 0) or 0)
                + int(metrics.get("output_tokens", 0) or 0)
            )
            if _is_retryable_error(row.get("error")):
                continue
            keys.add(
                (
                    str(spec["query_id"]),
                    str(spec["condition"]),
                    int(spec["seed"]),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return keys


def _is_retryable_error(error: str | None) -> bool:
    """Return true for infrastructure errors that should not satisfy resume."""
    if not error:
        return False
    retryable_markers = (
        "RateLimitError",
        "Quota exceeded",
        "RESOURCE_EXHAUSTED",
        "Too Many Requests",
        "authentication method",
        "missing api key",
    )
    return any(marker in error for marker in retryable_markers)


def trial_cost_usd(result: SiloTrialResult, model: str | None = None) -> float:
    """Estimate a Silo trial's USD cost from recorded token counts."""
    model_name = model or _DEFAULT_MODEL
    pricing = _DEFAULT_PRICING.get(model_name, {"input": 3.00, "output": 15.00})
    return (
        result.total_input_tokens * pricing["input"]
        + result.total_output_tokens * pricing["output"]
    ) / _MTOK


# ---------------------------------------------------------------------------
# Core trial runner
# ---------------------------------------------------------------------------


def run_silo_trial(
    spec: SiloTrialSpec,
    results_dir: Path,
    max_rounds: int = 20,
    model: str | None = None,
) -> SiloTrialResult:
    """Run one Silo-Bench trial via the vendored engine.

    The vendored engine reads the protocol from the case workspace's
    metadata.json (written by init_case), so run_round/evaluate need
    only the case_dir string.
    """
    # Import here so that the vendor sys.path additions above are in effect.
    from src.engine import evaluate, init_case, run_round  # noqa: PLC0415

    _model = model or _DEFAULT_MODEL
    trial_id = str(ULID())
    workspace = Path(results_dir) / "workspaces"
    workspace.mkdir(parents=True, exist_ok=True)

    task_path = str(_TASKS_DIR / spec.task_file)

    start = time.perf_counter()
    case_dir: str | None = None
    try:
        case_dir = init_case(
            task_file=task_path,
            protocol=spec.protocol,
            model=_model,
            api_base="",       # ignored — bridge uses Anthropic SDK
            api_key="",        # ignored — bridge reads ANTHROPIC_API_KEY
            max_rounds=max_rounds,
            workspace=str(workspace),
        )

        for _ in range(max_rounds):
            done = run_round(case_dir)
            if done:
                break

        result_dict = evaluate(case_dir)
        wall = time.perf_counter() - start

        metrics = result_dict.get("metrics", {})
        s = float(metrics.get("S_success_rate", 0.0))
        p = float(metrics.get("P_partial_correctness", 0.0))
        c = float(metrics.get("C_token_consumption", 0.0))
        d = float(metrics.get("D_communication_density", 0.0))

        in_tok, out_tok = _read_tokens_from_metadata(case_dir)

        return SiloTrialResult(
            spec=spec,
            trial_id=trial_id,
            completed=(s >= 1.0),
            s_score=s,
            p_score=p,
            c_score=c,
            d_score=d,
            total_input_tokens=in_tok,
            total_output_tokens=out_tok,
            wall_time_s=wall,
            error=None,
        )

    except Exception as exc:
        wall = time.perf_counter() - start
        # Still try to recover tokens if init_case succeeded.
        in_tok, out_tok = (0, 0)
        if case_dir is not None:
            try:
                in_tok, out_tok = _read_tokens_from_metadata(case_dir)
            except Exception:
                pass
        return SiloTrialResult(
            spec=spec,
            trial_id=trial_id,
            completed=False,
            s_score=0.0,
            p_score=0.0,
            c_score=0.0,
            d_score=0.0,
            total_input_tokens=in_tok,
            total_output_tokens=out_tok,
            wall_time_s=wall,
            error=f"{type(exc).__name__}: {exc}",
        )


def _read_tokens_from_metadata(case_dir: str) -> tuple[int, int]:
    """Read accumulated token counts from the engine's metadata.json."""
    meta_path = Path(case_dir) / "metadata.json"
    try:
        meta = json.loads(meta_path.read_text())
        execution = meta.get("execution", {})
        return (
            int(execution.get("total_input_tokens", 0)),
            int(execution.get("total_output_tokens", 0)),
        )
    except (json.JSONDecodeError, OSError):
        return 0, 0


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------


def write_trial_result(
    result: SiloTrialResult,
    results_dir: Path,
) -> None:
    """Append one JSONL line in our standard schema."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    line = {
        "trial_id": result.trial_id,
        "spec": {
            "condition": result.spec.protocol,   # protocol IS the condition
            "benchmark": "silo_bench",
            "query_id": result.spec.task_file.replace(".json", ""),
            "seed": result.spec.seed,
        },
        "completed": result.completed,
        "metrics": {
            "input_tokens": result.total_input_tokens,
            "output_tokens": result.total_output_tokens,
            "redundant_call_rate": 0.0,
            "trace_event_count": 0,
            "s_score": result.s_score,
            "p_score": result.p_score,
            "c_score": result.c_score,
            "d_score": result.d_score,
            "comm_reasoning_gap": result.p_score - result.s_score,
        },
        "wall_time_s": result.wall_time_s,
        "transcript_path": None,   # workspace path varies; omitted for portability
        "error": result.error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with (results_dir / "trials.jsonl").open("a") as fh:
        fh.write(json.dumps(line) + "\n")


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_sweep(
    *,
    tasks: list[str],
    protocols: list[str],
    seeds: list[int],
    results_dir: Path,
    max_rounds: int = 20,
    model: str | None = None,
    max_cost_usd: float | None = None,
    max_trials: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
    cooldown_seconds: float = 0.0,
    initial_cooldown_seconds: float = 0.0,
    zero_token_retries: int = 0,
) -> list[SiloTrialResult]:
    """Run a sweep across tasks × protocols × seeds, writing JSONL incrementally."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out: list[SiloTrialResult] = []
    cumulative_cost = 0.0
    effective_model = model or _DEFAULT_MODEL
    seen = existing_trial_keys(results_dir) if resume else set()
    if initial_cooldown_seconds > 0 and not dry_run:
        print(f"[cooldown  initial] sleeping {initial_cooldown_seconds:.1f}s")
        time.sleep(initial_cooldown_seconds)
    for task in tasks:
        for protocol in protocols:
            for seed in seeds:
                key = trial_key(task, protocol, seed)
                if key in seen:
                    print(f"[skip      {protocol:9s} {task:18s} seed={seed}] already present")
                    continue
                if max_trials is not None and len(out) >= max_trials:
                    return out
                if dry_run:
                    print(f"[dry-run   {protocol:9s} {task:18s} seed={seed}] would run")
                    continue
                if max_cost_usd is not None and cumulative_cost >= max_cost_usd:
                    raise CostBudgetExceeded(
                        total_cost=cumulative_cost,
                        ceiling=max_cost_usd,
                        trials_completed=len(out),
                    )
                spec = SiloTrialSpec(task_file=task, protocol=protocol, seed=seed)
                result = _run_trial_with_zero_token_retries(
                    spec=spec,
                    results_dir=results_dir,
                    max_rounds=max_rounds,
                    model=effective_model,
                    retries=zero_token_retries,
                    cooldown_seconds=cooldown_seconds,
                )
                out.append(result)
                tokens = result.total_input_tokens + result.total_output_tokens
                if not _is_retryable_error(result.error):
                    write_trial_result(result, results_dir)
                cumulative_cost += trial_cost_usd(result, effective_model)
                tag = "OK" if result.completed else ("ERR" if result.error else "FAIL")
                print(
                    f"[{protocol:9s} {task:18s} seed={seed}] {tag} "
                    f"S={result.s_score:.2f} P={result.p_score:.2f} "
                    f"tok={tokens:6d} "
                    f"cost=${cumulative_cost:6.2f} "
                    f"({result.wall_time_s:5.1f}s)"
                )
                if max_cost_usd is not None and cumulative_cost > max_cost_usd:
                    raise CostBudgetExceeded(
                        total_cost=cumulative_cost,
                        ceiling=max_cost_usd,
                        trials_completed=len(out),
                    )
                if cooldown_seconds > 0:
                    print(f"[cooldown  next] sleeping {cooldown_seconds:.1f}s")
                    time.sleep(cooldown_seconds)
    return out


def _run_trial_with_zero_token_retries(
    *,
    spec: SiloTrialSpec,
    results_dir: Path,
    max_rounds: int,
    model: str,
    retries: int,
    cooldown_seconds: float,
) -> SiloTrialResult:
    """Retry setup/quota failures that happen before token usage is recorded."""
    attempts = retries + 1
    result: SiloTrialResult | None = None
    for attempt in range(1, attempts + 1):
        result = run_silo_trial(spec, results_dir, max_rounds=max_rounds, model=model)
        tokens = result.total_input_tokens + result.total_output_tokens
        if not (result.error and tokens == 0):
            return result
        if attempt < attempts and cooldown_seconds > 0:
            print(
                f"[retry     {spec.protocol:9s} {spec.task_file:18s} seed={spec.seed}] "
                f"zero-token error; sleeping {cooldown_seconds:.1f}s before retry {attempt + 1}/{attempts}"
            )
            time.sleep(cooldown_seconds)
    if result is None:
        raise RuntimeError("unreachable: no Silo trial attempt was run")
    return result
