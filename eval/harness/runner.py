"""TrialRunner — orchestrates one trial end-to-end.

Per-trial sequence:
1. Resolve benchmark adapter
2. Load query
3. Build tool registry from benchmark
4. Instantiate baseline (via factory) + planner/executor agents
5. Run orchestration → Transcript
6. Score
7. Compute metrics
8. Write JSONL line + transcript artifact
9. Return TrialResult
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from ulid import ULID

from benchmarks.base import BenchmarkProtocol


def _parse_failed_path_approach(approach: str) -> dict[str, Any] | None:
    """Parse a FAILED_PATH approach string back into {tool_name, input}.

    The ET-MCP baseline records approach as: `{tool_name}({input_json})`.
    Returns None if the string doesn't match this pattern.
    """
    import json
    import re
    m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$", approach.strip(), re.DOTALL)
    if not m:
        return None
    try:
        return {"tool_name": m.group(1), "input": json.loads(m.group(2))}
    except json.JSONDecodeError:
        return None


@dataclass
class TrialSpec:
    condition: str
    benchmark: str
    query_id: str
    seed: int


@dataclass
class TrialResult:
    spec: TrialSpec
    trial_id: str
    completed: bool
    metrics: dict[str, Any]
    transcript_path: str
    wall_time_s: float
    error: str | None


class TokenBudgetExceeded(Exception):
    """Raised when cumulative trial tokens exceed the configured ceiling."""
    def __init__(self, total_tokens: int, ceiling: int, trials_completed: int) -> None:
        super().__init__(
            f"Token budget exceeded: {total_tokens:,} > {ceiling:,} "
            f"after {trials_completed} trials"
        )
        self.total_tokens = total_tokens
        self.ceiling = ceiling
        self.trials_completed = trials_completed


# Per-million-token pricing (USD). Updated 2026-05.
# Override via MODEL_PRICING env var as JSON string if needed.
_DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5":   {"input": 0.80, "output": 4.00},
    "claude-haiku-4-5-0": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-6":  {"input": 3.00, "output": 15.00},
    "claude-opus-4-1":    {"input": 15.00, "output": 75.00},
}

_MTok = 1_000_000


def _trial_cost_usd(metrics: dict[str, Any], model: str) -> float:
    pricing = _DEFAULT_PRICING.get(model, {"input": 3.00, "output": 15.00})
    input_tok = metrics.get("input_tokens", 0) or 0
    output_tok = metrics.get("output_tokens", 0) or 0
    return (input_tok * pricing["input"] + output_tok * pricing["output"]) / _MTok


class CostBudgetExceeded(Exception):
    """Raised when cumulative spend exceeds the configured USD ceiling."""
    def __init__(self, total_cost: float, ceiling: float, trials_completed: int) -> None:
        super().__init__(
            f"Cost budget exceeded: ${total_cost:.2f} > ${ceiling:.2f} "
            f"after {trials_completed} trials"
        )
        self.total_cost = total_cost
        self.ceiling = ceiling
        self.trials_completed = trials_completed


class _LLMLike(Protocol):
    async def call(self, *, system: str, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]] | None = None,
                   temperature: float = 0.0, max_tokens: int = 4096): ...


BaselineFactory = Callable[[str, str, _LLMLike], Any]
"""(condition, task_id, llm) -> BaselineProtocol instance"""


_PLANNER_SYSTEM = """You are the PLANNER agent in a two-agent task.

Your job: decide what the EXECUTOR should do next, and emit a clear,
actionable sub-task as your response. The EXECUTOR has access to all
the tools and will do the actual work.

You MUST delegate to the executor by emitting a clear sub-task in your
response. Do NOT call tools yourself.

Only emit <final_answer>...</final_answer> tags AFTER the executor has
done the work and reported back AND you have verified everything is
complete (e.g. both flight and hotel are booked successfully).

When emitting a final answer, include the booking IDs verbatim
(e.g. F2 and H_LON_1) so the scorer can extract them.

# Task
{task}

# Handoff context from previous agent
{handoff}

# Peer notes (negative knowledge from other agents, if any)
{peer_notes}
"""


_EXECUTOR_SYSTEM = """You are the EXECUTOR agent in a two-agent task.

Your job: take the planner's sub-task and use the provided tools
(search_flights, search_hotels, book_flight, book_hotel) to do it.
Always call tools — never make up data.

When the entire task is complete (both flight AND hotel successfully
booked AND total cost within budget), output your result inside
<final_answer>...</final_answer> tags. Include the booking IDs
verbatim (e.g. F2 and H_LON_1) so they can be extracted.

# Sub-task from planner
{task}

# Handoff context
{handoff}

# Peer notes (negative knowledge from other agents, if any)
{peer_notes}
"""


@dataclass
class TrialRunner:
    llm: _LLMLike
    results_dir: Path
    benchmarks: dict[str, BenchmarkProtocol]
    baseline_factory: BaselineFactory

    async def run_trial(self, spec: TrialSpec) -> TrialResult:
        from harness.agent import Agent
        from harness.metrics import compute_all_metrics
        from harness.orchestration import run_two_agent_trial
        from harness.results import write_trial_result
        from harness.tools import ToolRegistry

        trial_id = str(ULID())
        start = time.perf_counter()
        results_dir = Path(self.results_dir)

        try:
            benchmark = self.benchmarks[spec.benchmark]
            query = benchmark.load_query(spec.query_id)
            query.payload["seed"] = spec.seed

            tools = {t.name: t for t in benchmark.tools_for(query)}
            registry = ToolRegistry(tools=tools)
            # Planner has no tools — forces delegation to executor
            empty_registry = ToolRegistry(tools={})

            baseline = self.baseline_factory(spec.condition, trial_id, self.llm)
            if hasattr(baseline, "setup"):
                await baseline.setup()

            planner = Agent(
                agent_id=f"{trial_id}_planner", role="planner",
                llm=self.llm, tools=empty_registry,
                system_prompt_template=_PLANNER_SYSTEM,
            )
            executor = Agent(
                agent_id=f"{trial_id}_executor", role="executor",
                llm=self.llm, tools=registry,
                system_prompt_template=_EXECUTOR_SYSTEM,
            )

            transcript = await run_two_agent_trial(
                planner=planner, executor=executor, query=query, baseline=baseline,
            )
            score = benchmark.score(query, transcript.final_output)

            # Extract failed paths from ET-MCP trace store (if applicable)
            failed_paths: list[dict[str, Any]] = []
            trace_event_count = 0
            if hasattr(baseline, "server"):
                events = await baseline.server.store.list_for_task(trial_id)
                trace_event_count = len(events)
                for ev in events:
                    if ev.event_type.value == "FAILED_PATH":
                        # The FAILED_PATH payload's "approach" field encodes the
                        # tool call as `book_flight({"flight_id": "F2"})`.
                        # Parse it back into (tool_name, input) shape.
                        approach = ev.payload.get("approach", "")
                        parsed = _parse_failed_path_approach(approach)
                        if parsed is not None:
                            failed_paths.append(parsed)

            metrics = compute_all_metrics(transcript, score, failed_paths=failed_paths)
            metrics["trace_event_count"] = trace_event_count

            if hasattr(baseline, "teardown"):
                await baseline.teardown()

            wall = time.perf_counter() - start
            result = TrialResult(
                spec=spec, trial_id=trial_id, completed=score.completed,
                metrics=metrics, transcript_path="",
                wall_time_s=wall, error=None,
            )
            transcript_path = write_trial_result(result, transcript, results_dir)
            result.transcript_path = str(transcript_path.relative_to(results_dir))
            return result

        except Exception as exc:
            wall = time.perf_counter() - start
            result = TrialResult(
                spec=spec, trial_id=trial_id, completed=False,
                metrics={}, transcript_path="",
                wall_time_s=wall, error=f"{type(exc).__name__}: {exc}",
            )
            try:
                from harness.transcript import Transcript
                write_trial_result(result, Transcript(), results_dir)
            except Exception:
                pass
            return result

    async def run_many(
        self, specs: list[TrialSpec], max_concurrency: int = 10,
        max_total_tokens: int | None = None,
        max_cost_usd: float | None = None,
        model: str = "claude-haiku-4-5",
    ) -> list[TrialResult]:
        """Run trials concurrently with optional cost/token ceiling.

        `max_cost_usd` (preferred) — abort when cumulative USD spend exceeds
        this value; raises CostBudgetExceeded.
        `max_total_tokens` — legacy token-based ceiling; raises TokenBudgetExceeded.

        Both are checked at batch boundaries (every max_concurrency trials).
        """
        use_cost_cap = max_cost_usd is not None
        use_token_cap = max_total_tokens is not None and not use_cost_cap

        if not use_cost_cap and not use_token_cap:
            sem = asyncio.Semaphore(max_concurrency)

            async def _bounded(spec: TrialSpec) -> TrialResult:
                async with sem:
                    return await self.run_trial(spec)

            return await asyncio.gather(*(_bounded(s) for s in specs))

        sem = asyncio.Semaphore(max_concurrency)
        results: list[TrialResult] = []
        cumulative_tokens = 0
        cumulative_cost = 0.0

        async def _bounded(spec: TrialSpec) -> TrialResult:
            async with sem:
                return await self.run_trial(spec)

        for i in range(0, len(specs), max_concurrency):
            batch = specs[i : i + max_concurrency]
            batch_results = await asyncio.gather(*(_bounded(s) for s in batch))
            results.extend(batch_results)
            for r in batch_results:
                tok_in = r.metrics.get("input_tokens", 0) or 0
                tok_out = r.metrics.get("output_tokens", 0) or 0
                cumulative_tokens += tok_in + tok_out
                cumulative_cost += _trial_cost_usd(r.metrics, model)

            if use_cost_cap and cumulative_cost > max_cost_usd:  # type: ignore[operator]
                raise CostBudgetExceeded(
                    total_cost=cumulative_cost,
                    ceiling=max_cost_usd,  # type: ignore[arg-type]
                    trials_completed=len(results),
                )
            if use_token_cap and cumulative_tokens > max_total_tokens:  # type: ignore[operator]
                raise TokenBudgetExceeded(
                    total_tokens=cumulative_tokens,
                    ceiling=max_total_tokens,  # type: ignore[arg-type]
                    trials_completed=len(results),
                )

        return results
