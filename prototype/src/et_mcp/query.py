"""Three-stage filter → rank → project pipeline for trace.query."""

from __future__ import annotations

import json
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from et_mcp.events import Event, EventType
from et_mcp.store import TraceStore


@dataclass
class QueryResult:
    events: list[Event]
    summaries: list[str]   # parallel array, one per event
    scores: list[float]    # parallel array, one per event


def _event_text(event: Event) -> str:
    parts = [event.event_type.value, event.agent_id]
    parts.append(json.dumps(event.payload, sort_keys=True))
    return " ".join(parts)


def _summarize(event: Event) -> str:
    p = event.payload
    et = event.event_type
    if et == EventType.FAILED_PATH:
        return f"FAILED_PATH({event.agent_id}): {p.get('approach')} — {p.get('reason')}"
    if et == EventType.CONSTRAINT_VIOLATION:
        return (
            f"CONSTRAINT_VIOLATION({event.agent_id}): {p.get('constraint')} "
            f"(attempted={p.get('value_attempted')})"
        )
    if et == EventType.ABANDONED_APPROACH:
        return (
            f"ABANDONED_APPROACH({event.agent_id}): {p.get('description')} — "
            f"{p.get('why_abandoned')}"
        )
    if et == EventType.INTERMEDIATE_DECISION:
        return (
            f"INTERMEDIATE_DECISION({event.agent_id}): {p.get('decision')} "
            f"(conf={p.get('confidence')}, reversible={p.get('reversible')})"
        )
    if et == EventType.TOOL_ERROR:
        return (
            f"TOOL_ERROR({event.agent_id}): {p.get('tool_name')} → "
            f"{p.get('error')}"
        )
    return f"{et.value}({event.agent_id})"


def _rank(question: str, events: list[Event]) -> list[float]:
    """TF-IDF cosine similarity. Returns one score per event."""
    if not events:
        return []
    corpus = [_event_text(e) for e in events] + [question]
    vec = TfidfVectorizer(lowercase=True, stop_words="english")
    try:
        matrix = vec.fit_transform(corpus)
    except ValueError:
        # Empty vocabulary (e.g. question is only stop words and events also)
        return [0.0] * len(events)
    sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
    return [float(s) for s in sims]


async def run_query(
    store: TraceStore,
    *,
    task_id: str,
    question: str,
    event_types: list[EventType] | None = None,
    agent_id: str | None = None,
    limit: int = 10,
) -> QueryResult:
    all_events = await store.list_for_task(task_id)

    # Stage 1: filter
    filtered = [
        e for e in all_events
        if (event_types is None or e.event_type in event_types)
        and (agent_id is None or e.agent_id == agent_id)
    ]

    # Stage 2: rank
    scores = _rank(question, filtered)
    ranked = sorted(
        zip(filtered, scores), key=lambda pair: pair[1], reverse=True
    )

    # Stage 3: project
    top = ranked[:limit]
    events = [e for e, _ in top]
    return QueryResult(
        events=events,
        summaries=[_summarize(e) for e in events],
        scores=[s for _, s in top],
    )
