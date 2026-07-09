"""In-process ET-MCP trace store for the tau2-bench adapter.

This is a simplified version of the production EtMcpServer; it speaks the same
typed event API but skips the MCP HTTP transport for in-process speed during
evaluation. Production code path is unchanged.

The store is task-scoped: allocate on agent init, tear down at task close.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal


EventType = Literal[
    "FAILED_PATH",
    "CONSTRAINT_VIOLATION",
    "ABANDONED_APPROACH",
    "INTERMEDIATE_DECISION",
    "TOOL_ERROR",
]


@dataclass
class TraceEvent:
    event_id: str
    task_id: str
    event_type: EventType
    agent_id: str
    timestamp: float
    payload: dict[str, Any]
    version: int = 1

    def text_for_ranking(self) -> str:
        """Flattened text used by the TF-IDF ranker."""
        parts = [self.event_type.lower(), self.agent_id]
        for k, v in self.payload.items():
            parts.append(f"{k} {v}")
        return " ".join(str(p) for p in parts)


@dataclass
class TraceStore:
    """Task-scoped append-only event store with TF-IDF query.

    Single-process; no locking; intended for in-process eval use. The
    production EtMcpServer enforces concurrency + lifecycle via MCP Tasks.
    """

    task_id: str
    events: list[TraceEvent] = field(default_factory=list)

    def write(
        self,
        event_type: EventType,
        agent_id: str,
        payload: dict[str, Any],
    ) -> TraceEvent:
        evt = TraceEvent(
            event_id=str(uuid.uuid4()),
            task_id=self.task_id,
            event_type=event_type,
            agent_id=agent_id,
            timestamp=time.time(),
            payload=payload,
        )
        self.events.append(evt)
        return evt

    def query(
        self,
        question: str,
        event_types: list[EventType] | None = None,
        agent_id: str | None = None,
        limit: int = 5,
    ) -> list[TraceEvent]:
        """Filter then TF-IDF rank then top-k."""
        candidates = [
            e
            for e in self.events
            if (event_types is None or e.event_type in event_types)
            and (agent_id is None or e.agent_id == agent_id)
        ]
        if not candidates:
            return []
        return self._tfidf_rank(question, candidates)[:limit]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())

    @classmethod
    def _tfidf_rank(
        cls, question: str, events: list[TraceEvent]
    ) -> list[TraceEvent]:
        docs = [cls._tokenize(e.text_for_ranking()) for e in events]
        q_tokens = cls._tokenize(question)
        if not q_tokens:
            return events
        # Document frequency
        df: Counter[str] = Counter()
        for doc in docs:
            for tok in set(doc):
                df[tok] += 1
        n_docs = len(docs)

        def score(doc: list[str]) -> float:
            tf = Counter(doc)
            return sum(
                tf[tok] * (1.0 / (1 + df[tok]))
                for tok in q_tokens
                if tok in tf
            )

        scored = sorted(zip(docs, events), key=lambda x: score(x[0]), reverse=True)
        return [e for _, e in scored]

    def summary(self) -> str:
        """One-line per event, suitable for prompt inlining."""
        if not self.events:
            return "(no peer events yet)"
        lines = []
        for e in self.events:
            payload_str = json.dumps(e.payload, separators=(",", ":"))
            lines.append(f"- [{e.event_type}] {payload_str}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.events)
