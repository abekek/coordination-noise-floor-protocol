"""Common interface for handoff baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class HandoffContext:
    """What an agent makes available to its successor."""
    payload: dict[str, Any]
    token_cost: int

    def payload_as_text(self) -> str:
        import json
        return json.dumps(self.payload, indent=2, sort_keys=True)


class BaselineProtocol(ABC):
    name: str

    @abstractmethod
    async def prepare_handoff(
        self, agent_id: str, agent_state: dict[str, Any],
    ) -> HandoffContext: ...

    @abstractmethod
    async def consume_handoff(
        self, agent_id: str, context: HandoffContext,
    ) -> None: ...

    async def query_during_step(
        self, agent_id: str, question: str,
    ) -> str | None:
        """Return peer knowledge to inject as `peer_notes`, or None.

        Default: None (push-based baselines have no pull channel).
        Only EtMcpBaseline overrides.
        """
        return None
