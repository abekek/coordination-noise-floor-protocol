"""Common interface for benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class BenchmarkQuery:
    query_id: str
    payload: dict[str, Any]
    difficulty: str   # "easy" | "medium" | "hard"


@dataclass
class BenchmarkResult:
    query_id: str
    completed: bool
    raw_output: Any


class BenchmarkProtocol(ABC):
    name: str

    @abstractmethod
    def load_queries(self, *, subset: str | None = None) -> list[BenchmarkQuery]: ...

    @abstractmethod
    def score(self, query: BenchmarkQuery, output: Any) -> BenchmarkResult: ...
