"""REALM-Bench Wedding Logistics adapter (stub for Phase 2)."""

from __future__ import annotations

from typing import Any

from benchmarks.base import BenchmarkProtocol, BenchmarkQuery, BenchmarkResult


class RealmBenchWeddingBenchmark(BenchmarkProtocol):
    name = "realm_bench_wedding"

    def load_queries(self, *, subset: str | None = None) -> list[BenchmarkQuery]:
        raise NotImplementedError("Phase 2: load Wedding Logistics scenarios")

    def score(self, query: BenchmarkQuery, output: Any) -> BenchmarkResult:
        raise NotImplementedError("Phase 2: REALM-Bench scoring routine")
