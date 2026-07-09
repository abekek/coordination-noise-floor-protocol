"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def task_id() -> str:
    return "task_test_001"


@pytest.fixture
def agent_id() -> str:
    return "agent_alpha"
