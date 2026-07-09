"""CLI wrapper: ensure our adapter registers before tau2.cli.main runs.

Usage:
    eval/benchmarks/tau2_bench/_vendor/.venv/bin/python \\
        eval/adapters/tau2_etmcp/tau2_etmcp_cli.py \\
        run --domain mock --agent et_mcp_agent --num-trials 1 ...

This is a thin wrapper around `tau2 run` that imports our factory module
before delegating to tau2.cli:main, so the et_mcp_agent factory is
registered by the time tau2 looks it up.
"""

from __future__ import annotations

import pathlib
import sys


def _ensure_repo_on_path() -> None:
    """Insert the repo root onto sys.path so `eval.adapters.*` resolves
    even when this script is invoked from the tau2 venv."""
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main() -> int:
    _ensure_repo_on_path()
    # Register our agent before tau2 reads the registry.
    from eval.adapters.tau2_etmcp import factory  # noqa: F401

    from tau2.cli import main as tau2_main
    tau2_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
