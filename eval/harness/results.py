"""JSONL writer/reader for trial artifacts."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from harness.runner import TrialResult
from harness.transcript import Transcript


def write_trial_result(
    result: TrialResult, transcript: Transcript, results_dir: Path,
) -> Path:
    """Append one JSONL line + write the full transcript JSON.

    Returns the path to the transcript file.
    """
    results_dir = Path(results_dir)
    (results_dir / "transcripts").mkdir(parents=True, exist_ok=True)

    transcript_path = results_dir / "transcripts" / f"{result.trial_id}.json"
    transcript_path.write_text(
        json.dumps(_transcript_to_dict(transcript), indent=2, default=str)
    )

    line = {
        "trial_id": result.trial_id,
        "spec": asdict(result.spec),
        "completed": result.completed,
        "metrics": result.metrics,
        "wall_time_s": result.wall_time_s,
        "transcript_path": str(transcript_path.relative_to(results_dir)),
        "error": result.error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with (results_dir / "trials.jsonl").open("a") as fh:
        fh.write(json.dumps(line) + "\n")
    return transcript_path


def read_trial_results(results_dir: Path) -> Iterator[dict[str, Any]]:
    path = Path(results_dir) / "trials.jsonl"
    if not path.exists():
        return
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _transcript_to_dict(transcript: Transcript) -> dict[str, Any]:
    return {
        "final_output": transcript.final_output,
        "hit_max_handoffs": transcript.hit_max_handoffs,
        "totals": {
            "llm_calls": transcript.llm_calls_total(),
            "input_tokens": transcript.input_tokens_total(),
            "output_tokens": transcript.output_tokens_total(),
            "cache_read_tokens": transcript.cache_read_tokens_total(),
        },
        "steps": [_step_to_dict(s) for s in transcript.steps],
    }


def _step_to_dict(step) -> dict[str, Any]:
    d = asdict(step) if dataclasses.is_dataclass(step) else dict(step.__dict__)
    # Drop raw_messages from the public transcript (debugging only,
    # would explode file size).
    d.pop("raw_messages", None)
    return d
