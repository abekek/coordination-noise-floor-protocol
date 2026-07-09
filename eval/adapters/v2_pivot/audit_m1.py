"""Writer-attribution audit for M1 (writer mis-attribution).

For each trial-0 failure under pull and intercept on the n=100
retail data, LLM-judge whether the last-K=3 tool calls recorded as
FAILED_TRIAL_ACTION events contain the causally-relevant call (the
one that broke the run), or whether the relevant call is upstream
of the recorded window (the M1 hypothesis).

Run from project root:
    /Users/alibek/anaconda3/bin/python3 -m \\
        eval.adapters.v2_pivot.audit_m1 \\
        --root eval/results/v2pivot_w1_n100 \\
        --proto pull \\
        --max-trials 50

Requires save_transcripts=True in the original sweep, OR we
re-derive tool calls from the harness logs.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import anthropic

_REPO = pathlib.Path(__file__).resolve().parents[3]


SYSTEM = """You are auditing a multi-trial tau2-bench retail
coordination experiment. For each failed trial, I will show you (a) the
agent's recorded tool calls in chronological order, (b) the recorded
last-K=3 calls flagged by the harness as FAILED_TRIAL_ACTION events
(these are the events the writer wrote to the coordination store), and
(c) the final reward (always 0.0 since the trial failed).

Your job: judge whether the last-K=3 window CONTAINS the causally-
relevant tool call — the call that, had it not happened or had it been
called differently, would most likely have led to a successful trial.

Return one of:
- CONTAINED  : The causally-relevant call is in the last-K window.
- UPSTREAM   : The causally-relevant call is OUTSIDE the last-K window
               (typically an early state-mutation that left the env
               unrecoverable; the recorded last-K calls are downstream
               symptoms).
- UNCLEAR    : You cannot tell from the trajectory alone.

Output exactly one of these three words on a single line, no other
content."""


def audit_trial(client, tool_calls, last_k_events, model="claude-haiku-4-5"):
    user = "Tool calls (chronological):\n"
    for i, (name, args) in enumerate(tool_calls):
        args_compact = json.dumps(args, default=str)[:200]
        user += f"  {i}: {name}({args_compact})\n"
    user += "\nLast-K=3 events recorded by writer:\n"
    for e in last_k_events:
        args_compact = json.dumps(e.get("args", {}), default=str)[:200]
        user += f"  {e.get('name', '?')}({args_compact})\n"
    user += "\nReward: 0.0 (failed trial). Judge: CONTAINED / UPSTREAM / UNCLEAR."
    resp = client.messages.create(
        model=model,
        max_tokens=20,
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip().split()[0]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--proto", choices=["pull", "intercept"], default="pull")
    p.add_argument("--max-trials", type=int, default=50)
    p.add_argument("--model", default="claude-haiku-4-5")
    args = p.parse_args(argv)

    root = pathlib.Path(args.root)
    trials_path = root / args.proto / "trials.jsonl"
    if not trials_path.exists():
        print(f"missing {trials_path}", file=sys.stderr)
        return 2

    failed = []
    for line in open(trials_path):
        r = json.loads(line)
        if r.get("trial") == 0 and r.get("reward") != 1.0:
            failed.append(r)

    print(f"{len(failed)} trial-0 failures in {args.proto}")
    print("Note: agent message transcripts not saved in this sweep "
          "(save_transcripts=False); writer-attribution audit requires "
          "re-running with save_transcripts=True to get tool_calls.")
    print()
    print("Falling back to: count last-K=3 events per failed trial and "
          "report 'store_events_at_end' distribution.")
    sizes = [r.get("store_events_at_end", 0) for r in failed]
    print(f"store_events_at_end distribution across {len(failed)} failures:")
    print(f"  mean: {sum(sizes)/max(len(sizes),1):.2f}")
    print(f"  min: {min(sizes)}, max: {max(sizes)}")
    print()
    print("To run the full M1 audit, re-run a smaller sweep with "
          "--save-transcripts and pipe through this script with the "
          "actual tool_calls field. The infrastructure is in place; "
          "the audit run is the missing piece.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
