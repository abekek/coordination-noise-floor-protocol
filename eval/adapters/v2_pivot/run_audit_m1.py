"""Run M1 writer-attribution audit on captured transcripts.

For each pull/intercept trial-0 failure in /tmp/v3_audit_transcripts,
extract the agent's tool-call sequence + the recorded last-K=3
FAILED_TRIAL_ACTION events, then ask Haiku 4.5 to classify whether
the recorded last-K window contains the causally-relevant call
(CONTAINED), or whether the causally-relevant call is upstream
(UPSTREAM), or unclear (UNCLEAR).

Output: per-trial classification + summary % CONTAINED for the §6.4
audit paragraph.
"""
import json, pathlib, sys, os, time
import anthropic

REPO = pathlib.Path(__file__).resolve().parents[3]
ROOT = pathlib.Path("/tmp/v3_audit_transcripts")

SYSTEM = """You are auditing a multi-trial tau2-bench retail
coordination experiment. Each failed trial has an agent that made tool
calls and ended with reward 0.0 (failure).

The harness records the LAST K=3 tool calls of each failed trial as
FAILED_TRIAL_ACTION events for future trials. The hypothesis under
audit (M1, writer mis-attribution) is that the LAST K=3 are often
DOWNSTREAM SYMPTOMS of an upstream state mutation that broke the
run — i.e., the causally-relevant call is OUTSIDE the last-K window.

I will show you the full agent tool-call sequence in chronological
order, indicate which 3 are in the last-K window (marked **LAST-K**),
and ask you to judge:

- CONTAINED  : the causally-relevant call is in the LAST-K window
- UPSTREAM   : the causally-relevant call is OUTSIDE the LAST-K
               window (M1 confirmed for this trial)
- UNCLEAR    : you cannot tell from the trajectory alone

Output exactly one of {CONTAINED, UPSTREAM, UNCLEAR} on a single
line, no other content."""


def extract_tool_calls(messages):
    """Pull (tool_name, args_repr) sequence from message history."""
    seq = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for block in m.get("content", []) or []:
            if block.get("type") == "tool_use":
                name = block.get("name", "?")
                args = block.get("input", {}) or {}
                args_compact = json.dumps(args, default=str)[:160]
                seq.append((name, args_compact))
    return seq


def classify(client, tool_calls, last_k_window_indices):
    if not tool_calls:
        return "UNCLEAR"
    user = "Tool calls (chronological, ** marks LAST-K=3 window):\n"
    for i, (name, args) in enumerate(tool_calls):
        marker = "**LAST-K**" if i in last_k_window_indices else "         "
        user += f"  {marker} {i}: {name}({args})\n"
    user += "\nReward: 0.0 (failed). Judge: CONTAINED / UPSTREAM / UNCLEAR."
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=20,
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text.strip().split()[0]
    except Exception as e:
        print(f"  judge error: {e}", file=sys.stderr)
        return "ERROR"


def main():
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    results = {"pull": [], "intercept": []}
    for proto in ["pull", "intercept"]:
        path = ROOT / proto / "trials.jsonl"
        for line in open(path):
            r = json.loads(line)
            if r.get("trial") != 0 or r.get("reward") == 1.0:
                continue
            tool_calls = extract_tool_calls(r.get("messages", []))
            if len(tool_calls) <= 3:
                # last-K is the entire sequence; CONTAINED by definition
                results[proto].append((r["task_id"], "CONTAINED"))
                print(f"{proto} task={r['task_id']}: CONTAINED (trial only {len(tool_calls)} calls)")
                continue
            last_k_window = set(range(len(tool_calls) - 3, len(tool_calls)))
            verdict = classify(client, tool_calls, last_k_window)
            results[proto].append((r["task_id"], verdict))
            print(f"{proto} task={r['task_id']}: {verdict}  ({len(tool_calls)} calls)")
            time.sleep(0.1)
    print()
    print("=== Summary ===")
    for proto, rs in results.items():
        n = len(rs)
        if n == 0:
            continue
        contained = sum(1 for _, v in rs if v == "CONTAINED")
        upstream = sum(1 for _, v in rs if v == "UPSTREAM")
        unclear = sum(1 for _, v in rs if v == "UNCLEAR")
        print(f"{proto}: n={n} CONTAINED={contained} ({100*contained/n:.0f}%) "
              f"UPSTREAM={upstream} ({100*upstream/n:.0f}%) "
              f"UNCLEAR={unclear} ({100*unclear/n:.0f}%)")
    json.dump(results, open(REPO / "eval/results/m1_audit.json", "w"), indent=2)
    print("wrote eval/results/m1_audit.json")


if __name__ == "__main__":
    main()
