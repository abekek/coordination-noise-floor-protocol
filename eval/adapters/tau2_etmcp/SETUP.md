# tau2-bench adapter setup

The vendored tau2-bench clone is gitignored. To bring it up locally:

```bash
mkdir -p eval/benchmarks/tau2_bench
cd eval/benchmarks/tau2_bench
git clone --depth 1 https://github.com/sierra-research/tau2-bench.git _vendor

cd _vendor
uv venv --python 3.12 .venv
uv pip install -e . --python .venv/bin/python
```

Pinned tau2-bench version: latest `main` as of 2026-06-05 (we follow MIT
upstream; the adapter has no dependencies on internal tau2 details that
could break across minor versions).

## Smoke check

From repo root:
```bash
eval/benchmarks/tau2_bench/_vendor/.venv/bin/python \
    eval/adapters/tau2_etmcp/smoke_register.py
```
should print:
```
Registered: True; available agents: [...et_mcp_agent...]
TraceStore wrote 1, query returned 1 hit(s).
OK: smoke checks passed.
```

## Running tau2 with the ET-MCP agent

The agent factory is registered as a side effect of importing
`eval.adapters.tau2_etmcp.factory`. The wrapper CLI ensures this happens
before tau2 looks up the agent:

```bash
eval/benchmarks/tau2_bench/_vendor/.venv/bin/python \
    eval/adapters/tau2_etmcp/tau2_etmcp_cli.py \
    run \
    --domain mock \
    --agent et_mcp_agent \
    --agent-llm "claude-haiku-4-5" \
    --agent-llm-args '{"coord_protocol": "et_mcp"}' \
    --user dummy_user \
    --num-trials 1 \
    --task-ids mock_0001
```

Coordination protocol is selected via `--agent-llm-args`:
- `{"coord_protocol": "no_coord"}` — no writes, no reads (baseline)
- `{"coord_protocol": "push_scratchpad"}` — CA-MCP-style shared dict (Day 2)
- `{"coord_protocol": "message_passing"}` — explicit handoff message (Day 2)
- `{"coord_protocol": "et_mcp"}` — pull-based trace store (default)

For the cross-trial coordination experiment, the trace store is allocated
per-task and shared across the N trials of that task. See Day 4 sweep
script (TBD).

## API keys

tau2 uses `litellm`, which picks up provider keys from environment:
- `ANTHROPIC_API_KEY` for Claude models
- `OPENAI_API_KEY` for GPT models (and for the user simulator default)

For the Day 4 full sweep on Uber's gen-AI gateway, set instead:
```bash
export ANTHROPIC_BASE_URL=https://genai-api.uberinternal.com/
export ANTHROPIC_AUTH_TOKEN=<token from get_usso_token>
```
