# How Much Coordination Gain Is Real?

**A Paired Noise-Floor Protocol for Multi-Agent LLM Benchmarks**

Alibek Kaliyev (UT Austin) · Artem Maryanskyy (Uber)

Accepted at the **KDD 2026 Workshop on Agentic AI Evaluation and Trustworthiness**.

📄 **[Interactive paper page](https://abekek.github.io/coordination-noise-floor-protocol/)** ·
[PDF](https://abekek.github.io/coordination-noise-floor-protocol/et-mcp-paper.pdf) ·
[Poster](https://abekek.github.io/coordination-noise-floor-protocol/poster.pdf)

## TL;DR

Multi-agent coordination papers report small benchmark deltas as evidence one
architecture beats another. We ask the prior question: how much do two protocols
disagree when their API inputs are **configuration-equivalent** (matched by code
inspection plus SHA-256 byte audits) — i.e., when any gap is noise, not
architecture? On τ²-bench retail with Claude Haiku 4.5, the observed paired-gap
envelope spans **[−3, +18] pp across two n=100 seeds** (pooled clean-contrast CI
[−2, +12] pp; largest pooled upper Wilson CI ≈ 15 pp). Seven of ten recent
multi-agent coordination architectures report headline gains below this
envelope. We propose **coordination-active pass^k** — pass^k restricted to
trials where the coordination mechanism is logically active — as the minimum
reporting protocol, and show that even an **oracle** injecting ground-truth
golden actions does not separate from baseline at feasible n: the benchmark's
coordination-active subset, not the reader mechanisms, is the binding
constraint.

## Repository layout

| Path | What it is |
|---|---|
| `paper/latex/` | LaTeX source (workshop version `et-mcp.tex`, arXiv variant `et-mcp-arxiv.tex`, shared `sections/` + `figures/`) |
| `paper/arxiv-submission/` | Flattened, self-contained arXiv upload tree |
| `paper/poster/` | KDD workshop poster (pure-TikZ XeLaTeX source + PDF; needs Helvetica Neue and Menlo, i.e. macOS fonts) |
| `site/` | The interactive paper page (single self-contained HTML file) |
| `eval/adapters/v2_pivot/` | **The measurement harness**: Anthropic-native τ²-bench orchestrator, the four protocols, sweep runner, paired-stats analysis, request-equivalence audits |
| `eval/results/` | Raw per-trial JSONL artifacts behind the paper's tables/figures (see inventory below) |
| `eval/benchmarks/tau2_bench/_vendor/` | Vendored [τ²-bench](https://github.com/sierra-research/tau2-bench) (MIT, unmodified — we replace only the conversation-state layer). Ships source + the retail/airline/mock domain data the paper uses; telecom/banking domains and voice data are omitted — fetch from upstream if needed |
| `prototype/` | ET-MCP, the MCP-conformant task-scoped trace-store substrate |
| `eval/analysis/`, `eval/harness/` | Earlier general harness + stats/table utilities |

"ET-MCP" is the Ephemeral Trace store substrate the measurements run on; the
paper's contribution is the measurement protocol, not the substrate.

## Reproducing the results

### Requirements

- Python 3.12, [uv](https://docs.astral.sh/uv/)
- An Anthropic API key in `.env` at the repo root: `ANTHROPIC_API_KEY=...`
- Rough costs at Haiku 4.5 prices: ~$0.10/trial, ~31 s/trial serial

### Setup

```bash
# 1. τ²-bench vendored environment (runs the sweeps)
cd eval/benchmarks/tau2_bench/_vendor
python3.12 -m venv .venv
.venv/bin/pip install -e . anthropic

# 2. (optional) prototype + eval environments for unit tests / analysis utils
cd prototype && uv sync && cd ..
cd eval && uv sync && cd ..
```

### Headline sweep (Tables 1–2, Figures 2–3; ~600 trials, ~$56, ~5 h)

From the repo root:

```bash
set -a && source .env && set +a
VENV=eval/benchmarks/tau2_bench/_vendor/.venv/bin/python

$VENV -m eval.adapters.v2_pivot.run_matrix \
  --domain retail \
  --task-ids-file eval/results/v2pivot_w1_n100/task_ids.json \
  --num-trials 2 --protocols no_coord,pull,intercept \
  --writer last_k --writer-last-k 3 \
  --out-root eval/results/my_retail_n100
```

The Messages API exposes no sampling seed, so T=0 runs are not
bit-deterministic — expect per-cell rates to move within the paper's
noise envelope. That is, in fact, the point of the paper.

Cross-domain / cross-model probes (§6.4): same command with
`--domain airline`, or `--agent-model claude-sonnet-4-5
--user-model claude-sonnet-4-5`, on 30-task subsets.

### Oracle positive control (§7 P0; 200 trials, ~$19, ~1.7 h)

```bash
$VENV -m eval.adapters.v2_pivot.run_matrix \
  --domain retail \
  --task-ids-file eval/results/v2pivot_w1_n100/task_ids.json \
  --num-trials 2 --protocols oracle \
  --writer last_k --writer-last-k 3 \
  --out-root eval/results/my_oracle_n100
```

### Analysis and figures

```bash
# Paired sign tests, Cliff's δ, corrected p-values, LaTeX fragments
$VENV -m eval.adapters.v2_pivot.analyze \
  --root eval/results/v2pivot_w1_n100 \
  --out /tmp/summary_paired.json

# Regenerate the paper's matplotlib figures from raw trials
python3 paper/latex/figures/make_plots.py

# Request-equivalence SHA-256 audit (§6.3.1, E1)
$VENV -m eval.adapters.v2_pivot.audit_request_equivalence --help
```

### Tests (no API calls)

```bash
eval/benchmarks/tau2_bench/_vendor/.venv/bin/python eval/adapters/v2_pivot/test_oracle.py
cd prototype && uv run pytest -q
cd eval && uv run pytest -q
```

## Data inventory

| Directory | Trials | Backs |
|---|---|---|
| `eval/results/v2pivot_w1_n100/` | 600 | Headline seed-1: Tables 1–2, Fig. 2, Fig. 3 group 1 |
| `eval/results/v3_airline_n30/` | 180 | Haiku airline probe (§6.4, Fig. 3) |
| `eval/results/v3_sonnet_n30/` | 180 | Sonnet retail probe (§6.4, Fig. 3) |
| `eval/results/v3_oracle_n100/` | 200 | Oracle positive control (§7 P0) |
| `eval/results/m1_audit.json` | — | M1 writer mis-attribution judge audit (§6.3.2) |

**Transparency note (seed 2).** The second n=100 seed reported in §6.3.2
(trial-0 sign tests 13/13/74, 15/18/67, 18/21/61; trial-1 rates
0.52/0.54/0.51) was run on 2026-06-09, but its raw per-trial JSONLs were
written to a temporary directory and not retained. The paired W/L/T counts
and derived statistics are recorded in the paper and in
`paper/latex/figures/make_plots.py` (`noise_rows`); the raw files for that
seed cannot be re-shared. Re-running the sweep produces a fresh seed drawn
from the same envelope, not a byte-identical replication — consistent with
the paper's own account of API nondeterminism (§6.3.1, E2).

## Building the paper

```bash
cd paper/latex && make          # → et-mcp.pdf (workshop version)
latexmk -pdf et-mcp-arxiv.tex   # → arXiv variant
```

Requires TeX Live with `acmart` and `latexmk`.

## Citation

```bibtex
@inproceedings{kaliyev2026noisefloor,
  title     = {How Much Coordination Gain Is Real? A Paired Noise-Floor
               Protocol for Multi-Agent {LLM} Benchmarks},
  author    = {Kaliyev, Alibek and Maryanskyy, Artem},
  booktitle = {KDD 2026 Workshop on Agentic AI Evaluation
               and Trustworthiness},
  year      = {2026},
  url       = {https://abekek.github.io/coordination-noise-floor-protocol/}
}
```

## License

MIT — see [LICENSE](LICENSE). The vendored τ²-bench under
`eval/benchmarks/tau2_bench/_vendor/` retains its upstream MIT license and
attribution.
