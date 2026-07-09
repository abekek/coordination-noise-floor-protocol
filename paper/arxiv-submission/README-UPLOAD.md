# arXiv upload notes

**Upload file:** `paper/et-mcp-arxiv.tar.gz` (contains everything in this
directory except this README).

- Main file: `et-mcp-arxiv.tex` (arXiv auto-detects it).
- `et-mcp-arxiv.bbl` is included because arXiv does not run BibTeX.
- `acmart.cls` is pinned so arXiv's TeX Live version differences can't
  change the layout.
- Verified self-contained: builds to 9 pages with two pdflatex passes and
  no undefined references.

## Metadata form

- **Title:** How Much Coordination Gain Is Real? A Paired Noise-Floor
  Protocol for Multi-Agent LLM Benchmarks
- **Authors:** Alibek Kaliyev (The University of Texas at Austin),
  Artem Maryanskyy (Uber)
- **Primary category:** cs.MA (Multiagent Systems)
- **Cross-lists:** cs.AI, cs.CL
- **License:** arXiv non-exclusive license (default) is fine for a
  preprint later published at an ACM workshop.

## Abstract (plain text for the form)

Multi-agent LLM coordination papers report small benchmark deltas as
evidence that one architecture beats another. A prior question: how much
paired trial-0 disagreement do two protocols produce on the same model
and benchmark when their API inputs are configuration-equivalent (matched
by code inspection plus a SHA-256 byte audit), short of full
identity-replay? On Claude Haiku 4.5 against $\tau^2$-bench retail, the
clean configuration-equivalent contrast (no_coord vs. intercept, both
inert at trial 0) gives signed paired gaps of +10 pp and 0 pp across two
n=100 seeds; pooled across both, +5 pp with Wilson CI [-2,+12], not
significant. The largest single-seed contrast (+18 pp pull-vs-intercept,
p_corr=0.012) did not reproduce at the second seed (-3 pp, p_corr=1.0);
no trial-0 contrast is significant after Bonferroni at either seed or
pooled. The envelope of observed paired gaps spans [-3,+18] pp across two
seeds, with pooled upper Wilson CI $\lesssim$15 pp. Seven of ten recent
multi-agent coordination architectures report headline effects below this
local floor, and one more sits inside the envelope; whether they survive
a same-model paired replication is, by construction, untested in their
original settings. We define coordination-active pass^k, pass^k
restricted to trials where the coordination mechanism is logically
active, as the minimum reporting protocol, with sample-size targets and
runtime hooks in the body. Measurements run on ET-MCP, a task-scoped
negative-knowledge store conformant with MCP 2026-07-28, used as a
substrate to isolate reader-side choices, not as a contribution. On
Haiku 4.5 the candidate readers (pull, intercept) do not improve trial-1
recovery; we give a preliminary diagnosis of failure modes with
refinements on existing production hook surfaces.

## Caveat before posting

The paper is under double-blind review at the KDD '26 Workshop on Agentic
AI Evaluation and Trustworthiness. Many KDD workshops follow the main
conference's policy, which permits non-anonymous preprints, but check the
workshop CFP's anonymity/dual-posting clause before the arXiv submission
goes live.
