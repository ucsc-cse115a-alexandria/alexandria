# Release Plan

- **Product:** Alexandria
- **Team:** Alexandria Team
- **Release:** 1.0
- **Release date:** 2026-07-21 (end of Sprint 3)
- **Revision:** 5 (2026-07-03)

See [`spec.md`](spec.md) for the design and [`tech-stack.md`](tech-stack.md) for the stack.

## High-level goals

- **G1 Reduction**: compress a prompt as efficiently as possible — cut tokens without dropping
  instructions.
- **G2 Accuracy**: on a benchmark that cannot leak, show the shortened prompt keeps the agent at
  least as accurate as the original — and raise that accuracy as far as the token savings allow.
- **G3 Usability**: make the library and CLI easy enough that anyone can compress a prompt and see
  the accuracy/cost trade-off they get.

## Approach

Four one-week sprints. Each sprint delivers one or more user stories sized to fit the sprint. Sprint 0
ships a working command end to end; each later sprint adds value on top of it. Sprints 1–3 form a
build → improve → release arc: first build the benchmark and make accuracy measurable, then use it to
improve compression and round out the CLI, then release and broaden the accuracy evidence. The
implementation tasks that deliver each story are estimated in ideal hours in that sprint's Sprint Plan.

| Sprint | Dates | Focus |
|--------|-------|-------|
| 0 | Jun 24-30 | Shorten a prompt end to end via one CLI command |
| 1 | Jul 1-7 | Build a leak-proof benchmark, prove accuracy holds, and add trade-off controls |
| 2 | Jul 8-14 | Improve accuracy/compression against the benchmark and add convenience CLI commands |
| 3 | Jul 15-21 | Publish the package and broaden the accuracy evaluation |

## Sprints

### Sprint 0: Shorten a prompt end to end

- **User story** (G1, G3): As an engineer using Cursor or Claude Code whose `CLAUDE.md` / `AGENT.md` has grown bloated, I want a one CLI command that cuts the token count of that agent-instruction file by removing redundant instructions while keeping meaning intact, so that I cut the per-token cost I pay on every request and avoid the accuracy loss that comes with a bloated prompt.
- **Spike**: Survey prompt-optimization work, how token count affects LLM accuracy, and existing
  prompt/agent benchmarks — prepares the Sprint 1 benchmark.
- **Infrastructure**: Repo scaffold and dependencies; CI running lint, type check, and tests on
  every push.

### Sprint 1: Build the benchmark, prove accuracy, add trade-off controls

See [`sprint-1-plan.md`](sprint-1-plan.md) for the detailed task breakdown.

- **User story 1** (G2): As an engineer who compresses my `CLAUDE.md` / `AGENT.md` with Alexandria, I want to see proof that the shortened prompt keeps my coding agent just as accurate as the original, so that I can switch to the smaller prompt without quietly trading reliability for a lower token bill.
- **User story 2** (G3): As an engineer building an LLM application whose system prompt is sent on every API call at scale, I want CLI options to cap how hard Alexandria compresses my prompt (a `--min-similarity` floor and a `--max-tokens` budget) and to see the token savings I get, so that I can push compression as far as my cost target needs, even when I am willing to spend a little accuracy for a cheaper API bill.
- **Spike**: Choose a leak-proof, length-sensitive benchmark against explicit acceptance criteria: established (20+ citations or 100+ stars), score degrades on 2× and 10× inflated prompts, and a subset yields a significant result in 10 minutes or less at $1 or less.
- **Infrastructure**: (a) The leak-proof fidelity dataset — a skill-corpus download script, a redundancy-inflation script, and a versioned, label-by-construction dataset with a 99%-similarity fidelity check. (b) Split the library from the CLI so the CLI only parses arguments and calls the public API.

### Sprint 2: Improve compression and add CLI convenience

- **User story 1** (G1, G2): As an engineer building an LLM application where every extra percent of prompt reduction is real money, I want the command to compress harder while the benchmark still confirms my agent is as accurate — by combining multiple compression strategies into one pass and keeping only the edits the benchmark backs up — so that I save more tokens without giving up accuracy or hand-picking a method.
- **User story 2** (G3): As an engineer using Cursor or Claude Code with a bloated `CLAUDE.md`, I want convenience commands that show me what compression did before I adopt the output (e.g. a preview of which instructions were dropped and a report of similarity and token metrics), so that I can switch to the compressed prompt with confidence. Exact commands are decided in the Sprint 2 Plan.
- **Spike**: Explore packaging and distribution options (PyPI, pipx) — prepares the Sprint 3 release.
- **Infrastructure**: Support for running multiple optimizers together, concatenating their proposed edits into one ranked stack for Select to fold. Any Sprint 1 spillover from the fidelity dataset lands here.

### Sprint 3: Release and broaden the accuracy evaluation

- **User story 1** (G3): As an engineer who wants to drop Alexandria into my own scripts and pipelines, I want to install it in one command and use it as both a CLI and a Python library, with clear setup docs and published benchmark numbers on the default strategy, so that I get verified prompt compression and know the accuracy and cost I am getting.
- **User story 2** (G2): As an engineer weighing Alexandria for my own setup, I want the accuracy proof to generalize beyond a single benchmark (e.g. additional benchmarks or tasks and a compression-strength sweep that shows the accuracy/cost curve), so that I can trust the result applies to my prompts. The evaluations to add are decided in the Sprint 3 Plan.
- **Infrastructure**: Re-run and extend the benchmark evaluation for the chosen default strategy to
  confirm its accuracy/compression numbers, then publish the package to PyPI so it installs in one command.

## Capacity sanity check

- Team of 4, one-week sprints.
- Each sprint carries one or more user stories sized to fit; spikes and infrastructure tasks fill out
  the remaining capacity and are detailed in the Sprint Plan.
- Totals are intentionally light because evaluation and improvement are open-ended and run alongside
  other course work.

## Product backlog (not in Release 1.0)

- Hosted embedding API backend.
- Per-model exact tokenizers.
- Rewriting instructions beyond drop and merge.
- A configurable redundancy metric.

We revisit and update the release plan and product backlog after each sprint.
