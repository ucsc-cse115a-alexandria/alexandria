# Release Plan

- **Product:** Alexandria
- **Team:** Alexandria Team
- **Release:** 1.0
- **Release date:** 2026-07-21 (end of Sprint 4)
- **Revision:** 4 (2026-06-30)

See [`spec.md`](spec.md) for the design and [`tech-stack.md`](tech-stack.md) for the stack.

## High-level goals

- **G1 Reduction**: compress a prompt as efficiently as possible — cut tokens without dropping
  instructions.
- **G2 Accuracy**: on a benchmark that cannot leak, show the shortened prompt keeps the agent at
  least as accurate as the original — and raise that accuracy as far as the token savings allow.
- **G3 Usability**: make the library and CLI easy enough that anyone can compress a prompt and see
  the accuracy/cost trade-off they get.

## Approach

Four one-week sprints, one user story each. Sprint 1 ships a working command end to end; each later
sprint adds value on top of it. Sprints 2–4 form a measure → improve → confirm arc: first make
accuracy measurable, then push it up, then lock in and ship the gain. Each story is sized to fit one
sprint; the implementation tasks that deliver it are estimated in ideal hours in that sprint's Sprint
Plan.

| Sprint | Dates | Focus |
|--------|-------|-------|
| 1 | Jun 24-30 | Shorten a prompt end to end via one CLI command |
| 2 | Jul 1-7 | Measure accuracy before and after on a leak-proof benchmark |
| 3 | Jul 8-14 | Raise accuracy by trying multiple compression strategies |
| 4 | Jul 15-21 | Re-measure the best strategy and ship the library + CLI |

## Sprints

### Sprint 1: Shorten a prompt end to end

- **User story** (G1, G3): As an engineer who uses Cursor or Claude Code, I want a one CLI command
  that cuts the token count of my agent-instruction file (`AGENT.md` / `CLAUDE.md`) by removing redundant instructions while keeping meaning intact, so that I cut cost and avoid the accuracy loss that comes with a bloated prompt.
- **Spike**: Survey prompt-optimization work, how token count affects LLM accuracy, and existing
  prompt/agent benchmarks — prepares the Sprint 2 benchmark.
- **Infrastructure**: Repo scaffold and dependencies; CI running lint, type check, and tests on
  every push.

### Sprint 2: Measure the accuracy of a shortened prompt

- **User story** (G2): As that engineer, I want to know whether the shortened prompt keeps my agent just as accurate before I rely on it, so that I'm not trading reliability for a smaller prompt without realizing it.
- **Spike**: Pick an evaluation task with published prompts and ground truth so accuracy is
  measurable, and design label-by-construction edit pairs the benchmark cannot leak on.
- **Infrastructure**: A benchmark runner that executes an agent on a prompt, records its outputs,
  and scores accuracy against ground truth.

### Sprint 3: Push accuracy higher with more strategies

- **User story** (G1, G2): As that engineer, I want the command to compress harder while the benchmark still shows my agent is as accurate — by combining multiple compression strategies into one pass and keeping only the edits the benchmark backs up — so that I save more tokens without giving up accuracy or hand-picking a method.
- **Spike**: Explore packaging and distribution options (PyPI, pipx) — prepares the Sprint 4 release.
- **Infrastructure**: Support for running multiple optimizers together, concatenating their proposed edits into one ranked stack for Select to fold.

### Sprint 4: Confirm the gain and ship

- **User story** (G3): I want to use Alexandria as both a CLI and a Python library, with clear setup docs so I can get going quickly, so that I can drop verified prompt compression into my own scripts and pipelines and know the accuracy and cost I am getting.
- **Infrastructure**: Re-run the benchmark on the chosen default strategy to confirm its
  accuracy/compression numbers, then package and publish the CLI so it installs in one command.

## Capacity sanity check

- Team of 4, one-week sprints.
- One user story per sprint keeps each sprint focused; spikes and infrastructure tasks fill out the
  remaining capacity and are detailed in the Sprint Plan.
- Totals are intentionally light because evaluation and improvement are open-ended and run alongside
  other course work.

## Product backlog (not in Release 1.0)

- Hosted embedding API backend.
- Per-model exact tokenizers.
- Rewriting instructions beyond drop and merge.
- A configurable redundancy metric.
- User control over how aggressively the prompt is compressed.

We revisit and update the release plan and product backlog after each sprint.
