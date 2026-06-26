# Release Plan

- **Product:** Alexandria
- **Team:** Alexandria Team
- **Release:** 1.0
- **Release date:** 2026-07-21 (end of Sprint 4)
- **Revision:** 2 (2026-06-26)

See [`spec.md`](spec.md) for the design and [`tech-stack.md`](tech-stack.md) for the stack.

## High-level goals

1. **Scoring**: show which instructions in a prompt overlap, with no labels and no training.
2. **Reduction**: remove redundant instructions while preserving meaning.
3. **Evaluation**: report concrete scores. How many tokens were cut, and whether an agent
   performs the same on the reduced prompt as on the original.
4. **CLI**: run the whole thing as one simple Unix-style command.

## Approach

Four one-week sprints. Sprint 1 builds a thin end-to-end version that works and produces a first
number. Each later sprint improves it. Evaluation starts in Sprint 1 and grows every sprint, so
progress is measured rather than assumed.

| Sprint | Dates | Focus |
|--------|-------|-------|
| 1 | Jun 24-30 | Working end-to-end pipeline and first numbers |
| 2 | Jul 1-7 | Smarter reduction and an agent-performance benchmark |
| 3 | Jul 8-14 | Improvement driven by the benchmark |
| 4 | Jul 15-21 | Improvement and release to Release 1.0 |

## User stories

Priority P1 (highest) to P3. Story points in brackets. Each story is tagged with the
high-level goal it serves (G1-G4).

### Sprint 1: Working end-to-end pipeline and first numbers

- `ALX-1.1` [P1] (G1) As a prompt author, I want a redundancy score for each instruction, so I can
  see which instructions overlap. [10]
- `ALX-1.2` [P1] (G2, G4) As a user, I want one command that removes redundant instructions, so my
  prompt gets shorter in a single step. [5]
- `ALX-1.3` [P2] (G3) As a maintainer, I want token reduction reported over a few known prompts,
  so we have a concrete result from day one. [2]
- **Spike:** literature survey of prompt optimization, how token count affects LLM accuracy, and
  existing prompt/agent benchmarks. [5]
- **Infrastructure:** dependencies, CI, and quality gates (lint, types, tests). [3]

### Sprint 2: Smarter reduction and an agent-performance benchmark

- `ALX-2.1` [P1] (G2) As a prompt author, I want reduction to keep the load-bearing instruction of
  a redundant pair and never drop a unique one, so meaning is preserved, not just length cut. [8]
- `ALX-2.2` [P1] (G3) As a maintainer, I want a benchmark that compares an agent on the original
  prompt versus the reduced one, summarizing tokens cut and before/after redundancy. [8]

### Sprint 3: Improvement driven by the benchmark

- `ALX-3.1` [P1] (G2, G3) As a maintainer, I want reduction improved from the Sprint 2 benchmark
  findings, so it cuts tokens without hurting agent performance. [5]
- `ALX-3.2` [P2] (G2, G4) As a user, I want to control how aggressively redundancy is removed and
  see what was removed and why, so I can trade length against safety and trust the output. [6]

### Sprint 4: Improvement and release

- `ALX-4.1` [P1] (G3) As a maintainer, I want final tuning from the benchmark with the key results
  reproduced, so the release is our strongest version and the numbers are trustworthy. [8]
- `ALX-4.2` [P2] (G4) As a user, I want a README, install steps, and examples, so I can adopt the
  tool. [3]

## Capacity sanity check

- Team of 5, one-week sprints, roughly 20 story points each. Every sprint's planned work fits
  that budget.
- Sprints 3 and 4 are lighter on purpose: improvement and evaluation are open-ended and run
  alongside other course work.

## Product backlog (not in Release 1.0)

- Hosted embedding API backend.
- Per-model exact tokenizers.
- Rewriting instructions beyond drop and merge.
- A web UI or hosted service.
- A configurable redundancy metric.

We revisit and update the release plan and product backlog after each sprint.
