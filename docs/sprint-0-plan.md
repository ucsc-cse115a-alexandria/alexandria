# Sprint 0 Plan

**Product:** Alexandria (Prompt Optimization for LLM Applications / Coding Agent) ·
**Team:** Alexandria ·
**Sprint completion date:** Tue, Jun 30, 2026 ·
**Revision:** 1.1 (2026-06-26)

## Goal

Prototype the optimization pipeline end to end: turn a prompt into the `Document` IR
(Represent), compute per-instruction `redundancy` scores (Score), and drop redundant
instructions with `greedy_pairwise` (Optimize). In parallel, produce short research notes
(the spike).

## Task listing (by user story, priority order)

Highest priority at the top. Estimates are ideal hours (each task ≤ 6h).

### User story 1: Project setup & CI

> As an open-source contributor, I want a project scaffold and a CI pipeline,
> so that multiple developers can work on the codebase while holding a minimum code-quality bar.

- Set up repo scaffold and packaging (3h)
- Set up CI to run lint, type check, and tests on every push (4h)
- Write install / run instructions (2h)

**Total for user story 1: 9 hours**

### User story 2: Optimize a prompt

> As a user, I want to pass in a prompt and get back a shorter, optimized prompt
> with redundant instructions removed.

> _Built incrementally: first a PoC that runs end to end, then each step is improved
> (better segmentation, scoring, and optimization) in later increments._

- Represent: split the prompt into instructions and embed each one (6h)
- Score: rate how redundant each instruction is (4h)
- Optimize: drop redundant instructions while preserving meaning (6h)
- CLI: run the whole pipeline, prompt in and reduced prompt out (4h)
- Report token reduction over a few known prompts (2h)

**Total for user story 2: 22 hours**

### User story 3: Research

> As a developer, I want to ground our scoring and evaluation design in prior research,
> so that we build on existing work instead of guessing.

Each task produces a note following [docs/research/TEMPLATE.md](research/TEMPLATE.md).

- Prompt optimization: 2-3 works on prompt compression/optimization (5h)
- Long-prompt effects: 2-3 papers on long-context degradation (5h)
- Prompt-writing techniques (2026 papers only): extract reproducible techniques (5h)
- Accuracy benchmarks: find one publishing exact eval prompts + ground truth (6h)

**Total for user story 3: 21 hours**

## Team roles

- Masa Ishihara: Product Owner
- Matthew Zerner: _(TBD)_
- Virinchi Chintala: _(TBD)_
- Marc Dylan Tan: _(TBD)_
- Jack Dao: Scrum _(TBD)_

## Initial task assignment

- Masa Ishihara: User story 2 (Optimize a prompt), build the end-to-end PoC
- Matthew Zerner: _(TBD)_
- Virinchi Chintala: _(TBD)_
- Marc Dylan Tan: _(TBD)_
- Jack Dao: _(TBD)_

## Initial burnup chart

_(TBD: burnup chart for Sprint 0, labeled with sprint number and product name, posted in the lab.)_

## Initial scrum board

_(TBD: task board labeled with sprint number and product name, posted in the lab. Four columns:
user stories · tasks not started · tasks in progress · tasks completed. Tasks sit in the same row
as their user story.)_

## Scrum times

_(TBD: at least three weekly Scrum meeting days/times; indicate which meeting the TA/tutor attends,
expected during the lab-time Scrum.)_
