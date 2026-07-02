# Sprint 2 Plan

**Product:** Alexandria
**Team:** Alexandria ·
**Sprint completion date:** Tue, Jul 7, 2026 ·
**Revision:** 1.0 (2026-07-02)

## Goal

Let a CLI user trust and control the compression: prove a shortened prompt keeps their agent just as
accurate, cut more tokens while meaning stays intact, and give them CLI controls to steer the
accuracy/cost trade-off. Most of the sprint is the benchmark, dataset, and scripts that make this
possible.

> Scope note: the release plan frames Sprint 2 as G2 (behavioral accuracy) only. This plan also
> delivers G1 (more token savings) and G3 (CLI controls). If the team commits to all of it, revise
> [`release-plan.md`](release-plan.md) alongside this plan.

## Task listing (by user story, priority order)

Highest priority at the top. User stories are written from the perspective of the person using the
CLI. Estimates are ideal hours (each task ≤ 6h) and are the Product Owner's starting proposal; the
team finalizes them via planning poker at the planning meeting.

### User story 1: Trust that a shortened prompt stays accurate (G2) — highest priority

> As an engineer who compresses my `CLAUDE.md` / `AGENT.md` with Alexandria, I want proof that the
> shortened prompt keeps my agent just as accurate, so that I can rely on it without secretly trading
> reliability for a smaller prompt.

- Before/after accuracy report: run the original vs. compressed prompt through the benchmark runner
  and publish the accuracy delta so a user can see performance is preserved (4h)
- Document the trust evidence in the user-facing docs / README (2h)

**Total for user story 1: 6 hours**

### User story 2: Save more tokens without losing meaning (G1)

> As that engineer, I want Alexandria to remove as many tokens as possible while the meaning stays
> intact, so that I cut more cost without giving up quality.

- Baseline: run the current reducer over the fidelity dataset and record token reduction at the
  ≥ 99% similarity gate (3h)
- Improve the scorer / optimizer / selector to raise token reduction at the same ≥ 99% gate,
  iterating against that metric (6h)
- Regression guard: add a test/experiment that fails if fidelity drops below 99% on the dataset (3h)

**Total for user story 2: 12 hours**

### User story 3: Steer and see the compression (G3) — lowest priority

> As that engineer, I want to cap how far the prompt is compressed — keep overall similarity ≥ 99%,
> or hit a target token count — and see the token savings, so that I control the accuracy/cost
> trade-off I get.

- Add a `--min-similarity` option (e.g. `0.99`) to `reduce`/`select` so reduction stops before it
  crosses the fidelity floor (4h)
- Add a `--max-tokens N` option so reduction targets a token budget (4h)
- Add token counting to the CLI: report tokens before/after and the reduction percentage (3h)

**Total for user story 3: 11 hours**

## Infrastructure & spike tasks (not tied to a user story)

These deliver no direct CLI-user value on their own, but they are what US1–US3 stand on. Sequence
them early. Listed in the priority order given for the sprint.

### Benchmark: prove an `A → A'` compression preserves performance (enables US1)

Acceptance criteria for the chosen benchmark:

- **Must have**
  - Established: ≥ 20 citations **or** ≥ 100 GitHub stars.
  - Length-sensitive: inflating a prompt to 2× and 10× its length makes the score worse.
  - A subset shows a significant result in ≤ 10 minutes, at ≤ $1 in LLM cost.
- **Nice to have**
  - An even smaller subset shows a significant result in ≤ 1 minute, at ≤ $0.1.
  - The benchmark bundles multiple prompts.

- [spike] Survey candidate benchmarks that clear the must-have bar (≥ 20 citations or ≥ 100 stars);
  record each in a research note per [docs/research/TEMPLATE.md](research/TEMPLATE.md) (4h)
- [spike] Hypothesis-based analysis: rate each candidate against every acceptance criterion above
  and pick the optimal base benchmark, documenting the rationale (5h)
- Verify length-sensitivity on the chosen benchmark: inflate its prompt to 2× redundant length
  (inflation script below) and confirm the score drops — this is what lets the benchmark detect a
  bad compression (4h)
- Benchmark runner: run a fixed subset, execute the agent per prompt, score against ground truth,
  and record each experiment (config, tokens, cost, score) to a results file; enforce the
  ≤ 10 min / ≤ $1 budget (6h)

### Fidelity dataset & scripts (enables US2 and US3)

- Skill-corpus download script: fetch the SKILL.md of the top-N (≤ 100) skill repos into a local
  corpus, building on [`scripts/search_skill_repos.py`](../scripts/search_skill_repos.py) and its
  `data/skill_repos.json` output (5h)
- Redundancy inflation script: given a prompt, produce an n× longer version by adding redundant
  restatements that carry no new instruction (6h)
- Dataset generator: take the top-10 skills and emit 1.2×, 1.5×, 2×, and 10× inflated variants as a
  versioned, label-by-construction dataset the benchmark cannot leak on (4h)
- Compression-only fidelity metric: for a reduced prompt, compute overall cosine similarity to the
  original plus token reduction, and define the ≥ 99% similarity check (4h)

**Total for infrastructure & spike: 38 hours**

## Capacity sanity check

- Team of 4, one-week sprint, with the Jul 3 holiday reducing capacity. At a conservative
  8–12 ideal hours/person/week, realistic capacity is roughly **32–44 ideal hours**.
- Everything above totals **67 hours** — over capacity. The team must trim at the planning meeting.
  Recommended commitment order: benchmark infrastructure (unblocks everything) → fidelity dataset &
  scripts → US1 → US2 → US3. Cut from the bottom (US3 first, then US2's improvement iteration) until
  the committed total fits capacity; the rest returns to the product backlog for Sprint 3.
- Dependencies: US1 needs the benchmark runner; the length-sensitivity check needs the inflation
  script; US2 needs the dataset and fidelity metric. Sequence the infrastructure tasks early so the
  user stories are not blocked.

## Team roles

- Masa Ishihara: Product Owner
- Matthew Zerner:
- Virinchi Chintala:
- Marc Dylan Tan: Scrum Master

## Initial task assignment

- Masa Ishihara:
- Matthew Zerner:
- Virinchi Chintala:
- Marc Dylan Tan:

## Initial burnup chart

_(TBD)_

## Initial scrum board

_(TBD)_

## Scrum times

_(TBD)_
