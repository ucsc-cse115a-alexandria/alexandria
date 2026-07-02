# Sprint 2 Plan

**Product:** Alexandria
**Team:** Alexandria ·
**Sprint completion date:** Tue, Jul 7, 2026 ·
**Revision:** 1.0 (2026-07-02)

## Goal

Make the accuracy of a compressed prompt measurable. Build a leak-proof benchmark that proves an
`A → A'` compression preserves agent task performance, plus a label-by-construction fidelity dataset
that measures compression without running an agent, and use both to push the reducer to cut more
tokens while holding overall similarity ≥ 99% — exposed through new CLI controls.

> Scope note: the release plan frames Sprint 2 as G2 (behavioral accuracy) only. This plan also pulls
> in G1 compression-fidelity data and G3 CLI controls. If the team commits to all four stories, revise
> [`release-plan.md`](release-plan.md) alongside this plan.

## Task listing (by user story, priority order)

### User story 1: Prove a compression preserves performance (G2) — highest priority

> As an engineer relying on Alexandria, I want a benchmark that proves compressing my prompt from
> `A` to `A'` does not change my agent's task performance, so that I can trust the shortened prompt
> before I depend on it.

Acceptance criteria for the chosen benchmark:

- **Must have**
  - Established: ≥ 20 citations **or** ≥ 100 GitHub stars.
  - Length-sensitive: inflating a prompt to 2× and 10× its length makes the score worse.
  - A subset shows a significant result in ≤ 10 minutes, at ≤ $1 in LLM cost.
- **Nice to have**
  - An even smaller subset shows a significant result in ≤ 1 minute, at ≤ $0.1.
  - The benchmark bundles multiple prompts.

Tasks:

- Survey candidate benchmarks that clear the must-have bar (≥ 20 citations or ≥ 100 stars); record
  each in a research note per [docs/research/TEMPLATE.md](research/TEMPLATE.md) (4h)
- Hypothesis-based analysis: rate each candidate against every acceptance criterion above and pick
  the optimal base benchmark, documenting the rationale (5h)
- Verify length-sensitivity on the chosen benchmark: inflate its prompt to 2× redundant length
  (US2 script) and confirm the score drops — this is what makes the benchmark able to detect a bad
  compression (4h)
- Benchmark runner: run a fixed subset, execute the agent per prompt, score against ground truth,
  and record each experiment (config, tokens, cost, score) to a results file; enforce the
  ≤ 10 min / ≤ $1 budget (6h)
- Before/after report: run the original vs. compressed prompt through the runner and report the
  accuracy delta (4h)

**Total for user story 1: 23 hours**

### User story 2: Measure compression fidelity without an agent (G1)

> As a developer improving the reducer, I want a label-by-construction dataset where a known-good
> prompt is padded with redundancy, so that I can measure compression fidelity without running an
> agent — the redundant version must compress back to roughly the original.

Tasks:

- Skill-corpus download script: fetch the SKILL.md of the top-N (≤ 100) skill repos into a local
  corpus, building on [`scripts/search_skill_repos.py`](../scripts/search_skill_repos.py) and its
  `data/skill_repos.json` output (5h)
- Redundancy inflation script: given a prompt, produce an n× longer version by adding redundant
  restatements that carry no new instruction (6h)
- Dataset generator: take the top-10 skills and emit 1.2×, 1.5×, 2×, and 10× inflated variants as a
  versioned dataset (4h)
- Compression-only fidelity metric: for a reduced prompt, compute overall cosine similarity to the
  original plus token reduction, and define the ≥ 99% similarity check (4h)

**Total for user story 2: 19 hours**

### User story 3: Cut more tokens at ≥ 99% fidelity (G1)

> As that engineer, I want the reducer to remove as many tokens as possible while overall similarity
> to the original stays ≥ 99%, so that I get the maximum savings without losing meaning.

Tasks:

- Baseline: run the current reducer over the US2 dataset and record token reduction at the ≥ 99%
  fidelity gate (3h)
- Improve the scorer / optimizer / selector to raise token reduction at the same ≥ 99% gate,
  iterating against that metric (6h)
- Regression guard: add a test/experiment that fails if fidelity drops below 99% on the dataset (3h)

**Total for user story 3: 12 hours**

### User story 4: CLI controls for targeted compression (G3) — lowest priority

> As a CLI user, I want to cap how far the prompt is compressed — keep overall similarity ≥ 99%, or
> hit a target token count — and see token counts, so that I control the accuracy/cost trade-off I get.

Tasks:

- Add a `--min-similarity` option (e.g. `0.99`) to `reduce`/`select` so reduction stops before it
  crosses the fidelity floor (4h)
- Add a `--max-tokens N` option so reduction targets a token budget (4h)
- Add token counting to the CLI: report tokens before/after and the reduction percentage (3h)

**Total for user story 4: 11 hours**

## Capacity sanity check

- Team of 4, one-week sprint, with the Jul 3 holiday reducing capacity.
- Committed total (US1–US3): **54 ideal hours**. US4 (11h) is a stretch: if the team cannot fit it,
  US4 returns to the product backlog first and is picked up in Sprint 3, per the "when there isn't
  enough time" rule.
- US1's length-sensitivity verification depends on US2's inflation script, and US3 depends on the
  US2 dataset and fidelity metric — sequence US2's scripts early so US1 and US3 are not blocked.

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
