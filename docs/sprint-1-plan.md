# Sprint 1 Plan

**Product:** Alexandria ·
**Team:** Alexandria ·
**Sprint completion date:** Tue, Jul 7, 2026 ·
**Revision:** 2.0 (2026-07-02)

## Goal

Show that a prompt Alexandria shortened keeps the agent as accurate as the original, measured on a
benchmark that cannot leak. Add CLI options so a user can trade some accuracy for lower cost when
they want to. Most of the sprint is choosing the benchmark and building the leak-proof dataset the
proof rests on.

## Task listing (by user story, priority order)

### User story 1: Trust that a shortened prompt stays accurate (G2 · Customer/User value · Priority 1)

> As an engineer who compresses my `CLAUDE.md` / `AGENT.md` with Alexandria, I want to see proof that
> the shortened prompt keeps my coding agent just as accurate as the original, so that I can switch to
> the smaller prompt without quietly trading reliability for a lower token bill.

- Run the original prompt and its Alexandria-compressed version through the benchmark runner and
  publish the accuracy difference, so the user can see for themselves that performance holds (4h)
- Write that result into the user-facing README / docs, so a user can read the evidence before they
  adopt compression (2h)

**Total for user story 1: 6 hours**

### User story 2: Steer the accuracy/cost trade-off (G3 · Customer/User value · Priority 2)

> As an engineer building an LLM application, I want CLI options to cap how hard Alexandria compresses
> my prompt (hold similarity above a floor I set, or hit a token budget) and to see the token savings
> I get, so that I can push compression as far as my cost target needs, even when I am willing to
> spend a little accuracy for a cheaper API bill.

- Add a `--min-similarity` option (e.g. `0.99`) to `reduce` so reduction stops before it crosses the
  similarity floor the user sets (2h). Depends on the compression fidelity check in Enabler B.
- Add a `--max-tokens N` option so reduction targets a token budget (2h)
- Add token counting to the CLI: report tokens before and after, plus the reduction percentage (2h)

**Total for user story 2: 6 hours**

## Enabling work (not user stories)

These items deliver no value to the user on their own, but the user stories cannot land without them.
Each is an imperative backlog item tagged with the kind of value it delivers. Do them first; they are
listed in priority order.

### Enabler A: Choose a benchmark that cannot leak (Spike / Technical value · Priority 3)

Acceptance criteria for the chosen benchmark:

- **Must have**
  - Established: at least 20 citations, or at least 100 GitHub stars.
  - Length-sensitive: inflating a prompt to 2× and 10× its length makes the score worse.
  - A subset shows a significant result in 10 minutes or less, at $1 or less in LLM cost.
- **Nice to have**
  - An even smaller subset shows a significant result in 1 minute or less, at $0.1 or less.
  - The benchmark bundles multiple prompts.

- [spike] Survey candidate benchmarks that clear the must-have bar (20+ citations or 100+ stars) and
  record each in a research note per [docs/research/TEMPLATE.md](research/TEMPLATE.md) (6h)
- [spike] Rate each candidate against every acceptance criterion above and shortlist the strongest (4h)
- [spike] Run a small trial on the top candidate to confirm the time/cost budget and that its score
  drops on an inflated prompt, then pick the base benchmark and write down the rationale (4h)

**Total for Enabler A: 14 hours**

### Enabler B: Leak-proof fidelity dataset (Technical value · Priority 4)

- Skill-corpus download script: fetch the SKILL.md of the top N (≤ 100) skill repos into a local
  corpus, building on [`scripts/search_skill_repos.py`](../scripts/search_skill_repos.py) and its
  `data/skill_repos.json` output (5h)
- Redundancy inflation script: given a prompt, produce an n× longer version by adding redundant
  restatements that carry no new instruction (6h)
- Dataset generator: take the top 10 skills and emit 1.2×, 1.5×, 2×, and 10× inflated variants as a
  versioned, label-by-construction dataset the benchmark cannot leak on (4h)
- Compression fidelity check: for a compressed prompt, compute cosine similarity to the original and
  the token reduction, and use the 99% similarity gate to confirm each dataset variant keeps its
  meaning (4h)

**Total for Enabler B: 19 hours**

### Enabler C: Split the library from the CLI (Refactoring · Priority 5)

Today `cli.py` wires up internals directly: it builds the embedder and picks the default scorer,
optimizer, and selector. Move that wiring behind the library so the CLI only parses arguments and
calls the public API. Land this before US2 so the new options sit on the thin CLI layer.

- Widen the public API so `reduce()` and `score_report()` take plain options (model, `min_similarity`,
  `max_tokens`) and build the embedder and defaults themselves, leaving callers no internals to wire
  up (4h)
- Move the CLI into its own `alexandria/cli/` package that imports only `alexandria`'s public API, and
  drop its direct imports of `core` / `phases` / `runtime` (3h)
- Add an import contract (or test) that fails if the CLI package reaches into `core` / `phases` /
  `runtime`, locking the seam (2h)

**Total for Enabler C: 9 hours**

## Capacity sanity check

- Team of 4, one-week sprint, with the Jul 3 holiday cutting into capacity: roughly **32 to 44 ideal
  hours** at 8 to 12 per person.
- The five items total **54 hours**, over the band. Commit in priority order: Enabler A → US1 (20h)
  proves accuracy, then Enabler C → US2 (15h) splits out the library and adds the CLI controls on top,
  then Enabler B as capacity allows.
- If Enabler B is cut, keep its compression fidelity check, which US2's `--min-similarity` needs, and
  push the rest to Sprint 2.

## Deferred to the product backlog

The [release plan](release-plan.md) scopes Sprint 1 to accuracy measurement (G2). This plan also
pulls the CLI controls (G3) forward as US2, since they deliver direct user value on their own, so the
release plan should be revised to move G3 into Sprint 1. Improving the compression itself stays out:

- **G1 (save more tokens):** improve the scorer / optimizer / selector to raise token reduction at
  the fidelity gate → Sprint 2.

## Team roles

- Masa Ishihara: Product Owner
- Matthew Zerner: _(TBD)_
- Virinchi Chintala: _(TBD)_
- Marc Dylan Tan: Scrum Master

## Initial task assignment

- Masa Ishihara: Enabler C (split library from CLI), then US2's CLI compression options
- Matthew Zerner: Enabler B, skill-corpus download script
- Virinchi Chintala: US1 experiment only — run the before/after accuracy experiment (original vs.
  compressed); writing the result into the README is assigned separately
- Marc Dylan Tan: Enabler A, benchmark selection

## Initial burnup chart

_(TBD)_

## Initial scrum board

_(TBD)_

## Scrum times

Three weekly scrum meetings (daily-scrum equivalent):

- **Monday 5:30pm**, right after the TA meeting with Scott (5:00 to 5:30pm). TA and tutor present.
- **Thursday 5:15pm**, right after the TA meeting with Scott (4:45 to 5:15pm). TA and tutor present.
- **Saturday 5:00pm**, team only.
