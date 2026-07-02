# Sprint 2 Plan

**Product:** Alexandria ·
**Team:** Alexandria ·
**Sprint completion date:** Tue, Jul 7, 2026 ·
**Revision:** 2.0 (2026-07-02)

## Goal

Give an engineer two things they can act on: proof that a shortened prompt keeps their coding agent
just as accurate as the original (on a benchmark that cannot leak), and CLI options to steer the
accuracy/cost trade-off when they would rather spend a little accuracy for a cheaper API bill. Most
of the sprint builds the benchmark, leak-proof dataset, and runner behind the proof.

## Task listing (by user story, priority order)

This sprint carries **two** pieces of user-facing value — the user stories below — plus the
technical-value and spike work they stand on. The user stories are written from the perspective of
the person who runs Alexandria on their own prompt, not from ours. Everything that delivers no direct
value to that person is listed separately under [Enabling work](#enabling-work-not-user-stories),
tagged with the kind of value it delivers (Customer, Business, Technical, Refactoring, or a research
spike — see the "Multiple Aspects of Value" framing in the planning guide).

Highest priority at the top. Estimates are ideal hours (each task ≤ 6h) and are the Product Owner's
starting proposal; the team finalizes them via planning poker at the planning meeting.

### User story 1 — Trust that a shortened prompt stays accurate (G2 · Customer/User value · Priority 1)

> As an engineer who compresses my `CLAUDE.md` / `AGENT.md` with Alexandria, I want to see proof that
> the shortened prompt keeps my coding agent just as accurate as the original, so that I can switch to
> the smaller prompt without quietly trading reliability for a lower token bill.

- Run the original prompt and its Alexandria-compressed version through the benchmark runner and
  publish the accuracy delta, so the user can see for themselves that performance is preserved (4h)
- Write that result into the user-facing README / docs as the trust evidence a user reads before
  they adopt compression (2h)

**Total for user story 1: 6 hours**

### User story 2 — Steer the accuracy/cost trade-off (G3 · Customer/User value · Priority 2)

> As an engineer building an LLM application, I want CLI options to cap how hard Alexandria compresses
> my prompt — hold similarity above a floor I set, or hit a token budget — and see the token savings I
> get, so that I can push compression as far as my cost target needs, even when I am willing to spend
> a little accuracy for a cheaper API bill.

- Add a `--min-similarity` option (e.g. `0.99`) to `reduce` so reduction stops before it crosses the
  similarity floor the user sets (4h) — depends on the compression fidelity check in Enabler C
- Add a `--max-tokens N` option so reduction targets a token budget (4h)
- Add token counting to the CLI: report tokens before/after and the reduction percentage (3h)

**Total for user story 2: 11 hours**

## Enabling work (not user stories)

These items deliver no value to the user on their own, but User Story 1 cannot exist without them.
Each is an imperative backlog item tagged with the kind of value it delivers. Sequence them before
US1; they are listed in priority order.

### Enabler A — Choose a benchmark that cannot leak (Spike / Technical value · Priority 3)

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
- [spike] Hypothesis-based analysis: rate each candidate against every acceptance criterion above and
  pick the base benchmark, documenting the rationale (5h)

**Total for Enabler A: 9 hours**

### Enabler B — Benchmark runner (Technical value · Priority 4)

- Benchmark runner: run a fixed subset, execute the agent per prompt, score against ground truth, and
  record each experiment (config, tokens, cost, score) to a results file; enforce the
  ≤ 10 min / ≤ $1 budget (6h)
- Verify length-sensitivity on the chosen benchmark: inflate its prompt to 2× redundant length
  (inflation script in Enabler C) and confirm the score drops — this is what lets the benchmark
  detect a bad compression (4h)

**Total for Enabler B: 10 hours**

### Enabler C — Leak-proof fidelity dataset (Technical value · Priority 5)

- Skill-corpus download script: fetch the SKILL.md of the top-N (≤ 100) skill repos into a local
  corpus, building on [`scripts/search_skill_repos.py`](../scripts/search_skill_repos.py) and its
  `data/skill_repos.json` output (5h)
- Redundancy inflation script: given a prompt, produce an n× longer version by adding redundant
  restatements that carry no new instruction (6h)
- Dataset generator: take the top-10 skills and emit 1.2×, 1.5×, 2×, and 10× inflated variants as a
  versioned, label-by-construction dataset the benchmark cannot leak on (4h)
- Compression fidelity check: for a compressed prompt, compute cosine similarity to the original plus
  token reduction, and use the ≥ 99% similarity gate to validate that each dataset variant preserves
  meaning (4h)

**Total for Enabler C: 19 hours**

## Capacity sanity check

- Team of 4, one-week sprint, with the Jul 3 holiday reducing capacity. At a conservative
  8–12 ideal hours/person/week, realistic capacity is roughly **32–44 ideal hours**.
- Everything above totals **55 hours** — well over that band. The team trims at the planning meeting.
- Recommended commitment order: **Enabler A → Enabler B → US1** (25h) delivers a real proof on the
  chosen benchmark, then **US2** (11h) gives users the CLI controls. That core is 36h and fits the top
  of capacity. Add **Enabler C** as capacity allows.
- Recommended trim order if over capacity, mostly within Enabler C: drop the corpus download script
  (hand-pick ~10 skills instead), then the 10× variant — but keep the compression fidelity check,
  since US2's `--min-similarity` depends on it. Returned work goes to the product backlog for Sprint 3.
- Dependencies: US1 needs the benchmark runner (Enabler B) and at least a small dataset; the
  length-sensitivity check needs the inflation script (Enabler C); US2's `--min-similarity` needs the
  compression fidelity check (Enabler C). Sequence the enablers first so the user stories are not
  blocked.

## Deferred to the product backlog

The [release plan](release-plan.md) scopes Sprint 2 to accuracy measurement (G2). This plan also
pulls the CLI controls (G3) forward as US2, since they deliver direct user value on their own — so the
release plan should be revised to move G3 into Sprint 2. Improving the compression itself stays out:

- **G1 — save more tokens:** improve the scorer / optimizer / selector to raise token reduction at the
  fidelity gate → Sprint 3.

## Team roles

- Masa Ishihara: Product Owner
- Matthew Zerner: _(TBD)_
- Virinchi Chintala: _(TBD)_
- Marc Dylan Tan: Scrum Master

## Initial task assignment

_(Proposal — team members sign up for their own work at the planning meeting.)_

- Masa Ishihara: US2, add the CLI compression options (`--min-similarity` / `--max-tokens` / token counting)
- Matthew Zerner: Enabler C, skill-corpus download script
- Virinchi Chintala: US1, before/after accuracy experiment (original vs. compressed)
- Marc Dylan Tan: Enabler A / B, benchmark selection and runner

## Initial burnup chart

_(TBD)_

## Initial scrum board

_(TBD)_

## Scrum times

_(TBD)_
