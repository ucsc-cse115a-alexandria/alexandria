# Release Summary

**Product:** Alexandria ·
**Team:** Alexandria ·
**Date:** 2026-07-20 ·
**Release:** 1.0 (target 2026-07-21, end of Sprint 4)

Alexandria is a label-free prompt optimizer. It shortens instruction-heavy prompts while keeping
their meaning: it uses sentence embeddings to find overlapping instructions, then merges each
near-duplicate pair into one sentence rewritten by an LLM. Redundancy means the cosine similarity of
an instruction to its closest peer, and every rewrite is checked against a whole-document
`cos_sim_diff` budget. It runs on Python 3.14 with OpenAI `text-embedding-3-small` (embeddings) and
`gpt-5.6-luna` (merging), and ships as a CLI (`alexandria`) and a Python library
(`import alexandria`).

## Key user stories and acceptance criteria

Use these as an acceptance-test guide. They map to the release goals: G1 token reduction, G2 accuracy
retention, G3 usability.

- **US1 (G1, G3): Shorten a bloated prompt in one command.** Acceptance: `alexandria reduce` returns
  a prompt with fewer tokens than the source, and meaning is preserved within a bounded whole-document
  `cos_sim_diff`.
- **US2 (G3): Control the trade-off and see the savings.** Acceptance: `--keep`, `--save-tokens`,
  `--min-similarity`, `--max-tokens`, and `--target-reduction` work, where `--target-reduction`
  guarantees the token ceiling; `tokens` lists per-instruction counts.
- **US3 (G3): Review a compression before adopting it.** Acceptance: `reduce --interactive` (terminal
  accept/reject), `reduce --browser` (browser accept/reject), `score --table` (redundancy report),
  and phase-by-phase JSON (`represent`, `score`, `optimize`, `select`) with `--out` to save and
  resume.
- **US4 (G1, G2): Compress harder with an accuracy check, and inspect quality.** Acceptance: the
  multi-optimizer pass keeps only budget-backed edits, and `report` emits token metrics, quality
  scores, and a baseline comparison.
- **US5 (G3): Install outside the checkout, use as CLI and library, with published numbers.**
  Acceptance: `uv tool install git+...` installs the CLI, the library `reduce()` runs, and the docs
  publish accuracy, token, and cost numbers.

## Known Problems

See the [Test Plan and Report](test-plan-and-report.md) for the underlying runs. Per the review
policy, there is no penalty for the failures listed here.

1. **Accuracy retention is not proven at the release threshold.** The Sprint 4 benchmark (BABILong 8k
   and RULERv2, 50 cases each, seed 42, `gpt-5.6-luna` for both compression and answers) tested
   best-effort `cos_sim_diff` budgets from 0.0025 to 0.02: average task accuracy fell from 76.0% on
   the original prompts to roughly 56% to 63%, while mean token reduction was only 0.40% to 0.51%. A
   separate hard-target study that forced a BABILong prompt to 75% through 95% of its length also
   failed, with every setting's accuracy-retention confidence interval below the release threshold
   (original 72%, keep75 48%, keep90 64%). On these benchmarks Alexandria does not yet meet the G2
   promise. We did not adopt a more aggressive default to mask this.
2. **Quality-monitoring CI is not on `main`.** `CONTRIBUTING.md` describes an "Optimization
   quality" workflow (`.github/workflows/optimization-quality.yml`) and a committed
   `benchmarks/optimization_baseline.json`, but neither file is on `main`. CI currently runs lint,
   format, pyright, import-linter, and pytest only. The docs and the actual CI need to be reconciled.
3. **The pipeline requires an OpenAI key and network access.** Offline use exists only through the
   library, by injecting your own embedder and merger. The built-in offline `HashEmbedder` is not
   semantic (it catches only exact-duplicate lines), so offline runs need a generous `cos_sim_diff`
   budget.
4. **Compression can be slow and cost real money.** The merge model is called once per near-duplicate
   pair. Some benchmark reduction runs cost more than \$1 each and took several minutes.
5. **Merge output is nondeterministic.** Because merging goes through an LLM, the same input can
   produce different output across runs.
6. **Pre-1.0 software.** The version is 0.1.0 and the API may still change. It is not on PyPI, so the
   only install path is the git URL.

## Product Backlog

High-priority follow-on work, ordered by priority.

1. **Reach accuracy retention at the release threshold.** Tune the default operating point and widen
   benchmark coverage so the G2 promise holds. This is the one release goal not met.
2. **Land the quality-monitoring CI.** Commit the `optimization-quality.yml` workflow and the
   `optimization_baseline.json` baseline, then reconcile `CONTRIBUTING.md`.
3. **Publish to PyPI** so install is a single command instead of a git URL.
4. **Add a hosted embedding API backend** as an alternative to the OpenAI-only path.
5. **Add per-model exact tokenizers** for accurate counts across models.
6. **Rewrite instructions beyond drop-and-merge**, so compression is not limited to removing
   near-duplicates.
7. **Make the redundancy metric configurable** instead of fixed to nearest-neighbor cosine
   similarity.
