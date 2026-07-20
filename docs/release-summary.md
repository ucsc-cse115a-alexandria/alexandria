# Release Summary

**Product:** Alexandria
**Team:** Alexandria
**Date:** 2026-07-20
**Release:** 1.0 (target 2026-07-21, end of Sprint 4)

**Team members:** Masa Ishihara (Product Owner), Matthew Zerner (Scrum Master), Virinchi Chintala, Marc Dylan Tan

## What Alexandria is

Alexandria is a label-free prompt optimizer. It shortens instruction-heavy prompts while trying to preserve their meaning, and it does this without any training labels. It uses sentence embeddings to score which instructions overlap, where redundancy is the cosine similarity of an instruction to the most similar other instruction. It then merges each near-duplicate pair into one LLM-rewritten sentence, keeping the first occurrence and dropping the second. Every rewrite is checked by measuring the whole-document `cos_sim_diff` against a budget, so an edit that moves the document's meaning too far is rejected.

It runs on Python 3.14 and uses OpenAI `text-embedding-3-small` for embeddings and `gpt-5.6-luna` for merging. It ships two ways: a CLI (`alexandria`) and a Python library (`import alexandria`).

## Key user stories and acceptance criteria

These stories map to the release plan goals: G1 (token reduction), G2 (accuracy retention), G3 (usability). Use the acceptance criteria as an acceptance-test guide.

**US1 (G1, G3): Shorten a bloated prompt in one command.** A user points `alexandria reduce` at a large `CLAUDE.md` or `AGENT.md` and gets back a version with the redundant instructions removed.
Acceptance: `alexandria reduce` returns a prompt with fewer tokens than the source, and meaning is preserved within a bounded whole-document `cos_sim_diff`.

**US2 (G3): Control the trade-off and see the savings.** A user can cap how hard the tool compresses and inspect token counts.
Acceptance: the flags `--keep`, `--save-tokens`, `--min-similarity`, `--max-tokens`, and `--target-reduction` all work, where `--target-reduction` guarantees the token ceiling. The `tokens` command lists counts.

**US3 (G3): Review a compression before adopting it.** A user can inspect and approve edits instead of accepting them blind.
Acceptance: `reduce --interactive` gives terminal accept/reject, `reduce --browser` gives browser accept/reject, `score --table` prints a redundancy report, and the pipeline can run phase by phase (`represent`, `score`, `optimize`, `select`) with `--out` to save and resume.

**US4 (G1, G2): Compress harder with an accuracy check, and watch quality.** A user runs a stronger pass that only keeps edits the budget backs, and gets metrics.
Acceptance: the multi-optimizer pass keeps only budget-backed edits, and `report` emits token metrics, quality scores, and a baseline comparison.

**US5 (G3): Install outside the checkout, use as CLI and library, with published numbers.** A user installs Alexandria without cloning it and can call it from Python.
Acceptance: `uv tool install git+…` installs the CLI, the library `reduce()` runs, and the docs publish accuracy, token, and cost numbers.

## Known Problems

See the Test Report for the underlying benchmark runs. There is no review penalty for the failures listed here.

**1. Accuracy retention is not proven at the release threshold.** This is the important one. The Sprint 4 benchmark used BABILong 8k and RULERv2, 50 cases each, seed 42, with `gpt-5.6-luna` doing both the compression and the answering. Across best-effort `cos_sim_diff` budgets from 0.0025 to 0.02, average task accuracy fell from 76.0% on the original prompts to roughly 56% to 63%, while mean token reduction was only 0.40% to 0.51%. A separate hard-target study that forced a BABILong prompt down to between 75% and 95% of its length also failed: every setting's accuracy-retention confidence interval stayed below the release threshold. For example, the original scored 72%, keep75 scored 48%, and keep90 scored 64%. So on these benchmarks Alexandria does not yet meet the G2 promise that a shortened prompt stays as accurate as the original. We are publishing the numbers as they are and did not adopt a more aggressive compression default to paper over them.

**2. Quality-monitoring CI is not on `main`.** `docs/contributing.md` describes an "Optimization quality" workflow at `.github/workflows/optimization-quality.yml` plus a committed `benchmarks/optimization_baseline.json`, but neither file is on `main`. CI on `main` currently runs lint, format, pyright, import-linter, and pytest only. The docs and the actual CI need to be reconciled.

**3. The pipeline needs an OpenAI API key and network access.** Every pipeline command calls OpenAI. Offline use exists only through the library, by injecting your own embedder and merger. The built-in offline `HashEmbedder` is not semantic: it only catches exact-duplicate lines, so offline runs need a generous `cos_sim_diff` budget to do anything useful.

**4. Compression can be slow and cost real money.** The merge model is called once per near-duplicate pair. Some benchmark reduction runs cost more than $1 each and took several minutes.

**5. Merge output is nondeterministic.** Because merging goes through an LLM, the same input can produce different output across runs.

**6. This is pre-1.0 software.** The version is 0.1.0 and the API may still change. It is not on PyPI, so the only install path is the git URL.

## Product Backlog

High-priority follow-on work for the next team, ordered by priority.

1. **Reach accuracy retention at the release threshold.** Tune the default operating point and widen the benchmark coverage so the G2 promise holds. This is the top priority because it is the one release goal we did not meet.
2. **Land the quality-monitoring CI.** Commit the `optimization-quality.yml` workflow and the `optimization_baseline.json` baseline, then reconcile `docs/contributing.md` so it describes what actually runs.
3. **Publish to PyPI** so install is a single `pip`/`uv` command instead of a git URL.
4. **Add a hosted embedding API backend** as an alternative to the current OpenAI-only path.
5. **Add per-model exact tokenizers** for accurate token counts across models.
6. **Support rewriting instructions beyond drop-and-merge**, so compression is not limited to removing near-duplicates.
7. **Make the redundancy metric configurable** instead of fixed to cosine similarity to the nearest neighbor.
