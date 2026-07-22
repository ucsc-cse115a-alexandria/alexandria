# Release summary

| Field | Value |
| --- | --- |
| Product | Alexandria |
| Team | Alexandria |
| Date | 2026-07-21 |
| Course release | 1.0, end of Sprint 4 |
| Package version | 0.1.0 |

Alexandria shortens instruction-heavy prompts without labeled examples or expected outputs. It finds similar instructions with sentence embeddings and asks an LLM to merge them. The CLI command is `alexandria`, and the Python package is imported as `alexandria`.

## User stories and acceptance criteria

| ID | User outcome | Acceptance criteria |
| --- | --- | --- |
| US1 | Shorten a prompt with one command. | `alexandria reduce` returns a prompt with fewer tokens for the accepted duplicate-input test case. The JSON output reports source and reduced token counts. |
| US2 | Control token savings and semantic change. | `reduce` supports `--save-tokens`, `--keep`, `--target-reduction`, and `--cos-sim-diff-budget`. Best-effort requests report the result they reached. `--target-reduction` requires the calculated token ceiling. The `tokens` command counts matching instruction files in a directory. |
| US3 | Review changes before using them. | Terminal and browser review apply only accepted edits. `score --table` shows redundancy information. `represent`, `score`, `optimize`, and `select` exchange saved JSON envelopes. |
| US4 | Measure compression quality. | `report` returns token and instruction-preservation metrics. It can compare the result with a compatible user-provided baseline. Published benchmark runs include accuracy, token, time, and cost data. |
| US5 | Install and use the CLI and library outside the checkout. | The Git URL installs the `alexandria` command and importable package. The clean-package smoke test checks both interfaces. |

The [Test Plan and Report](test-plan-and-report.md) gives the commands, expected results, and automated evidence.

## Known problems

1. The repository does not run an optimization-quality baseline comparison in CI. The `report` command supports a baseline supplied by the user, but no baseline is committed.
2. The default reduction path needs an OpenAI API key and network access. Library users can run offline by injecting an embedder and merger. `HashEmbedder` is deterministic but is intended for tests, not semantic comparison.
3. Compression may take several minutes and make multiple paid model calls.
4. LLM rewrites are nondeterministic. The same input may produce a different rewrite on another run.
5. The package version is 0.1.0. Its API may change, and installation currently uses the Git repository because the package is not on PyPI.

## Product backlog

1. Add a committed optimization baseline and run the comparison in CI.
2. Publish `alexandria-prompt` to PyPI.
3. Add another hosted embedding provider.
4. Add exact tokenizers for more models.
5. Support rewrite methods that do more than delete and merge near-duplicate instructions.
6. Make the redundancy metric configurable.
