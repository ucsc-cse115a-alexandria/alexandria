# Final Project Submission: Alexandria

- **Product:** Alexandria
- **Team:** Alexandria (Masa Ishihara, Matthew Zerner, Virinchi Chintala, Marc Dylan Tan)
- **Release:** 1.0 (target 2026-07-21, end of Sprint 4)

This is the reviewer's entry point: each required submission element and where it lives in the
repository.

## Required elements

| Element | Location |
| --- | --- |
| **Source code** (install and run the system) | [`src/alexandria/`](src/alexandria/) (package), [`pyproject.toml`](pyproject.toml) (dependencies and the `alexandria` CLI entry point), [`README.md`](README.md) (install and quickstart) |
| **Test code** | Co-located unit tests `src/alexandria/**/*_test.py`; [`tests/pipeline_e2e_test.py`](tests/pipeline_e2e_test.py) (offline end-to-end); [`tests/ai_e2e_test.py`](tests/ai_e2e_test.py) + [`tests/fixtures/`](tests/fixtures/) (live end-to-end, `ai` marker); [`conftest.py`](conftest.py); CI [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |
| **Scrum: release plan** | [`docs/release-plan.md`](docs/release-plan.md) |
| **Scrum: sprint plans** | [1](docs/sprint-1-plan.md) · [2](docs/sprint-2-plan.md) · [3](docs/sprint-3-plan.md) · [4](docs/sprint-4-plan.md) |
| **Scrum: sprint reports** | [1](docs/sprint-1-report.md) · [2](docs/sprint-2-report.md) · [3](docs/sprint-3-report.md) · [4 (draft)](docs/sprint-4-report.md) |
| **Scrum: team working agreement** (roles, Definition of Done, cadence) | [`docs/working-agreement.md`](docs/working-agreement.md) |
| **Style guide / code conventions** | [`CONTRIBUTING.md`](CONTRIBUTING.md) (the tool config is the style guide: ruff, pyright, import-linter, 80% coverage) |
| **Test Plan and Report** | [`docs/test-plan-and-report.md`](docs/test-plan-and-report.md) |
| **Release Summary** (key user stories + acceptance criteria, known problems, product backlog) | [`docs/release-summary.md`](docs/release-summary.md) |
| **Release docs: installation** | [`README.md`](README.md) |
| **Release docs: dependencies** | [`pyproject.toml`](pyproject.toml), [`uv.lock`](uv.lock) |
| **Release docs: user guides** | [`docs/cli.md`](docs/cli.md), [`docs/library.md`](docs/library.md), [`examples/reduce_prompt.py`](examples/reduce_prompt.py) |
| **Release docs: design docs** | [`docs/spec.md`](docs/spec.md), [`docs/tech-stack.md`](docs/tech-stack.md) |
| **Release docs: contribution guide** | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| **Benchmark evidence** | [`README.md`](README.md) "Benchmark" section; raw runs under `benchmarks/*/results/` |

## Known gaps

The [Release Summary](docs/release-summary.md) records the open problems in full. The one that matters
for the review:

- The quality-monitoring CI gate documented in [`CONTRIBUTING.md`](CONTRIBUTING.md) is
  **not on `main`** yet.

The Sprint 4 report and a few Test Plan scenarios are drafts pending the final sprint day (Jul 21)
and a live run at the review.
