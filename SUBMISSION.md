# Final Project Submission

This page is an index for the final project review. It follows the submission requirements and points to the files the reviewer may want to inspect.

| Required element | Paths | What to review |
| --- | --- | --- |
| Source code | [`src/alexandria/`](src/alexandria/), [`pyproject.toml`](pyproject.toml), [`uv.lock`](uv.lock) | The installable Python package, CLI entry point, dependencies, and locked environment. Installation and the first run are in [`README.md`](README.md). |
| Test code | [`src/alexandria/`](src/alexandria/), [`tests/`](tests/), [`scripts/`](scripts/), [`benchmarks/`](benchmarks/), [`conftest.py`](conftest.py) | Unit tests use the `*_test.py` name and usually sit beside the code they test. Broader end-to-end tests are in `tests/`. Pytest settings and the coverage requirement are in [`pyproject.toml`](pyproject.toml). |
| Scrum documents | [`docs/release-plan.md`](docs/release-plan.md), sprint plans [1](docs/sprint-1-plan.md), [2](docs/sprint-2-plan.md), [3](docs/sprint-3-plan.md), and [4](docs/sprint-4-plan.md), sprint reports [1](docs/sprint-1-report.md), [2](docs/sprint-2-report.md), [3](docs/sprint-3-report.md), and [4](docs/sprint-4-report.md), [`docs/working-agreement.md`](docs/working-agreement.md), [`docs/definition-of-done.md`](docs/definition-of-done.md), [`CONTRIBUTING.md`](CONTRIBUTING.md) | The release plan, sprint records, team roles, Definition of Done, work schedule, and code conventions. |
| Test Plan and Report | [`docs/test-plan-and-report.md`](docs/test-plan-and-report.md) | System test scenarios, expected results, recorded results, and links to automated evidence. |
| Release Summary Document | [`docs/release-summary.md`](docs/release-summary.md) | Key user stories and acceptance criteria, known problems, and the product backlog. |
| Release documents | [`README.md`](README.md), [`docs/release-install.md`](docs/release-install.md), [`docs/cli.md`](docs/cli.md), [`docs/library.md`](docs/library.md), [`docs/spec.md`](docs/spec.md), [`docs/tech-stack.md`](docs/tech-stack.md), [`benchmarks/prompt_compression/README.md`](benchmarks/prompt_compression/README.md) | Installation, dependencies, user guides, design, technology choices, benchmark method, and published results. |
