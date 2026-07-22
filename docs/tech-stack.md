# Alexandria technology stack

This page records the main technology choices and their purpose. [`pyproject.toml`](../pyproject.toml) defines the versions and dependencies. See the [design specification](spec.md) for component contracts.

## Runtime and packaging

| Area | Choice | Purpose |
| --- | --- | --- |
| Language | Python 3.14 | Provides typing features used by the library and CLI. |
| Environment | `uv` | Installs dependencies and maintains the lockfile. |
| Build | Hatchling | Builds the `src`-layout wheel and source distribution. |
| Data contracts | Pydantic v2 | Validates the document model, edits, parameters, reports, and JSON envelopes. |
| Embeddings and rewrites | OpenAI Python SDK | Provides the default embedding and merge implementations behind injectable protocols. |
| Numeric operations | NumPy | Stores vectors and calculates cosine similarity. |
| Token counting | `tiktoken` | Applies the shared `cl100k_base` token measure. |
| CLI | Click | Defines commands, options, configuration, and review workflows. |

Library users can provide their own embedder and merger. Alexandria does not require the OpenAI implementation when those dependencies are injected.

## Development checks

| Check | Tool |
| --- | --- |
| Tests and coverage | pytest and pytest-cov |
| Lint and formatting | Ruff |
| Static typing | Pyright in strict mode |
| Package boundaries | import-linter |
| Continuous integration | GitHub Actions |

Tests use the `*_test.py` name and usually sit beside the code they test. Broader end-to-end tests are in `tests/`.

## Extension boundary

Scorers, optimizers, and selectors use in-process registries. The Python library can select registered implementations. Version `0.1.0` does not discover installed third-party plugins, and the CLI does not provide a strategy-selection option.
