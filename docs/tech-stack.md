# Alexandria technology stack

This page summarizes the current implementation. `pyproject.toml` is authoritative for package
metadata and dependencies. See the
[design specification](spec.md) for component contracts and the [CLI guide](cli.md) for commands.

## Runtime and packaging

| Area | Current choice | Role |
| --- | --- | --- |
| Language | Python 3.14+ | Typed library and Click command-line application. |
| Environment and lockfile | `uv` | Reproducible development and runtime dependency resolution. |
| Build backend | Hatchling | Builds the source distribution and `src`-layout wheel; co-located tests are excluded. |
| IR and validation | Pydantic v2 | Frozen `Document` / `Section` / `Sentence` models, edit contracts, envelopes, and reports. |
| Embeddings and rewrites | OpenAI API via the Python SDK | `text-embedding-3-small` embeddings and `gpt-5.6-luna` rewrites, isolated behind protocols. |
| Numerics | NumPy | `float32` vectors, normalization, cosine similarity, and semantic-change measurements. |
| Token metric | `tiktoken` | `cl100k_base` token counts used for budgets and reports. |
| CLI | Click | Phase commands, end-to-end reduction, reporting, review modes, and configuration. |
| Persistence | Pydantic JSON | Versioned `DocumentEnvelope`, `ScoredEnvelope`, and `PlanEnvelope` streams. |

Library callers can inject implementations of the embedder and merger protocols.

## Development and verification

| Area | Current choice |
| --- | --- |
| Tests | pytest and pytest-cov, with co-located `*_test.py` modules and integration tests under `tests/`. |
| Lint and format | Ruff. |
| Static types | Pyright in strict mode. |
| Architecture | import-linter contracts plus layer-specific imports. |
| CI | GitHub Actions runs repository quality checks. |
| Analysis scripts | NumPy, SciPy, and Matplotlib. |
| Research notebooks | JupyterLab, IPykernel, Matplotlib, UMAP, HDBSCAN, and scikit-learn analyze embeddings produced through Alexandria's OpenAI-backed default. |

## Strategy extension status

Scorers, optimizers, and selectors use small in-process registries. Built-ins register when their
modules are imported. End-to-end composition can ask an optimizer which scorers it requires; the
default optimizer currently requires none, while the standalone redundancy score remains available
for inspection.

There is no installed third-party plugin loading or CLI flag for choosing a strategy. Library callers
can select registered names and inject model boundaries.
