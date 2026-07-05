# Alexandria — Technology Stack Decision

**Product:** Alexandria · **Team:** Alexandria Team · **Revision:** 3 (2026-06-26)

Scoped to what the team can learn in a five-week course and to what the pipeline in
[`spec.md`](spec.md) actually needs.

## Decisions

| Area | Choice | Why |
|---|---|---|
| Language / packaging | Python 3.14, `uv` | Established in the repo; fast, reproducible envs. |
| IR / validation | Pydantic v2 | The `Document` / `Section` / `Sentence` tree. Validates on build and freezes nodes, so stored `text` / `token_count` / `embedding` can't drift from their children. |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Local, free, deterministic, no API key. The one nondeterministic dependency, isolated inside Represent. |
| Numerics | NumPy | `float32` vectors, the cosine-similarity matrix, and the ablation distances behind `greedy_pairwise`. |
| Persistence | Parquet via Polars | One row per node; stores embeddings as compact `float32` lists. JSON/JSONL bloat them ~3–5× and round lossily. |
| Reporting | Polars | Benchmark tables; lighter than pandas, already used for Parquet I/O. |
| Token metric | `tiktoken` | A **model-independent proxy** for token reduction, not an exact Claude count. |
| Plugin discovery | Python entry points | One registry discovers built-in **and** third-party scorers/optimizers by name; adding one never edits Alexandria. |
| CLI | [`click`](https://github.com/pallets/click) | Verbs `represent` / `score` / `optimize` / `select` / `reduce`; text in, text out, `--json` for machine-readable output. |
| Quality gates | `ruff`, `pyright` (strict), `pytest` + `pytest-cov` | Configured in `pyproject.toml`; coverage gated at 80%. |
| CI | GitHub Actions | Runs lint + types + tests. |

## Rejected alternatives

- **Hosted embedding API (OpenAI / Voyage):** higher quality, but adds cost, keys, and
  benchmark non-determinism. The IR keeps embeddings swappable, so this stays open for later.
- **JSON / JSONL for persistence:** readable, but bloats dense `float32` embeddings and rounds
  them lossily; Parquet stores typed lists and round-trips exactly.
- **Hand-built IR / dataclasses:** no validation at the boundary. Pydantic enforces the tree's
  invariants on construction, so impossible states are unrepresentable.
- **A hardcoded scorer↔optimizer table:** couples each new strategy to central wiring. The
  registry lets an optimizer declare what it needs (`requires=(...)`), so selection is data.
