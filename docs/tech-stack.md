# Alexandria — Technology Stack Decision

**Product:** Alexandria · **Team:** Alexandria Team · **Revision:** 1 (2026-06-24)

Scoped to what the team can realistically learn within a five-week course, and to what
the pipeline in [`spec.md`](spec.md) actually needs.

## Decisions

| Area | Choice | Why |
|---|---|---|
| Language / packaging | Python 3.14, `uv` | Already established in the repo; `uv` gives fast, reproducible envs. |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Local, free, deterministic, no API key or network. Small/fast (384-dim) baseline; runs in CI. |
| Numerics | NumPy | Cosine-similarity matrix and vector math behind the redundancy score. |
| Data / reporting | Polars | Sprint 3 benchmark tables; faster and lighter than pandas. |
| Token metric | `tiktoken` | Token-reduction measurement. Documented as a **model-independent proxy**, not an exact Claude token count. |
| CLI | `argparse` + stdin/stdout JSON | Stdlib (minimal deps); stages compose with Unix pipes. |
| Quality gates | `ruff`, `pyright` (strict), `pytest` | Already configured in `pyproject.toml`. |
| CI | GitHub Actions (`.github/workflows/ci.yml`) | Already present; runs lint + types + tests. |

## Rejected alternatives

- **Hosted embedding API (OpenAI / Voyage):** higher quality, but adds cost, API keys,
  network flakiness, and non-determinism in benchmarks. The IR keeps embeddings swappable,
  so this stays open for a later release.
- **pandas instead of Polars:** heavier; no feature we need that Polars lacks.
- **Typer / Click for CLI:** nicer ergonomics, but an extra dependency for a single
  entry point — `argparse` is enough for v1.
- **Exact per-model tokenizers everywhere:** prompts target many models; one proxy
  tokenizer keeps the metric comparable across the corpus.

## Notes

- Each stage in `spec.md` depends only on the IR, so any single choice above (model,
  token counter, CLI framework) can be replaced without rewriting other stages.
