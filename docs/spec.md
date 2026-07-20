# Alexandria design specification

**Status:** current implementation · **Package version:** `0.1.0` · **Updated:** 2026-07-19

This document describes the code that is present in this repository. It is not a roadmap and does
not describe unimplemented command-line flags, storage formats, or plugin systems.

## Purpose and scope

Alexandria compresses instruction-heavy prompts without labels or expected outputs. It represents a
prompt as a validated tree, finds redundant instructions with sentence embeddings, asks a merger to
rewrite compatible text, and applies edits only within the configured token and semantic-change
budgets.

The default runtime uses OpenAI for embeddings and merge rewrites. Library callers can inject their
own implementations of the `Embedder` and `SentenceMerger` protocols; the included `HashEmbedder` is
intended for deterministic offline tests, not semantic evaluation.

## Architecture

The code follows a functional-core, imperative-shell shape:

```text
CLI / library API
       |
       v
represent -> optional score -> optimize -> select
       |             shared validated IR          |
       +-------------------------------------------+
                 injected embedders and merger
```

The public phase commands remain independently composable. End-to-end `reduce` resolves which
scorers its selected optimizers require. The built-in `merge_rewrite` optimizer currently requires no
standalone scorer, so the default path skips the Score call and ranks pairs from the document's
embeddings directly. Score remains available for inspection and for future or injected strategies
that declare scorer requirements.

The main packages have these responsibilities:

- `alexandria.ir`: immutable data contracts, edit application, similarity helpers, and internal
  strategy registries.
- `alexandria.ops.features`: the independently callable Represent, Score, Optimize, Select, Compare,
  Diff, and hard-target operations.
- `alexandria.ops.pipe`: end-to-end composition, proposals, and reduction metrics.
- `alexandria.cli`: Click commands, interactive/browser review, JSON envelopes, and configuration.
- `alexandria.utils`: tokenization plus the OpenAI/configuration shell.

Import-linter contracts in `pyproject.toml` enforce the intended direction between these layers and
keep OpenAI client construction out of the IR and operations packages.

## Intermediate representation

The shared IR is a frozen Pydantic tree:

```text
Document
└── Section (Markdown, XML, or plain)
    ├── Section
    └── Sentence
```

Every encoded node carries its source text, token count, and a one-dimensional NumPy `float32`
embedding. A `Document` also records the embedding model ID. Construction validates that:

- parent text and token counts exactly roll up from their children;
- all embeddings in a tree have the same dimension;
- sentence IDs are unique within a document; and
- a document and every section retain at least one sentence.

Edits are `Delete` or `Replace` values targeting sentence IDs. `Document.apply` returns a rebuilt,
validated document, ignores edits whose targets no longer apply, and refuses edits that would empty
the document or a section. The IR itself never calls a model.

## Runtime contracts

`Params` controls the built-in reduction behavior:

| Field | Default | Meaning |
| --- | ---: | --- |
| `threshold` | `0.85` | Similarity threshold available to strategies. |
| `cos_sim_diff_budget` | `0.5` | Maximum accepted whole-document `1 - cosine_similarity`. |
| `max_tokens` | `None` | Optional stopping ceiling for the reduced prompt. |
| `require_target` | `False` | Make `max_tokens` a hard requirement; requires `max_tokens`. |

The principal injected protocols are:

- `Embedder.embed(texts)` with a stable `model_id`;
- `SentenceMerger.merge(first, second, feedback)`;
- scorer, optimizer, and selector callables registered inside the package; and
- `ReductionReporter` for optional progress events.

The same embedder model must be used throughout one document's run. This prevents comparisons across
incompatible vector spaces.

## Built-in behavior

### Represent

Represent preserves prompt text while identifying Markdown headings, XML blocks, plain sections, and
sentence-like leaves. It tokenizes with `cl100k_base`, embeds the leaves and rollups, and constructs a
validated `Document`.

### Score

The built-in `redundancy` scorer reports each optimizable sentence's similarity to its most similar
peer. It powers the standalone score report and is an extension point; it is not an obligatory step
in the default end-to-end path.

### Optimize

The built-in `merge_rewrite` optimizer considers similar sentence pairs. Exact duplicates can become
deletions without a merger call. Other pairs are sent to the merger, represented again, and checked
as whole-document candidates. A rejected rewrite can be retried with feedback, up to three model
attempts per pair.

### Select

The built-in `least_cos_sim_diff` selector ranks viable edits by whole-document semantic change. It
folds them over the source document, enforcing the cumulative semantic-change budget and stopping
when `max_tokens` is reached when possible.

### Hard target

When `require_target=True`, reduction uses the hard-target path instead of the normal
Optimize/Select path. Protected Markdown and XML boundaries remain intact. Generated candidates are
checked with the same tokenizer used for reporting, overshoot is repaired deterministically, and the
result must fit `max_tokens`. An infeasible protected structure raises `InfeasibleTargetError` before
generation. Metrics separately report whether the selected target-safe result also met the semantic
change budget.

## Strategy registration

Built-in strategies register by decorator when their modules are imported:

| Kind | Built-in name |
| --- | --- |
| Scorer | `redundancy` |
| Optimizer | `merge_rewrite` |
| Selector | `least_cos_sim_diff` |

The Python API accepts strategy names, but the CLI intentionally uses the built-in choices. There is
no Python entry-point discovery and no installed third-party plugin loading. Adding an out-of-tree
plugin mechanism would require a new public contract and is outside version `0.1.0`.

## Library surface

The top-level `alexandria` package exports the end-to-end `reduce` and `propose` operations, individual
phases, comparison/diff/report helpers, result models, and `Document`. Library callers can provide
their own embedder and merger or allow the defaults to resolve an OpenAI API key.

See [the library guide](library.md) for runnable examples and exact call signatures.

## Command-line surface

All phase commands read an optional file argument or standard input. Diagnostics go to standard
error so standard output remains pipeable.

| Command | Implemented options | Input / output |
| --- | --- | --- |
| `represent [FILE]` | `--out PATH` | Raw prompt -> `DocumentEnvelope` JSON. |
| `score [FILE]` | `--table`, `--out PATH` | `DocumentEnvelope` -> `ScoredEnvelope` JSON or table. |
| `optimize [FILE]` | `--cos-sim-diff-budget`, `--out PATH` | `ScoredEnvelope` -> `PlanEnvelope` JSON. |
| `select [FILE]` | `--cos-sim-diff-budget`, `--json` | `PlanEnvelope` -> reduced text or JSON summary. |
| `compare ORIGINAL EDITED` | `--min-similarity` | Prompt pair -> comparison JSON; optional exit gate. |
| `reduce [FILE]` | `--save-tokens`, `--keep`, `--target-reduction`, `--cos-sim-diff-budget`, `--json`, `--interactive`, `--browser`, `--no-open`, `--verbose` | Raw prompt -> reduced text or JSON summary. |
| `report [FILE]` | `--cos-sim-diff-budget`, `--baseline`, `--quality-tolerance`, `--token-tolerance` | Raw prompt -> optimization report JSON. |
| `tokens [DIRECTORY]` | none | Token counts for recognized instruction Markdown files. |
| `config set openai-api-key` | none | Save an API key with hidden input. |

There are no CLI options for selecting a model, scorer, optimizer, selector, or similarity threshold.
Use the library API for dependency injection and strategy selection. See [the CLI guide](cli.md) for
examples and option interactions.

## Persistence and reports

The phase wire format is JSON, not Parquet:

- `DocumentEnvelope` contains `schema_version = 1` and a `Document`.
- `ScoredEnvelope` adds a name-keyed score bundle.
- `PlanEnvelope` carries the document, candidate plan, and merge metrics.

Pydantic serializes embeddings as JSON arrays and validates them back to NumPy `float32` vectors.
Unknown envelope schema versions are rejected. The repository does not use Polars or an Arrow-backed
persistence layer.

Optimization reports use `schema_version = 3`. Their configuration records the embedder model,
merger identity, optimizer and selector names, threshold, semantic-change budget, `max_tokens`, and
`require_target`; token, quality, and applied-edit metrics follow. A baseline is simply a report JSON
saved by the user. No baseline is committed or enforced in CI.

## Credentials and local configuration

The default OpenAI boundary resolves a key in this order:

1. an explicit library argument;
2. `OPENAI_API_KEY`; then
3. the owner-only, XDG-aware config file written by `alexandria config set openai-api-key`.

Alexandria does not load `.env` files automatically. The example program opts into
`python-dotenv`, which is a development dependency.

## Verification

The repository uses ordinary pytest tests, strict Pyright checks, Ruff, coverage, and import-linter.
Tests are co-located as `*_test.py` plus integration tests under `tests/`. There is no Hypothesis test
suite and no generated `tests/strategies.py` module. Build configuration excludes co-located test
modules from both wheels and source distributions.

Benchmark runners and append-only evidence live under `benchmarks/prompt_compression`; they exercise
the public pipeline but are not part of the runtime package.

## Explicit non-features in 0.1.0

- no Parquet or Polars persistence/reporting;
- no Python entry-point or third-party plugin discovery;
- no CLI model/strategy/threshold selection flags;
- no automatic `.env` loading; and
- no committed optimization baseline or quality-baseline CI gate.
