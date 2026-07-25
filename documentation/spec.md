# Alexandria design specification

This document describes the implemented runtime contracts. User-facing commands and examples live in
the [CLI guide](cli.md) and [library guide](library.md).

## Scope

Alexandria compresses instruction-heavy prompts without labels or expected outputs. The normal path
finds semantically similar instructions, proposes deletions or rewrites, and applies compatible edits
within token and whole-prompt semantic-change budgets. The hard-target path instead guarantees a token
ceiling when feasible and reports whether the result also met the semantic budget.

The default runtime uses OpenAI for embeddings and rewrites. Library callers can inject compatible
implementations for offline or alternative-model use.

## Architecture

```text
CLI / public Python API
          |
          v
Represent -> optional Score -> Optimize -> Select
          |          validated IR          |
          +--------------------------------+
                 embedder and merger

Hard target: Represent -> prune and targeted rewrite
Review mode: Represent -> Optimize -> Diff -> human choice -> Apply
```

The packages have these responsibilities:

- `alexandria.ir`: immutable data contracts, edit application, similarity helpers, and strategy
  registries.
- `alexandria.ops.features`: independently callable pipeline operations.
- `alexandria.ops.pipe`: end-to-end reduction, proposals, and metrics.
- `alexandria.cli`: Click commands, review UIs, and JSON envelopes.
- `alexandria.utils`: tokenization, OpenAI adapters, and local configuration.

Import-linter rules in `pyproject.toml` enforce the layer direction and isolate OpenAI client
construction from the IR and operations packages.

## Intermediate representation

The frozen Pydantic tree is:

```text
Document
└── Section (Markdown, XML, or plain)
    ├── Section
    └── Sentence
```

Represent assigns each encoded node its text, a `cl100k_base` token count, and a one-dimensional NumPy
`float32` embedding. `Document` records the embedder model ID. The models validate non-negative token
counts, parent rollups, one embedding dimension across the tree, unique sentence IDs, and at least one
sentence in every document and section.

An edit is a `Delete` or `Replace` targeting sentence IDs. `Document.apply` rebuilds and validates the
tree, ignores stale targets, and rejects edits that would empty a document or section. It does not call
the embedder: rebuilt ancestor embeddings remain source snapshots. Optimize embeds trial text for
measurement without replacing those stored ancestor values; reporting represents the reduced text
again when it needs a fresh tree.

## Runtime controls

`Params` is the shared configuration contract:

| Field | Default | Behavior |
| --- | ---: | --- |
| `threshold` | `0.85` | Minimum pair similarity considered by the built-in optimizer and target pruning. |
| `cos_sim_diff_budget` | `0.5` | Maximum whole-prompt semantic change accepted by the normal selector. |
| `max_tokens` | `None` | Optional stopping ceiling. |
| `require_target` | `False` | Use the hard-target path; requires `max_tokens`. |

One document must use the same embedder model throughout a run. The main injected protocols are
`Embedder`, `SentenceMerger`, `TargetedMerger`, and `ReductionReporter`.

## Normal pipeline

### Represent

Represent splits text losslessly into Markdown, XML, or plain sections and sentence-like leaves.
Markdown headings and XML boundary lines are marked non-optimizable. It tokenizes and embeds sentences,
sections, and the whole document in batches.

### Score

The `redundancy` scorer assigns every sentence the cosine similarity of its most-similar peer. Score is
available for inspection and for optimizers that declare a scorer dependency. The built-in optimizer
declares none, so end-to-end `reduce` skips this phase by default.

### Optimize

`merge_rewrite` considers only optimizable pairs at or above `threshold`, ordered by pair similarity.
Exact duplicates may become `Delete` edits without a merger call. For other pairs, the merger output is
embedded, applied to a trial document, and checked against the source document. A rejected or non-shorter
rewrite receives feedback for up to three attempts.

### Select

`least_cos_sim_diff` ranks viable candidates by their individual whole-prompt semantic change, then
folds them over the source. Each cumulative result is embedded and accepted only when it remains within
`cos_sim_diff_budget`. Selection stops after reaching `max_tokens` when possible. A best-effort token
request may remain unmet when no acceptable edit reaches it.

## Hard targets

With `require_target=True`, `reduce` bypasses Score, Optimize, and Select. It first prunes sufficiently
redundant content while preserving non-optimizable structure, then requests targeted rewrite candidates
for compressible groups. Generated markup is rejected, token overshoot is repaired, and candidates are
ranked by feasibility and measured quality until the document fits `max_tokens`.

The token ceiling takes priority over `cos_sim_diff_budget`; `MergeMetrics.cos_sim_diff_budget_met`
records compliance. `InfeasibleTargetError` identifies targets blocked by protected structure or a lack
of rewritable content. `TargetMergeError` carries metrics when generation cannot make progress.

## Human review

`propose` runs the normal proposal path and renders candidates as discrete diffs. Terminal and browser
review modes apply exactly the accepted candidates through `apply_candidates`; human acceptance replaces
automatic selection, so edits are not re-filtered by the selector.

## Strategy registration

Built-ins register in process under these names:

| Kind | Name |
| --- | --- |
| Scorer | `redundancy` |
| Optimizer | `merge_rewrite` |
| Selector | `least_cos_sim_diff` |

The library accepts registered strategy names. The CLI uses the built-ins. There is no installed
third-party plugin discovery in version `0.1.0`.

## Serialization and reporting

Phase commands exchange versioned Pydantic JSON envelopes:

- `DocumentEnvelope` contains a represented document.
- `ScoredEnvelope` adds named score maps.
- `PlanEnvelope` adds candidates and merge metrics.

Optimization reports contain their configuration, source and reduced token counts, applied-edit count,
and token-weighted mean and minimum best-match similarity for source sentences. Reporting represents
the reduced text again so its quality embeddings are fresh. A baseline is a user-saved compatible report; the
repository does not commit or enforce one.

## Credentials

The default OpenAI adapters resolve an explicit library key first, then `OPENAI_API_KEY`, then the
owner-only XDG-aware file written by `alexandria config set openai-api-key`. Alexandria does not load
`.env` files automatically.
