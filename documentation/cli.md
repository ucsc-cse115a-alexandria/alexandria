# CLI guide

Run `alexandria --help` or `alexandria COMMAND --help` for the authoritative option reference. In a
development checkout, use `uv run alexandria` in place of `alexandria`.

## Configure OpenAI

Commands that embed or rewrite text use the OpenAI defaults and require a key:

```bash
alexandria config set openai-api-key
```

The command saves hidden input to an owner-only, XDG-aware config file. `OPENAI_API_KEY` takes
precedence over that file. The `score` and `tokens` commands do not make model calls.

## Reduce

`reduce` writes the reduced prompt to standard output and diagnostics to standard error:

```bash
alexandria reduce prompt.md > reduced.md
alexandria reduce prompt.md --json > reduction.json
```

Choose at most one token control:

- `--save-tokens N` asks the best-effort selector to save N tokens.
- `--keep P` asks it to retain P percent of the source tokens.
- `--target-reduction P` makes a P percent reduction a hard requirement.

Best-effort controls may fall short when no acceptable edits remain within
`--cos-sim-diff-budget`. A hard target prioritizes its token ceiling and reports
`merge_metrics.cos_sim_diff_budget_met` separately in JSON. Protected Markdown headings and XML tag
boundaries are not removed.

`-v`/`--verbose` streams automatic compaction progress to standard error.

Use `--interactive` or `--browser` to review proposed edits and apply only accepted ones:

```bash
alexandria reduce --interactive prompt.md > reduced.md
alexandria reduce --browser prompt.md > reduced.md
alexandria reduce --browser --no-open prompt.md
```

Review modes require a file path, are mutually exclusive, and cannot be combined with `--save-tokens`,
`--keep`, `--target-reduction`, or `--verbose`. `--no-open` requires `--browser`.

## Other commands

| Command | Purpose |
| --- | --- |
| `compare ORIGINAL EDITED` | Report cosine similarity and `cl100k_base` token reduction; `--min-similarity` adds an exit-code gate. |
| `report [FILE]` | Run best-effort reduction and emit token and instruction-preservation metrics as JSON. |
| `tokens [DIRECTORY]` | Count Markdown files named `CLAUDE.md` or `AGENT.md`, plus Markdown under a `skills` path. |
| `score [FILE] --table` | Display each represented sentence and its most-similar peer. |

`report --baseline FILE` compares a new report with a compatible report saved earlier. The baseline
must use the same configuration and source token count. No baseline is committed or checked by CI.

## Run phases separately

The phase commands exchange versioned JSON envelopes over standard input and output:

```bash
uv run alexandria represent prompt.md \
  | uv run alexandria score \
  | uv run alexandria optimize \
  | uv run alexandria select > reduced.md
```

| Command | Input | Output |
| --- | --- | --- |
| `represent` | Raw prompt | `DocumentEnvelope` |
| `score` | `DocumentEnvelope` | `ScoredEnvelope` |
| `optimize` | `ScoredEnvelope` | `PlanEnvelope` |
| `select` | `PlanEnvelope` | Reduced text, or a summary with `--json` |

This explicit pipeline includes Score. The end-to-end `reduce` command skips Score for the built-in
optimizer because that optimizer ranks sentence pairs directly.

`represent`, `score`, and `optimize` accept `--out PATH`. The option saves their JSON envelope without
changing the normal standard output. With `score --table`, the table goes to standard output while
`--out` still receives the JSON envelope.

```bash
uv run alexandria represent --out document.json < prompt.md > /dev/null
uv run alexandria score --out scored.json document.json > /dev/null
uv run alexandria optimize --out plan.json scored.json > /dev/null
uv run alexandria select plan.json > reduced.md
```

Each saved file is valid only as input to the next phase. Model-generated rewrites are stochastic, so
rerunning `optimize` may produce a different plan.

For offline execution, inject an embedder and merger through the Python API; see the
[library guide](library.md).
