# CLI guide

Each command reads either its optional `FILE` argument or standard input, writes its result to standard
output, and sends diagnostics to standard error. Run `uv run alexandria --help` or
`uv run alexandria COMMAND --help` for the complete option reference.

## Reduce a prompt

`reduce` is the usual entry point. It runs all four phases in one process and writes the reduced prompt.

```bash
uv run alexandria reduce prompt.txt > reduced.txt
```

By default, Alexandria downloads and uses the `all-MiniLM-L6-v2` sentence-transformer model on first
use. `--json` emits a machine-readable summary with `text`, `applied`, `source_tokens`, and
`reduced_tokens`.

```bash
uv run alexandria reduce prompt.txt --json
```

Use `--threshold` to control redundancy eligibility and `--drift-budget` to limit cumulative semantic
drift. `--min-similarity` is an alternative similarity floor; it cannot be combined with a non-default
`--drift-budget`. `--interactive` lets you accept or reject proposed edits in the terminal.

## Run phases separately

The phase commands compose as a Unix pipeline:

```bash
uv run alexandria represent < prompt.txt \
  | uv run alexandria score \
  | uv run alexandria optimize \
  | uv run alexandria select > reduced.txt
```

The first three commands write self-contained JSON envelopes:

| Command | Input | Output |
| --- | --- | --- |
| `represent` | Raw prompt | `DocumentEnvelope` |
| `score` | `DocumentEnvelope` | `ScoredEnvelope` |
| `optimize` | `ScoredEnvelope` | `PlanEnvelope` |
| `select` | `PlanEnvelope` | Reduced prompt text |

`represent` and `select` accept `--model`. `score` accepts `--scorer`, and `optimize` accepts
`--optimizer` and `--threshold`. `select` accepts `--drift-budget` and `--json`.

## Save an envelope and resume later

`represent`, `score`, and `optimize` accept `--out PATH`. The option saves the same JSON envelope that
the command writes to standard output, so it does not interrupt a pipe.

```bash
uv run alexandria represent --out document.json < prompt.txt \
  | uv run alexandria score --out scored.json \
  | uv run alexandria optimize --out plan.json > /dev/null
```

Pass a saved envelope to the next compatible phase as its `FILE` argument. This reruns the selected
phase without rerunning the earlier phases:

```bash
uv run alexandria optimize scored.json > new-plan.json
uv run alexandria optimize scored.json | uv run alexandria select > reduced.txt
```

For example, `scored.json` is the output of `score` and therefore the input of `optimize`; a raw prompt
or a `DocumentEnvelope` is not accepted by `optimize`. A save-then-load split run produces the same
result as an equivalent in-memory `reduce` run.

## Inspect redundancy

Use `score --table` for a per-instruction, human-readable redundancy report. If `--out` is also present,
the `ScoredEnvelope` is still saved while the table is printed.

```bash
uv run alexandria represent < prompt.txt | uv run alexandria score --table
```

## Deterministic offline runs

For tests, CI, or an offline smoke test, choose the deterministic hash embedder. It is not semantic, so
use a larger drift budget when you want edits to be accepted.

```bash
printf 'Be concise.\nBe concise.\nUse examples.\n' \
  | uv run alexandria reduce --model deterministic --drift-budget 2.0
```

This prints:

```text
Be concise.
Use examples.
```
