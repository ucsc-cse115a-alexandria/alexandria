# CLI guide

Each command reads either its optional `FILE` argument or standard input, writes its result to standard
output, and sends diagnostics to standard error. Run `uv run alexandria --help` or
`uv run alexandria COMMAND --help` for the complete option reference.

## Set your API key

Alexandria uses OpenAI for embeddings and merging, so every command that runs the pipeline needs a key.
Store it once with hidden input:

```bash
uv run alexandria config set openai-api-key
```

The key is saved to `~/.config/alexandria/config.toml` (owner-only, honoring `XDG_CONFIG_HOME`). You can
instead `export OPENAI_API_KEY=...`. Resolution order is explicit argument, then `OPENAI_API_KEY`, then
the config file. Without a key, commands fail before any work with:

```text
OpenAI API key not found. Set it with `alexandria config set openai-api-key` or export OPENAI_API_KEY.
```

## Reduce a prompt

`reduce` is the usual entry point. It runs all four phases in one process and writes the reduced prompt.

```bash
uv run alexandria reduce prompt.txt > reduced.txt
```

`--json` emits a machine-readable summary with `text`, `applied`, `source_tokens`, and `reduced_tokens`.

```bash
uv run alexandria reduce prompt.txt --json
```

Use `--keep P` to aim for P percent of the prompt's source tokens, or `--save-tokens N` to stop once N
tokens are saved (edits are applied least-drift-first). These options are mutually exclusive. They are
stopping points rather than guarantees: `--drift-budget` may stop compression sooner.

Use `--target-reduction P` when the percentage itself is required: `P` means “reduce by P percent,” and
the command exits with an error if that target is not reached. It is mutually exclusive with `--keep` and
`--save-tokens`; lower the target or raise `--drift-budget` when it reports that the target was unmet.

`--drift-budget` caps the cumulative whole-document embedding drift the reduction may accept (`0.01` =
1%). `--interactive` lets you accept or reject proposed edits in the terminal, and `--browser` does the
same in a browser.

```bash
uv run alexandria reduce --keep 95 prompt.txt > reduced.txt
uv run alexandria reduce prompt.txt --save-tokens 200 > reduced.txt
uv run alexandria reduce --target-reduction 10 prompt.txt > reduced.txt
```

## Run phases separately

The phase commands compose as a Unix pipeline:

```bash
cat prompt.md | alexandria represent | alexandria score | alexandria optimize | alexandria select
```

The first three commands write self-contained JSON envelopes:

| Command | Input | Output |
| --- | --- | --- |
| `represent` | Raw prompt | `DocumentEnvelope` |
| `score` | `DocumentEnvelope` | `ScoredEnvelope` |
| `optimize` | `ScoredEnvelope` | `PlanEnvelope` |
| `select` | `PlanEnvelope` | Reduced prompt text |

`optimize` and `select` each accept `--drift-budget`, and `select` also accepts `--json`.

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

## Offline runs

The CLI always uses the OpenAI defaults and therefore requires an API key. To run the pipeline offline
(for tests or CI) inject your own embedder and merger through the library API; see
[the library guide](library.md).
