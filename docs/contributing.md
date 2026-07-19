# Contributing

## Optimization quality gate

The [Optimization quality workflow](../.github/workflows/optimization-quality.yml) runs on every push
and pull request. It generates a deterministic report for
[`benchmarks/optimization_prompt.txt`](../benchmarks/optimization_prompt.txt) and compares it with the
committed [`benchmarks/optimization_baseline.json`](../benchmarks/optimization_baseline.json).

Run the same check locally from the repository root:

```bash
uv sync --dev --frozen
uv run alexandria report benchmarks/optimization_prompt.txt \
  --model deterministic \
  --optimizer greedy_pairwise \
  --selector auto \
  --threshold 0.85 \
  --drift-budget 2.0 \
  --baseline benchmarks/optimization_baseline.json \
  --token-tolerance 0 \
  --quality-tolerance 0.0
```

The command exits with status 1 when the current reduced token count is higher than the baseline or
when either monitored quality score is lower than the baseline. The workflow uses zero tolerance
because the built-in deterministic embedder and locked dependencies make this fixture reproducible.
The emitted JSON includes a `comparison` object, and any failing metrics are also printed to standard
error.

## Updating the report baseline

A baseline update is acceptable only when the metric change is intentional and explained by the same
pull request. Typical reasons are an approved optimizer or selector behavior change, a correction to
the report calculation, or a deliberate change to the benchmark prompt or report configuration. Do
not update the baseline merely to make an unexplained regression pass.

Generate a candidate baseline without comparing it to the existing file:

```bash
uv run alexandria report benchmarks/optimization_prompt.txt \
  --model deterministic \
  --optimizer greedy_pairwise \
  --selector auto \
  --threshold 0.85 \
  --drift-budget 2.0 \
  > benchmarks/optimization_baseline.json.new
```

Review the candidate before replacing the committed baseline:

```bash
python -m json.tool benchmarks/optimization_baseline.json.new > /dev/null
git diff --no-index \
  benchmarks/optimization_baseline.json \
  benchmarks/optimization_baseline.json.new || true
```

Reviewers should confirm that the code or fixture change explains every metric delta, that any rise in
`tokens.reduced` or fall in `quality.instruction_coverage` or
`quality.minimum_instruction_similarity` is an accepted tradeoff, and that the workflow tolerances
were not loosened to hide the change. The report configuration and source token count should change
only when the benchmark fixture or its documented command changes intentionally.

After approval of the candidate values, replace the baseline and rerun the exact CI check:

```bash
mv benchmarks/optimization_baseline.json.new benchmarks/optimization_baseline.json
uv run alexandria report benchmarks/optimization_prompt.txt \
  --model deterministic \
  --optimizer greedy_pairwise \
  --selector auto \
  --threshold 0.85 \
  --drift-budget 2.0 \
  --baseline benchmarks/optimization_baseline.json \
  --token-tolerance 0 \
  --quality-tolerance 0.0
```

Commit the baseline together with the implementation, test, or fixture change that justified it.
