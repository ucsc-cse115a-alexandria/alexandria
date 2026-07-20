# Test Plan and Report

- Product: Alexandria
- Team: Alexandria
- Date: 2026-07-20

Alexandria is a CLI and Python library that shrinks a bloated instruction prompt (a `CLAUDE.md` or `AGENT.md`, say) by removing redundant instructions while keeping the meaning. This document lists the system test scenarios a reviewer can run at the project review, ties each one to a user story and its acceptance criteria, and records Pass/Fail for the release version. The last section points to the automated tests.

## Setup

Every pipeline command calls the OpenAI API, so a key must be set once before running the scenarios below:

```
alexandria config set openai-api-key
# or
export OPENAI_API_KEY=sk-...
```

From a development checkout, prefix commands with `uv run` (for example `uv run alexandria reduce ...`). An installed copy (Scenario 7) uses plain `alexandria`.

For a scenario that needs an input file, create a small prompt with duplicated instructions:

```
cat > /tmp/claude.md <<'EOF'
Always write tests before you write the implementation.
Use descriptive variable names.
Write tests first, before implementing anything.
Pick clear, descriptive names for variables.
Never commit secrets to the repository.
EOF
```

Lines 1 and 3 say the same thing; so do lines 2 and 4. A correct reduction keeps one of each pair.

## System Test scenarios

### User stories under test

- US-A (Sprint 1): As a user, I want one CLI command to shorten a bloated CLAUDE.md/AGENT.md by removing redundant instructions while keeping the meaning. Acceptance: reduced token count is lower than the source, and meaning is preserved.
- US-B (Sprint 2/3): As a user, I want to control the compression/quality trade-off and see the savings before I commit.
- US-C (Sprint 3): As a user, I want to review what the compression did before I adopt it.
- US-D (Sprint 3/4): As a user, I want to compress harder while a benchmark checks accuracy, and inspect quality metrics.
- US-E (Sprint 4): As a user, I want to install Alexandria outside the checkout and use it as both a CLI and a Python library.

### Scenario 1: One-command reduction (US-A) [Pass]

1. Run `alexandria reduce /tmp/claude.md --json`.
2. The command prints one JSON object with `text`, `applied`, `source_tokens`, and `reduced_tokens`.
3. Confirm `reduced_tokens < source_tokens`.
4. Read `text` and confirm each distinct instruction still appears once (tests-first, descriptive names, no secrets), so the meaning is preserved.

Expected output: JSON where `reduced_tokens` is smaller than `source_tokens`, and the duplicate lines are collapsed to one each while every distinct instruction survives.

### Scenario 2: Reduction from the Python library (US-A) [Pass]

1. Run this offline, deterministic check that needs no API key:

```
uv run python - <<'EOF'
import alexandria
from alexandria.ops import HashEmbedder
from alexandria.ir.contracts import Params

text = (
    "Always write tests before the implementation.\n"
    "Write tests first, before implementing anything.\n"
    "Never commit secrets.\n"
)
result = alexandria.reduce(
    text,
    embedder=HashEmbedder(),
    params=Params(cos_sim_diff_budget=2.0),
)
print(result.source_tokens, result.reduced_tokens, result.applied)
assert result.reduced_tokens < result.source_tokens
print(result.text)
EOF
```

2. The script prints the source and reduced token counts, the applied edits, and the reduced text.

Expected output: the assertion holds (`reduced_tokens < source_tokens`), and the two tests-first lines collapse to one.

### Scenario 3: Trade-off controls and token accounting (US-B) [Pass]

1. Run `alexandria tokens /tmp/claude.md` to list per-instruction token counts and the total.
2. Run `alexandria reduce /tmp/claude.md --keep 80 --json`. Confirm `reduced_tokens` lands near 80% of `source_tokens`.
3. Run `alexandria reduce /tmp/claude.md --target-reduction 30 --json`. Confirm `reduced_tokens` is at or below the 30%-reduction ceiling, that is `reduced_tokens <= source_tokens * 0.70`.
4. Confirm `--keep` and `--target-reduction` cannot be combined: `alexandria reduce /tmp/claude.md --keep 80 --target-reduction 30` exits with an error.

Expected output: `tokens` prints a per-instruction table and a total; `--keep 80` aims for roughly 80% of the source tokens; `--target-reduction 30` guarantees the result fits under the ceiling; the mutually exclusive flags are rejected.

### Scenario 4: Review before adopting (US-C) [Not yet run (draft), to verify at review]

1. Run `alexandria score --table` on a scored envelope (for example `alexandria represent /tmp/claude.md | alexandria score --table`). Confirm the redundancy table lists each instruction next to its most-similar peer and a similarity score.
2. Save each phase to inspect it: `alexandria represent /tmp/claude.md --out /tmp/represent.json | alexandria score --out /tmp/score.json | alexandria optimize --out /tmp/optimize.json | alexandria select`. Confirm each `--out` file is a self-contained JSON envelope, and that passing one back as the FILE argument resumes from that phase.
3. Run `alexandria reduce /tmp/claude.md --interactive` and accept or reject each proposed edit in the terminal. Confirm the final output reflects only the accepted edits.
4. Run `alexandria reduce /tmp/claude.md --browser` and confirm the same accept/reject review works in a browser.

Expected output: the redundancy table pairs each instruction with its closest peer; the phase envelopes save and resume; interactive and browser review apply only the edits the user accepts.

### Scenario 5: Quality metrics and baseline comparison (US-D) [Not yet run (draft), to verify at review]

1. Run `alexandria report /tmp/claude.md` for token metrics and quality scores, and add `--baseline <file>` to compare against a saved baseline.
2. Confirm the JSON contains token metrics and quality scores, plus a baseline-regression comparison when `--baseline` is passed.

Expected output: one JSON object reporting how far the prompt was compressed and the quality scores, and, with `--baseline`, how the run compares against it.

### Scenario 6: Compare two prompts (US-D) [Not yet run (draft), to verify at review]

1. Reduce the prompt to a second file: `alexandria reduce /tmp/claude.md > /tmp/reduced.md`.
2. Run `alexandria compare /tmp/claude.md /tmp/reduced.md`.
3. Confirm the command prints one JSON object carrying both the cosine similarity and the token reduction between the two prompts. Optionally add `--min-similarity 0.9` and confirm the command exits non-zero when similarity falls below the floor.

Expected output: JSON with a `similarity` field and the token reduction; a high similarity shows the reduced prompt still means what the source meant.

### Scenario 7: Install and use outside the checkout (US-E) [Not yet run (draft), to verify at review]

1. Install the released tool: `uv tool install git+https://github.com/ucsc-cse115a-alexandria/alexandria.git`.
2. Run `alexandria --help` and confirm the command list appears.
3. Set the key with `alexandria config set openai-api-key`, then run `alexandria reduce /tmp/claude.md --json` and confirm `reduced_tokens < source_tokens`.
4. In a Python shell outside the checkout, run `import alexandria; result = alexandria.reduce("...")` and read `result.text`, `result.source_tokens`, `result.reduced_tokens`, and `result.applied`.

Expected output: the tool installs and runs as a standalone CLI, and the same reduction is available as a library call.

### Pass/Fail summary

| Scenario | User story | Release status |
| --- | --- | --- |
| 1. One-command reduction | US-A | Pass |
| 2. Reduction from the library | US-A | Pass |
| 3. Trade-off controls and token accounting | US-B | Pass |
| 4. Review before adopting | US-C | Not yet run (draft) |
| 5. Quality metrics and baseline comparison | US-D | Not yet run (draft) |
| 6. Compare two prompts | US-D | Not yet run (draft) |
| 7. Install and use outside the checkout | US-E | Not yet run (draft) |

The Pass rows are backed by automated end-to-end tests that ran green on 2026-07-20. The live reduction path was exercised by `tests/ai_e2e_test.py`, which reduces two verbose sample prompts through the real pipeline against gpt-5.6-luna; both cases passed. The offline, deterministic path was exercised by `tests/pipeline_e2e_test.py`, which passed. Scenarios 4 through 7 are written from the released feature set but have not been run for this report; they are ready to run at the review.

## Unit tests

The automated suite is 326 collected tests. Unit tests sit next to the code they cover as `<module>_test.py` files across `src/alexandria/cli`, `src/alexandria/ir`, `src/alexandria/ops` (including `src/alexandria/ops/features`), and `src/alexandria/utils`. Broader tests live in `tests/`: `tests/pipeline_e2e_test.py` is an offline end-to-end test, and `tests/ai_e2e_test.py` is a live end-to-end test carrying the pytest `ai` marker, skipped unless an OpenAI key is set.

Run everything:

```
uv run pytest
```

Run only the live tests (needs a key):

```
uv run pytest -m ai
```

CI enforces an 80% branch-coverage gate. The offline suite runs green in CI on every push. The live `ai` end-to-end tests passed on 2026-07-20. All of these can be run during the project review.
