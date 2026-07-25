# Test Plan and Report

**Product:** Alexandria ·
**Team:** Alexandria ·
**Date:** 2026-07-20

## Setup

Alexandria is a CLI and Python library that shortens a bloated instruction prompt (a `CLAUDE.md` or
`AGENT.md`) by removing redundant instructions while keeping the meaning. Every pipeline command
calls the OpenAI API, so set a key once before the scenarios below:

```bash
alexandria config set openai-api-key   # or: export OPENAI_API_KEY=sk-...
```

From a development checkout, prefix commands with `uv run` (for example `uv run alexandria reduce`).
Scenario 7 uses an installed copy and runs plain `alexandria`. Where a scenario needs an input file,
create a small prompt with duplicated instructions:

```bash
cat > /tmp/claude.md <<'EOF'
Always write tests before you write the implementation.
Use descriptive variable names.
Write tests first, before implementing anything.
Pick clear, descriptive names for variables.
Never commit secrets to the repository.
EOF
```

Lines 1 and 3 say the same thing, and so do lines 2 and 4. A correct reduction keeps one of each
pair.

## System Test scenarios

### User stories under test

A. **User story A (Sprint 1):** As an engineer with a bloated `CLAUDE.md`/`AGENT.md`, I want one CLI
command to remove redundant instructions while keeping the meaning, so that I cut per-request token
cost. Acceptance: the reduced token count is lower than the source, and every distinct instruction
survives.

B. **User story B (Sprint 2/3):** As an engineer, I want to cap how hard the prompt is compressed and
see the token savings, so that I can trade accuracy for cost on my terms. Acceptance: `--keep`,
`--save-tokens`, `--min-similarity`, `--max-tokens`, and `--target-reduction` work; `tokens` lists
counts.

C. **User story C (Sprint 3):** As an engineer, I want to review what the compression did before I
adopt it, so that I can switch with confidence. Acceptance: `reduce --interactive`, `reduce
--browser`, `score --table`, and phase-by-phase JSON with `--out` all work.

D. **User story D (Sprint 3/4):** As an engineer, I want to compress harder while a benchmark checks
accuracy and inspect quality metrics, so that I only keep the edits the metrics support. Acceptance: the
multi-optimizer pass keeps only budget-backed edits, and `report` emits token and quality metrics.

E. **User story E (Sprint 4):** As an engineer, I want to install Alexandria outside the checkout and
use it as a CLI and a Python library, so that I can drop it into my own workflow. Acceptance: it
installs from the repository, the CLI runs, and the same reduction is callable from Python.

### Scenario 1: One-command reduction (User story A) — Pass

1. Run `alexandria reduce /tmp/claude.md --json`.
2. Output: one JSON object with `text`, `applied`, `source_tokens`, and `reduced_tokens`, where
   `reduced_tokens < source_tokens`.
3. Read `text`: the duplicate lines are collapsed to one each, and every distinct instruction
   (tests-first, descriptive names, no secrets) still appears.

### Scenario 2: Reduction from the Python library, offline (User story A) — Pass

1. Run the deterministic check that needs no API key:

   ```bash
   uv run python - <<'EOF'
   import alexandria
   from alexandria.ir.contracts import Params
   from alexandria.ops import HashEmbedder


   class FirstWinsMerger:
       def merge(self, first: str, second: str, feedback: str | None = None) -> str:
           del second, feedback
           return first


   text = "Write tests first.\nWrite tests first.\nNever commit secrets.\n"
   result = alexandria.reduce(
       text,
       HashEmbedder(),
       FirstWinsMerger(),
       params=Params(cos_sim_diff_budget=2.0),
   )
   assert result.reduced_tokens < result.source_tokens
   print(result.source_tokens, result.reduced_tokens)
   print(result.text)
   EOF
   ```

2. Output: the assertion holds (`reduced_tokens < source_tokens`) and the exact duplicate collapses
   to one.

### Scenario 3: Trade-off controls and token accounting (User story B) — Pass

1. Run `alexandria tokens /tmp/claude.md`. Output: a per-instruction token count and a total.
2. Run `alexandria reduce /tmp/claude.md --keep 80 --json`. Output: `reduced_tokens` lands near 80%
   of `source_tokens`.
3. Run `alexandria reduce /tmp/claude.md --target-reduction 30 --json`. Output: `reduced_tokens <=
   source_tokens * 0.70` (the 30%-reduction ceiling is guaranteed).
4. Run `alexandria reduce /tmp/claude.md --keep 80 --target-reduction 30`. Output: the command exits
   with an error, because the two flags are mutually exclusive.

### Scenario 4: Review before adopting (User story C) — Not yet run (draft)

1. Run `alexandria represent /tmp/claude.md | alexandria score --table`. Output: a redundancy table
   listing each instruction, its most-similar peer, and a similarity score.
2. Run `alexandria represent /tmp/claude.md --out /tmp/represent.json | alexandria score --out
   /tmp/score.json | alexandria optimize --out /tmp/optimize.json | alexandria select`. Output: each
   `--out` file is a self-contained JSON envelope; passing one back as the FILE argument resumes from
   that phase.
3. Run `alexandria reduce /tmp/claude.md --interactive` and accept or reject each edit. Output: the
   final prompt reflects only the accepted edits.
4. Run `alexandria reduce /tmp/claude.md --browser`. Output: the same accept/reject review works in a
   browser.

### Scenario 5: Quality metrics (User story D) — Not yet run (draft)

1. Run `alexandria report /tmp/claude.md`, optionally with `--baseline <file>`.
2. Output: one JSON object with token metrics and quality scores, plus a baseline-regression
   comparison when `--baseline` is passed.

### Scenario 6: Compare two prompts (User story D) — Not yet run (draft)

1. Run `alexandria reduce /tmp/claude.md > /tmp/reduced.md`.
2. Run `alexandria compare /tmp/claude.md /tmp/reduced.md`.
3. Output: one JSON object with a `similarity` field and the token reduction. Adding
   `--min-similarity 0.9` makes the command exit non-zero when similarity falls below the floor.

### Scenario 7: Install and use outside the checkout (User story E) — Automated CI gate

1. Run `python scripts/release_smoke_test.py` from the repository checkout.
2. The script builds the release wheel and source distribution, then installs the wheel in a
   temporary Python 3.14 environment.
3. It runs the installed `alexandria` console script with a local deterministic embeddings endpoint.
   Output: the command exits successfully and `reduced_tokens < source_tokens`.
4. It runs `import alexandria` through the installed interpreter with `HashEmbedder` and an offline
   merger. Output: the API exits successfully, reduces the fixture, and reports a module path inside
   the temporary environment rather than the checkout.

### Release-version results

| Scenario | User story | Result |
| --- | --- | --- |
| 1. One-command reduction | A | Pass |
| 2. Reduction from the library (offline) | A | Pass |
| 3. Trade-off controls and token accounting | B | Pass |
| 4. Review before adopting | C | Not yet run (draft) |
| 5. Quality metrics | D | Not yet run (draft) |
| 6. Compare two prompts | D | Not yet run (draft) |
| 7. Install and use outside the checkout | E | Automated (pending CI) |

The Pass rows are backed by automated end-to-end tests that ran green on 2026-07-20:
`tests/ai_e2e_test.py` reduced two verbose sample prompts through the real pipeline against
`gpt-5.6-luna` (both cases passed), and `tests/pipeline_e2e_test.py` exercised the offline
deterministic path (passed). Scenarios 4 through 6 are written from the released feature set but
have not been run for this report. Scenario 7 is enforced by the clean-package smoke script in CI;
record it as Pass after the release workflow completes successfully.

## Unit tests

The automated suite is 326 collected tests. Unit tests sit next to the code they cover as
`<module>_test.py` files across `src/alexandria/cli`, `src/alexandria/ir`, `src/alexandria/ops`
(including `src/alexandria/ops/features`), and `src/alexandria/utils`. Broader tests live in
`tests/`: `tests/pipeline_e2e_test.py` is an offline end-to-end test, and `tests/ai_e2e_test.py` is a
live end-to-end test carrying the pytest `ai` marker (skipped unless an OpenAI key is set).

```bash
uv run pytest          # everything (live ai tests skip without a key)
uv run pytest -m ai    # only the live tests (needs a key)
```

No unit test failed for the released version. The offline suite runs green in CI on every push, and
the live `ai` end-to-end tests passed on 2026-07-20. CI enforces an 80% branch-coverage gate. All of
these can be run during the project review.
