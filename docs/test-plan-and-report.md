# Test Plan and Report

| Field | Value |
| --- | --- |
| Product | Alexandria |
| Date | 2026-07-21 |
| Code revision | `a1eabac` |
| Pull request | Draft PR #129 |

## Setup

Use Python 3.14 and install the development environment:

```bash
uv sync --dev --frozen
```

Commands that embed or rewrite text need an OpenAI API key. `tokens` and the offline library scenario do not need one.

```bash
alexandria config set openai-api-key
```

Create the test prompt in a directory so the same input works with `tokens`:

```bash
mkdir -p /tmp/alexandria-test
cat > /tmp/alexandria-test/CLAUDE.md <<'EOF'
Always write tests before you write the implementation.
Use descriptive variable names.
Write tests first, before implementing anything.
Pick clear, descriptive names for variables.
Never commit secrets to the repository.
EOF
```

Lines 1 and 3 have the same instruction. Lines 2 and 4 also have the same instruction. A correct reduction keeps the distinct requirements.

From a development checkout, add `uv run` before each `alexandria` command.

## System test scenarios

### Scenario 1: Reduce a prompt with one command

Run:

```bash
alexandria reduce /tmp/alexandria-test/CLAUDE.md --json
```

Expected result: the command returns one JSON object with `text`, `applied`, `source_tokens`, and `reduced_tokens`. The reduced prompt keeps the requirements about tests, variable names, and secrets.

### Scenario 2: Reduce a prompt through the offline Python API

Run:

```bash
uv run python - <<'PY'
import alexandria
from alexandria.ir.contracts import Params
from alexandria.ops import HashEmbedder


class FirstWinsMerger:
    def merge(self, first: str, second: str, feedback: str | None = None) -> str:
        del second, feedback
        return first


result = alexandria.reduce(
    "Write tests first.\nWrite tests first.\nNever commit secrets.\n",
    HashEmbedder(),
    FirstWinsMerger(),
    params=Params(cos_sim_diff_budget=2.0),
)
assert result.reduced_tokens < result.source_tokens
print(result.text)
PY
```

Expected result: the assertion passes and one copy of the duplicate instruction remains.

### Scenario 3: Use token controls and count instruction files

Run:

```bash
alexandria tokens /tmp/alexandria-test
alexandria reduce /tmp/alexandria-test/CLAUDE.md --keep 80 --json
alexandria reduce /tmp/alexandria-test/CLAUDE.md --target-reduction 30 --json
alexandria reduce /tmp/alexandria-test/CLAUDE.md --keep 80 --target-reduction 30
```

Expected results:

- `tokens` prints one count for `CLAUDE.md` and a total. It counts files, not individual instructions.
- `--keep 80` makes a best-effort request. The result may stop above the requested size when no acceptable edit remains.
- `--target-reduction 30` returns a result at or below 70% of the source token count, or exits with an error when the target cannot be reached.
- Combining `--keep` and `--target-reduction` returns an option error.

### Scenario 4: Review edits and save phase output

Run the phase pipeline:

```bash
alexandria represent /tmp/alexandria-test/CLAUDE.md \
  | alexandria score \
  | alexandria optimize \
  | alexandria select
```

Use `--out PATH` with `represent`, `score`, or `optimize` to save the JSON envelope without changing normal standard output. Run `reduce --interactive FILE` or `reduce --browser FILE` to review proposed edits.

Expected result: each phase accepts the previous envelope. Review modes apply only the selected edits.

### Scenario 5: Produce quality metrics

Run:

```bash
alexandria report /tmp/alexandria-test/CLAUDE.md
```

Expected result: one JSON object reports configuration, token counts, the applied-edit count, and instruction-preservation scores. `--baseline FILE` compares the output with a compatible saved report.

### Scenario 6: Compare two prompts

Run:

```bash
alexandria reduce /tmp/alexandria-test/CLAUDE.md > /tmp/alexandria-test/reduced.md
alexandria compare /tmp/alexandria-test/CLAUDE.md /tmp/alexandria-test/reduced.md
```

Expected result: the JSON output reports `similarity`, `cos_sim_diff`, `original_tokens`, `edited_tokens`, and `token_reduction`. `--min-similarity 0.9` returns exit code 1 when the similarity is below 0.9.

### Scenario 7: Install and run outside the checkout

Run:

```bash
python scripts/release_smoke_test.py
```

Expected result: the script builds the wheel and source distribution, installs the wheel in a temporary Python 3.14 environment, runs the installed CLI, and imports the installed library. The reported module path must point inside the temporary environment.

## Recorded results

| Scenario | Status | Date | Evidence |
| --- | --- | --- | --- |
| 1. One-command reduction | Pass | 2026-07-20 | [`tests/ai_e2e_test.py`](../tests/ai_e2e_test.py) ran the real OpenAI pipeline on two prompts. |
| 2. Offline Python API | Pass | 2026-07-21 | [`tests/pipeline_e2e_test.py`](../tests/pipeline_e2e_test.py) checks exact-duplicate reduction without a model call. |
| 3. Token controls and counts | Pass | 2026-07-21 | [`src/alexandria/cli/main_test.py`](../src/alexandria/cli/main_test.py) tests `tokens`, `--keep`, and `--target-reduction`. |
| 4. Review and phase output | Pass | 2026-07-21 | CLI tests cover saved envelopes, terminal review, and browser review. A live presentation remains a manual check. |
| 5. Quality metrics | Pass | 2026-07-21 | CLI and report tests check the JSON schema and baseline comparison. |
| 6. Prompt comparison | Pass | 2026-07-21 | CLI tests check the comparison output and the similarity exit-code gate. |
| 7. Installed package | Pass | 2026-07-21 | [`scripts/release_smoke_test.py`](../scripts/release_smoke_test.py) builds and tests a clean installed wheel. |

## Automated suite

Unit tests use the `*_test.py` name and usually sit beside the code they cover. Broader end-to-end tests are in [`tests/`](../tests/). CI runs the offline suite on every push and enforces at least 80% branch coverage.

```bash
uv run pytest
uv run pytest -m ai
```

The second command makes live OpenAI calls. It skips when an API key is not available.
