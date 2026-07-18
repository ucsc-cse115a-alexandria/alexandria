from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import click
import pytest
from click.testing import CliRunner

from alexandria.cli import cli
from alexandria.cli.envelope import DocumentEnvelope, PlanEnvelope, ScoredEnvelope
from alexandria.ir.contracts import Params
from alexandria.ir.registry import register_scorer
from alexandria.ops import HashEmbedder, count_tokens, default_embedder, default_merger
from alexandria.ops.features.represent import represent
from alexandria.ops.features.score import score
from alexandria.ops.pipe import ReduceResult, reduce

if TYPE_CHECKING:
    from alexandria.ir.contracts import Embedder, SentenceMerger
    from alexandria.ir.document import Document


def _constant_scorer(document: Document) -> list[float]:
    return [0.0 for _ in document.sentences]


register_scorer("cli_test_constant")(_constant_scorer)


class _FakeMerger:
    """Offline merger: the first sentence wins, so exact-duplicate pairs become deletes."""

    def merge(self, first: str, second: str, feedback: str | None = None) -> str:
        del second, feedback
        return first.strip()


@pytest.fixture
def offline_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("alexandria.cli.main.default_embedder", lambda: HashEmbedder())
    monkeypatch.setattr("alexandria.cli.main.default_merger", lambda: _FakeMerger())


@pytest.mark.usefixtures("offline_models")
def test_phase_verbs_pipe_to_the_same_text_as_reduce() -> None:
    runner = CliRunner()
    prompt = "keep one\nkeep one\nunique line\n"

    document = runner.invoke(cli, ["represent"], input=prompt)
    scored = runner.invoke(cli, ["score"], input=document.output)
    plan = runner.invoke(cli, ["optimize", "--drift-budget", "2.0"], input=scored.output)
    reduced = runner.invoke(cli, ["select", "--drift-budget", "2.0"], input=plan.output)

    assert [document.exit_code, scored.exit_code, plan.exit_code, reduced.exit_code] == [0, 0, 0, 0]
    expected = reduce(prompt, HashEmbedder(), _FakeMerger(), params=Params(drift_budget=2.0))
    assert reduced.output == expected.text


@pytest.mark.usefixtures("offline_models")
def test_out_is_a_tee_so_the_piped_run_still_matches_reduce() -> None:
    runner = CliRunner()
    prompt = "keep one\nkeep one\nunique line\n"
    with runner.isolated_filesystem():
        document = runner.invoke(cli, ["represent", "--out", "doc.json"], input=prompt)
        scored = runner.invoke(cli, ["score", "--out", "scored.json"], input=document.output)
        plan = runner.invoke(cli, ["optimize", "--drift-budget", "2.0", "--out", "plan.json"], input=scored.output)
        reduced = runner.invoke(cli, ["select", "--drift-budget", "2.0"], input=plan.output)

        assert [document.exit_code, scored.exit_code, plan.exit_code, reduced.exit_code] == [0, 0, 0, 0]
        assert [Path(name).exists() for name in ("doc.json", "scored.json", "plan.json")] == [True, True, True]

    expected = reduce(prompt, HashEmbedder(), _FakeMerger(), params=Params(drift_budget=2.0))
    assert reduced.output == expected.text


@pytest.mark.usefixtures("offline_models")
def test_a_phase_starts_from_a_saved_file_matching_the_in_memory_reduce() -> None:
    # #60: each phase can be rerun alone by loading the previous phase's saved envelope from a file.
    runner = CliRunner()
    prompt = "keep one\nkeep one\nunique line\n"
    with runner.isolated_filesystem():
        Path("d.json").write_text(runner.invoke(cli, ["represent"], input=prompt).output)
        Path("s.json").write_text(runner.invoke(cli, ["score", "d.json"]).output)  # starts from the saved file
        Path("pl.json").write_text(
            runner.invoke(cli, ["optimize", "--drift-budget", "2.0", "s.json"]).output
        )  # starts from the saved file
        reduced = runner.invoke(cli, ["select", "--drift-budget", "2.0", "pl.json"])

    assert reduced.exit_code == 0
    expected = reduce(prompt, HashEmbedder(), _FakeMerger(), params=Params(drift_budget=2.0))
    assert reduced.output == expected.text


@pytest.mark.usefixtures("offline_models")
def test_select_json_reports_the_reduction_summary() -> None:
    runner = CliRunner()
    prompt = "keep one\nkeep one\nunique line\n"
    document = runner.invoke(cli, ["represent"], input=prompt)
    scored = runner.invoke(cli, ["score"], input=document.output)
    plan = runner.invoke(cli, ["optimize", "--drift-budget", "2.0"], input=scored.output)

    result = runner.invoke(cli, ["select", "--drift-budget", "2.0", "--json"], input=plan.output)

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["text"].count("keep one") == 1
    assert len(payload["applied"]) == 1
    assert payload["reduced_tokens"] < payload["source_tokens"]
    assert payload["merge_metrics"] == {
        "calls": 0,
        "retries": 0,
        "pairs_attempted": 0,
        "proposed_edits": 1,
        "applied_edits": 1,
    }


def test_score_emits_a_scored_envelope_by_default() -> None:
    runner = CliRunner()
    document = DocumentEnvelope(document=represent("dup\ndup\n", HashEmbedder())).model_dump_json()

    result = runner.invoke(cli, ["score"], input=document)

    assert result.exit_code == 0
    assert '"scores"' in result.output
    assert '"redundancy"' in result.output


def test_score_out_saves_a_roundtrippable_scored_envelope() -> None:
    runner = CliRunner()
    document = DocumentEnvelope(document=represent("dup\ndup\n", HashEmbedder())).model_dump_json()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["score", "--out", "scored.json"], input=document)

        assert result.exit_code == 0
        saved = Path("scored.json").read_text()
        assert saved.strip() == result.output.strip()
        assert ScoredEnvelope.model_validate_json(saved).model_dump_json() == saved.strip()


def test_score_table_prints_a_human_report() -> None:
    runner = CliRunner()
    document = DocumentEnvelope(document=represent("dup\ndup\n", HashEmbedder())).model_dump_json()

    result = runner.invoke(cli, ["score", "--table"], input=document)

    assert result.exit_code == 0
    assert "redundancy=" in result.output
    assert "most_similar_id=" in result.output


@pytest.mark.usefixtures("offline_models")
def test_optimize_out_saves_a_roundtrippable_plan_envelope() -> None:
    runner = CliRunner()
    document = DocumentEnvelope(document=represent("dup\ndup\n", HashEmbedder())).model_dump_json()
    scored = runner.invoke(cli, ["score"], input=document).output
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["optimize", "--drift-budget", "2.0", "--out", "plan.json"], input=scored)

        assert result.exit_code == 0
        saved = Path("plan.json").read_text()
        assert saved.strip() == result.output.strip()
        assert PlanEnvelope.model_validate_json(saved).model_dump_json() == saved.strip()


@pytest.mark.usefixtures("offline_models")
def test_select_reports_a_model_mismatch_cleanly() -> None:
    document = represent("a\nb\n", HashEmbedder())  # model_id 'hash-64'
    payload = json.loads(PlanEnvelope(document=document, plan=()).model_dump_json())
    payload["document"]["embedding_model"] = "hash-32"  # a document built by a different embedder

    result = CliRunner().invoke(cli, ["select"], input=json.dumps(payload))  # the offline embedder is 'hash-64'

    assert result.exit_code == 1
    assert "does not match" in result.output


def test_score_rejects_a_malformed_envelope_cleanly() -> None:
    result = CliRunner().invoke(cli, ["score"], input="not an envelope")

    assert result.exit_code == 1
    assert "invalid input" in result.output


@pytest.mark.usefixtures("offline_models")
def test_optimize_rejects_an_unknown_schema_version_cleanly() -> None:
    scored: dict[str, object] = {"schema_version": 2, "document": {}, "scores": {}}

    result = CliRunner().invoke(cli, ["optimize"], input=json.dumps(scored))

    assert result.exit_code == 1
    assert "invalid input" in result.output


@pytest.mark.usefixtures("offline_models")
def test_optimize_reports_a_missing_required_scorer_cleanly() -> None:
    document = represent("dup\ndup\n", HashEmbedder())
    bundle = score(document, names=("cli_test_constant",))  # lacks the redundancy scorer merge_rewrite requires
    scored = ScoredEnvelope(document=document, scores=bundle).model_dump_json()

    result = CliRunner().invoke(cli, ["optimize"], input=scored)

    assert result.exit_code == 1
    assert "redundancy" in result.output
    assert "Traceback" not in result.output


@pytest.mark.usefixtures("offline_models")
def test_reduce_drops_a_duplicate_under_a_generous_budget() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["reduce", "--drift-budget", "2.0"], input="keep one\nkeep one\nunique\n")
    assert result.exit_code == 0
    assert result.output.count("keep one") == 1
    assert "unique" in result.output


@pytest.mark.usefixtures("offline_models")
def test_reduce_keeps_everything_under_the_default_budget() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["reduce"], input="keep one\nkeep one\nunique\n")
    assert result.exit_code == 0
    assert result.output.count("keep one") == 2


@pytest.mark.usefixtures("offline_models")
def test_reduce_json_reports_the_applied_edits_and_token_counts() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["reduce", "--drift-budget", "2.0", "--json"],
        input="keep one\nkeep one\nunique\n",
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["text"].count("keep one") == 1
    assert len(payload["applied"]) == 1
    assert payload["reduced_tokens"] < payload["source_tokens"]
    assert payload["merge_metrics"] == {
        "calls": 0,
        "retries": 0,
        "pairs_attempted": 0,
        "proposed_edits": 1,
        "applied_edits": 1,
    }


@pytest.mark.usefixtures("offline_models")
def test_report_outputs_machine_readable_token_and_quality_fields() -> None:
    result = CliRunner().invoke(
        cli,
        ["report", "--drift-budget", "2.0"],
        input="keep one\nkeep one\nunique\n",
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["tokens"]["source"] > payload["tokens"]["reduced"]
    assert "instruction_coverage" in payload["quality"]
    assert "minimum_instruction_similarity" in payload["quality"]


@pytest.mark.usefixtures("offline_models")
def test_report_baseline_passes_then_flags_a_regression() -> None:
    runner = CliRunner()
    prompt = "keep one\nkeep one\nunique\n"
    with runner.isolated_filesystem():
        baseline = runner.invoke(cli, ["report", "--drift-budget", "2.0"], input=prompt)
        assert baseline.exit_code == 0
        Path("baseline.json").write_text(baseline.output)

        passing = runner.invoke(cli, ["report", "--drift-budget", "2.0", "--baseline", "baseline.json"], input=prompt)
        assert passing.exit_code == 0
        assert json.loads(passing.output)["comparison"] == {"passed": True, "regressions": []}

        doctored = json.loads(baseline.output)
        doctored["tokens"]["reduced"] -= 1  # the baseline now claims a smaller prompt than we can produce
        Path("regressed.json").write_text(json.dumps(doctored))
        regressed = runner.invoke(
            cli, ["report", "--drift-budget", "2.0", "--baseline", "regressed.json"], input=prompt
        )

    assert regressed.exit_code == 1
    payload = json.loads(regressed.output.split("\nregression:", maxsplit=1)[0])
    assert not payload["comparison"]["passed"]
    assert "tokens.reduced" in regressed.output


def test_reduce_without_a_key_fails_with_setup_instructions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # default_embedder is lru_cached; clear it so a client cached by another test cannot bypass the key check.
    default_embedder.cache_clear()
    default_merger.cache_clear()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))  # empty config dir
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    runner = CliRunner()

    result = runner.invoke(cli, ["reduce"], input="a\nb\n")

    assert result.exit_code != 0
    assert "alexandria config set openai-api-key" in result.output


@pytest.mark.usefixtures("offline_models")
def test_save_tokens_stops_at_the_target() -> None:
    runner = CliRunner()
    prompt = "repeat me\nrepeat me\nrepeat me\nunique line\n"
    source = count_tokens(prompt)

    result = runner.invoke(cli, ["reduce", "--save-tokens", "3", "--drift-budget", "2.0", "--json"], input=prompt)

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["source_tokens"] == source
    assert payload["reduced_tokens"] <= source - 3


@pytest.mark.usefixtures("offline_models")
def test_keep_derives_the_max_token_budget_from_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    from alexandria.cli import main as main_module

    prompt = "repeat me\nrepeat me\nrepeat me\nunique line\n"
    source = count_tokens(prompt)
    seen_params: list[Params] = []

    def capture_params(
        prompt: str,
        embedder: Embedder | None = None,
        merger: SentenceMerger | None = None,
        *,
        params: Params | None = None,
    ) -> ReduceResult:
        assert params is not None
        seen_params.append(params)
        return reduce(prompt, embedder, merger, params=params)

    monkeypatch.setattr(main_module, "reduce", capture_params)

    result = CliRunner().invoke(cli, ["reduce", "--keep", "60"], input=prompt)

    assert result.exit_code == 0
    assert [params.max_tokens for params in seen_params] == [source * 60 // 100]
    assert not seen_params[0].require_target


@pytest.mark.usefixtures("offline_models")
def test_target_reduction_derives_the_max_token_budget_and_requires_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alexandria.cli import main as main_module

    prompt = "repeat me\nrepeat me\nrepeat me\nunique line\n"
    source = count_tokens(prompt)
    seen_params: list[Params] = []

    def capture_params(
        prompt: str,
        embedder: Embedder | None = None,
        merger: SentenceMerger | None = None,
        *,
        params: Params | None = None,
    ) -> ReduceResult:
        assert params is not None
        seen_params.append(params)
        return reduce(prompt, embedder, merger, params=params)

    monkeypatch.setattr(main_module, "reduce", capture_params)

    result = CliRunner().invoke(
        cli, ["reduce", "--target-reduction", "20", "--drift-budget", "2.0", "--json"], input=prompt
    )

    assert result.exit_code == 0
    assert [params.max_tokens for params in seen_params] == [source * 80 // 100]
    assert seen_params[0].require_target
    assert json.loads(result.output)["reduction_pct"] >= 0.20


@pytest.mark.usefixtures("offline_models")
def test_target_reduction_fails_when_the_drift_gate_prevents_the_target() -> None:
    result = CliRunner().invoke(
        cli,
        ["reduce", "--target-reduction", "10", "--json"],
        input="repeat me\nrepeat me\nunique line\n",
    )

    assert result.exit_code == 1
    assert "target reduction 10% was not met: achieved 0.0%" in result.output


def test_budget_options_are_mutually_exclusive() -> None:
    result = CliRunner().invoke(
        cli, ["reduce", "--keep", "95", "--save-tokens", "5", "--target-reduction", "10"], input="a\nb\n"
    )

    assert result.exit_code == 2
    assert "--keep, --save-tokens, and --target-reduction are mutually exclusive" in result.output


@pytest.mark.parametrize("keep", ["-1", "0", "100", "101"])
def test_keep_rejects_values_outside_the_open_percentage_range(keep: str) -> None:
    result = CliRunner().invoke(cli, ["reduce", "--keep", keep], input="a\nb\n")

    assert result.exit_code == 2
    assert "0.0<x<100.0" in result.output


@pytest.mark.parametrize("reduction", ["-1", "100", "101"])
def test_target_reduction_rejects_values_outside_the_percentage_range(reduction: str) -> None:
    result = CliRunner().invoke(cli, ["reduce", "--target-reduction", reduction], input="a\nb\n")

    assert result.exit_code == 2
    assert "0.0<=x<100.0" in result.output


@pytest.mark.usefixtures("offline_models")
def test_keep_json_reports_the_existing_reduction_summary() -> None:
    prompt = "repeat me\nrepeat me\nrepeat me\nunique line\n"

    result = CliRunner().invoke(
        cli,
        ["reduce", "--keep", "60", "--drift-budget", "2.0", "--json"],
        input=prompt,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["source_tokens"] == count_tokens(prompt)
    assert payload["reduced_tokens"] == count_tokens(payload["text"])
    assert payload["reduced_tokens"] <= payload["source_tokens"] * 60 // 100
    assert payload["reduction_pct"] == pytest.approx(1.0 - payload["reduced_tokens"] / payload["source_tokens"])


def test_removed_flags_are_gone_from_help() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["reduce", "--help"])

    assert result.exit_code == 0
    for flag in ("--model", "--optimizer", "--selector", "--threshold", "--min-similarity", "--max-tokens"):
        assert flag not in result.output


def test_reduce_interactive_rejects_a_stdin_prompt() -> None:
    result = CliRunner().invoke(cli, ["reduce", "--interactive"], input="a\nb\n")

    assert result.exit_code == 2
    assert "--interactive" in result.output
    assert "stdin" in result.output


@pytest.mark.usefixtures("offline_models")
def test_reduce_interactive_applies_only_accepted_edits(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = iter("\rd")  # accept the first (only) proposal, then done
    monkeypatch.setattr(click, "getchar", lambda: next(keys))
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("p.md").write_text("keep one\nkeep one\nunique\n")
        result = runner.invoke(cli, ["reduce", "--interactive", "--drift-budget", "2.0", "p.md"])

    assert result.exit_code == 0
    assert result.stdout.count("keep one") == 1
    assert "unique" in result.stdout


@pytest.mark.usefixtures("offline_models")
def test_reduce_interactive_quit_leaves_the_prompt_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(click, "getchar", lambda: "q")
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("p.md").write_text("keep one\nkeep one\nunique\n")
        result = runner.invoke(cli, ["reduce", "--interactive", "--drift-budget", "2.0", "p.md"])

    assert result.exit_code == 0
    assert result.stdout.count("keep one") == 2
    assert "aborted" in result.stderr


_TWO_PAIR_PROMPT = "# A\nrepeat me\nrepeat me\n# B\necho twice\necho twice\n"


def _interactive_reduce_run(monkeypatch: pytest.MonkeyPatch, keys: str) -> tuple[int, str]:
    feed = iter(keys)
    monkeypatch.setattr(click, "getchar", lambda: next(feed))
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("p.md").write_text(_TWO_PAIR_PROMPT)
        result = runner.invoke(cli, ["reduce", "--interactive", "--drift-budget", "2.0", "p.md"])
    return result.exit_code, result.stdout


@pytest.mark.usefixtures("offline_models")
def test_interactive_accept_all_matches_the_automatic_run(monkeypatch: pytest.MonkeyPatch) -> None:
    exit_code, stdout = _interactive_reduce_run(monkeypatch, "ad")  # check everything, done

    # A budget generous enough that the automatic selector rejects nothing (the #51 reading).
    automatic = reduce(_TWO_PAIR_PROMPT, HashEmbedder(), _FakeMerger(), params=Params(drift_budget=2.0))
    assert exit_code == 0
    assert stdout == automatic.text


@pytest.mark.usefixtures("offline_models")
def test_interactive_reject_all_returns_the_input_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    exit_code, stdout = _interactive_reduce_run(monkeypatch, "d")  # done with nothing checked

    assert exit_code == 0
    assert stdout == _TWO_PAIR_PROMPT


@pytest.mark.usefixtures("offline_models")
def test_interactive_mixed_selection_applies_exactly_the_accepted_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    exit_code, stdout = _interactive_reduce_run(monkeypatch, "\rd")  # accept the cursor row only, done

    assert exit_code == 0
    # One duplicate pair reduced, the other left untouched — exactly the accepted subset.
    reduced_pair, kept_pair = sorted((stdout.count("repeat me"), stdout.count("echo twice")))
    assert (reduced_pair, kept_pair) == (1, 2)
    assert "# A" in stdout and "# B" in stdout


def test_reduce_interactive_rejects_save_tokens() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("p.md").write_text("a\nb\n")
        result = runner.invoke(cli, ["reduce", "--interactive", "--save-tokens", "5", "p.md"])

    assert result.exit_code == 2
    assert "--interactive" in result.output
    assert "--save-tokens" in result.output


@pytest.mark.usefixtures("offline_models")
def test_reduce_browser_applies_only_accepted_edits(monkeypatch: pytest.MonkeyPatch) -> None:
    from alexandria.cli import main as main_module
    from alexandria.ir.contracts import Candidate
    from alexandria.ops.pipe import Proposal, propose

    prompt = "keep one\nkeep one\nunique\n"
    proposal = propose(prompt, HashEmbedder(), _FakeMerger(), params=Params(drift_budget=2.0))
    accepted = (proposal.diffs[0].candidate,)

    def fake_run_browser_review(_proposal: Proposal, **_: object) -> tuple[Candidate, ...]:
        return accepted

    monkeypatch.setattr(main_module, "run_browser_review", fake_run_browser_review)

    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("p.md").write_text(prompt)
        result = runner.invoke(cli, ["reduce", "--browser", "--drift-budget", "2.0", "p.md"])

    assert result.exit_code == 0
    assert result.stdout.count("keep one") == 1
    assert "unique" in result.stdout


@pytest.mark.usefixtures("offline_models")
def test_reduce_browser_abort_leaves_the_prompt_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    from alexandria.cli import main as main_module
    from alexandria.ops.pipe import Proposal

    def fake_run_browser_review(_proposal: Proposal, **_: object) -> None:
        return None

    monkeypatch.setattr(main_module, "run_browser_review", fake_run_browser_review)

    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("p.md").write_text("keep one\nkeep one\nunique\n")
        result = runner.invoke(cli, ["reduce", "--browser", "--drift-budget", "2.0", "p.md"])

    assert result.exit_code == 0
    assert result.stdout.count("keep one") == 2
    assert "aborted" in result.stderr


def test_reduce_browser_rejects_stdin_prompt() -> None:
    result = CliRunner().invoke(cli, ["reduce", "--browser"], input="a\nb\n")

    assert result.exit_code == 2
    assert "--browser" in result.output
    assert "stdin" in result.output


def test_reduce_browser_rejects_save_tokens() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("p.md").write_text("a\nb\n")
        result = runner.invoke(cli, ["reduce", "--browser", "--save-tokens", "5", "p.md"])

    assert result.exit_code == 2
    assert "--browser" in result.output
    assert "--save-tokens" in result.output


def test_reduce_browser_and_interactive_mutually_exclusive() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("p.md").write_text("a\nb\n")
        result = runner.invoke(cli, ["reduce", "--browser", "--interactive", "p.md"])

    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_reduce_no_open_requires_browser() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("p.md").write_text("a\nb\n")
        result = runner.invoke(cli, ["reduce", "--no-open", "p.md"])

    assert result.exit_code == 2
    assert "--no-open requires --browser" in result.output


@pytest.mark.usefixtures("offline_models")
def test_compare_min_similarity_gates_the_exit_code() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("a.txt").write_text("keep this exact instruction\n")
        Path("b.txt").write_text("keep this exact instruction\n")
        Path("c.txt").write_text("a completely unrelated instruction\n")

        same = runner.invoke(cli, ["compare", "a.txt", "b.txt", "--min-similarity", "0.99"])
        differ = runner.invoke(cli, ["compare", "a.txt", "c.txt", "--min-similarity", "0.99"])

    assert same.exit_code == 0
    assert json.loads(same.output)["similarity"] == pytest.approx(1.0)
    assert differ.exit_code == 1


@pytest.mark.usefixtures("offline_models")
def test_represent_out_saves_a_roundtrippable_document_envelope() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["represent", "--out", "doc.json"], input="dup\ndup\n")

        assert result.exit_code == 0
        saved = Path("doc.json").read_text()
        # The saved file matches stdout, and re-parsing then re-dumping reproduces it byte-for-byte.
        assert saved.strip() == result.output.strip()
        assert DocumentEnvelope.model_validate_json(saved).model_dump_json() == saved.strip()


@pytest.mark.usefixtures("offline_models")
def test_represent_rejects_an_empty_prompt_cleanly() -> None:
    result = CliRunner().invoke(cli, ["represent"], input="")

    assert result.exit_code == 1
    assert "empty prompt" in result.output


def test_tokens_counts_instruction_files_accurately() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Setup: Create our target instruction files
        Path("CLAUDE.md").write_text("this is a test prompt\n")

        # Setup: Create a skills directory with a file
        skills_dir = Path("skills")
        skills_dir.mkdir()
        (skills_dir / "python.md").write_text("another test prompt here\n")

        # Execution
        result = runner.invoke(cli, ["tokens", "."])

        # Verification
        assert result.exit_code == 0
        assert "CLAUDE.md:" in result.output
        assert "python.md:" in result.output
        assert "Total:" in result.output


def test_config_set_prompts_hidden_and_saves_the_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    runner = CliRunner()

    result = runner.invoke(cli, ["config", "set", "openai-api-key"], input="sk-test-123\n")

    assert result.exit_code == 0
    saved = tmp_path / "alexandria" / "config.toml"
    assert saved.read_text() == 'openai-api-key = "sk-test-123"\n'
    assert "sk-test-123" not in result.output  # hidden input never echoes the key


def test_tokens_ignores_non_instruction_files() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Setup: Create valid files AND a file that should be ignored (README.md)
        Path("CLAUDE.md").write_text("this is a test prompt\n")
        Path("README.md").write_text("this file should not be counted\n")

        # Execution
        result = runner.invoke(cli, ["tokens", "."])

        # Verification
        assert result.exit_code == 0
        assert "CLAUDE.md:" in result.output
        assert "README.md:" not in result.output
