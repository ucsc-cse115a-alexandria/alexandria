from __future__ import annotations

import json
from typing import TYPE_CHECKING

from click.testing import CliRunner

from alexandria.cli import cli
from alexandria.cli.envelope import DocumentEnvelope, PlanEnvelope
from alexandria.ir.contracts import Params
from alexandria.ir.registry import register_scorer
from alexandria.ops.features.represent import represent
from alexandria.ops.pipe import reduce
from alexandria.utils.embedders import HashEmbedder

if TYPE_CHECKING:
    from alexandria.ir.document import Document


def _constant_scorer(document: Document) -> list[float]:
    return [0.0 for _ in document.sentences]


register_scorer("cli_test_constant")(_constant_scorer)


def test_phase_verbs_pipe_to_the_same_text_as_reduce() -> None:
    runner = CliRunner()
    prompt = "keep one\nkeep one\nunique line\n"

    document = runner.invoke(cli, ["represent", "--model", "deterministic"], input=prompt)
    scored = runner.invoke(cli, ["score"], input=document.output)
    plan = runner.invoke(cli, ["optimize"], input=scored.output)
    reduced = runner.invoke(cli, ["select", "--model", "deterministic", "--drift-budget", "2.0"], input=plan.output)

    assert [document.exit_code, scored.exit_code, plan.exit_code, reduced.exit_code] == [0, 0, 0, 0]
    expected = reduce(prompt, HashEmbedder(), params=Params(drift_budget=2.0))
    assert reduced.output == expected.text


def test_select_json_reports_the_reduction_summary() -> None:
    runner = CliRunner()
    prompt = "keep one\nkeep one\nunique line\n"
    document = runner.invoke(cli, ["represent", "--model", "deterministic"], input=prompt)
    scored = runner.invoke(cli, ["score"], input=document.output)
    plan = runner.invoke(cli, ["optimize"], input=scored.output)

    result = runner.invoke(
        cli, ["select", "--model", "deterministic", "--drift-budget", "2.0", "--json"], input=plan.output
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["text"].count("keep one") == 1
    assert len(payload["applied"]) == 1
    assert payload["reduced_tokens"] < payload["source_tokens"]


def test_score_emits_a_scored_envelope_by_default() -> None:
    runner = CliRunner()
    document = DocumentEnvelope(document=represent("dup\ndup\n", HashEmbedder())).model_dump_json()

    result = runner.invoke(cli, ["score"], input=document)

    assert result.exit_code == 0
    assert '"scores"' in result.output
    assert '"redundancy"' in result.output


def test_score_table_prints_a_human_report() -> None:
    runner = CliRunner()
    document = DocumentEnvelope(document=represent("dup\ndup\n", HashEmbedder())).model_dump_json()

    result = runner.invoke(cli, ["score", "--table"], input=document)

    assert result.exit_code == 0
    assert "redundancy=" in result.output
    assert "most_similar_id=" in result.output


def test_select_reports_a_model_mismatch_cleanly() -> None:
    document = represent("a\nb\n", HashEmbedder(dim=32))  # model_id 'hash-32'
    envelope = PlanEnvelope(document=document, plan=()).model_dump_json()

    result = CliRunner().invoke(cli, ["select", "--model", "deterministic"], input=envelope)  # 'hash-64'

    assert result.exit_code == 1
    assert "does not match" in result.output


def test_score_rejects_a_malformed_envelope_cleanly() -> None:
    result = CliRunner().invoke(cli, ["score"], input="not an envelope")

    assert result.exit_code == 1
    assert "invalid input" in result.output


def test_optimize_rejects_an_unknown_schema_version_cleanly() -> None:
    scored: dict[str, object] = {"schema_version": 2, "document": {}, "scores": {}}

    result = CliRunner().invoke(cli, ["optimize"], input=json.dumps(scored))

    assert result.exit_code == 1
    assert "invalid input" in result.output


def test_optimize_reports_a_missing_required_scorer_cleanly() -> None:
    runner = CliRunner()
    document = DocumentEnvelope(document=represent("dup\ndup\n", HashEmbedder())).model_dump_json()
    scored = runner.invoke(cli, ["score", "--scorer", "cli_test_constant"], input=document)

    result = runner.invoke(cli, ["optimize"], input=scored.output)

    assert result.exit_code == 1
    assert "redundancy" in result.output
    assert "Traceback" not in result.output


def test_reduce_drops_a_duplicate_under_a_generous_budget() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli, ["reduce", "--model", "deterministic", "--drift-budget", "2.0"], input="keep one\nkeep one\nunique\n"
    )
    assert result.exit_code == 0
    assert result.output.count("keep one") == 1
    assert "unique" in result.output


def test_reduce_keeps_everything_under_the_default_budget() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["reduce", "--model", "deterministic"], input="keep one\nkeep one\nunique\n")
    assert result.exit_code == 0
    assert result.output.count("keep one") == 2


def test_reduce_json_reports_the_applied_edits_and_token_counts() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["reduce", "--model", "deterministic", "--drift-budget", "2.0", "--json"],
        input="keep one\nkeep one\nunique\n",
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["text"].count("keep one") == 1
    assert len(payload["applied"]) == 1
    assert payload["reduced_tokens"] < payload["source_tokens"]


def test_represent_rejects_an_empty_prompt_cleanly() -> None:
    result = CliRunner().invoke(cli, ["represent", "--model", "deterministic"], input="")

    assert result.exit_code == 1
    assert "empty prompt" in result.output
