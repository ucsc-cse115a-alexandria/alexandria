from __future__ import annotations

from click.testing import CliRunner

from alexandria.cli import cli


def test_reduce_drops_a_duplicate() -> None:
    runner = CliRunner()
    # deterministic embeddings re-embed to unrelated vectors, so a generous budget is needed to delete.
    result = runner.invoke(
        cli,
        ["reduce", "--model", "deterministic", "--drift-budget", "2.0"],
        input="keep one\nkeep one\nunique line\n",
    )
    assert result.exit_code == 0
    assert result.output.count("keep one") == 1
    assert "unique line" in result.output


def test_reduce_keeps_everything_under_the_default_budget() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["reduce", "--model", "deterministic"], input="keep one\nkeep one\nunique line\n")
    assert result.exit_code == 0
    assert result.output.count("keep one") == 2


def test_score_emits_json() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["score", "--model", "deterministic", "--json"], input="dup\ndup\n")
    assert result.exit_code == 0
    assert '"redundancy"' in result.output
    assert '"most_similar_id"' in result.output
