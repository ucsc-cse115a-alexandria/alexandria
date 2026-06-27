"""The Alexandria CLI: reduce and score verbs. A thin wrapper over the library."""

from __future__ import annotations

import json
import sys
from typing import IO

import click

from alexandria.embedding import build_embedder
from alexandria.pipeline import reduce as reduce_prompt
from alexandria.pipeline import score_prompt

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


@click.group()
def cli() -> None:
    """Alexandria — label-free prompt optimization."""


@cli.command()
@click.argument("file", type=click.File("r"), default="-")
@click.option("--optimizer", "optimizers", default="greedy_pairwise", help="comma-separated optimizer names")
@click.option("--threshold", type=float, default=0.85, help="redundancy threshold")
@click.option("--model", default=_DEFAULT_MODEL, help="embedding model id, or 'deterministic'")
def reduce(file: IO[str], optimizers: str, threshold: float, model: str) -> None:
    """Reduce a prompt: prompt in, reduced prompt out."""
    names = tuple(n.strip() for n in optimizers.split(",") if n.strip())
    reduced = reduce_prompt(file.read(), build_embedder(model), optimizers=names, threshold=threshold)
    click.echo(reduced, nl=False)


@cli.command()
@click.argument("file", type=click.File("r"), default="-")
@click.option("--scorer", "scorers", default="redundancy", help="comma-separated scorer names")
@click.option("--model", default=_DEFAULT_MODEL, help="embedding model id, or 'deterministic'")
@click.option("--json", "as_json", is_flag=True, help="emit JSON instead of a table")
def score(file: IO[str], scorers: str, model: str, as_json: bool) -> None:
    """Score a prompt: per-instruction scores out."""
    names = tuple(n.strip() for n in scorers.split(",") if n.strip())
    document, bundle = score_prompt(file.read(), build_embedder(model), scorers=names)
    rows: list[dict[str, object]] = []
    for i, sentence in enumerate(document.sentences):
        row: dict[str, object] = {"id": sentence.id, "text": sentence.text.strip()}
        for name in names:
            row[name] = round(bundle[name][i], 4)
        rows.append(row)
    if as_json or not sys.stdout.isatty():
        click.echo(json.dumps(rows, indent=2))
    else:  # pragma: no cover
        for row in rows:
            joined = "  ".join(f"{name}={row[name]}" for name in names)
            click.echo(f"{row['id']}  {joined}  {row['text']}")
