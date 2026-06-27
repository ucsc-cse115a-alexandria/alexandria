"""The Alexandria CLI: reduce and score verbs. A thin wrapper over the library."""

from __future__ import annotations

import json
import sys
from typing import IO

import click

from alexandria.embedding import DEFAULT_MODEL, DETERMINISTIC, build_embedder
from alexandria.optimize import DEFAULT_OPTIMIZER, OptimizerParams
from alexandria.pipeline import reduce as reduce_prompt
from alexandria.pipeline import score_report
from alexandria.score import DEFAULT_SCORER

_DEFAULTS = OptimizerParams()
_MODEL_HELP = f"embedding model id, or {DETERMINISTIC!r}"


@click.group()
def cli() -> None:
    """Alexandria — label-free prompt optimization."""


@cli.command()
@click.argument("file", type=click.File("r"), default="-")
@click.option("--optimizer", "optimizers", default=DEFAULT_OPTIMIZER, help="comma-separated optimizer names")
@click.option("--threshold", type=float, default=_DEFAULTS.threshold, help="redundancy threshold")
@click.option(
    "--max-drift",
    type=float,
    default=_DEFAULTS.max_drift,
    help="max cosine drift from the original prompt allowed per deletion (2.0 = no limit)",
)
@click.option("--model", default=DEFAULT_MODEL, help=_MODEL_HELP)
def reduce(file: IO[str], optimizers: str, threshold: float, max_drift: float, model: str) -> None:
    """Reduce a prompt: prompt in, reduced prompt out."""
    names = tuple(n.strip() for n in optimizers.split(",") if n.strip())
    params = OptimizerParams(threshold=threshold, max_drift=max_drift)
    reduced = reduce_prompt(file.read(), build_embedder(model), optimizers=names, params=params)
    click.echo(reduced, nl=False)


@cli.command()
@click.argument("file", type=click.File("r"), default="-")
@click.option("--scorer", "scorers", default=DEFAULT_SCORER, help="comma-separated scorer names")
@click.option("--model", default=DEFAULT_MODEL, help=_MODEL_HELP)
@click.option("--json", "as_json", is_flag=True, help="emit JSON instead of a table")
def score(file: IO[str], scorers: str, model: str, as_json: bool) -> None:
    """Score a prompt: per-instruction scores out."""
    names = tuple(n.strip() for n in scorers.split(",") if n.strip())
    rows = score_report(file.read(), build_embedder(model), scorers=names)
    if as_json or not sys.stdout.isatty():
        click.echo(json.dumps(rows, indent=2))
    else:  # pragma: no cover
        for row in rows:
            click.echo("  ".join(f"{key}={value}" for key, value in row.items()))
