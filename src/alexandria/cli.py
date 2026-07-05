from __future__ import annotations

import json
import sys
from typing import IO

import click

from alexandria.core.protocols import Params
from alexandria.phases.optimize import DEFAULT_OPTIMIZER
from alexandria.phases.score import DEFAULT_SCORER
from alexandria.phases.select import DEFAULT_SELECTOR
from alexandria.runtime.embedding import DEFAULT_MODEL, DETERMINISTIC, build_embedder
from alexandria.runtime.pipeline import reduce as reduce_prompt
from alexandria.runtime.pipeline import score_report

_DEFAULTS = Params()
_MODEL_HELP = f"embedding model id, or {DETERMINISTIC!r}"


@click.group()
def cli() -> None:
    """Alexandria — label-free prompt optimization."""

@cli.command()
@click.argument("file", type=click.File("r"), default="-")
@click.option("--optimizer", "optimizers", default=DEFAULT_OPTIMIZER, help="comma-separated optimizer names")
@click.option("--selector", default=DEFAULT_SELECTOR, help="selector name")
@click.option("--threshold", type=float, default=_DEFAULTS.threshold, help="redundancy threshold")
@click.option(
    "--drift-budget",
    type=float,
    default=_DEFAULTS.drift_budget,
    help="max cosine drift from the original prompt the reduction may accumulate (0.01 = 1%)",
)
@click.option(
    "--min-similarity",
    type=float,
    default=None,
    help="Stop reduction before dropping below this similarity floor (e.g., 0.99). Mutually exclusive with --drift-budget.",
)
@click.option("--model", default=DEFAULT_MODEL, help=_MODEL_HELP)
def reduce(
    file: IO[str],
    optimizers: str,
    selector: str,
    threshold: float,
    drift_budget: float,
    min_similarity: float | None,
    model: str
) -> None:
    """Reduce a prompt: prompt in, reduced prompt out."""
    
    # NEW: Validate mutually exclusive options
    # We check if min_similarity was provided AND if drift_budget was altered from its default
    if min_similarity is not None and drift_budget != _DEFAULTS.drift_budget:
        raise click.UsageError("Options --min-similarity and --drift-budget are mutually exclusive.")

    names = tuple(n.strip() for n in optimizers.split(",") if n.strip())
    
    # Convert min-similarity to drift budget, or fall back to drift_budget
    final_drift_budget = (1.0 - min_similarity) if min_similarity is not None else drift_budget
    
    params = Params(threshold=threshold, drift_budget=final_drift_budget)
    reduced = reduce_prompt(file.read(), build_embedder(model), optimizers=names, selector=selector, params=params)
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
