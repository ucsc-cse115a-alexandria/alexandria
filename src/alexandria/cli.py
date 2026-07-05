from __future__ import annotations

import json
from contextlib import contextmanager
from typing import TYPE_CHECKING

import click
from pydantic import ValidationError

from alexandria.ir.contracts import Params
from alexandria.ir.envelope import DocumentEnvelope, PlanEnvelope, ScoredEnvelope
from alexandria.phases.optimize import DEFAULT_OPTIMIZER
from alexandria.phases.optimize import optimize as optimize_phase
from alexandria.phases.represent import represent as represent_phase
from alexandria.phases.score import DEFAULT_SCORER
from alexandria.phases.score import score as score_phase
from alexandria.phases.select import DEFAULT_SELECTOR
from alexandria.phases.select import select as select_phase
from alexandria.runtime.embedding import DEFAULT_MODEL, DETERMINISTIC, build_embedder
from alexandria.runtime.pipeline import reduce as reduce_prompt
from alexandria.runtime.pipeline import score_rows

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import IO

    from alexandria.ir.contracts import Plan

_DEFAULTS = Params()
_MODEL_HELP = f"embedding model id, or {DETERMINISTIC!r}"


@contextmanager
def _clean_errors() -> Generator[None]:
    """Turn expected boundary failures (bad envelope JSON, unknown names, empty prompt) into a
    clean CLI error and exit code instead of a traceback."""
    try:
        yield
    except ValidationError as error:
        raise click.ClickException(f"invalid input: {error}") from error
    except ValueError as error:
        raise click.ClickException(str(error)) from error


def _names(raw: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in raw.split(",") if name.strip())


def _reduction_json(text: str, applied: Plan, source_tokens: int, reduced_tokens: int) -> str:
    payload = {
        "text": text,
        "applied": [candidate.model_dump(mode="json") for candidate in applied],
        "source_tokens": source_tokens,
        "reduced_tokens": reduced_tokens,
    }
    return json.dumps(payload, indent=2)


@click.group()
def cli() -> None:
    """Alexandria — label-free prompt optimization."""


@cli.command(name="represent")
@click.argument("file", type=click.File("r"), default="-")
@click.option("--model", default=DEFAULT_MODEL, help=_MODEL_HELP)
def represent_cmd(file: IO[str], model: str) -> None:
    """Raw prompt in, a DocumentEnvelope (JSON) out."""
    with _clean_errors():
        document = represent_phase(file.read(), build_embedder(model))
    click.echo(DocumentEnvelope(document=document).model_dump_json())


@cli.command(name="score")
@click.argument("file", type=click.File("r"), default="-")
@click.option("--scorer", "scorers", default=DEFAULT_SCORER, help="comma-separated scorer names")
@click.option("--table", "as_table", is_flag=True, help="print a human-readable report instead of a ScoredEnvelope")
def score_cmd(file: IO[str], scorers: str, as_table: bool) -> None:
    """DocumentEnvelope in, a ScoredEnvelope (JSON) out, or a report with --table."""
    names = _names(scorers)
    with _clean_errors():
        document = DocumentEnvelope.model_validate_json(file.read()).document
        bundle = score_phase(document, names=names)
    if as_table:
        for row in score_rows(document, bundle, names):
            click.echo("  ".join(f"{key}={value}" for key, value in row.items()))
    else:
        click.echo(ScoredEnvelope(document=document, scores=bundle).model_dump_json())


@cli.command(name="optimize")
@click.argument("file", type=click.File("r"), default="-")
@click.option("--optimizer", "optimizers", default=DEFAULT_OPTIMIZER, help="comma-separated optimizer names")
@click.option("--threshold", type=float, default=_DEFAULTS.threshold, help="redundancy threshold")
def optimize_cmd(file: IO[str], optimizers: str, threshold: float) -> None:
    """ScoredEnvelope in, a PlanEnvelope (JSON) out."""
    names = _names(optimizers)
    with _clean_errors():
        scored = ScoredEnvelope.model_validate_json(file.read())
        plan = optimize_phase(scored.document, scored.scores, names=names, params=Params(threshold=threshold))
    click.echo(PlanEnvelope(document=scored.document, plan=plan).model_dump_json())


@cli.command(name="select")
@click.argument("file", type=click.File("r"), default="-")
@click.option("--model", default=DEFAULT_MODEL, help=_MODEL_HELP)
@click.option(
    "--drift-budget",
    type=float,
    default=_DEFAULTS.drift_budget,
    help="max cumulative cosine drift from the original prompt the reduction may accept (0.01 = 1%)",
)
@click.option("--json", "as_json", is_flag=True, help="emit a JSON reduction summary instead of the reduced text")
def select_cmd(file: IO[str], model: str, drift_budget: float, as_json: bool) -> None:
    """PlanEnvelope in, the reduced prompt out (or a JSON reduction summary with --json)."""
    with _clean_errors():
        envelope = PlanEnvelope.model_validate_json(file.read())
        selection = select_phase(
            envelope.document, envelope.plan, build_embedder(model), params=Params(drift_budget=drift_budget)
        )
    if as_json:
        click.echo(
            _reduction_json(
                selection.document.text,
                selection.applied,
                envelope.document.token_count,
                selection.document.token_count,
            )
        )
    else:
        click.echo(selection.document.text, nl=False)


@cli.command(name="reduce")
@click.argument("file", type=click.File("r"), default="-")
@click.option("--optimizer", "optimizers", default=DEFAULT_OPTIMIZER, help="comma-separated optimizer names")
@click.option("--selector", default=DEFAULT_SELECTOR, help="selector name")
@click.option("--threshold", type=float, default=_DEFAULTS.threshold, help="redundancy threshold")
@click.option(
    "--drift-budget",
    type=float,
    default=_DEFAULTS.drift_budget,
    help="max cumulative cosine drift from the original prompt the reduction may accept (0.01 = 1%)",
)
@click.option("--model", default=DEFAULT_MODEL, help=_MODEL_HELP)
@click.option("--json", "as_json", is_flag=True, help="emit a JSON reduction summary instead of the reduced text")
def reduce_cmd(
    file: IO[str], optimizers: str, selector: str, threshold: float, drift_budget: float, model: str, as_json: bool
) -> None:
    """Reduce a prompt end to end: prompt in, reduced prompt out (or a JSON summary with --json)."""
    names = _names(optimizers)
    params = Params(threshold=threshold, drift_budget=drift_budget)
    with _clean_errors():
        result = reduce_prompt(file.read(), build_embedder(model), optimizers=names, selector=selector, params=params)
    if as_json:
        click.echo(_reduction_json(result.text, result.applied, result.source_tokens, result.reduced_tokens))
    else:
        click.echo(result.text, nl=False)
