from __future__ import annotations

import json
from contextlib import contextmanager
from typing import TYPE_CHECKING

import click
from pydantic import ValidationError

from alexandria.cli.envelope import DocumentEnvelope, PlanEnvelope, ScoredEnvelope
from alexandria.ir.contracts import Params
from alexandria.ops import (
    DEFAULT_MODEL,
    DETERMINISTIC,
    OptimizationReport,
    build_embedder,
    compare_reports,
    optimization_report,
)
from alexandria.ops.features.optimize import DEFAULT_OPTIMIZER
from alexandria.ops.features.optimize import optimize as optimize_phase
from alexandria.ops.features.represent import represent as represent_phase
from alexandria.ops.features.score import DEFAULT_SCORER, score_rows
from alexandria.ops.features.score import score as score_phase
from alexandria.ops.features.select import DEFAULT_SELECTOR
from alexandria.ops.features.select import select as select_phase
from alexandria.ops.pipe import reduce as reduce_prompt

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


@cli.command(name="report")
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
@click.option(
    "--baseline",
    type=click.File("r"),
    help="compare against a previously generated report and exit non-zero on regression",
)
@click.option(
    "--quality-tolerance",
    type=click.FloatRange(min=0.0),
    default=0.0,
    show_default=True,
    help="allowed decrease in each quality score",
)
@click.option(
    "--token-tolerance",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="allowed increase in reduced token count",
)
def report_cmd(
    file: IO[str],
    optimizers: str,
    selector: str,
    threshold: float,
    drift_budget: float,
    model: str,
    baseline: IO[str] | None,
    quality_tolerance: float,
    token_tolerance: int,
) -> None:
    """Run optimization and emit token metrics plus quality scores as JSON."""
    names = _names(optimizers)
    params = Params(threshold=threshold, drift_budget=drift_budget)
    comparison = None
    with _clean_errors():
        report = optimization_report(
            file.read(),
            build_embedder(model),
            optimizers=names,
            selector=selector,
            params=params,
        )
        if baseline is not None:
            baseline_report = OptimizationReport.model_validate_json(baseline.read())
            comparison = compare_reports(
                report,
                baseline_report,
                quality_tolerance=quality_tolerance,
                token_tolerance=token_tolerance,
            )

    payload = report.model_dump(mode="json")
    if comparison is not None:
        payload["comparison"] = comparison.model_dump(mode="json")
    click.echo(json.dumps(payload, indent=2))

    if comparison is not None and not comparison.passed:
        for regression in comparison.regressions:
            relation = ">=" if regression.expected == "at_least" else "<="
            click.echo(
                f"regression: {regression.metric}={regression.current} "
                f"(expected {relation} {regression.baseline} with tolerance {regression.tolerance})",
                err=True,
            )
        raise click.exceptions.Exit(1)
