from __future__ import annotations

import json
from contextlib import contextmanager
from typing import TYPE_CHECKING

import click
from pydantic import ValidationError

from alexandria.cli.envelope import DocumentEnvelope, PlanEnvelope, ScoredEnvelope
from alexandria.cli.interactive import apply_candidates, review
from alexandria.ir.contracts import Params
from alexandria.ops import DEFAULT_MODEL, DETERMINISTIC, build_embedder
from alexandria.ops.features.compare import compare
from alexandria.ops.features.optimize import DEFAULT_OPTIMIZER, optimize
from alexandria.ops.features.represent import represent
from alexandria.ops.features.score import DEFAULT_SCORER, score, score_rows
from alexandria.ops.features.select import DEFAULT_SELECTOR, select
from alexandria.ops.pipe import ReduceResult, propose, reduce

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import IO

    from alexandria.ir.contracts import Candidate, Plan

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


def _interactive_reduce(prompt: str, optimizers: tuple[str, ...], params: Params, model: str) -> ReduceResult:
    """Propose edits, review them in the terminal, and apply exactly the accepted ones."""
    proposal = propose(prompt, build_embedder(model), optimizers=optimizers, params=params)
    accepted: tuple[Candidate, ...] = ()
    if not proposal.diffs:
        click.echo("no proposed edits; the prompt is unchanged", err=True)
    else:
        chosen = review(proposal.document, proposal.diffs)
        if chosen is None:
            click.echo("review aborted; the prompt is unchanged", err=True)
        else:
            accepted = chosen
    document = apply_candidates(proposal.document, accepted)
    return ReduceResult(document=document, source=proposal.document, applied=accepted)


@click.group()
def cli() -> None:
    """Alexandria — label-free prompt optimization."""


@cli.command(name="represent")
@click.argument("file", type=click.File("r"), default="-")
@click.option("--model", default=DEFAULT_MODEL, help=_MODEL_HELP)
def represent_cmd(file: IO[str], model: str) -> None:
    """Raw prompt in, a DocumentEnvelope (JSON) out."""
    with _clean_errors():
        document = represent(file.read(), build_embedder(model))
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
        bundle = score(document, names=names)
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
        plan = optimize(scored.document, scored.scores, names=names, params=Params(threshold=threshold))
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
        selection = select(
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


@cli.command(name="compare")
@click.argument("original", type=click.File("r"))
@click.argument("edited", type=click.File("r"))
@click.option(
    "--min-similarity",
    type=float,
    default=None,
    help="turn the command into a gate: exit 1 when similarity falls below this (0.99 = the 99% fidelity gate)",
)
@click.option("--model", default=DEFAULT_MODEL, help=_MODEL_HELP)
@click.pass_context
def compare_cmd(
    ctx: click.Context, original: IO[str], edited: IO[str], min_similarity: float | None, model: str
) -> None:
    """Compare two prompts: cosine similarity and token reduction as JSON (at most one FILE may be '-').

    This command talks in similarity where reduce talks in --drift-budget; the JSON carries both.
    """
    with _clean_errors():
        result = compare(original.read(), edited.read(), build_embedder(model))
    click.echo(result.model_dump_json(indent=2))
    if min_similarity is not None and result.similarity < min_similarity:
        ctx.exit(1)


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
@click.option(
    "--min-similarity",
    type=float,
    default=None,
    help=(
        "Stop reduction before dropping below this similarity floor (e.g., 0.99). "
        "Mutually exclusive with --drift-budget."
    ),
)
@click.option("--model", default=DEFAULT_MODEL, help=_MODEL_HELP)
@click.option("--json", "as_json", is_flag=True, help="emit a JSON reduction summary instead of the reduced text")
@click.option(
    "--interactive", is_flag=True, help="review each proposed edit in the terminal and apply only the accepted ones"
)
def reduce_cmd(
    file: IO[str],
    optimizers: str,
    selector: str,
    threshold: float,
    drift_budget: float,
    min_similarity: float | None,
    model: str,
    as_json: bool,
    interactive: bool,
) -> None:
    """Reduce a prompt end to end: prompt in, reduced prompt out (or a JSON summary with --json)."""
    if interactive and getattr(file, "name", None) == "<stdin>":
        raise click.UsageError(
            "--interactive reads keys from the terminal, so FILE cannot be stdin; pass a file path."
        )
    if interactive and (
        min_similarity is not None or drift_budget != _DEFAULTS.drift_budget or selector != DEFAULT_SELECTOR
    ):
        raise click.UsageError(
            "--interactive replaces the selector with your choices; "
            "drop --selector, --drift-budget, and --min-similarity."
        )
    # NEW: Validate mutually exclusive options
    if min_similarity is not None and drift_budget != _DEFAULTS.drift_budget:
        raise click.UsageError("Options --min-similarity and --drift-budget are mutually exclusive.")

    names = _names(optimizers)

    # NEW: Convert min-similarity to drift budget, or fall back to drift_budget
    final_drift_budget = (1.0 - min_similarity) if min_similarity is not None else drift_budget

    params = Params(threshold=threshold, drift_budget=final_drift_budget)

    with _clean_errors():
        if interactive:
            result = _interactive_reduce(file.read(), names, params, model)
        else:
            result = reduce(file.read(), build_embedder(model), optimizers=names, selector=selector, params=params)

    if as_json:
        click.echo(_reduction_json(result.text, result.applied, result.source_tokens, result.reduced_tokens))
    else:
        click.echo(result.text, nl=False)
