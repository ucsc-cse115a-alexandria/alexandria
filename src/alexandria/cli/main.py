from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import click
from pydantic import ValidationError

from alexandria.cli.browser_review import run_browser_review
from alexandria.cli.envelope import DocumentEnvelope, PlanEnvelope, ScoredEnvelope
from alexandria.cli.interactive import apply_candidates, review
from alexandria.ir.contracts import MergeMetrics, Params, TrackedMerger
from alexandria.ops import (
    OptimizationReport,
    compare_reports,
    count_tokens,
    default_embedder,
    default_merger,
    optimization_report,
    write_config_key,
)
from alexandria.ops.features.compare import compare
from alexandria.ops.features.optimize import optimize
from alexandria.ops.features.represent import represent
from alexandria.ops.features.score import DEFAULT_SCORER, score, score_rows
from alexandria.ops.features.select import select
from alexandria.ops.pipe import ReduceResult, propose, reduce

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import IO

    from alexandria.ir.contracts import Candidate, Embedder, Plan, SentenceMerger

_DEFAULTS = Params()

_DRIFT_HELP = "max cumulative cosine drift from the original prompt the reduction may accept (0.01 = 1%)"


@contextmanager
def _clean_errors() -> Generator[None]:
    """Turn expected boundary failures (missing API key, bad envelope JSON, empty prompt) into a
    clean CLI error and exit code instead of a traceback."""
    try:
        yield
    except ValidationError as error:
        raise click.ClickException(f"invalid input: {error}") from error
    except ValueError as error:
        raise click.ClickException(str(error)) from error


_out_option = click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="also save this phase's envelope JSON to PATH (stdout is unchanged)",
)


def _emit_envelope(payload: str, out_path: Path | None) -> None:
    """Echo the phase envelope to stdout and, when --out is given, also save it to that path."""
    if out_path is not None:
        out_path.write_text(payload + "\n")
    click.echo(payload)


def _reduction_json(
    text: str,
    applied: Plan,
    source_tokens: int,
    reduced_tokens: int,
    merge_metrics: MergeMetrics | None = None,
) -> str:
    reduction_pct = 1.0 - (reduced_tokens / source_tokens) if source_tokens > 0 else 0.0
    payload = {
        "text": text,
        "applied": [candidate.model_dump(mode="json") for candidate in applied],
        "source_tokens": source_tokens,
        "reduced_tokens": reduced_tokens,
        "reduction_pct": reduction_pct,
        "merge_metrics": (merge_metrics or MergeMetrics()).model_dump(),
    }
    return json.dumps(payload, indent=2)


def _interactive_reduce(prompt: str, embedder: Embedder, merger: SentenceMerger, params: Params) -> ReduceResult:
    """Propose edits, review them in the terminal, and apply exactly the accepted ones."""
    proposal = propose(prompt, embedder, merger, params=params)
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


def _browser_reduce(
    prompt: str, embedder: Embedder, merger: SentenceMerger, params: Params, *, open_browser: bool
) -> ReduceResult:
    """Propose edits, review them in a browser, and apply exactly the accepted ones."""
    proposal = propose(prompt, embedder, merger, params=params)
    accepted: tuple[Candidate, ...] = ()
    if not proposal.diffs:
        click.echo("no proposed edits; the prompt is unchanged", err=True)
    else:
        chosen = run_browser_review(proposal, open_browser=open_browser)
        if chosen is None:
            click.echo("review aborted; the prompt is unchanged", err=True)
        else:
            accepted = chosen
    document = apply_candidates(proposal.document, accepted)
    return ReduceResult(document=document, source=proposal.document, applied=accepted)


@click.group()
def cli() -> None:
    """Alexandria — label-free prompt optimization.

    \b
    Quick start:
      alexandria config set openai-api-key         # store your OpenAI API key once
      alexandria reduce prompt.md                  # reduce automatically
      alexandria reduce --interactive prompt.md    # review each edit, apply only what you accept
      alexandria reduce --browser prompt.md        # review each edit in a browser, apply only what you accept
      alexandria compare original.md reduced.md    # check similarity and token reduction

    \b
    The phase verbs pipe into each other for step-by-step runs:
      cat prompt.md | alexandria represent | alexandria score | alexandria optimize | alexandria select
    """


@cli.command(name="represent")
@click.argument("file", type=click.File("r"), default="-")
@_out_option
def represent_cmd(file: IO[str], out_path: Path | None) -> None:
    """Raw prompt in, a DocumentEnvelope (JSON) out."""
    with _clean_errors():
        embedder = default_embedder()
        document = represent(file.read(), embedder)
        payload = DocumentEnvelope(document=document).model_dump_json()
    _emit_envelope(payload, out_path)


@cli.command(name="score")
@click.argument("file", type=click.File("r"), default="-")
@click.option("--table", "as_table", is_flag=True, help="print a human-readable report instead of a ScoredEnvelope")
@_out_option
def score_cmd(file: IO[str], as_table: bool, out_path: Path | None) -> None:
    """DocumentEnvelope in, a ScoredEnvelope (JSON) out, or a report with --table."""
    names = (DEFAULT_SCORER,)
    with _clean_errors():
        document = DocumentEnvelope.model_validate_json(file.read()).document
        bundle = score(document, names=names)
        payload = ScoredEnvelope(document=document, scores=bundle).model_dump_json()
    if out_path is not None:
        out_path.write_text(payload + "\n")
    if as_table:
        for row in score_rows(document, bundle, names):
            click.echo("  ".join(f"{key}={value}" for key, value in row.items()))
    else:
        click.echo(payload)


@cli.command(name="optimize")
@click.argument("file", type=click.File("r"), default="-")
@click.option("--drift-budget", type=float, default=_DEFAULTS.drift_budget, help=_DRIFT_HELP)
@_out_option
def optimize_cmd(file: IO[str], drift_budget: float, out_path: Path | None) -> None:
    """ScoredEnvelope in, a PlanEnvelope (JSON) out."""
    with _clean_errors():
        embedder = default_embedder()
        merger = default_merger()
        tracked_merger = TrackedMerger(merger)
        scored = ScoredEnvelope.model_validate_json(file.read())
        plan = optimize(
            scored.document,
            scored.scores,
            embedder,
            tracked_merger,
            params=Params(drift_budget=drift_budget),
        )
        payload = PlanEnvelope(
            document=scored.document,
            plan=plan,
            merge_metrics=tracked_merger.metrics(proposed_edits=len(plan), applied_edits=0),
        ).model_dump_json()
    _emit_envelope(payload, out_path)


@cli.command(name="select")
@click.argument("file", type=click.File("r"), default="-")
@click.option("--drift-budget", type=float, default=_DEFAULTS.drift_budget, help=_DRIFT_HELP)
@click.option("--json", "as_json", is_flag=True, help="emit a JSON reduction summary instead of the reduced text")
def select_cmd(file: IO[str], drift_budget: float, as_json: bool) -> None:
    """PlanEnvelope in, the reduced prompt out (or a JSON reduction summary with --json)."""
    with _clean_errors():
        embedder = default_embedder()
        envelope = PlanEnvelope.model_validate_json(file.read())
        selection = select(envelope.document, envelope.plan, embedder, params=Params(drift_budget=drift_budget))
    if as_json:
        click.echo(
            _reduction_json(
                selection.document.text,
                selection.applied,
                envelope.document.token_count,
                selection.document.token_count,
                envelope.merge_metrics.model_copy(update={"applied_edits": len(selection.applied)}),
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
@click.pass_context
def compare_cmd(ctx: click.Context, original: IO[str], edited: IO[str], min_similarity: float | None) -> None:
    """Compare two prompts: cosine similarity and token reduction as JSON (at most one FILE may be '-').

    This command talks in similarity where reduce talks in --drift-budget; the JSON carries both.
    """
    with _clean_errors():
        embedder = default_embedder()
        result = compare(original.read(), edited.read(), embedder)
    click.echo(result.model_dump_json(indent=2))
    if min_similarity is not None and result.similarity < min_similarity:
        ctx.exit(1)


@cli.command(name="reduce")
@click.argument("file", type=click.File("r"), default="-")
@click.option(
    "--save-tokens",
    type=int,
    default=None,
    help="reduce the prompt by N tokens (applied least-drift-first)",
)
@click.option(
    "--keep",
    type=click.FloatRange(min=0.0, max=100.0, min_open=True, max_open=True),
    default=None,
    help="keep P percent of the prompt's source tokens",
)
@click.option(
    "--target-reduction",
    type=click.FloatRange(min=0.0, max=100.0, max_open=True),
    default=None,
    help="require reducing the prompt by P percent; exit with an error if the target is not met",
)
@click.option("--drift-budget", type=float, default=_DEFAULTS.drift_budget, help=_DRIFT_HELP)
@click.option("--json", "as_json", is_flag=True, help="emit a JSON reduction summary instead of the reduced text")
@click.option(
    "--interactive", is_flag=True, help="review each proposed edit in the terminal and apply only the checked ones"
)
@click.option(
    "--browser", is_flag=True, help="review each proposed edit in a browser and apply only the accepted ones"
)
@click.option(
    "--no-open",
    is_flag=True,
    help="with --browser, print the review URL but do not open the default browser",
)
def reduce_cmd(
    file: IO[str],
    save_tokens: int | None,
    keep: float | None,
    target_reduction: float | None,
    drift_budget: float,
    as_json: bool,
    interactive: bool,
    browser: bool,
    no_open: bool,
) -> None:
    """Reduce a prompt end to end: prompt in, reduced prompt out (or a JSON summary with --json).

    \b
    Examples:
      alexandria reduce prompt.md                    # automatic: apply every edit within --drift-budget
      alexandria reduce --keep 95 prompt.md          # aim to keep 95% of the source tokens
      alexandria reduce prompt.md --save-tokens 200  # stop once 200 tokens are saved (least-drift-first)
      alexandria reduce prompt.md --json             # machine-readable summary of what was applied
      alexandria reduce --interactive prompt.md      # review each proposed edit yourself (FILE required)
      alexandria reduce --browser prompt.md          # review each proposed edit in a browser (FILE required)

    \b
    Interactive keys:
      up/down or k/j  move          enter/space  check or uncheck the edit
      s               show detail   a            toggle all
      d               done — apply the checked edits and exit
      q               quit without changing the prompt
    """
    if no_open and not browser:
        raise click.UsageError("--no-open requires --browser.")
    if interactive and browser:
        raise click.UsageError("--interactive and --browser are mutually exclusive.")
    budget_options = (keep, save_tokens, target_reduction)
    if sum(option is not None for option in budget_options) > 1:
        raise click.UsageError("--keep, --save-tokens, and --target-reduction are mutually exclusive.")

    manual_review = interactive or browser
    review_flag = "--interactive" if interactive else "--browser"
    if manual_review and getattr(file, "name", None) == "<stdin>":
        raise click.UsageError(f"{review_flag} needs a review UI, so FILE cannot be stdin; pass a file path.")
    if manual_review and any(option is not None for option in budget_options):
        raise click.UsageError(
            f"{review_flag} replaces the selector with your choices; "
            "drop --save-tokens, --keep, and --target-reduction."
        )

    with _clean_errors():
        embedder = default_embedder()
        merger = default_merger()
        prompt = file.read()
        if interactive:
            result = _interactive_reduce(prompt, embedder, merger, Params(drift_budget=drift_budget))
        elif browser:
            result = _browser_reduce(
                prompt, embedder, merger, Params(drift_budget=drift_budget), open_browser=not no_open
            )
        else:
            source_tokens = count_tokens(prompt)
            if keep is not None:
                max_tokens = math.floor(source_tokens * keep / 100)
            elif target_reduction is not None:
                # ``Params.max_tokens`` cannot be zero because a Document must retain at least one
                # sentence. The post-reduction check below still reports a 100%-ish target as unmet.
                max_tokens = max(math.floor(source_tokens * (100 - target_reduction) / 100), 1)
            else:
                max_tokens = max(source_tokens - save_tokens, 1) if save_tokens is not None else None
            params = Params(
                drift_budget=drift_budget,
                max_tokens=max_tokens,
                require_target=target_reduction is not None,
            )
            result = reduce(prompt, embedder, merger, params=params)

    if target_reduction is not None:
        actual_reduction = 1.0 - (result.reduced_tokens / result.source_tokens)
        if actual_reduction + 1e-12 < target_reduction / 100:
            raise click.ClickException(
                f"target reduction {target_reduction:g}% was not met: achieved {actual_reduction:.1%} "
                f"({result.source_tokens} -> {result.reduced_tokens} tokens). "
                "Increase --drift-budget or use a lower --target-reduction."
            )

    if as_json:
        click.echo(
            _reduction_json(
                result.text,
                result.applied,
                result.source_tokens,
                result.reduced_tokens,
                result.merge_metrics,
            )
        )
    else:
        reduction_pct = 1.0 - (result.reduced_tokens / result.source_tokens) if result.source_tokens > 0 else 0.0
        # Stats go to stderr so piping the reduced text on stdout stays clean.
        click.echo(
            f"Tokens: {result.source_tokens} -> {result.reduced_tokens} ({reduction_pct:.1%} reduction)", err=True
        )
        click.echo(
            f"Merge calls: {result.merge_metrics.calls} "
            f"({result.merge_metrics.retries} retries across {result.merge_metrics.pairs_attempted} pairs)",
            err=True,
        )
        click.echo(result.text, nl=False)


@cli.command(name="report")
@click.argument("file", type=click.File("r"), default="-")
@click.option("--drift-budget", type=float, default=_DEFAULTS.drift_budget, help=_DRIFT_HELP)
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
    drift_budget: float,
    baseline: IO[str] | None,
    quality_tolerance: float,
    token_tolerance: int,
) -> None:
    """Run optimization and emit token metrics plus quality scores as JSON."""
    params = Params(drift_budget=drift_budget)
    comparison = None
    with _clean_errors():
        embedder = default_embedder()
        merger = default_merger()
        report = optimization_report(file.read(), embedder, merger, params=params)
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


@cli.command(name="tokens")
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=".",
)
def tokens_cmd(directory: Path) -> None:
    """List token counts of instruction files in a directory."""
    total_tokens = 0
    target_names = {"CLAUDE.md", "AGENT.md"}

    for file_path in directory.rglob("*.md"):
        if file_path.name in target_names or "skills" in file_path.parts:
            count = count_tokens(file_path.read_text(encoding="utf-8"))
            click.echo(f"{file_path.name}: {count} tokens")
            total_tokens += count

    click.echo(f"{'-' * 20}\nTotal: {total_tokens} tokens")


@cli.group(name="config")
def config_group() -> None:
    """Manage Alexandria's stored configuration."""


@config_group.command(name="set")
@click.argument("field", type=click.Choice(["openai-api-key"]))
def config_set_cmd(field: str) -> None:
    """Prompt for a value with hidden input and save it to the config file."""
    value = click.prompt("OpenAI API key", hide_input=True)
    with _clean_errors():
        path = write_config_key(value)
    click.echo(f"saved {field} to {path}")
