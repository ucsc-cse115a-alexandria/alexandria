#!/usr/bin/env python3
"""Generate the redundancy-inflation evaluation dataset from the top-starred skill-corpus repos.

Runs ``inflate_redundancy.inflate`` over the top-N skills by GitHub stars in a local skill-corpus
checkout and writes paired flat files under ``{corpus_root}/inflate/`` with a provenance manifest.
Idempotent and resumable: an output file that already exists is not regenerated. Every committed
variant is gate-passed by construction; a variant that never clears the gate is skipped and recorded
as excluded in the manifest.

Requires:
- OPENAI_API_KEY in the environment (see .env.example)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import click
import tiktoken
from inflate_redundancy import PROMPT_VERSION, build_generate, inflate
from pydantic import BaseModel, ConfigDict, TypeAdapter

from alexandria import compare
from alexandria.utils.embedders import default_embedder

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from alexandria.ir.contracts import Embedder

    Generate = Callable[[str], str]

GENERATOR_VERSION = "inflate-dataset-v1"
ENCODING_NAME = "cl100k_base"
FACTORS: tuple[float, ...] = (1.2, 1.5, 2.0, 10.0)
DEFAULT_MODEL = "gpt-5.4-mini"  # matches inflate_redundancy's CLI default
# Skip pathologically large originals: a 10x inflation re-restates the whole prompt many times, so a
# huge source (e.g. a 48k-token doc) stalls the LLM on output-token limits. Such repos are backfilled.
DEFAULT_MAX_SOURCE_TOKENS = 20000


class InflateFn(Protocol):
    """The ``inflate_redundancy.inflate`` call shape, injected so tests can substitute a fake."""

    def __call__(
        self,
        prompt: str,
        factor: float,
        generate: Generate,
        embedder: Embedder,
        encoding: tiktoken.Encoding,
        max_attempts: int = ...,
    ) -> str: ...


class RowStatus(StrEnum):
    """Whether a variant file is present on disk (and therefore gate-passed) or was excluded."""

    INCLUDED = "included"
    EXCLUDED = "excluded"


class _RepoEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")
    full_name: str
    stars: int


class _DownloadedFile(BaseModel):
    model_config = ConfigDict(extra="ignore")
    repo: str
    local_path: str


class SkillSelection(BaseModel):
    """The one SKILL.md chosen to represent a repo: the largest by cl100k_base tokens."""

    model_config = ConfigDict(frozen=True)
    full_name: str
    stars: int
    source_path: str  # local_path within the corpus checkout
    dir_name: str  # owner__repo, the corpus directory name and the dataset filename stem
    tokens: int
    text: str


class ManifestRow(BaseModel):
    """One (repo, factor) variant's provenance and fidelity."""

    full_name: str
    stars: int
    source_path: str
    kind: str  # f"inflate-{factor:g}x"
    status: RowStatus
    similarity: float | None
    tokens_original: int
    tokens_edited: int | None


class Manifest(BaseModel):
    """The dataset provenance record written to ``inflate/manifest.json``."""

    generator_version: str
    model_id: str
    prompt_version: str
    encoding: str
    factors: list[float]
    generated_at: str
    rows: list[ManifestRow]


def _corpus_dir_name(local_path: str) -> str:
    """The owner__repo directory under ``corpus/`` — the dataset filename stem for this repo."""
    segments = [segment for segment in local_path.split("/") if segment]
    if segments and segments[0] == "corpus":
        segments = segments[1:]
    return segments[0] if segments else local_path


def factor_dir(factor: float) -> str:
    """The output subdirectory name for a factor (1.2 -> "1_2", 2.0 -> "2", 10.0 -> "10")."""
    return f"{factor:g}".replace(".", "_")


def select_skills(
    corpus_root: Path,
    limit: int,
    encoding: tiktoken.Encoding,
    max_source_tokens: int = DEFAULT_MAX_SOURCE_TOKENS,
) -> list[SkillSelection]:
    """The ``limit`` highest-starred repos with a non-empty SKILL.md at most ``max_source_tokens`` long.

    Per repo the largest qualifying SKILL.md is chosen; a repo whose only files exceed the cap is skipped
    and the next-starred repo backfills its slot.
    """
    repos = TypeAdapter(list[_RepoEntry]).validate_json((corpus_root / "data" / "skill_repos.json").read_bytes())
    manifest = json.loads((corpus_root / "data" / "download_manifest.json").read_text())
    files = TypeAdapter(list[_DownloadedFile]).validate_python(manifest["downloaded_files"])

    paths_by_repo: dict[str, list[str]] = {}
    for file in files:
        if Path(file.local_path).name == "SKILL.md":
            paths_by_repo.setdefault(file.repo, []).append(file.local_path)

    selections: list[SkillSelection] = []
    for repo in sorted(repos, key=lambda r: r.stars, reverse=True):
        if len(selections) >= limit:
            break
        best: SkillSelection | None = None
        for local_path in paths_by_repo.get(repo.full_name, []):
            path = corpus_root / local_path
            if not path.is_file():
                continue
            text = path.read_text()
            if not text.strip():
                continue
            tokens = len(encoding.encode(text))
            if tokens > max_source_tokens:
                continue
            if best is None or tokens > best.tokens:
                best = SkillSelection(
                    full_name=repo.full_name,
                    stars=repo.stars,
                    source_path=local_path,
                    dir_name=_corpus_dir_name(local_path),
                    tokens=tokens,
                    text=text,
                )
        if best is not None:
            selections.append(best)
    return selections


def _write_if_absent(path: Path, text: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _manifest_rows(
    inflate_root: Path,
    selections: Sequence[SkillSelection],
    embedder: Embedder,
    factors: Sequence[float],
) -> list[ManifestRow]:
    """Scan the written files and record each (repo, factor) as included (with fidelity) or excluded."""
    rows: list[ManifestRow] = []
    for selection in selections:
        for factor in factors:
            variant_path = inflate_root / factor_dir(factor) / f"{selection.dir_name}.md"
            kind = f"inflate-{factor:g}x"
            if variant_path.is_file():
                result = compare(selection.text, variant_path.read_text(), embedder)
                rows.append(
                    ManifestRow(
                        full_name=selection.full_name,
                        stars=selection.stars,
                        source_path=selection.source_path,
                        kind=kind,
                        status=RowStatus.INCLUDED,
                        similarity=result.similarity,
                        tokens_original=result.original_tokens,
                        tokens_edited=result.edited_tokens,
                    )
                )
            else:
                rows.append(
                    ManifestRow(
                        full_name=selection.full_name,
                        stars=selection.stars,
                        source_path=selection.source_path,
                        kind=kind,
                        status=RowStatus.EXCLUDED,
                        similarity=None,
                        tokens_original=selection.tokens,
                        tokens_edited=None,
                    )
                )
    return rows


def build_dataset(
    corpus_root: Path,
    selections: Sequence[SkillSelection],
    generate: Generate,
    embedder: Embedder,
    encoding: tiktoken.Encoding,
    *,
    model_id: str,
    factors: Sequence[float] = FACTORS,
    max_attempts: int = 2,
    inflate_fn: InflateFn = inflate,
) -> Manifest:
    """Write ``inflate/{1,<factor>}/<repo>.md`` for each selection (skipping existing) and the manifest."""
    inflate_root = corpus_root / "inflate"
    for selection in selections:
        _write_if_absent(inflate_root / "1" / f"{selection.dir_name}.md", selection.text)
        for factor in factors:
            variant_path = inflate_root / factor_dir(factor) / f"{selection.dir_name}.md"
            if variant_path.exists():
                continue
            try:
                variant = inflate_fn(selection.text, factor, generate, embedder, encoding, max_attempts=max_attempts)
            except Exception as exc:  # gate miss (RuntimeError) or a persistent API error: exclude, keep going
                click.echo(
                    f"excluded {selection.dir_name} @ {factor_dir(factor)}: {type(exc).__name__}: {exc}", err=True
                )
                continue  # no file written -> recorded as excluded when the manifest scans disk
            _write_if_absent(variant_path, variant)

    return Manifest(
        generator_version=GENERATOR_VERSION,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        encoding=ENCODING_NAME,
        factors=list(factors),
        generated_at=datetime.now(UTC).isoformat(),
        rows=_manifest_rows(inflate_root, selections, embedder, factors),
    )


@click.command(help=__doc__)
@click.option(
    "--corpus-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Local skill-corpus checkout.",
)
@click.option("--limit", type=int, default=20, show_default=True, help="Top-N repos by GitHub stars.")
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="OpenAI model id.")
@click.option("--max-attempts", type=int, default=2, show_default=True, help="inflate regenerations before excluding.")
@click.option(
    "--max-source-tokens",
    type=int,
    default=DEFAULT_MAX_SOURCE_TOKENS,
    show_default=True,
    help="Skip (and backfill) repos whose smallest usable SKILL.md exceeds this many cl100k_base tokens.",
)
def main(corpus_root: Path, limit: int, model: str, max_attempts: int, max_source_tokens: int) -> None:
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    selections = select_skills(corpus_root, limit, encoding, max_source_tokens)
    manifest = build_dataset(
        corpus_root,
        selections,
        build_generate(model),
        default_embedder(),
        encoding,
        model_id=model,
        max_attempts=max_attempts,
    )
    manifest_path = corpus_root / "inflate" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    included = sum(1 for row in manifest.rows if row.status is RowStatus.INCLUDED)
    click.echo(f"wrote {manifest_path}: {len(selections)} repos, {included}/{len(manifest.rows)} variants included")


if __name__ == "__main__":
    main()
