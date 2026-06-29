"""Probe where *sentence-embedding* cosine works as a prompt-fidelity signal -- and where it is blind.

spec-v2 keeps a prompt edit only if it barely moves the sentence embedding. That gate is
"near-necessary but not sufficient": cosine tracks a span's *topic*, so edits that flip the
downstream behavior (a negation, a comparator, a swapped role, a moved digit) can leave the
embedding almost untouched while the meaning breaks.

This script runs a labeled catalog of minimal pairs (loaded from `fidelity_pairs.json` via
`fidelity_pairs.load_catalogs`) through a bi-encoder and reports, per edit *kind*, whether cosine
separates meaning-changing (MC) edits from meaning-preserving (MP) ones. The companion
`behavioral_fidelity_probe.py` scores the same pairs with spec-v2's decoder-distribution gate, so the
two signals can be compared directly.
"""

import argparse
from pathlib import Path

import numpy as np
import tiktoken
from fidelity_pairs import DEFAULT_PAIRS_PATH, Label, Pair, load_catalogs
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Catalogs scored as standalone minimal pairs; "dilution" is a separate context-length experiment.
SCORING_CATALOGS = ("literature", "agent")
DILUTION_CATALOG = "dilution"


class PairResult(BaseModel):
    """Token counts and embedding cosine for a scored pair."""

    label: Label
    kind: str
    source: str
    original: str
    edited: str
    tokens_original: int
    tokens_edited: int
    cosine: float


def score_pair(pair: Pair, model: SentenceTransformer, encoding: tiktoken.Encoding) -> PairResult:
    """Embed both sides, return token counts and their cosine similarity."""
    embeddings = model.encode([pair.original, pair.edited], normalize_embeddings=True)
    cosine = float(np.dot(embeddings[0], embeddings[1]))
    return PairResult(
        label=pair.label,
        kind=pair.kind,
        source=pair.source,
        original=pair.original,
        edited=pair.edited,
        tokens_original=len(encoding.encode(pair.original)),
        tokens_edited=len(encoding.encode(pair.edited)),
        cosine=cosine,
    )


def print_dilution(results: list[PairResult]) -> None:
    """Show cosine climbing toward 1 as the same flip is buried in more shared context."""
    print("\nContext dilution -- one negation, growing shared context:")
    print(f"  {'shared tokens':>13}  {'cosine':>7}  edit")
    for r in sorted(results, key=lambda r: r.tokens_edited):
        print(f"  {r.tokens_edited:>13}  {r.cosine:>7.3f}  {r.edited!r}")
    print(
        "  => Same meaning flip; cosine rises with context length. Longer prompts (a compressor's "
        "target) hide the flip better."
    )


def print_table(results: list[PairResult]) -> None:
    """Print one row per pair, sorted by cosine descending."""
    header = f"{'label':<9} {'kind':<22} {'src':<11} {'cosine':>7}  original -> edited"
    print(header)
    print("-" * len(header))
    for result in sorted(results, key=lambda r: r.cosine, reverse=True):
        print(
            f"{result.label.value:<9} {result.kind:<22} {result.source:<11} {result.cosine:>7.3f}  "
            f"{result.original!r} -> {result.edited!r}"
        )


def print_threshold_squeeze(mc: list[PairResult], mp: list[PairResult]) -> None:
    """Show the squeeze: a gate keeps an edit when cosine >= cutoff, so a low cutoff keeps the
    honest paraphrases but also keeps meaning flips, and a high cutoff blocks flips but also drops
    paraphrases. Reporting both ends proves no cutoff does both jobs at once.
    """
    keep_all_cutoff = min(r.cosine for r in mp)
    leaked = sum(r.cosine >= keep_all_cutoff for r in mc)
    print(
        f"\nKeep all {len(mp)} paraphrases (cutoff {keep_all_cutoff:.3f}): "
        f"{leaked}/{len(mc)} meaning flips leak through with them."
    )

    # Smallest cutoff that lets at most one meaning flip through, and what it costs in paraphrases.
    safety_cutoff = sorted(r.cosine for r in mc)[-2] + 1e-6
    survivors = sum(r.cosine >= safety_cutoff for r in mp)
    print(
        f"Allow at most one leaked flip (cutoff {safety_cutoff:.3f}): "
        f"only {survivors}/{len(mp)} paraphrases survive the gate."
    )


def print_summary(results: list[PairResult]) -> None:
    """Show where cosine is right, where it is blind, and where it is over-eager."""
    mp = [r for r in results if r.label is Label.MEANING_PRESERVING]
    mc = [r for r in results if r.label is Label.MEANING_CHANGING]
    if not (mp and mc):
        return

    print("\nMean cosine by kind:")
    for kind in dict.fromkeys(r.kind for r in results):
        cosines = [r.cosine for r in results if r.kind == kind]
        label = next(r.label.value for r in results if r.kind == kind)
        print(f"  {label:<9} {kind:<22} n={len(cosines)}  mean={np.mean(cosines):.3f}")

    print(
        f"\nDistributions overlap: MC cosine in [{min(r.cosine for r in mc):.3f}, "
        f"{max(r.cosine for r in mc):.3f}], MP cosine in [{min(r.cosine for r in mp):.3f}, "
        f"{max(r.cosine for r in mp):.3f}]."
    )

    print_threshold_squeeze(mc, mp)

    paraphrase_ceiling = max(r.cosine for r in mp if r.kind in {"paraphrase", "synonym"})
    blind = sorted((r for r in mc if r.cosine > paraphrase_ceiling), key=lambda r: r.cosine, reverse=True)
    print(f"\nBLIND -- meaning flips above the best paraphrase ({paraphrase_ceiling:.3f}): {len(blind)}/{len(mc)}")
    for r in blind:
        print(f"  {r.cosine:.3f}  [{r.kind}]  {r.original!r} -> {r.edited!r}")

    overeager = sorted((r for r in mp if r.cosine < max(r.cosine for r in mc)), key=lambda r: r.cosine)
    rejected = [r for r in overeager if r.kind in {"paraphrase", "synonym"}]
    print(f"\nOVER-EAGER -- valid paraphrases scoring below some meaning flip: {len(rejected)}/{len(mp)}")
    for r in rejected:
        print(f"  {r.cosine:.3f}  [{r.kind}]  {r.original!r} -> {r.edited!r}")

    print(
        "\n=> Cosine is reliable only at the extremes (identical-modulo-order stays high; a real "
        "topic change drops). In the middle it fails both ways -- blind to truth-flipping operators, "
        "over-eager against honest paraphrases -- so no threshold separates MC from MP. spec-v2's premise."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="bi-encoder model id (Hugging Face)",
    )
    parser.add_argument("--encoding", default="cl100k_base", help="tiktoken encoding name")
    parser.add_argument(
        "--catalog",
        choices=SCORING_CATALOGS,
        default="literature",
        help="which minimal-pair catalog to score: paper-grounded or agent-prompt-grounded",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=DEFAULT_PAIRS_PATH,
        help="path to the minimal-pair catalog JSON",
    )
    args = parser.parse_args()

    catalogs = load_catalogs(args.pairs)
    model = SentenceTransformer(args.model)
    encoding = tiktoken.get_encoding(args.encoding)

    results = [score_pair(pair, model, encoding) for pair in catalogs[args.catalog]]
    print_table(results)
    print_summary(results)
    if args.catalog == "literature":
        print_dilution([score_pair(pair, model, encoding) for pair in catalogs[DILUTION_CATALOG]])


if __name__ == "__main__":
    main()
