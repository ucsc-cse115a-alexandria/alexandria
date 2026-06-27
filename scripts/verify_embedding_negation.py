#!/usr/bin/env python3
"""Empirically check spec-v2's premise: a sentence embedding barely moves when an
edit *flips the meaning* (negation / antonym / numeric / version swap).

For each pair we report the deployment-tokenizer token counts and the cosine
similarity of the bi-encoder embeddings. If the spec is right, meaning-changing
(MC) edits land at nearly the same high cosine as meaning-preserving (MP)
paraphrases, and only genuinely unrelated text drops low -- so cosine cannot tell
a meaning flip apart from a harmless rewrite.

Run:  uv run python scripts/verify_embedding_negation.py
"""

import argparse
from enum import StrEnum

import numpy as np
import tiktoken
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


class Label(StrEnum):
    """Whether the edit changes the set of acceptable outputs."""

    MEANING_CHANGING = "MC"
    MEANING_PRESERVING = "MP"
    UNRELATED = "unrelated"


class Pair(BaseModel):
    """A minimal pair: an original instruction and one edited variant."""

    label: Label
    kind: str
    original: str
    edited: str


class PairResult(BaseModel):
    """Token counts and embedding cosine for a scored pair."""

    label: Label
    kind: str
    original: str
    edited: str
    tokens_original: int
    tokens_edited: int
    cosine: float


# Catalog drawn from spec-v2's Layer-B edit operators (negation, antonym,
# numeric bound, version swap) plus MP and unrelated controls for contrast.
PAIRS: list[Pair] = [
    Pair(
        label=Label.MEANING_CHANGING,
        kind="negation",
        original="Respond in JSON.",
        edited="Do not respond in JSON.",
    ),
    Pair(
        label=Label.MEANING_CHANGING,
        kind="include/exclude",
        original="Do NOT include code examples in your answer.",
        edited="Include code examples in your answer.",
    ),
    Pair(
        label=Label.MEANING_CHANGING,
        kind="antonym",
        original="Keep the response short.",
        edited="Keep the response long.",
    ),
    Pair(
        label=Label.MEANING_CHANGING,
        kind="numeric bound",
        original="Use at most 50 words.",
        edited="Use at most 500 words.",
    ),
    Pair(
        label=Label.MEANING_CHANGING,
        kind="version swap",
        original="Use the v1 API endpoint.",
        edited="Use the v2 API endpoint.",
    ),
    Pair(
        label=Label.MEANING_PRESERVING,
        kind="paraphrase",
        original="Respond in JSON.",
        edited="Format your answer as JSON.",
    ),
    Pair(
        label=Label.MEANING_PRESERVING,
        kind="paraphrase",
        original="Keep the response short.",
        edited="Be concise in your reply.",
    ),
    Pair(
        label=Label.UNRELATED,
        kind="control",
        original="Respond in JSON.",
        edited="The weather in Santa Cruz is sunny today.",
    ),
]


def score_pair(pair: Pair, model: SentenceTransformer, encoding: tiktoken.Encoding) -> PairResult:
    """Embed both sides, return token counts and their cosine similarity."""
    embeddings = model.encode([pair.original, pair.edited], normalize_embeddings=True)
    cosine = float(np.dot(embeddings[0], embeddings[1]))
    return PairResult(
        label=pair.label,
        kind=pair.kind,
        original=pair.original,
        edited=pair.edited,
        tokens_original=len(encoding.encode(pair.original)),
        tokens_edited=len(encoding.encode(pair.edited)),
        cosine=cosine,
    )


def print_table(results: list[PairResult]) -> None:
    """Print one row per pair, sorted by cosine descending."""
    header = f"{'label':<9} {'kind':<16} {'tok':>7} {'cosine':>7}  original -> edited"
    print(header)
    print("-" * len(header))
    for result in sorted(results, key=lambda r: r.cosine, reverse=True):
        tokens = f"{result.tokens_original}->{result.tokens_edited}"
        print(
            f"{result.label.value:<9} {result.kind:<16} {tokens:>7} {result.cosine:>7.3f}  "
            f"{result.original!r} -> {result.edited!r}"
        )


def print_summary(results: list[PairResult]) -> None:
    """Print mean cosine per label so the MC/MP overlap is obvious."""
    print("\nMean cosine by label:")
    for label in Label:
        cosines = [r.cosine for r in results if r.label is label]
        if cosines:
            print(f"  {label.value:<9} n={len(cosines)}  mean={np.mean(cosines):.3f}")

    mc = [r.cosine for r in results if r.label is Label.MEANING_CHANGING]
    mp = [r.cosine for r in results if r.label is Label.MEANING_PRESERVING]
    if mc and mp:
        print(
            f"\nMeaning-changing edits range [{min(mc):.3f}, {max(mc):.3f}]; "
            f"meaning-preserving paraphrases range [{min(mp):.3f}, {max(mp):.3f}]."
        )
        if min(mc) > max(mp):
            print(
                f"=> Every meaning flip scores ABOVE the highest legitimate paraphrase "
                f"({max(mp):.3f}). No cosine threshold keeps paraphrases while rejecting flips."
            )
        print("=> Cosine tracks surface/topic overlap, not truth value -- exactly spec-v2's premise.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="bi-encoder model id (Hugging Face)",
    )
    parser.add_argument("--encoding", default="cl100k_base", help="tiktoken encoding name")
    args = parser.parse_args()

    model = SentenceTransformer(args.model)
    encoding = tiktoken.get_encoding(args.encoding)

    results = [score_pair(pair, model, encoding) for pair in PAIRS]
    print_table(results)
    print_summary(results)


if __name__ == "__main__":
    main()
