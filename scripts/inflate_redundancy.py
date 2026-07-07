#!/usr/bin/env python3
"""Inflate a prompt with LLM-generated redundant restatements, verified against the 99% similarity gate.

Requires:
- OPENAI_API_KEY in the environment (see .env.example)
"""

import argparse
import sys
from typing import TYPE_CHECKING

import tiktoken

from alexandria import compare
from alexandria.utils.embedders import default_embedder

if TYPE_CHECKING:
    from collections.abc import Callable

    from alexandria.ir.contracts import Embedder

type Generate = Callable[[str], str]

MIN_SIMILARITY = 0.99  # mirrors 1 - Params.drift_budget, the library's 99% gate

PROMPT_VERSION = "inflate-v1"
_INFLATE_INSTRUCTIONS = """\
Rewrite the prompt below to roughly {factor}x its current length by adding redundancy only: restate
and duplicate its existing instructions, and add filler preambles and summaries. Every sentence you
add must already be entailed by the prompt — introduce no negation, comparator, number, tool name,
or permission that is absent from it. Output only the rewritten prompt.

{prompt}"""
_EXPAND_INSTRUCTIONS = """\
The prompt below is a redundantly restated version of an original. Append more redundant
restatements and summaries of its existing instructions, under the same rule: introduce no negation,
comparator, number, tool name, or permission that is absent from it. Output only the full expanded prompt.

{text}"""


def build_generate(model: str) -> Generate:
    """Return a Generate backed by the OpenAI Responses API (reads OPENAI_API_KEY from the environment)."""
    from openai import OpenAI

    client = OpenAI()

    def generate(prompt: str) -> str:
        return client.responses.create(model=model, input=prompt).output_text

    return generate


def inflate(
    prompt: str,
    factor: float,
    generate: Generate,
    embedder: Embedder,
    encoding: tiktoken.Encoding,
    max_attempts: int = 3,
) -> str:
    """Return an ~factor-times-longer prompt of redundant restatements, gated at MIN_SIMILARITY."""
    target_tokens = factor * len(encoding.encode(prompt))
    for _ in range(max_attempts):
        inflated = generate(_INFLATE_INSTRUCTIONS.format(factor=factor, prompt=prompt))
        while len(encoding.encode(inflated)) < target_tokens:
            inflated = generate(_EXPAND_INSTRUCTIONS.format(text=inflated))
        if compare(prompt, inflated, embedder).similarity >= MIN_SIMILARITY:
            return inflated
    raise RuntimeError(f"inflated text stayed below {MIN_SIMILARITY} similarity after {max_attempts} attempts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", type=float, required=True, help="target token ratio, e.g. 1.5")
    parser.add_argument("--model", default="gpt-5.4-mini", help="OpenAI model id")
    parser.add_argument("--input", type=argparse.FileType("r"), default=sys.stdin, help="prompt file (default stdin)")
    parser.add_argument(
        "--output", type=argparse.FileType("w"), default=sys.stdout, help="output file (default stdout)"
    )
    parser.add_argument("--max-attempts", type=int, default=3, help="regenerations before giving up")
    args = parser.parse_args()

    inflated = inflate(
        args.input.read(),
        args.factor,
        build_generate(args.model),
        default_embedder(),
        tiktoken.get_encoding("cl100k_base"),
        max_attempts=args.max_attempts,
    )
    args.output.write(inflated)


if __name__ == "__main__":
    main()
