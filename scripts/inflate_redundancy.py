#!/usr/bin/env python3
"""Inflate a prompt with LLM-generated redundant restatements, verified against the 99% similarity gate.

Requires:
- OPENAI_API_KEY in the environment (see .env.example)
"""

from typing import TYPE_CHECKING

import click
import numpy as np
import tiktoken

from alexandria import compare, represent
from alexandria.ir.document import Section
from alexandria.ir.similarity import cosine_distance, normalize
from alexandria.utils.embedders import default_embedder

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TextIO

    from numpy.typing import NDArray

    from alexandria.ir.contracts import Embedder

type Generate = Callable[[str], str]

MIN_SIMILARITY = 0.99  # 1 - Params.drift_budget

PROMPT_VERSION = "inflate-v2"
_INFLATE_INSTRUCTIONS = """\
Rewrite the prompt below to roughly {factor}x its current length by adding redundancy only: restate
and duplicate its existing instructions, and add filler preambles and summaries. Every sentence you
add must already be entailed by the prompt — introduce no negation, comparator, number, tool name,
or permission that is absent from it. Keep every markdown header and XML tag exactly as in the
original: same text, same order, none added or removed. Output only the rewritten prompt.
{feedback}
{prompt}"""
_EXPAND_INSTRUCTIONS = """\
The prompt below is a redundantly restated version of an original. Append more redundant
restatements and summaries of its existing instructions, under the same rules: introduce no negation,
comparator, number, tool name, or permission that is absent from it, and keep every markdown header
and XML tag unchanged. Output only the full expanded prompt.

{text}"""
_FEEDBACK_HEADER = "\nYour previous attempt drifted in meaning. Fix the following, still adding redundancy only:\n"


def build_generate(model: str) -> Generate:
    """The production Generate: OpenAI Responses API, key from OPENAI_API_KEY."""
    from openai import OpenAI

    client = OpenAI()

    def generate(prompt: str) -> str:
        return client.responses.create(model=model, input=prompt).output_text

    return generate


def _flatten(sections: tuple[Section, ...]) -> list[Section]:
    flat: list[Section] = []
    for section in sections:
        flat.append(section)
        flat.extend(_flatten(tuple(child for child in section.children if isinstance(child, Section))))
    return flat


def _pooled(section: Section) -> NDArray[np.float32]:
    # not Section.embedding: one vector gets truncated at the model's 256-token window
    weights = np.array([sentence.token_count for sentence in section.sentences], dtype=np.float32)
    vectors = normalize(np.stack([sentence.embedding for sentence in section.sentences]))
    return (vectors * weights[:, None]).sum(axis=0) / weights.sum()


def section_feedback(original: str, inflated: str, embedder: Embedder) -> list[str]:
    """Notes for the next attempt: the sections that drifted, or the structure change itself."""
    original_sections = _flatten(represent(original, embedder).sections)
    inflated_sections = _flatten(represent(inflated, embedder).sections)
    if [(s.kind, s.header) for s in original_sections] != [(s.kind, s.header) for s in inflated_sections]:
        return ["the section structure changed: keep every markdown header and XML tag exactly as in the original"]
    return [
        f'section "{before.header or before.kind}" drifted (similarity {similarity:.3f}):'
        " restate only what it already says"
        for before, after in zip(original_sections, inflated_sections, strict=True)
        if (similarity := 1.0 - cosine_distance(_pooled(before), _pooled(after))) < MIN_SIMILARITY
    ]


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
    feedback = ""
    for _ in range(max_attempts):
        inflated = generate(_INFLATE_INSTRUCTIONS.format(factor=factor, feedback=feedback, prompt=prompt))
        while len(encoding.encode(inflated)) < target_tokens:
            inflated = generate(_EXPAND_INSTRUCTIONS.format(text=inflated))
        if compare(prompt, inflated, embedder).similarity >= MIN_SIMILARITY:
            return inflated
        notes = section_feedback(prompt, inflated, embedder)
        feedback = _FEEDBACK_HEADER + "".join(f"- {note}\n" for note in notes) if notes else ""
    raise RuntimeError(f"inflated text stayed below {MIN_SIMILARITY} similarity after {max_attempts} attempts")


@click.command(help=__doc__)
@click.option("--factor", type=float, required=True, help="Target token ratio, e.g. 1.5.")
@click.option("--model", default="gpt-5.4-mini", show_default=True, help="OpenAI model id.")
@click.option("--input", "input_file", type=click.File("r"), default="-", help="Prompt file (default stdin).")
@click.option("--output", "output_file", type=click.File("w"), default="-", help="Output file (default stdout).")
@click.option("--max-attempts", type=int, default=3, show_default=True, help="Regenerations before giving up.")
def main(factor: float, model: str, input_file: TextIO, output_file: TextIO, max_attempts: int) -> None:
    inflated = inflate(
        input_file.read(),
        factor,
        build_generate(model),
        default_embedder(),
        tiktoken.get_encoding("cl100k_base"),
        max_attempts=max_attempts,
    )
    output_file.write(inflated)


if __name__ == "__main__":
    main()
