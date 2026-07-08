#!/usr/bin/env python3
"""Inflate a prompt with LLM-generated redundant restatements, verified against a similarity gate.

Genuine paraphrase cannot clear the library's 0.99 fidelity gate: under chunk-pooled all-MiniLM-L6-v2
cosine, a faithful restatement lands around 0.95-0.98, and only near-verbatim copying reaches 0.99.
So inflation is gated at its own, looser MIN_SIMILARITY, which keeps the text genuinely reworded.

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

MIN_SIMILARITY = 0.95  # inflation's own gate; genuine restatement can't clear the library's 0.99 (MiniLM cosine)

PROMPT_VERSION = "inflate-v4"
_INFLATE_INSTRUCTIONS = """\
Restate the prompt below redundantly to make it roughly {factor}x its current length. Reword it in
your own words — do not copy sentences verbatim — while preserving its exact meaning and reusing its
key nouns, verbs, numbers, tool names, and markdown headers. Cover the same topics in the same order
and proportions, then add further independent restatements of the same content until the text reaches
about {factor}x the original length. Introduce no negation, comparator, number, tool name, or
permission that is absent from the prompt. Output only the restated prompt.
{feedback}
{prompt}"""
_EXPAND_INSTRUCTIONS = """\
The text below redundantly restates an original prompt. Append another independent restatement of the
same content in fresh wording, reusing the original's key terms and markdown headers and introducing
nothing new. Output only the full expanded text.

{text}"""
_FEEDBACK_HEADER = (
    "\nYour previous attempt drifted in meaning. Restate only what the prompt already says, reusing its key terms:\n"
)


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
    max_attempts: int = 2,
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
def main(factor: float, model: str, input_file: TextIO, output_file: TextIO) -> None:
    inflated = inflate(
        input_file.read(),
        factor,
        build_generate(model),
        default_embedder(),
        tiktoken.get_encoding("cl100k_base"),
    )
    output_file.write(inflated)


if __name__ == "__main__":
    main()
