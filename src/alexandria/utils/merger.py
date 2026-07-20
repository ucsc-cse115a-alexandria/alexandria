"""Sentence merger implementations — the only place the merge LLM is constructed."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import TYPE_CHECKING

from alexandria.utils.config import require_api_key

if TYPE_CHECKING:
    from alexandria.ir.contracts import SentenceMerger

MERGE_MODEL = "gpt-5.6-luna"
TARGET_CANDIDATES_PER_CALL = 3
_INSTRUCTIONS = (
    "You merge two overlapping prompt instructions. Rewrite them as ONE instruction that "
    "preserves the full meaning of both in as few tokens as possible. "
    "Reply with only the rewritten instruction, no explanations."
)

# One strategy line per candidate; the empty first line is the plain baseline. Diversity across
# these lines is what makes best-of-3 selection outperform a single generation.
_TARGET_STRATEGIES: tuple[str, ...] = (
    "",
    "Reuse the source's exact wording wherever possible; prefer deleting whole sentences over "
    "paraphrasing what you keep.",
    "Use dense paraphrasing to pack as much of the source's meaning as possible into the budget.",
)

_SENTENCE_FINAL_CHARS = (".", "!", "?", "`", '"')
# A cap-cut sentence ends at one of these markers; the whitespace is dropped, the punctuation kept.
_SENTENCE_ENDINGS = (". ", ".\n", "! ", "? ", ".)", ":\n")


def trim_to_last_sentence(text: str) -> str:
    """Trim a cap-cut completion back to its last complete sentence.

    Returns the stripped text unchanged if it already ends on sentence-final punctuation or if no
    interior sentence boundary is found.
    """
    stripped = text.strip()
    if stripped.endswith(_SENTENCE_FINAL_CHARS):
        return stripped
    cut = -1
    for ending in _SENTENCE_ENDINGS:
        index = stripped.rfind(ending)
        if index != -1:
            cut = max(cut, index + len(ending.rstrip()))
    if cut == -1:
        return stripped
    return stripped[:cut]


def _target_instructions(max_tokens: int, strategy: str) -> str:
    lines = [
        f"Compress the passage below to at most {max_tokens} cl100k_base tokens, preserving its "
        "meaning and every distinct rule and fact.",
        "Merge overlapping sentences rather than deleting information when possible.",
    ]
    if strategy:
        lines.append(strategy)
    lines.append(
        "Do not add XML tags, Markdown headings, wrappers, explanations, or code fences; "
        "return only the rewritten passage."
    )
    return " ".join(lines)


class OpenAIMerger:
    def __init__(self, api_key: str | None = None, *, model: str = MERGE_MODEL) -> None:  # pragma: no cover
        from openai import OpenAI

        self._client = OpenAI(api_key=require_api_key(api_key))
        self._model = model

    def merge(self, first: str, second: str, feedback: str | None = None) -> str:  # pragma: no cover
        from openai import OpenAIError

        prompt = f"Instruction 1: {first.strip()}\nInstruction 2: {second.strip()}"
        if feedback is not None:
            prompt += f"\n\n{feedback}"
        try:
            response = self._client.responses.create(
                model=self._model,
                reasoning={"effort": "medium"},
                instructions=_INSTRUCTIONS,
                input=prompt,
            )
        except OpenAIError as error:
            # cli/ops may not import openai (import-linter), so the boundary error type is ValueError.
            raise ValueError(f"OpenAI merge request failed: {error}") from error
        return response.output_text.strip()

    def merge_candidates_to_target(self, prompt: str, max_tokens: int) -> tuple[str, ...]:  # pragma: no cover
        source = prompt.strip()
        with ThreadPoolExecutor(max_workers=TARGET_CANDIDATES_PER_CALL) as executor:
            # Futures preserve strategy order; result() re-raises the first failure, so any failed
            # request fails the whole invocation.
            futures = [
                executor.submit(self._generate, source, max_tokens, strategy) for strategy in _TARGET_STRATEGIES
            ]
            return tuple(future.result() for future in futures)

    def _generate(self, source: str, max_tokens: int, strategy: str) -> str:  # pragma: no cover
        from openai import OpenAIError

        # max_tokens is a cl100k_base budget, but the model tokenizer is denser, so this cap makes a
        # completed response fit that budget by construction.
        cap = max(32, int(max_tokens / 1.1))
        try:
            response = self._client.responses.create(
                model=self._model,
                reasoning={"effort": "none"},
                max_output_tokens=cap,
                instructions=_target_instructions(max_tokens, strategy),
                input=source,
            )
        except OpenAIError as error:
            raise ValueError(f"OpenAI target merge request failed: {error}") from error
        text = response.output_text.strip()
        if response.status == "incomplete":
            return trim_to_last_sentence(text)
        return text


@lru_cache(maxsize=4)
def default_merger(api_key: str | None = None, *, model: str = MERGE_MODEL) -> SentenceMerger:  # pragma: no cover
    """The process-wide default merger, built lazily on first use."""
    return OpenAIMerger(api_key, model=model)
