"""Sentence merger implementations — the only place the merge LLM is constructed."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from alexandria.utils.config import require_api_key

if TYPE_CHECKING:
    from alexandria.ir.contracts import SentenceMerger

MERGE_MODEL = "gpt-5.6-luna"
_INSTRUCTIONS = (
    "You merge two overlapping prompt instructions. Rewrite them as ONE instruction that "
    "preserves the full meaning of both in as few tokens as possible. "
    "Reply with only the rewritten instruction, no explanations."
)


class OpenAIMerger:
    def __init__(self, api_key: str | None = None) -> None:  # pragma: no cover
        from openai import OpenAI

        self._client = OpenAI(api_key=require_api_key(api_key))

    def merge(self, first: str, second: str, feedback: str | None = None) -> str:  # pragma: no cover
        from openai import OpenAIError

        prompt = f"Instruction 1: {first.strip()}\nInstruction 2: {second.strip()}"
        if feedback is not None:
            prompt += f"\n\n{feedback}"
        try:
            response = self._client.responses.create(
                model=MERGE_MODEL,
                reasoning={"effort": "medium"},
                instructions=_INSTRUCTIONS,
                input=prompt,
            )
        except OpenAIError as error:
            # cli/ops may not import openai (import-linter), so the boundary error type is ValueError.
            raise ValueError(f"OpenAI merge request failed: {error}") from error
        return response.output_text.strip()


@lru_cache(maxsize=2)
def default_merger(api_key: str | None = None) -> SentenceMerger:  # pragma: no cover
    """The process-wide default merger, built lazily on first use."""
    return OpenAIMerger(api_key)
