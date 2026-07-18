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
_TARGET_INSTRUCTIONS = (
    "You compress one content segment from a larger prompt by merging and rewriting its contents. Preserve all "
    "information and representative original terminology within the requested token range. Do not over-compress: "
    "use nearly the full token budget to maximize semantic and embedding fidelity. Do not add XML tags, Markdown "
    "headings, wrappers, explanations, or code fences. Reply with only the merged replacement segment."
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

    def merge_to_target(self, prompt: str, max_tokens: int, feedback: str | None = None) -> str:  # pragma: no cover
        from openai import OpenAIError

        min_tokens = max(1, int(max_tokens * 0.95))
        request = (
            f"Merge and rewrite the content segment below to between {min_tokens} and {max_tokens} tokens. "
            "Use the full range: shorter is not better. The token range and semantic preservation are mandatory. "
            "Dense paraphrasing is allowed; information deletion is not.\n\n"
            f"<source_segment>\n{prompt}\n</source_segment>"
        )
        if feedback is not None:
            request += f"\n\nPrevious attempt feedback:\n{feedback}"
        try:
            response = self._client.responses.create(
                model=MERGE_MODEL,
                reasoning={"effort": "medium"},
                instructions=_TARGET_INSTRUCTIONS,
                input=request,
            )
        except OpenAIError as error:
            raise ValueError(f"OpenAI target merge request failed: {error}") from error
        return response.output_text.strip()


@lru_cache(maxsize=2)
def default_merger(api_key: str | None = None) -> SentenceMerger:  # pragma: no cover
    """The process-wide default merger, built lazily on first use."""
    return OpenAIMerger(api_key)
