"""Sentence merger implementations — the only place the merge LLM is constructed."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from alexandria.utils.config import require_api_key

if TYPE_CHECKING:
    from alexandria.ir.contracts import SentenceMerger

MERGE_MODEL = "gpt-5.6-luna"
TARGET_CANDIDATES_PER_CALL = 2
_INSTRUCTIONS = (
    "You merge two overlapping prompt instructions. Rewrite them as ONE instruction that "
    "preserves the full meaning of both in as few tokens as possible. "
    "Reply with only the rewritten instruction, no explanations."
)
_TARGET_INSTRUCTIONS = (
    f"Generate exactly {TARGET_CANDIDATES_PER_CALL} controlled variants for a token-constrained semantic search. "
    "On the first round, create "
    "diverse compressions from the source. On later rounds, mutate the single current base; do not restart or "
    "summarize the source from scratch. Copy the base nearly verbatim and make only the localized "
    f"substitution requested by the feedback. Candidate N must use feedback excerpt N, so the "
    f"{TARGET_CANDIDATES_PER_CALL} outputs explore {TARGET_CANDIDATES_PER_CALL} "
    "different local edits. Preserve all untouched base wording exactly. Remove redundant wording wherever needed "
    "to stay at or below the requested hard token limit. Do not add "
    "XML tags, Markdown headings, wrappers, explanations, or code fences. Return only the structured output."
)


class TargetMergeCandidates(BaseModel):
    """Structured output for one target-search generation round."""

    model_config = ConfigDict(frozen=True)
    candidates: list[str] = Field(
        min_length=TARGET_CANDIDATES_PER_CALL,
        max_length=TARGET_CANDIDATES_PER_CALL,
        description=(
            f"Exactly {TARGET_CANDIDATES_PER_CALL} distinct, meaning-preserving rewritten segments. "
            "Every item must fit the requested token range."
        ),
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

    def merge_candidates_to_target(  # pragma: no cover
        self,
        prompt: str,
        max_tokens: int,
        feedback: str | None = None,
        base_candidate: str | None = None,
    ) -> tuple[str, ...]:
        from openai import OpenAIError

        min_tokens = max(1, int(max_tokens * 0.95))
        request = (
            f"Merge and rewrite the content segment below to between {min_tokens} and {max_tokens} tokens. "
            "The upper limit is mandatory and will be checked with cl100k_base. Prefer the top of the range when "
            "quality is equal. Dense paraphrasing is allowed; information deletion is not.\n\n"
            f"<source_segment>\n{prompt}\n</source_segment>"
        )
        if base_candidate is not None:
            request += (
                f"\n\nUse this as the single base for all {TARGET_CANDIDATES_PER_CALL} local mutations. "
                "Keep every strong part unless the "
                "feedback identifies a replacement that should improve semantic similarity:\n"
                f"<current_base>\n{base_candidate}\n</current_base>"
            )
        if feedback is not None:
            request += f"\n\nPrevious round feedback:\n{feedback}"
        try:
            response = self._client.responses.parse(
                model=MERGE_MODEL,
                reasoning={"effort": "low"},
                instructions=_TARGET_INSTRUCTIONS,
                input=request,
                text_format=TargetMergeCandidates,
            )
        except OpenAIError as error:
            raise ValueError(f"OpenAI target merge request failed: {error}") from error
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("OpenAI target merge request returned no structured output")
        return tuple(candidate.strip() for candidate in parsed.candidates)


@lru_cache(maxsize=2)
def default_merger(api_key: str | None = None) -> SentenceMerger:  # pragma: no cover
    """The process-wide default merger, built lazily on first use."""
    return OpenAIMerger(api_key)
