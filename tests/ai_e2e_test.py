"""Live end-to-end reductions of verbose sample prompts through the real OpenAI defaults.

These exercise the full `reduce` path — real embeddings and real merges — on the kind of bloated
prompt Alexandria targets: a consumer travel-bot system prompt and a dev-infra `AGENTS.md`. Both
fixtures repeat each rule several ways, so a working pipeline should merge the near-duplicates and
return a shorter prompt. Marked `ai`; skipped without an OpenAI key (see the root conftest).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import alexandria
from alexandria.ir.contracts import Params

_FIXTURES = Path(__file__).parent / "fixtures"
_VERBOSE_PROMPTS = ("travel_bot_prompt.txt", "dev_infra_agents.md")

# These prompts repeat rules by paraphrase, not verbatim, so text-embedding-3-small scores the
# near-duplicate pairs around 0.65-0.74 — below the conservative 0.85 default. Lowering the
# similarity floor is the same knob the CLI exposes as `--min-similarity`; it lets the merger fold
# the paraphrased repeats the way a user compressing a bloated prompt would.
_PARAMS = Params(threshold=0.65)


@pytest.mark.ai
@pytest.mark.parametrize("filename", _VERBOSE_PROMPTS)
def test_reduce_shortens_a_verbose_prompt(filename: str) -> None:
    prompt = (_FIXTURES / filename).read_text()

    result = alexandria.reduce(prompt, params=_PARAMS)

    assert result.text.strip(), "the reduced prompt should not be empty"
    assert result.applied, "a redundant prompt should yield at least one applied edit"
    assert 0 < result.reduced_tokens < result.source_tokens
