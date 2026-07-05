"""Reduce a redundant prompt with the Alexandria library API.

Run: uv run python examples/reduce_prompt.py
"""

import alexandria
from alexandria.ir.contracts import Params
from alexandria.utils.embedders import HashEmbedder

PROMPT = """\
# Rules
Always write tests for new code.
Prefer small functions.

# More rules
Always write tests for new code.
Keep functions small and focused.
"""


def main() -> None:
    # Real usage is one line with the default semantic model
    # (downloads all-MiniLM-L6-v2 on first use):
    #   result = alexandria.reduce(PROMPT)
    # This example stays offline: the deterministic hash embedder re-embeds
    # reduced text to an unrelated vector, so it needs a generous drift budget
    # to accept deletions (a semantic model works with the default budget).
    result = alexandria.reduce(PROMPT, HashEmbedder(), params=Params(drift_budget=2.0))
    print(f"tokens: {result.source_tokens} -> {result.reduced_tokens}")
    print(result.text)


if __name__ == "__main__":
    main()
