"""Reduce a redundant prompt with the Alexandria library API.

Copy .env.example to .env, add your OpenAI API key, then run:
    uv run python examples/reduce_prompt.py
"""

from dotenv import load_dotenv

import alexandria

PROMPT = """\
# Rules
Always write tests for new code.
Prefer small functions.

# More rules
Always write tests for new code.
Keep functions small and focused.
"""


def main() -> None:
    load_dotenv()
    result = alexandria.reduce(PROMPT)
    print(f"tokens: {result.source_tokens} -> {result.reduced_tokens}")
    print(result.text)


if __name__ == "__main__":
    main()
